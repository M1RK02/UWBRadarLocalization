import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# Set GTX 1070 environment flags FIRST before TF loads
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


import numpy as np
import tensorflow as tf

# Add project root to sys.path so 'evaluation' and 'src' can be imported easily
sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_performance import MATCH_THRESHOLD, match_hungarian
from src.channels import ALL_RADARS, CHANNEL_SETS, format_radars, parse_radars
from src.data_loader import (
    calculate_energy_gate_from_files,
    calculate_p99_from_files,
    get_kfold_splits,
    list_windows,
    load_fold_data,
    load_seated_windows,
    make_dataset,
    n_radars,
    oversample_seated,
    window_channels,
)
from src.metrics import prf
from src.model import build_grid_model
from src.preprocessing import (
    extract_peaks_from_grid,
    frame_energy,
    normalize_channels,
)
from src.quantize import convert_to_int8
from src.tracker import Persistence, Tracker

# Enforce determinism for reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Lower bound matters: past sweeps picked the list's minimum, so keep headroom below it.
THRESHOLDS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
MODELS_DIR = "src/models"
LOG_ROOT = "logs"  # TensorBoard: one timestamped dir per run, folds as sub-runs
MODEL_PATH = os.path.join(MODELS_DIR, "best_grid_model.keras")
P99_PATH = os.path.join(MODELS_DIR, "best_p99.npy")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "best_threshold.npy")
# Frame-energy gate below which inference is skipped entirely (Phase
# 5.1). A calibration constant exactly like p99: derived from training
# windows only, written here so code.py and the parity guard read it back
# rather than carrying a literal.
ENERGY_GATE_PATH = os.path.join(MODELS_DIR, "best_energy_gate.npy")
# Which input channel set the saved weights expect. Written next to p99 so
# quantize.py and the submission cannot silently pair a model with the wrong
# preprocessing -- p99 is now a per-channel vector, and its length alone is not
# enough to tell "mag+aoa" from "mag+delta32".
CHANNELS_PATH = os.path.join(MODELS_DIR, "best_channels.txt")
# Which radars the saved weights expect, 1-indexed as in project_spec.md §3.
# Same reason as CHANNELS_PATH: the radar subset changes the input SHAPE, so
# quantizing or deploying against the wrong one is either a hard crash or -- if
# the count happens to match -- silently wrong. Config travels with the model.
RADARS_PATH = os.path.join(MODELS_DIR, "best_radars.txt")
# How many times seated-containing windows were repeated in each training set.
# Unlike the two above this changes NOTHING about the model's shape or the
# preprocessing, so nothing downstream has to read it back -- it is recorded
# purely so a saved model states the recipe that produced it. Kept beside them
# for exactly that reason: the training config travels with the weights.
SEATED_OVERSAMPLE_PATH = os.path.join(MODELS_DIR, "best_seated_oversample.txt")


def train_model(
    train_files,
    p99,
    epochs,
    batch_size,
    val_files=None,
    log_dir=None,
    channel_set="mag",
    radar_indices=None,
):
    """Train one occupancy model on `train_files` (inputs normalized with `p99`).

    `val_files`, when given, is used only for EarlyStopping's val_loss; the final
    all-data refit passes none and early-stops on the training loss instead.
    `log_dir`, when given, writes TensorBoard loss curves there.
    """
    X_train, y_train = load_fold_data(train_files, p99, channel_set, radar_indices)
    train_ds = make_dataset(X_train, y_train, batch_size=batch_size, is_training=True)

    val_ds = None
    monitor = "loss"
    if val_files:
        X_val, y_val = load_fold_data(val_files, p99, channel_set, radar_indices)
        val_ds = make_dataset(X_val, y_val, batch_size=batch_size, is_training=False)
        monitor = "val_loss"

    model = build_grid_model(
        input_shape=(
            n_radars(radar_indices),
            3,
            47,
            len(CHANNEL_SETS[channel_set]),
        )
    )

    # Keras 3 (shipped with TF >= 2.16) removed both of the old per-platform
    # workarounds: `optimizers.legacy` raises on instantiation, and `jit_compile`
    # is no longer an optimizer argument -- it moved to `Model.compile`. The
    # macOS legacy-Adam path existed for a Keras 2 Apple Silicon slowdown that no
    # longer applies, so one optimizer now serves both platforms. XLA stays off,
    # which is what the Linux branch wanted for the GTX 1070.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3),
        loss="mse",
        jit_compile=False,
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        shuffle=False,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor, patience=5, restore_best_weights=True
            ),
            *(
                [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)]
                if log_dir
                else []
            ),
        ],
    )
    return model


def int8_predict(tflite_model, X):
    """Run `X` through the quantized graph frame by frame, as code.py does.

    Returns dequantized float32 heatmaps, so the caller can score the int8 path
    with exactly the same code that scores the float one. `code.py`'s rounding
    convention is reproduced here deliberately -- a bare astype truncates toward
    zero and costs 1 LSB of bias, which is a real difference at 256 levels.
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    info = np.iinfo(inp["dtype"])

    preds = np.empty((len(X), *out["shape"][1:]), dtype=np.float32)
    for i in range(len(X)):
        frame = X[i : i + 1]
        if in_scale > 0:
            frame = np.clip(
                np.round(frame / in_scale + in_zp), info.min, info.max
            ).astype(inp["dtype"])
        else:
            frame = frame.astype(inp["dtype"])
        interpreter.set_tensor(inp["index"], frame)
        interpreter.invoke()
        q = interpreter.get_tensor(out["index"])
        preds[i] = (
            (q.astype(np.float32) - out_zp) * out_scale if out_scale > 0 else q
        )[0]
    return preds


def accumulate_oof_counts(
    model,
    val_files,
    p99,
    counts,
    run_stamp=None,
    channel_set="mag",
    radar_indices=None,
    tflite_model=None,
    energy_gate=0.0,
):
    """Predict each val window ONCE and add its detections to `counts`.

    `counts[th]` is a mutable [tp, fp, fn]. Predictions don't depend on the
    threshold, so we predict once per window and score every threshold on the
    stored heatmaps. These are out-of-fold predictions: `model` never trained on
    `val_files`. Input matches training/deployment (cache-backed, decluttered).

    Scoring runs the FULL deployed post-processing: peak extraction + Tracker,
    frames in temporal order (windows are contiguous 25 Hz captures). The
    tracker shifts the precision/recall operating point (coasting adds FPs at
    low thresholds), so sweeping without it picks the wrong threshold. One
    fresh tracker per (window, threshold) -- state never crosses windows.

    When `run_stamp` is given, each window's heatmaps and the ground truth
    needed to score them are ALSO written to logs/<run_stamp>/oof/. This is a
    pure side effect -- `counts` is untouched -- but it is what makes every
    post-processing knob (decode threshold, tracker alpha/max_coast/
    max_distance, NMS min_distance) a seconds-long sweep over cached arrays
    instead of a full 6-fold retrain. The cache is self-contained: replaying it
    needs neither the model nor the raw dataset.

    When `tflite_model` is given, the SCORED predictions come from the quantized
    graph, not from Keras (K1/3.4). The threshold was previously chosen on the
    float path and deployed on the int8 one, and the two curves peak at
    different thresholds -- so the number this function returns was measuring a
    model that never ships. Both are cached: `preds` is the int8 path and
    `preds_float` is Keras, which makes the float/int8 gap an out-of-fold
    measurement for the first time.

    `energy_gate` applies 5.1: frames whose input energy falls below it are
    scored as an all-zero heatmap, which is exactly what skipping inference
    produces. The gate is applied to the COUNTS but not to the cached `preds`,
    so a sweep can still explore gates in both directions from the cache.
    """
    oof_dir = None
    if run_stamp is not None:
        oof_dir = os.path.join(LOG_ROOT, run_stamp, "oof")
        os.makedirs(oof_dir, exist_ok=True)

    for filepath in val_files:
        data = np.load(filepath)
        people_xy = data["people_xy"]
        people_mask = data["people_mask"]

        X = normalize_channels(
            window_channels(filepath, channel_set, radar_indices)[0], p99
        )
        preds_float = model.predict(X, verbose=0)
        preds = (
            int8_predict(tflite_model, X) if tflite_model is not None else preds_float
        )
        energy = frame_energy(X)
        gated = energy < energy_gate

        gts = [
            [
                list(people_xy[i, p])
                for p in range(people_xy.shape[1])
                if people_mask[i, p]
            ]
            for i in range(len(X))
        ]

        if oof_dir is not None:
            # ~6.5 MB per window uncompressed; exact and instant to reload.
            # people_xy/people_mask are stored rather than the ragged per-frame
            # lists so every array stays fixed-shape (no pickle on load); `gts`
            # above is the one-line derivation a sweep repeats.
            np.savez(
                os.path.join(oof_dir, os.path.basename(filepath) + ".oof.npz"),
                preds=np.asarray(preds, dtype=np.float32),
                preds_float=np.asarray(preds_float, dtype=np.float32),
                energy=np.asarray(energy, dtype=np.float32),
                energy_gate=energy_gate,
                people_xy=people_xy[: len(X)],
                people_mask=people_mask[: len(X)],
                p99=p99,
                channel_set=channel_set,
                radars=format_radars(
                    ALL_RADARS if radar_indices is None else radar_indices
                ),
                window=os.path.basename(filepath),
            )

        for th in counts:
            # One fresh pair per (window, threshold): `Persistence` carries EMA
            # state and its gain is threshold-dependent, so state must never
            # cross either boundary.
            tracker = Tracker()
            persistence = Persistence(th)
            c = counts[th]
            for i in range(len(X)):
                # A gated frame is scored as the all-zero heatmap that skipping
                # inference produces -- persistence and the tracker still run on
                # it, so their state stays causal and correct.
                heatmap = np.zeros_like(preds[i]) if gated[i] else preds[i]
                pred = extract_peaks_from_grid(
                    persistence.update(heatmap), threshold=th
                )
                pred = tracker.update(pred)
                dists, n_fn, n_fp = match_hungarian(gts[i], pred, MATCH_THRESHOLD)
                c[0] += len(dists)
                c[1] += n_fp
                c[2] += n_fn


def train_and_evaluate(
    data_dir,
    epochs=20,
    batch_size=32,
    fast_dev=False,
    channel_set="mag",
    radar_indices=None,
    seated_oversample=1,
):
    os.makedirs(MODELS_DIR, exist_ok=True)
    radar_indices = ALL_RADARS if radar_indices is None else tuple(radar_indices)
    print(f"Input channel set: {channel_set} = {CHANNEL_SETS[channel_set]}")
    print(
        f"Radars: {format_radars(radar_indices)} "
        f"({len(radar_indices)} of {len(ALL_RADARS)})"
    )
    print(f"Seated oversample: x{seated_oversample}")

    # --- 1. Cross-validation: pooled out-of-fold detection counts --------------
    # Each window is validated by a model that never trained on it. Pooling every
    # fold's val detections yields ONE honest, global (micro-averaged) score --
    # not the max over folds. This estimates generalization; it is not the model
    # we ship.
    folds = get_kfold_splits(
        data_dir,
        k=6,
        seed=SEED,
        channel_set=channel_set,
        radar_indices=radar_indices,
        seated_oversample=seated_oversample,
    )
    counts = {th: [0, 0, 0] for th in THRESHOLDS}
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for fold_idx, train_files, val_files, p99 in folds:
        print(f"\n{'=' * 40}")
        print(f"CV Fold {fold_idx + 1} (p99={np.array2string(p99, precision=4)})")
        print(f"{'=' * 40}")
        model = train_model(
            train_files,
            p99,
            epochs,
            batch_size,
            val_files=val_files,
            log_dir=os.path.join(LOG_ROOT, run_stamp, f"fold_{fold_idx + 1}"),
            channel_set=channel_set,
            radar_indices=radar_indices,
        )

        # Both calibration constants come from this fold's TRAINING files only.
        gate = calculate_energy_gate_from_files(
            train_files, p99, channel_set=channel_set, radar_indices=radar_indices
        )
        # Calibrated on the fold's training windows too -- calibrating on all 24
        # would draw the int8 ranges from the very windows about to be scored.
        print(f"Quantizing fold {fold_idx + 1} (energy gate {gate:.6f})...")
        tflite_model = convert_to_int8(
            model, train_files, p99,
            channel_set=channel_set, radar_indices=radar_indices,
        )

        accumulate_oof_counts(
            model,
            val_files,
            p99,
            counts,
            run_stamp=run_stamp,
            channel_set=channel_set,
            radar_indices=radar_indices,
            tflite_model=tflite_model,
            energy_gate=gate,
        )
        del model, tflite_model
        if fast_dev:
            print("\n[FAST-DEV MODE] Single fold for the OOF estimate.")
            break

    # --- 2. Choose the decode threshold on the pooled OOF predictions ----------
    # Scored on the INT8 path and with the energy gate applied, i.e. on the
    # pipeline that actually ships (K1/3.4 + 5.1). Picking here on the float
    # path is what made the previous threshold a number for a model that was
    # never deployed.
    print("\nOut-of-fold threshold sweep (int8 path, energy gate applied):")
    best_threshold = THRESHOLDS[0]
    best_f1 = -1.0
    for th in THRESHOLDS:
        tp, fp, fn = counts[th]
        precision, recall, f1 = prf(tp, fp, fn)
        print(
            f"Threshold: {th:.2f} | F1: {f1:.4f} | "
            f"Precision: {precision:.4f} | Recall: {recall:.4f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
    print(f"\n-> CV out-of-fold F1: {best_f1:.4f} at threshold {best_threshold:.2f}")

    # --- 3. Refit the DEPLOYED model on ALL windows ----------------------------
    # CV only estimated generalization and picked the threshold. The shipped model
    # should use every window, so retrain once on the full dataset; p99 is
    # recomputed over all windows.
    all_files = list_windows(data_dir)
    p99_all = calculate_p99_from_files(
        all_files, channel_set=channel_set, radar_indices=radar_indices
    )
    energy_gate = calculate_energy_gate_from_files(
        all_files, p99_all, channel_set=channel_set, radar_indices=radar_indices
    )
    # Same recipe as the folds, so the shipped weights match what CV measured:
    # p99 on the distinct windows first, oversampling only for the fit itself.
    # There is no held-out set here, so this is a duplication, not a leak.
    fit_files = oversample_seated(
        all_files, load_seated_windows(data_dir), seated_oversample
    )
    print(
        f"\nRefitting deployed model on all {len(all_files)} windows "
        f"({len(fit_files)} with oversampling) "
        f"(p99={np.array2string(p99_all, precision=4)})..."
    )
    final_model = train_model(
        fit_files,
        p99_all,
        epochs,
        batch_size,
        val_files=None,
        log_dir=os.path.join(LOG_ROOT, run_stamp, "final"),
        channel_set=channel_set,
        radar_indices=radar_indices,
    )

    final_model.save(MODEL_PATH)
    np.save(P99_PATH, np.asarray(p99_all))
    np.save(THRESHOLD_PATH, np.array([best_threshold]))
    np.save(ENERGY_GATE_PATH, np.array([energy_gate]))
    with open(CHANNELS_PATH, "w") as f:
        f.write(channel_set + "\n")
    with open(RADARS_PATH, "w") as f:
        f.write(format_radars(radar_indices) + "\n")
    with open(SEATED_OVERSAMPLE_PATH, "w") as f:
        f.write(str(seated_oversample) + "\n")
    print(
        f"\nDone. Deployed model -> {MODEL_PATH}\n"
        f"  channels={channel_set}  radars={format_radars(radar_indices)}  "
        f"seated_oversample=x{seated_oversample}  "
        f"p99={np.array2string(p99_all, precision=4)}  "
        f"threshold={best_threshold:.2f}  energy_gate={energy_gate:.6f}  "
        f"(CV out-of-fold F1 {best_f1:.4f})\n"
        f"  OOF cache -> {os.path.join(LOG_ROOT, run_stamp, 'oof')}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the occupancy grid model.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="multi-person-localization/data",
        help="Path to .npz data files",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument(
        "--channels",
        type=str,
        default="mag",
        choices=sorted(CHANNEL_SETS),
        help="Input channel set (see channels.CHANNEL_SETS)",
    )
    parser.add_argument(
        "--radars",
        type=str,
        default=format_radars(ALL_RADARS),
        help=(
            "Comma-separated 1-indexed radar subset, e.g. '1,2,4,6' "
            "(project_spec.md §3 numbering). Default: all 6."
        ),
    )
    parser.add_argument(
        "--seated-oversample",
        type=int,
        default=1,
        metavar="K",
        help=(
            "Repeat the 8 seated-containing windows K times inside each fold's "
            "TRAINING set (never the held-out one), so the loss sees a seated "
            "frame K times per epoch. Default 1 = no-op. See Phase 4.1b."
        ),
    )
    parser.add_argument(
        "--fast-dev",
        action="store_true",
        help="Use a single CV fold for the OOF estimate (final refit still runs)",
    )
    args = parser.parse_args()

    train_and_evaluate(
        args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fast_dev=args.fast_dev,
        channel_set=args.channels,
        radar_indices=parse_radars(args.radars),
        seated_oversample=args.seated_oversample,
    )

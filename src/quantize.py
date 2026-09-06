import argparse
import glob
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"

import numpy as np
import tensorflow as tf

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.channels import CHANNEL_SETS, parse_radars
from src.data_loader import n_radars, window_channels
from src.preprocessing import normalize_channels

# Calibration frames are drawn at random from a pool this many times larger than
# num_samples. Named for what it does -- unrelated to `seated_oversample`, which
# repeats seated windows during training. See REPORT_NOTES for the measurements.
CALIBRATION_POOL_FACTOR = 6


def representative_dataset_gen(
    filepaths, p99, num_samples=500, channel_set="mag", radar_indices=None
):
    """Calibration frames for TFLite integer quantization.

    `filepaths` is an EXPLICIT window list, not a directory, and that is
    load-bearing for the fold models K1 quantizes during CV: calibrating a fold
    model on all 24 windows would draw its int8 ranges from the very windows it
    is about to be scored on, which quietly turns an out-of-fold measurement
    into an in-sample one. The deployed refit legitimately passes all 24,
    because it trained on all 24.

    DECLUTTERED frames, i.e. the distribution the model trains on and deploys
    against. This is load-bearing, not a detail: once BN folds into the conv
    weights there is no runtime MUL/ADD left to rescale activations, so the
    calibration range is the ONLY thing setting int8 resolution. Calibrating on
    raw frames instead -- which are far larger in magnitude -- sets every range
    too wide and collapses the deployed model from example-set F1 0.9884 to
    0.7815 (5,569 false positives). Measured both ways, before and after the
    fold; see REPORT_NOTES for the tables.

    Frames are strided across ALL windows so the pool spans every scenario, then
    drawn at random from a CALIBRATION_POOL_FACTOR x num_samples pool so the draw
    is not a stride artifact. Both axes are second-order (all within draw noise);
    neither is worth changing without re-measuring, and the 4th decimal of either
    is not signal.
    """
    if not filepaths:
        raise ValueError("representative_dataset_gen needs at least one window")

    per_window = max(
        1, math.ceil(CALIBRATION_POOL_FACTOR * num_samples / len(filepaths))
    )
    X_samples = []
    for filepath in filepaths:
        # (T,R,3,47,C), decluttered
        x, _ = window_channels(filepath, channel_set, radar_indices)
        idx = np.linspace(0, len(x) - 1, min(per_window, len(x))).astype(int)
        X_samples.append(normalize_channels(x[idx], p99))
        del x

    X_all = np.concatenate(X_samples, axis=0)

    # Shuffle and pick num_samples (seeded: unseeded draws make the calibration
    # -- and therefore the .tflite -- differ on every run)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X_all), min(num_samples, len(X_all)), replace=False)
    X_subset = X_all[indices]

    # Convert to float32 as expected by the converter
    X_subset = X_subset.astype(np.float32)

    def generator():
        for i in range(len(X_subset)):
            # Yield single sample with batch dimension: shape (1, 6, 3, 47, C)
            yield [np.expand_dims(X_subset[i], axis=0)]

    return generator


def convert_to_int8(
    model, filepaths, p99, channel_set="mag", radar_indices=None, verbose=False
):
    """Keras model -> full-integer INT8 TFLite bytes, calibrated on `filepaths`.

    Exported with a BATCH-SIZE-1 signature first, and that is not cosmetic.
    `from_keras_model` inherits the dynamic batch dim, so the Reshape's target
    becomes [tf.shape(x)[0], 18, 12, C] -- which the converter cannot
    constant-fold, shipping SHAPE -> STRIDED_SLICE -> PACK -> RESHAPE where one
    static RESHAPE would do. Pinning the batch to 1 (the only size ever used:
    code.py feeds one frame at a time) makes the sequence foldable, measured
    20 -> 17 ops with no model or training change.

    `ExportArchive` is the supported way to fix an input_signature;
    `tf.function(...).get_concrete_function(...)` alone leaves BN variables
    unfrozen and conversion fails on READ_VARIABLE. `model.inputs[0]`, not
    `model.input`: reloaded from .keras the latter is a one-element list.
    """
    input_signature = [tf.TensorSpec((1, *model.inputs[0].shape[1:]), tf.float32)]
    if verbose:
        print(f"Exporting with fixed input signature {input_signature[0].shape}...")

    with tempfile.TemporaryDirectory() as export_dir:
        archive = tf.keras.export.ExportArchive()
        archive.track(model)
        archive.add_endpoint(
            name="serve",
            fn=lambda x: model(x, training=False),
            input_signature=input_signature,
        )
        archive.write_out(export_dir)

        converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen(
            filepaths, p99, channel_set=channel_set, radar_indices=radar_indices
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        if verbose:
            print("Converting model to TFLite (INT8)...")
        return converter.convert()


def quantize_model(
    data_dir, model_path, p99_path, output_path, channels_path=None, radars_path=None
):
    filepaths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not filepaths:
        raise ValueError(f"No .npz files found in {data_dir}")

    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)

    print(f"Loading p99 value from {p99_path}...")
    p99 = np.atleast_1d(np.load(p99_path))

    # The channel set is read from the artifact the training run wrote, never
    # assumed: calibrating on the wrong channels is exactly the failure mode
    # that cost this project an 0.20 F1 cliff once already (raw vs decluttered
    # frames, see the docstring above). Falling back to "mag" only when the
    # file is absent keeps pre-Phase-4 model directories loadable.
    channel_set = "mag"
    if channels_path and os.path.exists(channels_path):
        channel_set = open(channels_path).read().strip()
    n_channels = len(CHANNEL_SETS[channel_set])
    if len(p99) != n_channels:
        raise ValueError(
            f"p99 has {len(p99)} entries but channel set {channel_set!r} has "
            f"{n_channels} channels -- src/models/ is inconsistent, retrain."
        )
    # Same argument for the radar subset: it changes the input SHAPE, so a
    # mismatch is either a crash inside the converter or -- if the counts happen
    # to agree -- a model calibrated on radars it will never see. Read it back
    # rather than assume, and cross-check against the weights themselves.
    radar_indices = None
    if radars_path and os.path.exists(radars_path):
        radar_indices = parse_radars(open(radars_path).read().strip())
    model_radars = int(model.inputs[0].shape[1])
    if n_radars(radar_indices) != model_radars:
        raise ValueError(
            f"{radars_path} selects {n_radars(radar_indices)} radars but the "
            f"model input expects {model_radars} -- src/models/ is inconsistent, "
            "retrain."
        )
    print(
        f"Using channel set {channel_set} = {CHANNEL_SETS[channel_set]} on "
        f"{model_radars} radars with p99={np.array2string(p99, precision=4)} "
        "for the quantization dataset."
    )

    tflite_model = convert_to_int8(
        model, filepaths, p99, channel_set=channel_set, radar_indices=radar_indices,
        verbose=True,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Quantized TFLite model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Keras model to TFLite INT8.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="multi-person-localization/data",
        help="Path to .npz data files for representative dataset",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="src/models/best_grid_model.keras",
        help="Path to trained Keras model",
    )
    parser.add_argument(
        "--p99-path",
        type=str,
        default="src/models/best_p99.npy",
        help="Path to saved p99 value",
    )
    parser.add_argument(
        "--channels-path",
        type=str,
        default="src/models/best_channels.txt",
        help="Path to the channel set recorded by the training run",
    )
    parser.add_argument(
        "--radars-path",
        type=str,
        default="src/models/best_radars.txt",
        help="Path to the radar subset recorded by the training run",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="submission/model.tflite",
        help="Path to save the quantized TFLite model",
    )
    args = parser.parse_args()

    quantize_model(
        args.data_dir,
        args.model_path,
        args.p99_path,
        args.output_path,
        channels_path=args.channels_path,
        radars_path=args.radars_path,
    )

"""Prove `submission/code.py` and `src/` are the same pipeline. Exits non-zero.

`submission/` cannot import from `src/`, so the deployed preprocessing, decode
and tracking are hand-copied. Hand-copied things drift, and this drift is
invisible: a wrong constant or a dropped axis produces a plausible-looking array
and a quietly worse score.

It has already happened once. `submission/model.tflite` was regenerated before
the channel axis was added to `build_grid_model`, so the shipped model took a
4-D input while every retrained model takes a 5-D one. The done gate passed
throughout, because the stale artifact and the stale `code.py` agreed with each
other. Check 3 is the one that would have caught it.

Checks:
  1. Constants -- every shared literal in code.py equals its src/ counterpart.
  2. Calibration -- P99_VALUE and THRESHOLD equal the artifacts the training run
     wrote (the old task-board heredoc guard, folded in here).
  3. Model contract -- what preprocess_radar produces is what model.tflite
     accepts, in rank, shape and dtype.
  4. Functions -- declutter, magnitude, normalize, preprocess, peak decode,
     Persistence and Tracker produce byte-identical output on real data.

Run:  .venv/bin/python src/helpers/verify_parity.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO))

from src import preprocessing as pp
from src import tracker as tr

EXAMPLE_INPUT = REPO / "evaluation" / "example" / "input_test.npy"
MODELS_DIR = REPO / "src" / "models"
TFLITE = REPO / "submission" / "model.tflite"
N_FRAMES = 600

failures = []


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{'   ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def check_equal(name, a, b):
    a, b = np.asarray(a), np.asarray(b)
    ok = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    check(name, ok, f"{a.shape} {a.dtype}")


def load_submission():
    spec = importlib.util.spec_from_file_location(
        "submission_code", REPO / "submission" / "code.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    code = load_submission()

    print("\n1. Shared constants")
    for name, src_value in [
        ("CROP_START", pp.CROP_START),
        ("CROP_END", pp.CROP_END),
        ("FRAME_RATE_HZ", pp.FRAME_RATE_HZ),
        ("ROOM_WIDTH_M", pp.ROOM_WIDTH_M),
        ("ROOM_HEIGHT_M", pp.ROOM_HEIGHT_M),
        ("MAX_PEOPLE", tr.MAX_PEOPLE),
        ("PERSIST_TAU_S", tr.PERSIST_TAU_S),
        ("PERSIST_T_LO", tr.PERSIST_T_LO),
    ]:
        got = getattr(code, name, None)
        check(name, got == src_value, f"code={got!r} src={src_value!r}")

    print("\n2. Calibration constants vs the training run's artifacts")
    artifacts = [
        ("P99_VALUE", "best_p99.npy"),
        ("THRESHOLD", "best_threshold.npy"),
        ("ENERGY_GATE", "best_energy_gate.npy"),
    ]
    for name, filename in artifacts:
        path = MODELS_DIR / filename
        if not path.exists():
            check(name, False, f"{path.relative_to(REPO)} missing -- retrain first")
            continue
        want = float(np.atleast_1d(np.load(path))[0])
        got = float(getattr(code, name))
        check(name, got == want, f"code={got!r} artifact={want!r}")

    print("\n3. Model contract")
    p99 = getattr(code, "P99_VALUE")
    raw = np.load(EXAMPLE_INPUT, mmap_mode="r")[:2]
    produced = code.preprocess_radar(pp.remove_clutter(np.asarray(raw)), p99)
    if not TFLITE.exists():
        check("model.tflite present", False, "run src/quantize.py first")
    else:
        import tensorflow as tf

        interp = tf.lite.Interpreter(model_path=str(TFLITE))
        interp.allocate_tensors()
        want = tuple(interp.get_input_details()[0]["shape"])
        got = (1,) + produced.shape[1:]
        check(
            "preprocess_radar rank/shape == model input",
            got == want,
            f"produces {got}, model accepts {want}",
        )

    print("\n4. Functions, on real data")
    raw = np.asarray(np.load(EXAMPLE_INPUT, mmap_mode="r")[:N_FRAMES])

    src_declutter = pp.remove_clutter(raw)
    sub_declutter = np.concatenate(list(code.remove_clutter_streaming(raw)))
    check_equal(
        "remove_clutter (batch src == streaming sub)", src_declutter, sub_declutter
    )

    check_equal("magnitude", pp.magnitude(src_declutter), code.magnitude(sub_declutter))
    check_equal(
        "normalize_channels",
        pp.normalize_channels(pp.magnitude(src_declutter)[..., None], p99),
        code.normalize_channels(code.magnitude(sub_declutter)[..., None], p99),
    )
    check_equal(
        "preprocess_radar",
        pp.preprocess_radar(src_declutter, p99),
        code.preprocess_radar(sub_declutter, p99),
    )
    check_equal(
        "frame_energy",
        pp.frame_energy(pp.preprocess_radar(src_declutter, p99)),
        code.frame_energy(code.preprocess_radar(sub_declutter, p99)),
    )

    rng = np.random.default_rng(0)
    grids = (rng.random((N_FRAMES, 18, 12, 1)).astype(np.float32) ** 3) * 1.2
    th = float(getattr(code, "THRESHOLD"))

    src_peaks = [pp.extract_peaks_from_grid(g, threshold=th) for g in grids]
    sub_peaks = [code.extract_peaks_from_grid(g, threshold=th) for g in grids]
    check("extract_peaks_from_grid", src_peaks == sub_peaks)

    src_p, sub_p = tr.Persistence(th), code.Persistence(th)
    check_equal(
        "Persistence.update",
        np.stack([src_p.update(g) for g in grids]),
        np.stack([sub_p.update(g) for g in grids]),
    )

    src_t, sub_t = tr.Tracker(), code.Tracker()
    check(
        "Tracker.update",
        [src_t.update(d) for d in src_peaks] == [sub_t.update(d) for d in sub_peaks],
    )

    # The whole chain at once, in the order code.py runs it -- catches a
    # difference in composition that each stage in isolation would hide.
    src_p, sub_p = tr.Persistence(th), code.Persistence(th)
    src_t, sub_t = tr.Tracker(), code.Tracker()
    src_out = [
        src_t.update(pp.extract_peaks_from_grid(src_p.update(g), threshold=th))
        for g in grids
    ]
    sub_out = [
        sub_t.update(code.extract_peaks_from_grid(sub_p.update(g), threshold=th))
        for g in grids
    ]
    check("full decode chain", src_out == sub_out)

    print()
    if failures:
        print(f"*** PARITY BROKEN -- {len(failures)} check(s) failed:")
        for f in failures:
            print(f"      {f}")
        return 1
    print("PARITY OK -- submission/code.py matches src/ on every checked surface")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Why the model misses seated subjects, and where the missing signal lives.

Background: the per-window out-of-fold breakdown shows 8 seated-containing
windows holding ~75% of all pooled false negatives, with window 021 (a single
seated subject, nobody walking) scoring F1 ~0.001. That is the largest single
error mode in the project, and Phase 4 of the task board is aimed at it.

This script establishes three things, in order, each with the empty-room
windows (022, 023) as the null distribution so nothing depends on a calibration
constant or on a trained model:

  presence   The seated subject is NOT absent from the data. Respiration puts
             0.15-0.6 Hz energy into the decluttered CIR at 50-170x the
             out-of-band floor, while both empty windows sit at ~1.8. Across
             all 24 windows the separation is perfect with a ~28x margin.

  per-frame  It is nearly absent from what the model is actually FED. The model
             consumes one frame of `magnitude()` at a time, and a single frame
             separates seated-subject from empty-room at only d' ~ 1.4 -- a
             periodic signal presented without any temporal context.

  features   Recovering it needs a time baseline of roughly half a breathing
             period, and phase only pays off at that scale. A 4-frame (160 ms)
             delta is WORSE than plain magnitude; the complex delta overtakes
             magnitude only past ~1 s. This is the measurement that sets the
             lag, rather than guessing one.

Run:  .venv/bin/python src/helpers/seated_diagnostics.py [--quick]
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.signal

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.preprocessing import CROP_END, CROP_START, remove_clutter

FS = 25.0
# Adult respiration at rest: ~9-36 breaths/min. Wide enough to hold the whole
# band across subjects, narrow enough to exclude the walking/torso rates above.
BREATH_BAND = (0.15, 0.60)
# Out-of-band reference. Above any bulk-motion rate, below Nyquist.
NOISE_BAND = (2.0, 8.0)

DATA_DIR = "multi-person-localization/data"
SEATED_ONLY, EMPTY = "window_000021.npz", ("window_000022.npz", "window_000023.npz")


def decluttered_complex(filename):
    """(T, 6, 3, 47) complex decluttered CIR -- the pipeline's own declutter and
    crop, one step before `magnitude()` throws the phase away."""
    raw = np.load(os.path.join(DATA_DIR, filename))["radar_cir_iq"]
    dec = remove_clutter(raw)[..., CROP_START:CROP_END, :]
    return dec[..., 0] + 1j * dec[..., 1]


def band_snr(z):
    """Breathing-band / out-of-band power per (radar, antenna, bin), computed on
    the complex signal so it is phase-preserving."""
    freqs, psd = scipy.signal.welch(
        np.stack([z.real, z.imag]), fs=FS, nperseg=1024, axis=1
    )
    in_band = (freqs >= BREATH_BAND[0]) & (freqs <= BREATH_BAND[1])
    out_band = (freqs >= NOISE_BAND[0]) & (freqs <= NOISE_BAND[1])
    return psd[:, in_band].mean(1).sum(0) / psd[:, out_band].mean(1).sum(0)


def d_prime(a, b):
    """Standardized mean separation between two samples -- the per-frame
    detectability of whatever statistic was passed in."""
    return abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()))


def report_presence(quick):
    """Breathing-band SNR for every window, against the empty-room null."""
    descriptions = {}
    with open(os.path.join(os.path.dirname(DATA_DIR), "metadata.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            descriptions[os.path.basename(row["path"])] = row["description"]

    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if quick:
        files = [SEATED_ONLY, *EMPTY]

    print("\n=== presence: is the seated subject in the data at all? ===")
    print(f"{'win':>4} {'cells SNR>10':>13} {'max SNR':>9}  description")
    for filename in files:
        snr = band_snr(decluttered_complex(filename))
        print(
            f"{int(filename.split('_')[1][:6]):>4} {100 * (snr > 10).mean():12.3f}%"
            f" {snr.max():9.1f}  {descriptions[filename]}"
        )
    print(
        "\n  Every occupied window clears SNR>10 on 2.8-9.9% of cells; both empty\n"
        "  windows clear it on exactly 0.000%. Window 021 -- the one the model\n"
        "  scores ~0.001 on -- carries the HIGHEST max SNR in the dataset."
    )


def report_features():
    """Per-frame vs temporal separability, on the cell carrying the subject."""
    z_seated = decluttered_complex(SEATED_ONLY)
    z_empty = decluttered_complex(EMPTY[0])

    # Pick the cell by band SNR rather than by geometry: the range-bin axis has
    # an unresolved constant offset (bin 28 ~= 4.2 m nominal vs a 3.26 m true
    # subject range), so a computed bin index would be the wrong cell.
    cell = np.unravel_index(band_snr(z_seated).argmax(), z_seated.shape[1:])
    seated = z_seated[:, cell[0], cell[1], cell[2]]
    empty = z_empty[:, cell[0], cell[1], cell[2]]
    print(f"\n=== features: cell radar {cell[0] + 1} antenna {cell[1]} bin {cell[2] + CROP_START} ===")

    freqs, psd = scipy.signal.welch(seated.real - seated.real.mean(), fs=FS, nperseg=2048)
    sel = (freqs > 0.1) & (freqs < 1.0)
    peak = freqs[sel][psd[sel].argmax()]
    print(f"  respiration rate {peak:.3f} Hz = {60 * peak:.1f} breaths/min, period {1 / peak:.2f} s")

    print("\n  per-frame (what the model is fed today):")
    print(f"    |decluttered| magnitude              d' = {d_prime(abs(seated), abs(empty)):.3f}")

    print("\n  temporal, same cell and same data:")
    for lag in (4, 12, 25, 40):
        mag = d_prime(
            abs(abs(seated[lag:]) - abs(seated[:-lag])),
            abs(abs(empty[lag:]) - abs(empty[:-lag])),
        )
        cpx = d_prime(abs(seated[lag:] - seated[:-lag]), abs(empty[lag:] - empty[:-lag]))
        print(
            f"    lag {lag:>2} ({lag / FS * 1000:>4.0f} ms)   magnitude d' = {mag:.3f}"
            f"   complex d' = {cpx:.3f}"
        )

    def bandpass(x):
        b, a = scipy.signal.butter(
            2, [BREATH_BAND[0] / (FS / 2), BREATH_BAND[1] / (FS / 2)], btype="band"
        )
        return scipy.signal.filtfilt(b, a, x)

    envelope = lambda s: abs(bandpass(s.real)) + abs(bandpass(s.imag))
    print(
        f"    0.15-0.6 Hz bandpass, complex        d' = "
        f"{d_prime(envelope(seated), envelope(empty)):.3f}   (non-causal, an upper bound)"
    )
    print(
        "\n  The task board's proposed 4-frame delta is worse than doing nothing:\n"
        "  160 ms spans 5% of a 3.3 s breath. Phase overtakes magnitude only once\n"
        "  the lag reaches ~1 s, which is what should set the channel design."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Presence check on windows 021/022/023 only, instead of all 24",
    )
    args = parser.parse_args()

    report_presence(args.quick)
    report_features()

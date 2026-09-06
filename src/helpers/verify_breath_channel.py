"""Prove the "breath" channel measures the respiration band and nothing else.

This exists because this project has already shipped one silently-wrong channel
from a plausible-looking axis argument: `mag+aoa`'s `np.roll(z, -1, axis=-3)`
differenced RADARS, not antennas, and a full CV run was spent measuring a
quantity nobody intended. It was caught weeks later by a synthetic test, not by
review. A filter is a strictly easier thing to get wrong than an axis index --
wrong band, wrong direction, non-causal, or numerically divergent all produce a
finite-looking array of plausible magnitudes -- so the channel gets its evidence
BEFORE it gets a CV run, not after.

What is checked, and why each one:

1. Frequency response. The band is the entire mechanism; if the -3 dB points are
   not at 0.15/0.60 Hz the channel measures something else.
2. Synthetic separation. A cell breathing at 0.3 Hz must read clearly HIGHER
   than an identical-amplitude cell moving at 2 Hz (the motion the two prior
   temporal channels collapsed onto) and than an empty cell. This is the
   property that distinguishes a breathing detector from a motion detector, so
   it is asserted rather than described.
3. Contract: shape, float32, non-negative -- `breath` is not in
   `SIGNED_CHANNELS`, so a negative value would be clipped to 0 by
   `normalize_channels` and quietly halve the channel.
4. Cold start. Inference must emit a position for frame 0, so the filter's
   `zi` initialization has to degrade gracefully instead of ringing, the same
   guarantee `lagged` gets from clamping to `z_0`.
5. Filter state cost in bytes -- real SRAM on the ESP32 that
   `evaluate_constraint.py` cannot see, the same class of cost 4.1d recorded
   (~183 KB) for its ring buffer.
6. A float32 divergence guard. This filter's poles sit at radius 0.9877 and a
   single-precision direct-form recursion does NOT survive a 300 s window. The
   guard pins that down so nobody "optimizes" the dtype later and silently
   fills the channel with NaN 150 s in.

Run:  .venv/bin/python src/helpers/verify_breath_channel.py
      .venv/bin/python src/helpers/verify_breath_channel.py --no-real
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.signal

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.channels import (
    BREATH_FILTER_ORDER,
    BREATH_HIGH_HZ,
    BREATH_LOW_HZ,
    channel,
)
from src.preprocessing import FRAME_RATE_HZ

# Grid the model actually sees: 6 radars x 3 antennas x 47 range bins.
N_RADARS, N_ANTENNAS, N_BINS = 6, 3, 47
N_CELLS = N_RADARS * N_ANTENNAS * N_BINS  # 846

# A 6th-order filter with poles at r=0.9877 has a ~81-frame (3.25 s) time
# constant, so a 10 s synthetic is still mid-transient. 60 s gives the filter
# ~18 time constants and leaves a real steady state to measure.
SYNTH_FRAMES = 1500
SETTLE_FRAMES = 500  # 20 s; everything before this is reported, not asserted.

BREATH_HZ = 0.30  # band centre, the measured respiration rate of w021
MOTION_HZ = 2.00  # clearly out of band: stands in for walking / bulk motion
AMPLITUDE = 0.1


def _design():
    return scipy.signal.butter(
        BREATH_FILTER_ORDER,
        [BREATH_LOW_HZ, BREATH_HIGH_HZ],
        btype="band",
        fs=FRAME_RATE_HZ,
    )


def report_response():
    """Check 1 -- the passband is where the respiration band is."""
    b, a = _design()
    order = len(a) - 1
    w, h = scipy.signal.freqz(b, a, worN=200_000, fs=FRAME_RATE_HZ)
    mag = np.abs(h)
    peak = mag.max()
    above = np.flatnonzero(mag >= peak / np.sqrt(2))
    f_lo, f_hi = w[above[0]], w[above[-1]]

    print("=== 1. frequency response ===")
    print(
        f"  butter(N={BREATH_FILTER_ORDER}, [{BREATH_LOW_HZ}, {BREATH_HIGH_HZ}] Hz, "
        f"'band', fs={FRAME_RATE_HZ}) -> filter order {order}, "
        f"{len(b)} b / {len(a)} a taps"
    )
    print(f"  max pole radius {np.abs(np.roots(a)).max():.6f}")
    print(f"  peak gain {peak:.4f} at {w[mag.argmax()]:.4f} Hz")
    print(f"  -3 dB band {f_lo:.4f} .. {f_hi:.4f} Hz   (target {BREATH_LOW_HZ} .. {BREATH_HIGH_HZ})")
    print(f"\n  {'freq (Hz)':>10} {'gain (dB)':>11}   note")
    notes = {
        0.0: "DC -- static clutter residue, must be rejected",
        0.05: "3 breaths/min, below any real subject",
        BREATH_LOW_HZ: "band edge",
        BREATH_HZ: "band centre: w021's measured 0.305 Hz",
        BREATH_HIGH_HZ: "band edge",
        1.0: "fidgeting",
        MOTION_HZ: "walking / bulk motion",
        5.0: "fast motion",
        12.0: "near Nyquist",
    }
    for f, note in notes.items():
        i = np.argmin(np.abs(w - f))
        db = 20 * np.log10(max(mag[i], 1e-300) / peak)
        print(f"  {f:10.2f} {db:11.2f}   {note}")

    assert abs(f_lo - BREATH_LOW_HZ) < 0.01, f"low -3dB point at {f_lo}, not {BREATH_LOW_HZ}"
    assert abs(f_hi - BREATH_HIGH_HZ) < 0.01, f"high -3dB point at {f_hi}, not {BREATH_HIGH_HZ}"
    print("\n  OK: -3 dB points land on the respiration band within 0.01 Hz.")


def _synthetic():
    """(T, 6, 3, 47) complex64 with one breathing, one moving and two empty cells."""
    t = np.arange(SYNTH_FRAMES) / FRAME_RATE_HZ
    z = np.zeros((SYNTH_FRAMES, N_RADARS, N_ANTENNAS, N_BINS), dtype=np.complex64)
    cells = {
        "breathing 0.3 Hz": (0, 0, 10),
        "moving 2.0 Hz": (1, 1, 20),
        "empty (flat zero)": (2, 2, 30),
        "empty (noise)": (3, 0, 40),
    }
    # Complex rotations, not real sinusoids: a chest displacement moves the CIR
    # phase, which is why the channel filters z rather than |z|. Identical
    # amplitude on both moving cells, so the comparison is purely about
    # frequency and never about level.
    z[(slice(None), *cells["breathing 0.3 Hz"])] = AMPLITUDE * np.exp(2j * np.pi * BREATH_HZ * t)
    z[(slice(None), *cells["moving 2.0 Hz"])] = AMPLITUDE * np.exp(2j * np.pi * MOTION_HZ * t)
    rng = np.random.default_rng(0)
    z[(slice(None), *cells["empty (noise)"])] = (
        AMPLITUDE * 0.01 * (rng.standard_normal(SYNTH_FRAMES) + 1j * rng.standard_normal(SYNTH_FRAMES))
    )
    return z, cells


def report_synthetic():
    """Checks 2-4 -- separation, output contract, cold start."""
    z, cells = _synthetic()
    out = channel(z, "breath")

    print("\n=== 2. synthetic separation ===")
    print(
        f"  {SYNTH_FRAMES} frames ({SYNTH_FRAMES / FRAME_RATE_HZ:.0f} s), amplitude "
        f"{AMPLITUDE} on both moving cells. Steady state = frames "
        f"{SETTLE_FRAMES}+ ({SETTLE_FRAMES / FRAME_RATE_HZ:.0f} s)."
    )
    print(f"\n  {'cell':<20} {'(r,a,bin)':>12} {'mean':>11} {'min':>11} {'max':>11}")
    stats = {}
    for label, cell in cells.items():
        v = out[(slice(SETTLE_FRAMES, None), *cell)]
        stats[label] = v
        print(f"  {label:<20} {str(cell):>12} {v.mean():11.6f} {v.min():11.6f} {v.max():11.6f}")

    breath = stats["breathing 0.3 Hz"]
    motion = stats["moving 2.0 Hz"]
    flat = stats["empty (flat zero)"]
    noise = stats["empty (noise)"]

    print(
        f"\n  breathing / moving   = {breath.mean() / motion.mean():10.1f}x"
        f"   (worst case, min/max = {breath.min() / motion.max():.1f}x)"
    )
    print(f"  breathing / noise    = {breath.mean() / max(noise.mean(), 1e-30):10.1f}x")
    print(f"  flat-zero cell       = {flat.max():.3e} (max over steady state)")

    # The 2 Hz cell must be suppressed by more than the ~39 dB (89x) the
    # response predicts is available; assert an order of magnitude to leave
    # room for the rectifier without ever letting a motion cell pass as breath.
    assert breath.min() > 10 * motion.max(), (
        f"2 Hz motion not suppressed: breath min {breath.min():.6g} "
        f"vs motion max {motion.max():.6g}"
    )
    assert flat.max() == 0.0, f"flat-zero cell is not zero: {flat.max():.6g}"
    assert breath.min() > 100 * noise.max(), (
        f"noise not suppressed: breath min {breath.min():.6g} vs noise max {noise.max():.6g}"
    )
    print("\n  OK: the 0.3 Hz cell dominates at every steady-state frame.")

    print("\n=== 3. output contract ===")
    print(f"  shape {out.shape}  dtype {out.dtype}  min {out.min():.6g}  max {out.max():.6g}")
    assert out.shape == z.shape, f"shape {out.shape} != input {z.shape}"
    assert out.dtype == np.float32, f"dtype {out.dtype}, expected float32"
    assert np.isfinite(out).all(), "non-finite values in the channel"
    assert (out >= 0).all(), f"negative values (min {out.min()}) -- breath must be unsigned"
    print("  OK: (T,6,3,47) float32, finite, non-negative everywhere.")

    print("\n=== 4. cold start ===")
    print(f"  {'frame':>6} {'breathing cell':>16} {'empty (flat)':>14}")
    for i in (0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000):
        print(f"  {i:>6} {out[(i, *cells['breathing 0.3 Hz'])]:16.6f} "
              f"{out[(i, *cells['empty (flat zero)'])]:14.6f}")
    # The real risk is a startup RINGING that reads as breathing before any
    # signal exists, so bound the whole transient against the steady state
    # rather than against itself.
    breathing = out[(slice(None), *cells["breathing 0.3 Hz"])]
    steady = breathing[SETTLE_FRAMES:].mean()
    overshoot = breathing.max() / steady
    assert np.isfinite(out[:SETTLE_FRAMES]).all(), "non-finite value during settling"
    assert overshoot < 1.5, f"cold-start transient overshoots {overshoot:.2f}x steady state"
    assert out[(0, *cells["breathing 0.3 Hz"])] < 1e-6, "frame 0 is not ~zero"
    print(
        f"\n  frame 0 = {out[(0, *cells['breathing 0.3 Hz'])]:.3e}; peak transient "
        f"{breathing.max():.6f} = {overshoot:.3f}x the steady state {steady:.6f}."
    )
    print(
        "  OK: rises INTO the band over ~4 s and settles with a 6% overshoot -- "
        "bounded,\n      never a blow-up, and never above the steady state by "
        "enough to read as a\n      detection. `lfilter_zi` freezes the scene at "
        "z[0] and a bandpass rejects DC,\n      so the cold start is ~0 rather "
        "than a step -- the same graceful degradation\n      `lagged` gets from "
        "clamping, with no explicit clamp needed."
    )


def report_state_cost():
    """Check 5 -- SRAM the constraint checker cannot see."""
    b, a = _design()
    order = len(a) - 1
    print("\n=== 5. filter state cost (real SRAM, invisible to evaluate_constraint) ===")
    for name, dtype, per_cell in (
        ("bandpass, complex128 (as computed here)", "complex128", order * 16 * 2),
        ("bandpass, complex64 (DIVERGES -- see 6)", "complex64", order * 8 * 2),
        ("bandpass, float32 sos biquads (deployable)", "float32", order * 4 * 2),
        ("smoothing EMA, float32", "float32", 1 * 4),
    ):
        total = per_cell * N_CELLS
        print(f"  {name:<44} {per_cell:>5} B/cell x {N_CELLS} = {total / 1024:8.1f} KB")

    # What a deployment would actually carry: an sos biquad cascade holding 2
    # float32 states per section per component (re, im), plus the 1-tap EMA.
    n_sections = order // 2
    sos_bytes = n_sections * 2 * 2 * 4 * N_CELLS
    ema_bytes = 4 * N_CELLS
    total = sos_bytes + ema_bytes
    print(
        f"\n  Deployable total: {n_sections} biquad sections x 2 states x "
        f"(re, im) x 4 B + 1 EMA state\n"
        f"                    = {total} B = {total / 1024:.1f} KB for all {N_CELLS} cells."
    )
    print(
        f"  Compare 4.1d's 83-frame magnitude ring buffer at ~183 KB: this is "
        f"{183 / (total / 1024):.0f}x SMALLER.\n"
        f"  A recursive filter carries {order} numbers per cell regardless of how "
        f"far back it\n  integrates; a ring buffer carries one per frame of history. "
        f"That is the whole\n  structural advantage of doing the integration in "
        f"the filter."
    )


def report_float32_guard():
    """Check 6 -- pin the divergence that forces float64, so nobody undoes it."""
    b, a = _design()
    t = np.arange(7500) / FRAME_RATE_HZ  # a real 300 s window
    x = (AMPLITUDE * np.exp(2j * np.pi * BREATH_HZ * t)).astype(np.complex64)

    y64, _ = scipy.signal.lfilter(b, a, x, zi=scipy.signal.lfilter_zi(b, a) * x[0])
    y32, _ = scipy.signal.lfilter(
        b.astype(np.float32),
        a.astype(np.float32),
        x,
        zi=(scipy.signal.lfilter_zi(b, a) * x[0]).astype(np.complex64),
    )
    sos = scipy.signal.butter(
        BREATH_FILTER_ORDER, [BREATH_LOW_HZ, BREATH_HIGH_HZ],
        btype="band", fs=FRAME_RATE_HZ, output="sos",
    )
    zi_sos = scipy.signal.sosfilt_zi(sos)[:, :, None] * x[0]
    ysos, _ = scipy.signal.sosfilt(sos.astype(np.float32), x, zi=zi_sos.astype(np.complex64)[..., 0])

    print("\n=== 6. float32 divergence guard (300 s window, 7500 frames) ===")
    print(f"  direct form, float64 coeffs : finite={np.isfinite(y64).all()}  |y|max {np.abs(y64).max():.6f}")
    bad = np.flatnonzero(~np.isfinite(y32))
    print(
        f"  direct form, float32 coeffs : finite={np.isfinite(y32).all()}"
        + (f"  FIRST NaN AT FRAME {bad[0]} ({bad[0] / FRAME_RATE_HZ:.1f} s)" if len(bad) else "")
    )
    print(
        f"  sos biquads,  float32 coeffs : finite={np.isfinite(ysos).all()}  "
        f"rel err vs float64 {np.abs(ysos - y64).max() / np.abs(y64).max():.2e}"
    )
    assert np.isfinite(y64).all(), "the float64 path must be stable"
    assert not np.isfinite(y32).all(), (
        "float32 direct form no longer diverges -- if scipy changed, re-derive "
        "the dtype note in channels.breath_envelope before relaxing it"
    )
    print(
        "\n  OK, and this is the load-bearing part: the complex128 state that "
        "`lfilter_zi`\n      produces is REQUIRED, not an accidental upcast to "
        "optimize away. Any\n      deployment must use the sos cascade, which is "
        "stable in float32."
    )


def report_real_data():
    """Feasibility only -- per-cell d', which discipline #9 says never predicts."""
    from src.cache import cached_complex
    from src.helpers.seated_diagnostics import band_snr, d_prime

    print("\n=== real data (feasibility check only -- see discipline #9) ===")
    z_seated = cached_complex("multi-person-localization/data/window_000021.npz")
    z_empty = cached_complex("multi-person-localization/data/window_000022.npz")

    # Same cell-selection rule as seated_diagnostics: by band SNR, because the
    # range-bin axis has an unresolved constant offset.
    cell = np.unravel_index(band_snr(z_seated).argmax(), z_seated.shape[1:])
    print(f"  w021 (seated alone) vs w022 (empty), cell radar {cell[0] + 1} "
          f"antenna {cell[1]} bin {cell[2]}")

    b_seated = channel(z_seated, "breath")[(slice(None), *cell)]
    b_empty = channel(z_empty, "breath")[(slice(None), *cell)]
    m_seated = channel(z_seated, "mag")[(slice(None), *cell)]
    m_empty = channel(z_empty, "mag")[(slice(None), *cell)]

    print(f"\n  {'feature':<40} {'d-prime':>9}")
    print(f"  {'mag (deployed, single frame)':<40} {d_prime(m_seated, m_empty):9.3f}")
    print(f"  {'breath (causal, this channel)':<40} {d_prime(b_seated, b_empty):9.3f}")
    print(
        "\n  Reference points from REPORT_NOTES Step 2, same cell rule: magnitude "
        "1.436,\n  non-causal filtfilt bandpass 2.452 (an upper bound -- it reads "
        "the future).\n  A per-cell d' is a feasibility check and never a "
        "prediction: the model sees\n  all 846 cells and the other 845 carry the noise."
    )


def main(real):
    report_response()
    report_synthetic()
    report_state_cost()
    report_float32_guard()
    if real:
        report_real_data()
    print("\nAll checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-real",
        action="store_true",
        help="Skip the real-data d' section (which needs the dataset cache)",
    )
    args = parser.parse_args()
    main(real=not args.no_real)

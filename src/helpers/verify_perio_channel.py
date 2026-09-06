"""Prove the "perio" channel keys on REPETITION, not on in-band energy.

This is the successor to `verify_breath_channel.py`, and the difference between
the two files is the whole lesson of REPORT_NOTES Step 7. That verifier passed
six asserted checks and still could not predict the failure, because every check
was synthetic and the failure was in the MODELLING ASSUMPTION: real walking is a
1-3 s broadband transient with large 0.15-0.6 Hz energy, not the out-of-band
2 Hz sinusoid the synthetic stood in for. Measured after the fact, walking
windows carry 5-13x MORE in-band energy than the seated window -- so the channel
built to detect breathing was dominated by walkers, and walking recall paid
5-10 paired SE for a 1-2 SE seated gain.

REPORT_NOTES discipline #13/#15/#16 are the rules that came out of that
and out of Step 8. Sections 5b and 6 below are the gates, they are the
load-bearing part of this file, and they EXIT NON-ZERO (discipline #17) -- if
they fail, no CV run is spent.

WHAT CHANGED FOR STEP 9, since this file judged Step 8's channel and killed it:
the channel now returns the COHERENCE |smoothed(z(t) conj(z(t-lag)))| /
smoothed(|z(t)|^2) rather than the raw smoothed product. Step 8's own synthetic
table printed both factors and showed the mechanism was never the problem: every
breather scored coherence 1.0000 and every one-off walker 0.0000-0.24, while the
returned |A|^2 x coherence handed the verdict to amplitude. So three checks here
are new, each one aimed at something Step 8's gate did not cover:

  * Check 2 asserts AMPLITUDE INVARIANCE directly -- the same periodic cell at
    two amplitudes must produce the same coherence. Step 8's gate never ran this,
    and it is the single property the fix exists to deliver.
  * Check 4 actually RUNS. It was unreachable dead code behind an early `return`
    in the Step 8 version, while the verdict block printed it as
    "PASS (asserted)". It matters more now: a ratio can produce NaN where a
    product cannot, and the frame-0 self-correlation |z_0|^2 / |z_0|^2 = 1.0 is a
    full-scale phantom that only exists once the statistic is normalized.
  * Check 5b is a SRAM GATE over the COMBINED total -- streaming declutter, the
    model's own freshly measured activation arena, and this channel's history
    buffer, together against 512 KB. Step 8 weighed the buffer in isolation
    (680.8 KB) and so could say neither how far over budget the real deployment
    was nor what would bring it under.
  * GATE 3 in section 6 promotes the int8-representability measurement from a
    printed number to a hard gate, and measures it as a CONTRAST above an empty
    cell rather than as an absolute fraction of p99. For `mag` the two agree
    (its empty cells sit near zero); for a normalized coherence they do not, and
    the contrast is the one that describes what the model can actually see.

What is checked, and why each one:

1. Synthetic mechanism. A genuinely periodic cell must read clearly HIGHER than
   a one-off broadband transient of the SAME and of 20x LARGER amplitude, and
   ~0 on an empty cell. Step 7 got exactly this comparison backwards on real
   data, so it is asserted rather than assumed.
2. Amplitude invariance and lag mismatch. Both are design premises, so both are
   measured: the same oscillation at 1x and 1000x amplitude must give the same
   number, and rates deliberately between the candidate lags must still respond.
3. Adversarial periodicity. A walker's gait (~2 Hz) and a serpentine walker who
   re-crosses the same cell are the two ways a WALKER could look periodic. These
   are reported, not asserted -- they are the mechanism to suspect first if the
   CV run shows a walking regression.
4. Contract: shape, float32, finite, non-negative, bounded -- "perio" is not in
   SIGNED_CHANNELS, so a negative value would be clipped to 0 by
   `normalize_channels` and quietly halve the channel. Plus a cold start bounded
   against the steady state, since inference must emit a position for frame 0.
5. Dtype, measured at both precisions over a full 7500-frame window (5a), and
   THE SRAM GATE (5b). 5a is the MIRROR IMAGE of `verify_breath_channel.py`'s
   check 6: that filter diverges to NaN in float32 and this one does not, and the
   guard pins down which is which so neither note gets copied onto the wrong
   channel.
6. THE ACCURACY GATE -- real-data distributions at ground-truth-occupied cells,
   split by walking vs seated, against genuinely empty cells. Three things must
   ALL hold: seated-occupied separates from empty, walking-occupied does not read
   higher than seated-occupied, and the seated cell survives p99 normalization
   with enough int8 levels to be representable at all. The second is what broke
   in Step 7; the third is what killed Step 8.

Run:  .venv/bin/python src/helpers/verify_perio_channel.py
      .venv/bin/python src/helpers/verify_perio_channel.py --no-real   (1-5 only)
      .venv/bin/python src/helpers/verify_perio_channel.py --arena-kb 27.6
                                          (skip the ~90 s arena re-measurement)
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.signal

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.channels import (
    PERIO_ENERGY_FLOOR,
    PERIO_LAGS,
    PERIO_SMOOTH_CYCLES,
    channel,
    format_radars,
    lagged,
)
from src.preprocessing import CROP_START, FRAME_RATE_HZ

# Grid the model actually sees: 6 radars x 3 antennas x 47 range bins.
N_RADARS, N_ANTENNAS, N_BINS = 6, 3, 47
N_CELLS = N_RADARS * N_ANTENNAS * N_BINS  # 846

# The radar subset this channel would actually DEPLOY on. Radars 1,2,4,6 (both
# short-wall radars plus one from each long-wall pair), 0-indexed here. Not an
# arbitrary trim: REPORT_NOTES' geometric feasibility check tested all 11 seated
# identities across all 8 seated windows against each radar's real +-60 deg FOV
# and found ZERO blind spots in all four possible pairings. It is the one lever
# in section 5b that costs no coverage, which is why it is taken before the lag
# trim. Section 6 measures the real-data gate on these radars only, because a
# statistic pooled over radars the deployment will not read is not the statistic
# the model gets fed.
DEPLOY_RADARS = (0, 1, 3, 5)

# ESP32-S3 with no external PSRAM (project_spec.md; the same 512 KB Phase
# 4.1d and 4.1e are written against). KB means KiB throughout this file, matching
# how every prior SRAM figure in REPORT_NOTES was computed.
SRAM_BUDGET_B = 512 * 1024

# Streaming-declutter working set, read off `remove_clutter_streaming` in
# submission/code.py (restated, not imported -- submission/ is a deliverable, not
# a library). Per raw CIR frame of 6 x 3 x 120 x [I,Q] = 4320 values it holds
# four arrays: the persistent lfilter state `zi` (float64, because `lfilter_zi`
# is float64 and the product with a float32 frame promotes), the input `frame`
# (float32), lfilter's `background` output (float64), and the yielded
# decluttered frame (float32).
DECLUTTER_VALUES = 6 * 3 * 120 * 2
DECLUTTER_BYTES = DECLUTTER_VALUES * (8 + 4 + 8 + 4)

SYNTH_FRAMES = 4500
SETTLE_FRAMES = 2000

AMPLITUDE = 0.1
# Real walkers are not equal-amplitude with breathers: Step 7 measured mean
# per-frame peak magnitude of 601-1652 in walking windows against 41 in the
# seated one. 20x is a conservative stand-in for that, and it is the honest
# version of the test -- an equal-amplitude comparison is the easy case.
WALKER_GAIN = 20.0
# For the amplitude-invariance check: the same breather, three decades apart.
INVARIANCE_GAINS = (1.0, 1000.0)


# --- Room geometry (project_spec.md SS3; same numbers as visualize_fov.py, ----
# restated rather than imported because that module pulls in matplotlib).
RADAR_XY = np.array(
    [(2.4, 0.0), (4.8, 1.8), (4.8, 5.4), (2.4, 7.2), (0.0, 5.4), (0.0, 1.8)]
)
# Boresight unit vectors: Up, Left, Left, Down, Right, Right.
RADAR_FACING = np.array([(0, 1), (-1, 0), (-1, 0), (0, -1), (1, 0), (1, 0)], dtype=float)
FOV_COS = np.cos(np.deg2rad(60.0))  # +-60 degrees
BIN_METRES = 0.15

DATA = "multi-person-localization/data/window_{:06d}.npz"
# Seated-only, walking-only (incl. the serpentine walker who deliberately
# re-crosses the same cells), walking-heavy, mixed, and both empty rooms.
GATE_WINDOWS = [16, 5, 19, 4, 10, 21, 22, 23]
# Skip each window's filter transient before aggregating, on the same rule as
# the synthetic: > 4x the longest EMA time constant.
REAL_SETTLE = 1500
# A cell counts as "empty background" only if its nominal range is at least this
# far from EVERY active person's range on that radar.
EMPTY_CLEARANCE_M = 1.5

# Gate thresholds.
GATE1_MIN_SEPARATION = 2.0   # seated-occupied vs empty, clean windows
GATE2_MAX_DOMINANCE = 1.5    # walking-occupied vs seated-occupied, clean windows
# Step 8 measured, on the median w021 seated cell: `mag` 10.6 int8 levels,
# `breath` 20.1 (and `breath` bought a real +1.2 to +2.1 SE seated gain on that),
# `perio` 0.8 (and 0.8 cannot buy anything). The bar is double digits, i.e. at
# least the deployed magnitude baseline's own 10.6 -- a channel that hands the
# model less signal than the input it already has is not worth a second channel's
# flash, whatever its ratios look like in absolute units.
GATE3_MIN_INT8_LEVELS = 10.0


# --- 1-4. synthetic ----------------------------------------------------------


def _synthetic():
    """(T, 6, 3, 47) complex64 with one cell per behaviour under test."""
    t = np.arange(SYNTH_FRAMES)
    z = np.zeros((SYNTH_FRAMES, N_RADARS, N_ANTENNAS, N_BINS), dtype=np.complex64)
    cells, kinds = {}, {}

    def put(label, cell, values, kind):
        cells[label], kinds[label] = cell, kind
        z[(slice(None), *cell)] = values

    def tone(period_frames, gain=1.0):
        return gain * AMPLITUDE * np.exp(2j * np.pi * t / period_frames)

    def transient(centre, width_frames, gain):
        """One broadband crossing: a Gaussian-enveloped Doppler chirp, once."""
        env = np.exp(-0.5 * ((t - centre) / (width_frames / 4)) ** 2)
        # Phase ramp = the range change of a body crossing the bin, which is
        # what makes a crossing broadband rather than a tone.
        return gain * AMPLITUDE * env * np.exp(2j * np.pi * 0.02 * (t - centre))

    # Genuine respiration, exactly at a candidate lag and between candidates.
    put("breathing, lag 82 (0.305 Hz)", (0, 0, 10), tone(82), "periodic")
    put("breathing, lag 70 (0.357 Hz)", (0, 0, 11), tone(70), "periodic")
    put("breathing, lag 91 (0.275 Hz)", (0, 0, 12), tone(91), "periodic")
    # THE amplitude-invariance pair: one oscillation, two amplitudes 1000x apart.
    # Both must return the same coherence. This is the property the Step 9 fix
    # exists to deliver and the one Step 8's gate never checked.
    for i, gain in enumerate(INVARIANCE_GAINS):
        put(f"breathing, lag 82, x{gain:g} amp", (0, 1, 20 + i), tone(82, gain), "invariance")
    # One-off crossings: what a walker does to a cell it passes through once.
    put("transient 1 s, x20 amp", (1, 1, 20), transient(2600, 25, WALKER_GAIN), "transient")
    put("transient 3 s, x20 amp", (1, 1, 21), transient(3400, 75, WALKER_GAIN), "transient")
    put("transient 3 s, x1 amp", (1, 1, 22), transient(3400, 75, 1.0), "transient")
    # Adversarial: the two ways a WALKER could look periodic.
    put(
        "gait 2 Hz burst, x20 amp",
        (2, 0, 25),
        transient(3000, 100, WALKER_GAIN) * np.exp(2j * np.pi * t / 12.5),
        "adversarial",
    )
    laps = sum(transient(c, 40, WALKER_GAIN) for c in range(600, SYNTH_FRAMES, 100))
    put("serpentine, re-crosses/4 s", (2, 0, 26), laps, "adversarial")
    # Nulls.
    put("empty (flat zero)", (3, 2, 30), 0.0, "empty")
    rng = np.random.default_rng(0)
    put(
        "empty (noise)",
        (3, 2, 31),
        AMPLITUDE * 0.01 * (rng.standard_normal(SYNTH_FRAMES) + 1j * rng.standard_normal(SYNTH_FRAMES)),
        "empty",
    )
    return z, cells, kinds


def report_synthetic():
    """Checks 1-4 -- mechanism, amplitude invariance, lag mismatch, contract."""
    z, cells, kinds = _synthetic()
    out = channel(z, "perio")

    print("=== 1-3. synthetic mechanism ===")
    print(
        f"  {SYNTH_FRAMES} frames ({SYNTH_FRAMES / FRAME_RATE_HZ:.0f} s), lags "
        f"{PERIO_LAGS}, steady state = frames {SETTLE_FRAMES}+ "
        f"({SETTLE_FRAMES / FRAME_RATE_HZ:.0f} s). Breathers at amplitude "
        f"{AMPLITUDE}, walkers at {WALKER_GAIN:g}x that."
    )
    # The channel now RETURNS the coherence, so `response` and `coherence` are
    # the same column and `|z|^2` is printed only to show what has been divided
    # out -- i.e. to show that cells 400x apart in energy land in the same place.
    print(
        f"\n  {'cell':<28} {'kind':<12} {'coherence':>10} {'|z|^2':>12} "
        f"{'vs breather':>12}"
    )
    stats, power = {}, {}
    for label, cell in cells.items():
        stats[label] = out[(slice(SETTLE_FRAMES, None), *cell)]
        power[label] = (np.abs(z[(slice(SETTLE_FRAMES, None), *cell)]) ** 2).mean()
    ref_mean = stats["breathing, lag 82 (0.305 Hz)"].mean()
    for label in cells:
        print(
            f"  {label:<28} {kinds[label]:<12} {stats[label].mean():10.4f} "
            f"{power[label]:12.6f} {stats[label].mean() / ref_mean:11.2f}x"
        )

    def group(kind):
        return {k: v for k, v in stats.items() if kinds[k] == kind}

    periodic, transients = group("periodic"), group("transient")
    empties, adversarial = group("empty"), group("adversarial")
    invariance = group("invariance")
    worst_periodic = min(v.mean() for v in periodic.values())
    worst_transient = max(v.mean() for v in transients.values())
    worst_empty = max(v.max() for v in empties.values())
    worst_adversarial = max(v.mean() for v in adversarial.values())

    # Contract-level facts: a genuine oscillation reads coherence ~1 whatever its
    # rate AND whatever its amplitude, and an empty cell reads ~0.
    assert worst_empty < 0.5, f"empty cells not suppressed: {worst_empty:.6g}"
    assert stats["empty (flat zero)"].max() == 0.0, (
        "flat-zero cell is not exactly zero -- the PERIO_ENERGY_FLOOR guard is "
        "not doing its job and a 0/0 has leaked through"
    )
    assert worst_periodic > 0.9, (
        f"a breather scored coherence {worst_periodic:.4f}, not ~1.0 -- the "
        f"normalization is wrong"
    )

    print("\n=== 2a. AMPLITUDE INVARIANCE -- the property the Step 9 fix exists for ===")
    inv = [(label, stats[label].mean(), power[label]) for label in invariance]
    print(f"\n  {'cell':<30} {'|z|^2':>14} {'coherence':>10}")
    for label, mean, pwr in inv:
        print(f"  {label:<30} {pwr:14.6f} {mean:10.4f}")
    lo, hi = min(m for _, m, _ in inv), max(m for _, m, _ in inv)
    p_lo, p_hi = min(p for _, _, p in inv), max(p for _, _, p in inv)
    drift = abs(hi - lo) / max(lo, 1e-30)
    print(
        f"\n  energy ratio between the two cells   {p_hi / p_lo:12.1f}x\n"
        f"  coherence ratio between the two cells {hi / max(lo, 1e-30):12.6f}x  "
        f"(relative drift {drift:.2e})"
    )
    assert drift < 1e-3, (
        f"coherence moved {drift:.3e} across a {p_hi / p_lo:.0f}x energy change "
        f"-- the channel is NOT amplitude-invariant, which is the entire point "
        f"of the Step 9 normalization"
    )
    print(
        "  OK: a 1e6x change in |z|^2 moves the output by less than 0.1%. That is\n"
        "      the algebraic identity (scale z by k and both sides scale by k^2)\n"
        "      confirmed NUMERICALLY, which is what Step 8's gate was missing --\n"
        "      it printed a coherence column but never asserted it was invariant."
    )

    ref = stats["breathing, lag 82 (0.305 Hz)"].mean()
    assert min(v.mean() for v in periodic.values()) > 0.5 * ref, (
        "an off-lag breather lost more than half its response -- the fixed-lag "
        "premise does not hold as stated"
    )
    print(
        "\n=== 2b. lag mismatch -- the premise that fixed lags suffice ===\n"
    )
    for label in periodic:
        print(
            f"    {label:<30} {stats[label].mean() / ref:6.3f} x the on-lag "
            f"response, coherence {stats[label].mean():.4f}"
        )
    print(
        "    OK: a rate BETWEEN the candidate lags keeps its full response. A\n"
        "        mismatched-but-consistent lag settles at a fixed phase offset and "
        "the\n        magnitude discards it -- only an unpredictably DRIFTING rate "
        "erodes this.\n"
        "        This is why dropping lag 100 for SRAM (section 5b) is cheap: the\n"
        "        multi-lag hedge was insurance against a risk this measurement says\n"
        "        is small."
    )

    # SEPARATION is a FINDING, not an assert: this file has to print its numbers
    # even -- especially -- when they are bad, because those numbers are the
    # result. Step 7's verifier asserted its way to a pass and the experiment
    # still failed on an axis no assert covered.
    print("\n=== 3. separation, the property the channel exists to have ===")
    print(
        f"    weakest periodic (mean)          {worst_periodic:10.4f}\n"
        f"    strongest one-off transient      {worst_transient:10.4f}  "
        f"= {worst_transient / worst_periodic:6.2f}x a breather\n"
        f"    strongest adversarial walker     {worst_adversarial:10.4f}  "
        f"= {worst_adversarial / worst_periodic:6.2f}x a breather\n"
        f"    strongest empty cell             {worst_empty:10.4f}"
    )
    ok_transient = worst_transient < worst_periodic
    ok_adversarial = worst_adversarial < worst_periodic
    print(
        f"\n    one-off transient below a breather : "
        f"{'PASS' if ok_transient else 'FAIL'}\n"
        f"    adversarial walker below a breather: "
        f"{'PASS' if ok_adversarial else 'FAIL'}"
    )
    if not ok_adversarial:
        print(
            "\n    NOTE: the adversarial cells are DESIGNED caricatures, not measured\n"
            "    lap times -- a walker who re-crosses one cell on a metronome is the\n"
            "    worst case a periodicity detector has, and it is reported rather\n"
            "    than asserted for that reason. w019 (the real serpentine walker) in\n"
            "    section 6 is the measurement that decides."
        )

    print("\n=== 4. output contract and cold start ===")
    print(f"  shape {out.shape}  dtype {out.dtype}  min {out.min():.6g}  max {out.max():.6g}")
    assert out.shape == z.shape, f"shape {out.shape} != input {z.shape}"
    assert out.dtype == np.float32, f"dtype {out.dtype}, expected float32"
    assert np.isfinite(out).all(), (
        "non-finite values in the channel -- a coherence is a RATIO, so unlike "
        "Step 8's raw product it can produce NaN/Inf; PERIO_ENERGY_FLOOR is the "
        "guard and it has failed"
    )
    assert (out >= 0).all(), f"negative values (min {out.min()}) -- perio must be unsigned"
    # Not a mathematical bound (Cauchy-Schwarz allows a decaying cell slightly
    # over 1), but a runaway ratio would show up here as a huge number rather
    # than as a 1.02.
    assert out.max() < 2.0, (
        f"coherence reached {out.max():.4g} -- far past the ~1 a normalized "
        f"correlation should approach, so the denominator is collapsing somewhere"
    )

    breather = out[(slice(None), *cells["breathing, lag 82 (0.305 Hz)"])]
    steady = breather[SETTLE_FRAMES:].mean()
    print(f"\n  {'frame':>6} {'breathing cell':>16} {'empty (flat)':>14} {'ALL-cell max':>14}")
    for i in (0, 1, 60, 100, 250, 500, 1000, 2000, 4000):
        print(
            f"  {i:>6} {breather[i]:16.6f} "
            f"{out[(i, *cells['empty (flat zero)'])]:14.6f} {out[i].max():14.6f}"
        )
    # The frame-0 bound is the interesting one and it is NOT free. `lagged`
    # clamps to z_0, so the frame-0 conjugate product is z_0 conj(z_0) = |z_0|^2
    # -- the denominator exactly. A numerator EMA warm-started there reads
    # coherence 1.0000 in EVERY cell on frame 0, a full-scale phantom on the one
    # frame inference cannot skip. `_perio` zero-starts the numerator instead, so
    # the bound below is 1/(cycles*shortest lag).
    frame0_bound = 1.0 / (PERIO_SMOOTH_CYCLES * min(PERIO_LAGS))
    assert out[0].max() <= frame0_bound * 1.001, (
        f"frame 0 peaks at {out[0].max():.6g} across all cells, above the "
        f"1/(cycles*min lag) = {frame0_bound:.6g} a zero-started numerator "
        f"allows -- the self-correlation cold start is back"
    )
    assert breather.max() <= steady * 1.05, (
        f"cold start overshoots: peak {breather.max():.6g} vs steady {steady:.6g}"
    )
    print(
        f"\n  OK: (T,6,3,47) float32, finite, non-negative, max {out.max():.4f}.\n"
        f"      Frame 0 peaks at {out[0].max():.6f} over ALL 846 cells, at or under "
        f"the\n      1/({PERIO_SMOOTH_CYCLES}*{min(PERIO_LAGS)}) = {frame0_bound:.6f} "
        f"bound, and the breathing cell rises\n      into its steady state "
        f"{steady:.4f} with no overshoot.\n"
        f"      (This check ran as dead code in the Step 8 version -- an early "
        f"`return`\n      sat above it while the verdict block reported it as "
        f"'PASS (asserted)'.)"
    )
    return ok_transient and ok_adversarial


# --- 5a. dtype ---------------------------------------------------------------


def report_dtype():
    """Check 5a -- the mirror image of verify_breath_channel's divergence guard."""
    print("\n=== 5a. dtype, measured at both precisions over a full 300 s window ===")
    t = np.arange(7500) / FRAME_RATE_HZ
    x = (AMPLITUDE * np.exp(2j * np.pi * 0.305 * t)).astype(np.complex64)
    prod = x * np.conj(lagged(x, 82))

    print(f"  {'lag':>5} {'alpha':>12} {'float64':>18} {'float32':>18} {'rel err':>10}")
    worst = 0.0
    for lag in PERIO_LAGS:
        alpha = 1.0 - 1.0 / (PERIO_SMOOTH_CYCLES * lag)
        b, a = [1 - alpha], [1, -alpha]
        zi = scipy.signal.lfilter_zi(b, a) * prod[0]
        y64, _ = scipy.signal.lfilter(b, a, prod, zi=zi)
        y32, _ = scipy.signal.lfilter(
            np.array(b, np.float32), np.array(a, np.float32), prod,
            zi=zi.astype(np.complex64),
        )
        bad = np.flatnonzero(~np.isfinite(y32))
        rel = np.abs(np.abs(y32) - np.abs(y64)).max() / np.abs(y64).max()
        worst = max(worst, rel)
        print(
            f"  {lag:>5} {alpha:12.8f} {'finite, |y|max ' + f'{np.abs(y64).max():.4f}':>18} "
            f"{('FIRST NaN @ ' + str(bad[0])) if len(bad) else 'finite':>18} {rel:10.2e}"
        )
        assert np.isfinite(y64).all(), f"lag {lag}: the float64 path must be stable"
        assert np.isfinite(y32).all(), (
            f"lag {lag}: float32 now diverges -- re-derive the dtype note in "
            f"channels.perio_coherence before trusting it"
        )

    # Why this differs from `breath`, which DOES diverge in float32 despite
    # having poles FURTHER from the unit circle. Pole radius is not the
    # criterion; direct-form tap cancellation is.
    b_br, a_br = scipy.signal.butter(3, [0.15, 0.6], btype="band", fs=FRAME_RATE_HZ)
    cancel_br = np.abs(a_br).sum() / abs(a_br.sum())
    a_pe = np.array([1.0, -(1.0 - 1.0 / (PERIO_SMOOTH_CYCLES * max(PERIO_LAGS)))])
    cancel_pe = np.abs(a_pe).sum() / abs(a_pe.sum())
    print(
        f"\n  Both precisions are stable, to {worst:.1e} relative error. That is "
        f"the OPPOSITE\n  of `breath`, whose float32 direct form diverges to NaN "
        f"at frame 3743 -- and it\n  is NOT because these poles are further out. "
        f"They are closer:\n"
        f"    breath  order 6, max|pole| {np.abs(np.roots(a_br)).max():.4f}, "
        f"denominator cancellation {cancel_br:.2e}x\n"
        f"    perio   order 1, max|pole| {-a_pe[1]:.4f}, "
        f"denominator cancellation {cancel_pe:.2e}x\n"
        f"  float32 resolves ~1e-7. A 6th-order direct form whose taps cancel to "
        f"1 part in\n  3.5e8 cannot survive it; a one-pole EMA cancelling to 1 "
        f"part in {cancel_pe:.0f} easily can.\n  So `breath` needs an sos biquad "
        f"cascade to deploy and `perio` does not.\n"
        f"  The energy denominator is a one-pole EMA on a REAL quantity, so it "
        f"inherits\n  the same margin and adds no new numerical hazard -- only the "
        f"division does,\n  and PERIO_ENERGY_FLOOR = {PERIO_ENERGY_FLOOR:g} is the "
        f"guard for that (check 4)."
    )


# --- 5b. THE SRAM GATE -------------------------------------------------------


def measure_arena_kb(n_radars, n_channels, override=None):
    """Activation arena of the candidate architecture, in KB, MEASURED.

    Built and converted here rather than quoted, because the figure has already
    been shown to vary by machine and by TF version and REPORT_NOTES discipline #2
    forbids citing a remembered one. Weight VALUES do not affect the arena --
    it is set by tensor shapes and the op graph -- so an untrained model
    converted through `quantize.py`'s exact export path measures the same number
    a trained one would, in ~90 s instead of ~45 min.

    Scored by `evaluation/evaluate_constraint.py` itself, via subprocess: that
    script is the authority on what counts as an activation tensor, and shelling
    out to it means this file cannot drift from it.
    """
    if override is not None:
        print(f"  arena: {override:.1f} KB (--arena-kb override, measurement skipped)")
        return override

    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    from src.model import build_grid_model

    print(
        f"  measuring the arena of a ({n_radars}, 3, 47, {n_channels}) model "
        f"through quantize.py's export path..."
    )
    model = build_grid_model(input_shape=(n_radars, 3, 47, n_channels))
    sig = [tf.TensorSpec((1, n_radars, 3, 47, n_channels), tf.float32)]
    with tempfile.TemporaryDirectory() as tmp:
        archive = tf.keras.export.ExportArchive()
        archive.track(model)
        archive.add_endpoint(
            name="serve", fn=lambda x: model(x, training=False), input_signature=sig
        )
        archive.write_out(f"{tmp}/export")
        conv = tf.lite.TFLiteConverter.from_saved_model(f"{tmp}/export")
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        rng = np.random.default_rng(0)
        samples = rng.random((200, 1, n_radars, 3, 47, n_channels), dtype=np.float32)
        conv.representative_dataset = lambda: ([s] for s in samples)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        path = Path(tmp) / "candidate.tflite"
        path.write_bytes(conv.convert())

        proc = subprocess.run(
            [sys.executable, "evaluation/evaluate_constraint.py", "--model-path", str(path)],
            capture_output=True,
            text=True,
        )
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("JSON_RESULT:")]
    if not line:
        raise RuntimeError(f"evaluate_constraint.py produced no JSON_RESULT:\n{proc.stdout}")
    res = json.loads(line[0].split("JSON_RESULT:", 1)[1])
    print(
        f"  evaluate_constraint.py on the candidate: {model.count_params():,} params, "
        f"flash {res['file_size_kb']:.1f} KB, arena {res['arena_kb']:.1f} KB, "
        f"hard checks {'PASS' if res['passed'] else 'FAIL'}"
    )
    return res["arena_kb"]


def report_sram_gate(arena_kb_override=None):
    """Check 5b -- the COMBINED SRAM total, not this channel's buffer alone.

    Step 8 weighed the history buffer on its own (680.8 KB against 512 KB) and
    stopped there, which could say neither how far over the real deployment was
    nor what would bring it under. Three costs share the 512 KB and all three are
    computed here from the code that would incur them.
    """
    print("\n=== 5b. THE SRAM GATE: declutter + arena + this channel, vs 512 KB ===")

    n_deploy = len(DEPLOY_RADARS)
    cells = n_deploy * N_ANTENNAS * N_BINS
    arena_kb = measure_arena_kb(n_deploy, 2, arena_kb_override)

    # Per cell: ONE complex64 history of the longest lag (shared by every lag --
    # one history of length max(PERIO_LAGS) serves all of them), one complex64
    # EMA state per lag for the numerators, and one float32 EMA state for the
    # shared energy denominator.
    ring_b = max(PERIO_LAGS) * 2 * 4
    num_ema_b = len(PERIO_LAGS) * 2 * 4
    den_ema_b = 4
    per_cell = ring_b + num_ema_b + den_ema_b
    buffer_b = per_cell * cells

    print(
        f"\n  Deployment under test: radars {format_radars(DEPLOY_RADARS)} "
        f"({n_deploy} of {N_RADARS}), lags {PERIO_LAGS},\n"
        f"  channel set mag+perio ({cells} cells of {N_CELLS}).\n"
    )
    print(f"  {'component':<44} {'bytes':>10} {'KB':>9}")
    print(f"  {'-' * 44} {'-' * 10} {'-' * 9}")
    print(
        f"  {'streaming declutter working set':<44} {DECLUTTER_BYTES:>10,} "
        f"{DECLUTTER_BYTES / 1024:9.1f}\n"
        f"  {'    zi state (6x3x120x2, float64)':<44} "
        f"{DECLUTTER_VALUES * 8:>10,} {DECLUTTER_VALUES * 8 / 1024:9.1f}\n"
        f"  {'    frame + background + output':<44} "
        f"{DECLUTTER_VALUES * 16:>10,} {DECLUTTER_VALUES * 16 / 1024:9.1f}"
    )
    print(
        f"  {'model activation arena (measured above)':<44} "
        f"{int(arena_kb * 1024):>10,} {arena_kb:9.1f}"
    )
    print(
        f"  {'perio history + filter state':<44} {buffer_b:>10,} "
        f"{buffer_b / 1024:9.1f}\n"
        f"  {f'    z history {max(PERIO_LAGS)} x (re,im) x 4 B':<44} "
        f"{ring_b * cells:>10,} {ring_b * cells / 1024:9.1f}\n"
        f"  {f'    {len(PERIO_LAGS)} numerator EMA states, complex':<44} "
        f"{num_ema_b * cells:>10,} {num_ema_b * cells / 1024:9.1f}\n"
        f"  {'    1 shared energy EMA state, real':<44} "
        f"{den_ema_b * cells:>10,} {den_ema_b * cells / 1024:9.1f}"
    )
    total_b = DECLUTTER_BYTES + int(arena_kb * 1024) + buffer_b
    print(f"  {'-' * 44} {'-' * 10} {'-' * 9}")
    print(f"  {'TOTAL':<44} {total_b:>10,} {total_b / 1024:9.1f}")
    print(f"  {'ESP32-S3 SRAM budget':<44} {SRAM_BUDGET_B:>10,} {SRAM_BUDGET_B / 1024:9.1f}")
    headroom = SRAM_BUDGET_B - total_b
    print(
        f"  {'headroom':<44} {headroom:>10,} {headroom / 1024:9.1f}"
        f"   ({100 * headroom / SRAM_BUDGET_B:+.1f}%)"
    )

    # What the two levers are worth, so the choice is visible rather than
    # asserted. Both are computed from the same formula as the row above.
    print("\n  What each lever buys (same formula, other configurations):\n")
    print(f"  {'radars':>7} {'lags':<14} {'buffer KB':>10} {'total KB':>9}  verdict")
    for radars in (N_RADARS, n_deploy):
        for lags in ((60, 82, 100), (60, 82), (82,)):
            c = radars * N_ANTENNAS * N_BINS
            b = (max(lags) * 8 + len(lags) * 8 + 4) * c
            tot = (DECLUTTER_BYTES + int(arena_kb * 1024) + b) / 1024
            mark = "OK" if tot < SRAM_BUDGET_B / 1024 else "over budget"
            here = "  <- this run" if radars == n_deploy and tuple(lags) == tuple(PERIO_LAGS) else ""
            print(
                f"  {radars:>7} {str(lags):<14} {b / 1024:10.1f} {tot:9.1f}  "
                f"{mark}{here}"
            )
    print(
        "\n  (The arena is held fixed across the rows above; it moves by ~1 KB "
        "between\n  4 and 6 radars, which changes no verdict.)"
    )

    ok = total_b < SRAM_BUDGET_B
    print(f"\n  SRAM GATE {'PASS' if ok else 'FAIL'}  -- "
          f"{total_b / 1024:.1f} KB {'<' if ok else '>='} {SRAM_BUDGET_B / 1024:.0f} KB")
    if ok:
        print(
            f"  Headroom is {headroom / 1024:.1f} KB ({100 * headroom / SRAM_BUDGET_B:.1f}%), "
            f"which is REAL but THIN: this accounting covers\n  the three costs "
            f"this project controls and not the IDF's own stack, heap\n  "
            f"fragmentation, or any driver buffers. Treat it as 'fits with nothing "
            f"to spare',\n  not as comfortable. Two further reductions exist and "
            f"neither is taken here:\n  decluttering only the "
            f"{n_deploy} deployed radars (a {DECLUTTER_BYTES * (N_RADARS - n_deploy) // N_RADARS / 1024:.1f} KB saving, "
            f"but a change to\n  submission/code.py, which this branch does not "
            f"touch) and storing the history\n  at reduced precision (a second "
            f"variable this run does not move)."
        )
    return ok


# --- 6. THE ACCURACY GATE: real-data distributions ---------------------------


def _gt_bins(people_xy, people_mask):
    """Nominal cropped range-bin index and FOV mask per (frame, person, radar)."""
    delta = people_xy[:, :, None, :] - RADAR_XY[None, None, :, :]  # (T,P,R,2)
    rng = np.hypot(delta[..., 0], delta[..., 1])
    cos = (delta * RADAR_FACING[None, None, :, :]).sum(-1) / np.maximum(rng, 1e-9)
    nominal = np.rint(rng / BIN_METRES).astype(np.int64) - CROP_START
    in_fov = (cos >= FOV_COS) & people_mask[:, :, None]
    # Radars the deployment will not read cannot contribute a sample.
    keep = np.zeros(N_RADARS, dtype=bool)
    keep[list(DEPLOY_RADARS)] = True
    return rng, nominal, in_fov & keep[None, None, :]


def _calibrate_offset(windows=(16, 17, 18)):
    """Where does a person ACTUALLY appear, relative to the nominal range bin?

    `seated_diagnostics` records an unresolved constant offset on the range axis
    (raw bin 28 ~ 4.2 m nominal against a 3.26 m true range), so a computed bin
    index is the wrong cell unless it is calibrated. Measured here on the
    single-walker windows using the `mag` channel ONLY -- independent of the
    channel under test, so the calibration cannot flatter it.
    """
    from src.cache import cached_complex

    print("\n=== 6a. range-bin offset, calibrated from data (uses `mag` only) ===")
    votes, profile = [], np.zeros(31)
    offsets = np.arange(-10, 21)
    for win in windows:
        raw = np.load(DATA.format(win))
        mag = channel(cached_complex(DATA.format(win)), "mag")
        _, nominal, in_fov = _gt_bins(raw["people_xy"], raw["people_mask"].astype(bool))
        # Antenna-summed magnitude, so the argmax is a per-(frame, radar) range.
        energy = mag.sum(axis=2)  # (T, R, 47)
        t_idx, p_idx, r_idx = np.nonzero(in_fov)
        keep = (nominal[t_idx, p_idx, r_idx] >= 0) & (nominal[t_idx, p_idx, r_idx] < N_BINS)
        t_idx, p_idx, r_idx = t_idx[keep], p_idx[keep], r_idx[keep]
        nom = nominal[t_idx, p_idx, r_idx]
        votes.append(energy[t_idx, r_idx].argmax(axis=1) - nom)
        for i, k in enumerate(offsets):
            b = np.clip(nom + k, 0, N_BINS - 1)
            profile[i] += energy[t_idx, r_idx, b].mean()
    votes = np.concatenate(votes)
    by_argmax = int(np.median(votes))
    by_profile = int(offsets[profile.argmax()])
    print(f"  windows {windows}, {len(votes)} (frame, person, radar) observations in FOV")
    print(f"  median(argmax bin - nominal bin) = {by_argmax:+d}   "
          f"(quartiles {np.percentile(votes, 25):+.0f} / {np.percentile(votes, 75):+.0f})")
    print(f"  argmax of the mean-magnitude-vs-offset profile = {by_profile:+d}")
    assert abs(by_argmax - by_profile) <= 3, (
        f"the two offset estimates disagree ({by_argmax} vs {by_profile}) -- the "
        f"GT-to-cell mapping is not trustworthy, so the gate below would be "
        f"measuring the wrong cells"
    )
    print(f"\n  OK: both estimates agree. Using offset {by_profile:+d} "
          f"({by_profile * BIN_METRES:+.2f} m).")
    return by_profile


def _window_stats(win, offset):
    """perio and mag at GT-occupied cells (by person kind) and at empty cells."""
    from src.cache import cached_complex
    from src.helpers.oof_breakdown import person_kinds

    raw = np.load(DATA.format(win))
    people_xy, people_mask = raw["people_xy"], raw["people_mask"].astype(bool)
    z = cached_complex(DATA.format(win))
    # `breath` is carried alongside purely as a REFERENCE SCALE for the p99
    # section below: it is the one Phase 4 channel measured to produce a real
    # seated gain (+1.2 to +2.1 paired SE, Step 7), so it turns an int8-levels
    # number from a bare figure into a comparison against a known outcome.
    feats = {"perio": channel(z, "perio"), "mag": channel(z, "mag"),
             "breath": channel(z, "breath")}
    kinds = person_kinds(people_xy, people_mask)

    rng, nominal, in_fov = _gt_bins(people_xy, people_mask)
    bins = nominal + offset
    sel = in_fov & (bins >= 1) & (bins < N_BINS - 1)
    sel[:REAL_SETTLE] = False  # drop each window's EMA transient

    out = {"occupied": {}, "empty": {}, "peak": {}, "sample": {}}
    for name, f in feats.items():
        # Everything below is measured on the DEPLOYED radar subset only -- the
        # p99 the model is normalized with is computed over exactly those radars
        # (data_loader.calculate_p99_from_files takes radar_indices), so pooling
        # over all 6 here would score a channel the deployment never sees.
        fd = f[:, DEPLOY_RADARS]
        # Peak-per-frame, the Step 7 comparison, so the two runs are readable
        # against each other directly.
        out["peak"][name] = fd[REAL_SETTLE:].reshape(len(fd) - REAL_SETTLE, -1).max(1).mean()
        # A flat 10%-of-frames draw, the same shape of sample
        # `calculate_p99_from_files` takes, so a p99 can be pooled over windows.
        out["sample"][name] = fd[REAL_SETTLE::10].ravel()

        for p, kind in enumerate(kinds):
            if kind is None:
                continue
            t_idx, r_idx = np.nonzero(sel[:, p, :])
            if not len(t_idx):
                continue
            b = bins[t_idx, p, r_idx]
            # Max over the 3 antennas and +-1 bin: a body is wider than one
            # 0.15 m bin, and the antennas are redundant by design.
            v = np.stack([f[t_idx, r_idx, a, b + d] for a in range(N_ANTENNAS)
                          for d in (-1, 0, 1)]).max(0)
            out["occupied"].setdefault(kind, {}).setdefault(name, []).append(v)

        # Empty background: bins whose nominal range clears EVERY active person
        # on that radar. Subsampled in time -- this is a distribution, not a sum.
        ts = np.arange(REAL_SETTLE, len(f), 25)
        bin_rng = (np.arange(N_BINS) + CROP_START - offset) * BIN_METRES
        far = np.ones((len(ts), N_RADARS, N_BINS), dtype=bool)
        for p, kind in enumerate(kinds):
            if kind is None:
                continue
            act = people_mask[ts, p]
            d = np.abs(bin_rng[None, None, :] - rng[ts, p, :, None])
            far &= (d > EMPTY_CLEARANCE_M) | ~act[:, None, None]
        far = np.repeat(far[:, :, None, :], N_ANTENNAS, axis=2)
        out["empty"][name] = f[ts][:, DEPLOY_RADARS][far[:, DEPLOY_RADARS]]

    for kind, d in out["occupied"].items():
        for name in d:
            d[name] = np.concatenate(d[name])
    return out, kinds


def report_gate():
    """Check 6 -- disciplines #13/#15/#16, applied literally. Decides the CV run."""
    offset = _calibrate_offset()

    print("\n=== 6b. THE GATE: `perio` on real data, at GT-occupied cells ===")
    print(
        f"  Windows {GATE_WINDOWS}, radars {format_radars(DEPLOY_RADARS)} only. "
        f"First {REAL_SETTLE} frames\n  dropped (EMA transient). An 'occupied' "
        f"sample is the max over 3 antennas and\n  +-1 bin at the calibrated GT "
        f"cell of an in-FOV radar; 'empty' is any cell\n  clearing every person "
        f"by {EMPTY_CLEARANCE_M:g} m."
    )
    print(
        f"\n  {'win':>4} {'scenario':<20} {'class':<9} {'n':>9} "
        f"{'p50':>10} {'p90':>10} {'p99':>10} {'perio/mag p50':>14}"
    )

    pooled = {"walking": [], "seated": [], "empty": []}
    # Windows containing exactly one class, so neither number is contaminated by
    # the other class's energy landing in the same range shell.
    clean = {"walking": [], "seated": [], "empty": []}
    peaks, samples = {}, {}
    for win in GATE_WINDOWS:
        stats, kinds = _window_stats(win, offset)
        present = {k for k in kinds if k is not None}
        scenario = " + ".join(
            f"{sum(1 for k in kinds if k == kind)} {kind}"
            for kind in ("walking", "seated")
            if any(k == kind for k in kinds)
        ) or "empty room"
        peaks[win] = (scenario, stats["peak"]["perio"], stats["peak"]["mag"])
        samples[win] = stats["sample"]

        rows = [(kind, d) for kind, d in stats["occupied"].items()]
        rows.append(("empty", stats["empty"]))
        for kind, d in rows:
            v, m = d["perio"], d["mag"]
            pooled[kind].append(v)
            if present == {"seated"} and kind in ("seated", "empty"):
                clean[kind].append(d)
            elif present == {"walking"} and kind == "walking":
                clean[kind].append((win, d))
            ratio = np.median(v) / max(np.median(m), 1e-30)
            print(
                f"  {win:>4} {scenario:<20} {kind:<9} {len(v):>9} "
                f"{np.median(v):10.4f} {np.percentile(v, 90):10.4f} "
                f"{np.percentile(v, 99):10.4f} {ratio:14.5f}"
            )

    print("\n  Mean per-frame PEAK of the channel, the Step 7 comparison table:")
    print(f"\n  {'win':>4} {'scenario':<20} {'perio peak':>12} {'mag peak':>12} {'ratio':>8}")
    for win, (scenario, p, m) in peaks.items():
        print(f"  {win:>4} {scenario:<20} {p:12.4f} {m:12.2f} {p / m:8.5f}")

    print("\n  Pooled across windows:")
    agg = {}
    for kind, vs in pooled.items():
        if not vs:
            continue
        v = np.concatenate(vs)
        agg[kind] = v
        print(
            f"    {kind:<9} n={len(v):>9}  p50 {np.median(v):8.4f}  "
            f"p90 {np.percentile(v, 90):8.4f}  mean {v.mean():8.4f}"
        )

    # The pooled "seated" number is CONTAMINATED and must not be read as-is: in
    # w004 and w010 a walker crossing a seated subject's range SHELL dumps its
    # own energy into that subject's cell. The only uncontaminated seated
    # measurement in the dataset is w021, the seated-only window -- which is also
    # the window this whole investigation is about. Discipline #16.
    clean_seated_by_channel = {
        name: np.concatenate([d[name] for d in clean["seated"]])
        for name in ("mag", "breath", "perio")
    }
    clean_empty_by_channel = {
        name: np.concatenate([d[name] for d in clean["empty"]])
        for name in ("mag", "breath", "perio")
    }
    clean_seated = clean_seated_by_channel["perio"]
    clean_empty = clean_empty_by_channel["perio"]
    print(
        "\n  CLEAN comparison -- seated-only window vs walking-only windows, no\n"
        "  cross-contamination in either direction:"
    )
    print(f"\n  {'':<34} {'p50':>10} {'p90':>10} {'vs w021 seated':>16}")
    ref = np.median(clean_seated)
    print(f"  {'w021 seated (no walkers present)':<34} {ref:10.4f} "
          f"{np.percentile(clean_seated, 90):10.4f} {1.0:15.2f}x")
    for win, d in clean["walking"]:
        v = d["perio"]
        print(
            f"  {'w%03d walking (no seated present)' % win:<34} {np.median(v):10.4f} "
            f"{np.percentile(v, 90):10.4f} {np.median(v) / max(ref, 1e-30):15.2f}x"
        )
    clean_walking = np.concatenate([d["perio"] for _, d in clean["walking"]])

    sep = ref / max(np.median(clean_empty), 1e-30)
    dom = np.median(clean_walking) / max(ref, 1e-30)
    dom_pooled = np.median(agg["walking"]) / max(np.median(agg["seated"]), 1e-30)
    print(
        f"\n  GATE 1  seated-occupied vs empty (both from w021)  = {sep:8.2f}x   "
        f"(need > {GATE1_MIN_SEPARATION:g}x)\n"
        f"  GATE 2  walking-occupied vs seated-occupied, clean  = {dom:8.2f}x   "
        f"(need <= {GATE2_MAX_DOMINANCE:g}x)\n"
        f"          same, pooled incl. contaminated mixed windows = {dom_pooled:6.2f}x   "
        f"(flattered -- see above)"
    )

    # What the MODEL sees, not what the channel computes. `normalize_channels`
    # clips at a single per-channel p99 pooled over every training window, so a
    # statistic that walking windows blow out leaves the seated subject squashed
    # against zero regardless of how well it separates in absolute terms.
    # Discipline #15, and it is what killed Step 8's version of this channel.
    print(
        "\n  GATE 3 -- what survives p99 normalization. The model sees "
        "clip(x, 0, p99)/p99\n  with ONE p99 pooled over all training windows "
        "(data_loader.calculate_p99_from_files):"
    )
    print(
        f"\n  {'channel':<8} {'pooled p99':>11} {'w021 seated':>12} {'w021 empty':>11} "
        f"{'norm seated':>12} {'norm empty':>11} {'levels vs empty':>16}   note"
    )
    levels = {}
    for name in ("mag", "breath", "perio"):
        p99 = float(np.percentile(np.concatenate([s[name] for s in samples.values()]), 99))
        seated_p50 = float(np.median(clean_seated_by_channel[name]))
        empty_p50 = float(np.median(clean_empty_by_channel[name]))
        # The CLIP is applied, not just the divide. `normalize_channels` is
        # clip(x, 0, p99)/p99, and for a bounded statistic like a coherence the
        # clip is not a formality: this channel's pooled p99 lands BELOW its own
        # seated value, so the seated cell saturates at 1.0 and an
        # unclipped seated/p99 would overstate what the model receives.
        norm_seated = min(seated_p50, p99) / p99
        norm_empty = min(empty_p50, p99) / p99
        # "Levels above an empty cell" is the number this measurement always
        # CLAIMED to be (see Step 8's own wording) and, for `mag`, it lands within
        # ~1 level of the absolute figure because an empty magnitude cell sits
        # near zero. For a NORMALIZED coherence it does not: an empty cell has a
        # non-zero floor of its own (uncorrelated noise still self-correlates at
        # ~1/sqrt(N_eff)), so an absolute fraction would flatter the channel. The
        # contrast is what the model can actually discriminate on, so it gates.
        levels[name] = 255 * (norm_seated - norm_empty)
        note = {
            "mag": "deployed baseline",
            "breath": "Step 7: +1.2 to +2.1 SE seated",
            "perio": "this channel",
        }[name]
        print(
            f"  {name:<8} {p99:11.4f} {seated_p50:12.4f} {empty_p50:11.4f} "
            f"{norm_seated:12.4f} {norm_empty:11.4f} {levels[name]:16.1f}   {note}"
        )
    print(
        "  The input quantizer spans [0, 1] in 255 int8 steps, so 'levels vs empty'\n"
        "  is how many quantization steps the median seated cell rises above an\n"
        "  empty one, AFTER clip(x, 0, p99)/p99. Below ~1 the subject is not\n"
        "  representable in the deployed input at all; Step 8's `perio` delivered\n"
        "  0.8 and was stopped here."
    )
    print(
        f"\n  GATE 3  seated int8 levels above empty = {levels['perio']:8.1f}     "
        f"(need >= {GATE3_MIN_INT8_LEVELS:g}; mag {levels['mag']:.1f}, "
        f"breath {levels['breath']:.1f})"
    )

    ok1 = sep > GATE1_MIN_SEPARATION
    ok2 = dom <= GATE2_MAX_DOMINANCE
    ok3 = levels["perio"] >= GATE3_MIN_INT8_LEVELS
    print(
        f"\n  GATE 1 {'PASS' if ok1 else 'FAIL'}   "
        f"GATE 2 {'PASS' if ok2 else 'FAIL'}   "
        f"GATE 3 {'PASS' if ok3 else 'FAIL'}"
    )
    if not (ok1 and ok2 and ok3):
        print(
            "\n  GATE FAILED -- do NOT spend a CV run. Step 7's whole lesson is "
            "that this\n  measurement is worth minutes and a CV run is worth "
            "~45 minutes, and that the\n  failure it catches is a design "
            "assumption a synthetic test cannot see."
        )
        return False
    print("\n  GATE PASSED on all three clauses -- a CV run is justified.")
    return True


def main(real, arena_kb):
    synthetic_ok = report_synthetic()
    report_dtype()
    sram_ok = report_sram_gate(arena_kb)
    gate_ok = report_gate() if real else None

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("  contract, cold start, lag-mismatch premise, dtype : PASS (asserted)")
    print("  amplitude invariance (check 2a)                   : PASS (asserted)")
    print(f"  synthetic separation (transient + adversarial)    : "
          f"{'PASS' if synthetic_ok else 'FAIL'}")
    print(f"  SRAM gate, combined total vs 512 KB (check 5b)    : "
          f"{'PASS' if sram_ok else 'FAIL'}")
    print(f"  real-data gate (disciplines #13/#15/#16)          : "
          f"{'skipped' if gate_ok is None else 'PASS' if gate_ok else 'FAIL'}")
    if synthetic_ok and sram_ok and gate_ok:
        print("\n  Every gate passes -- a CV run is justified.")
        return
    print(
        "\n  DO NOT SPEND A CV RUN. The point of this file is that a caught-"
        "before-the-run\n  negative result costs minutes where the run costs "
        "~45 minutes, and that it\n  catches design assumptions a synthetic pass "
        "cannot (REPORT_NOTES discipline #13,\n  #15, #17)."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-real",
        action="store_true",
        help="Skip section 6, the real-data gate (which needs the dataset cache)",
    )
    parser.add_argument(
        "--arena-kb",
        type=float,
        default=None,
        help=(
            "Skip the ~90 s arena re-measurement in 5b and use this figure. For "
            "iterating only -- a reported number must come from a real run."
        ),
    )
    args = parser.parse_args()
    main(real=not args.no_real, arena_kb=args.arena_kb)

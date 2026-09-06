"""The Phase 4 input-representation ablation: the deployed baseline and six
alternatives to it, none of which was adopted.

Kept in full as the experimental record. The deployed model uses `"mag"` and
nothing else; every other set here was built, measured on a full CV run (or
cancelled by its own pre-run gate), and rejected. `"mag+perio"` was tried in two
forms -- raw and amplitude-normalized -- so the six alternatives below amount to
seven experiments in all. `report/REPORT_NOTES.md` carries the reasoning and the
numbers -- the step number is cited next to each set below.

Nothing in `src/preprocessing.py` imports from this module; the dependency runs
one way only, so the deployed path cannot acquire an ablation dependency by
accident.
"""

import numpy as np
import scipy.signal

from src.preprocessing import FRAME_RATE_HZ

# --- Channel sets ------------------------------------------------------------
#
# A channel set is an ordered tuple of primitive channel names, each declared in
# `channel()` below. Signed channels normalize to [-1, 1] and unsigned ones to
# [0, 1]; clipping a signed channel at [0, p99] would flatten its negative half
# and destroy exactly the phase these sets exist to recover.
#
# Flash cost is (radars x 3 x 47 x channels) x 256 weights in the early-fusion
# Dense plus a fixed 114,275 after it, so channels and radars trade against each
# other. All 6 radars: 1 channel -> 330,851 params, 2 -> 547,427 (~578 KB),
# 3 -> 764,003 (~807 KB, over the limit). That is why the 4-channel set below
# only fits on a radar subset.
CHANNEL_SETS = {
    # The deployed baseline. Must stay byte-identical to `preprocessing.magnitude`.
    "mag": ("mag",),
    # Step 3 -- temporal phase at lag 32 (1.28 s, half a breath). Negative.
    "mag+delta32": ("mag", "delta32"),
    # Step 3 -- inter-antenna phase difference. Negative, and the axis bug that
    # made it difference RADARS rather than antennas is why the verifiers exist.
    "mag+aoa": ("mag", "aoa"),
    # Task board 4.1 read literally: raw I/Q instead of magnitude. Negative.
    "iq": ("re", "im"),
    # Step 7 -- respiration-band energy, integrated before the model sees a
    # frame. Negative: walking is a 1-3 s broadband transient whose spectrum
    # lands inside the band, so this became an in-band energy detector.
    "mag+breath": ("mag", "breath"),
    # Step 9 -- periodicity at the respiration lag, normalized to a coherence.
    # The largest regression in Phase 4: amplitude invariance leaves the
    # statistic no way to say "nobody is here", so empty cells self-correlate to
    # a 0.302 pedestal against `mag`'s 0.018.
    "mag+perio": ("mag", "perio"),
    # Step 5 -- one breathing cycle sampled at four instants, no differencing.
    # 82 frames = the measured 0.305 Hz period at 25 Hz. Only fits on 4 radars.
    "mag+mag27+mag55+mag82": ("mag", "maglag27", "maglag55", "maglag82"),
}

# Channels whose values straddle zero, and so normalize to [-1, 1]. "breath" and
# "perio" are deliberately absent: both are rectified magnitudes, non-negative by
# construction, so they take the same [0, 1] treatment as "mag".
SIGNED_CHANNELS = {"re", "im", "aoa"}


# --- Radar selection ---------------------------------------------------------
#
# The input is (T, 6, 3, 120, 2) regardless, so a subset is a slice, never a
# different capture. Radars are 1-indexed everywhere a HUMAN sees them (the
# spec's table, `--radars`, `best_radars.txt`) and 0-indexed everywhere an ARRAY
# sees them; these two functions are the only place the conventions meet.
N_RADARS = 6
ALL_RADARS = tuple(range(N_RADARS))


def parse_radars(text):
    """'1,2,4,6' -> (0, 1, 3, 5). Rejects anything that is not a real radar."""
    idx = tuple(int(t) - 1 for t in text.replace(",", " ").split())
    if not idx or any(i < 0 or i >= N_RADARS for i in idx) or len(set(idx)) != len(idx):
        raise ValueError(
            f"--radars {text!r}: expected a non-empty list of distinct radar "
            f"numbers in 1..{N_RADARS} (e.g. '1,2,4,6')"
        )
    return idx


def format_radars(radar_indices):
    """(0, 1, 3, 5) -> '1,2,4,6'. Inverse of `parse_radars`."""
    return ",".join(str(i + 1) for i in radar_indices)


# --- "breath": a causal respiration bandpass (REPORT_NOTES Step 7) ------------

# Adult respiration at rest (~9-36 breaths/min). `seated_diagnostics.py` measured
# a 52-168x SNR margin in this band across all 22 occupied windows, against
# 1.8-1.9x in both empty ones.
BREATH_LOW_HZ, BREATH_HIGH_HZ = 0.15, 0.6
# Order PER BAND EDGE -- scipy returns twice this. 3 gives -3 dB at the edges and
# -38.8 dB at 2 Hz. The stopband is the point: two prior temporal channels failed
# by becoming motion detectors, so bulk motion has to be rejected, not attenuated.
BREATH_FILTER_ORDER = 3
# ~10-frame (0.4 s) envelope smoother: takes the rectification ripple off without
# blurring the 3.3 s cycle.
BREATH_SMOOTH_ALPHA = 0.9


def breath_envelope(z):
    """0.15-0.6 Hz envelope of the complex CIR -- one number per cell per frame.

    Filters the COMPLEX `z`, not its magnitude: a sub-millimetre chest
    displacement is a phase rotation, and rectifying first discards it.

    Causal `lfilter`, never `filtfilt` -- filtfilt reads the future, so it could
    not run on the ESP32 and would leak later frames into earlier ones.

    DTYPE IS LOAD-BEARING. `butter`/`lfilter_zi` return float64, so this runs in
    double precision, and it must: forcing float32 makes this filter diverge to
    NaN at frame 3743 of a 7500-frame window. The cause is direct-form tap
    cancellation (denominator taps sum to ~3e-9, a 3.5e8 cancellation factor),
    not pole radius. Anything that deploys this needs an sos biquad cascade.
    """
    b, a = scipy.signal.butter(
        BREATH_FILTER_ORDER,
        [BREATH_LOW_HZ, BREATH_HIGH_HZ],
        btype="band",
        fs=FRAME_RATE_HZ,
    )
    zi = scipy.signal.lfilter_zi(b, a).reshape((-1, 1, 1, 1)) * z[0:1]
    filtered, _ = scipy.signal.lfilter(b, a, z, axis=0, zi=zi)

    envelope = np.sqrt(filtered.real**2 + filtered.imag**2)
    del filtered  # ~100 MB per window at complex128.

    b_s, a_s = [1 - BREATH_SMOOTH_ALPHA], [1, -BREATH_SMOOTH_ALPHA]
    zi_s = scipy.signal.lfilter_zi(b_s, a_s).reshape((-1, 1, 1, 1)) * envelope[0:1]
    smoothed, _ = scipy.signal.lfilter(b_s, a_s, envelope, axis=0, zi=zi_s)
    return smoothed.astype(np.float32)


# --- "perio": respiration periodicity (REPORT_NOTES Steps 8-9) ---------------

# Candidate lags in frames at 25 Hz: 60 = 0.417 Hz, 82 = 0.305 Hz (the rate
# measured on w021). A mismatched-but-consistent lag settles at a fixed phase
# offset that taking the magnitude discards, so fixed lags do not require the
# population to share a breathing rate. A third candidate at 100 was dropped for
# SRAM, not accuracy -- the history buffer is sized by the longest lag.
PERIO_LAGS = (60, 82)
# EMA time constant in cycles of whichever lag is being smoothed, so each
# candidate gets the same averaging measured in ITS OWN cycles.
PERIO_SMOOTH_CYCLES = 3
# Divide-by-zero guard for a cell that is EXACTLY zero. Deliberately tiny and
# ABSOLUTE: a floor near the real signal scale would manufacture a cold-start
# spike, which is the failure this guard exists to avoid.
PERIO_ENERGY_FLOOR = 1e-30


def _ema(x, alpha, warm_start=True):
    """Causal one-pole EMA over axis 0, with an explicit cold-start convention.

    `warm_start=True` scales `lfilter_zi` by the first frame, starting the filter
    as if the scene had been frozen there forever -- right when frame 0 is a real
    OBSERVATION. `warm_start=False` starts from zero -- right when frame 0 is an
    ARTIFACT, which is exactly `perio_coherence`'s numerator: `lagged` clamps, so
    its frame-0 product is |z_0|^2 and a warm start would read coherence 1.0 in
    every cell of the room.
    """
    b, a = [1 - alpha], [1, -alpha]
    zi = scipy.signal.lfilter_zi(b, a).reshape((-1, 1, 1, 1)) * x[0:1]
    if not warm_start:
        zi = np.zeros_like(zi)
    smoothed, _ = scipy.signal.lfilter(b, a, x, axis=0, zi=zi)
    return smoothed


def perio_coherence(z):
    """Does this cell REPEAT at a respiration lag? -- one number per frame.

    The conjugate product z(t) * conj(z(t-lag)) is constant for a genuine
    oscillation, so smoothing accumulates it coherently; for a one-off transient
    the cell's own lagged self is empty and the product is ~0.

    NORMALIZED by the smoothed |z|^2, which returns the coherence itself and is
    amplitude-invariant by construction -- scale z by k and both sides scale by
    k^2. The unnormalized form (Step 8) decomposes as |A|^2 x coherence, i.e. an
    energy statistic that walkers dominate. Amplitude invariance is what made
    this a distinct experiment, and also what killed it: see Step 9.

    ONE energy EMA shared by every lag, at the LONGEST lag's time constant --
    the denominator's memory must cover both samples the numerator correlates.
    Lags are reduced with a per-cell max on the NUMERATORS and divided once,
    which is identical output for one division instead of len(PERIO_LAGS).

    Unlike `breath_envelope`, the float64 promotion here is NOT load-bearing --
    a one-pole EMA's taps cancel by only ~500x, well inside float32.
    """
    power = z.real.astype(np.float64) ** 2 + z.imag.astype(np.float64) ** 2
    energy = _ema(power, 1.0 - 1.0 / (PERIO_SMOOTH_CYCLES * max(PERIO_LAGS)))
    del power  # ~50 MB per window at float64.
    np.maximum(energy, PERIO_ENERGY_FLOOR, out=energy)

    out = None
    for lag in PERIO_LAGS:
        prod = z * np.conj(lagged(z, lag))
        smoothed = _ema(
            prod, 1.0 - 1.0 / (PERIO_SMOOTH_CYCLES * lag), warm_start=False
        )
        del prod  # ~50 MB per window at complex64.

        response = np.sqrt(smoothed.real**2 + smoothed.imag**2)
        del smoothed  # ~100 MB per window at complex128.
        out = response if out is None else np.maximum(out, response, out=out)

    out /= energy
    return out.astype(np.float32)


def lagged(z, lag):
    """`z` delayed by `lag` frames, clamping to a repeated z_0 at the edge.

    Early frames clamp rather than being dropped: inference has to emit a
    position for frame 0, so training must see the same rule. Declutter leaves
    z_0 at ~0 (4e-12, not exactly 0 -- the EMA seeds from frame 0), so the clamp
    degrades gracefully: `delta<lag>` becomes |z_t| and `maglag<lag>` becomes an
    all-zero channel. The model falls back to what it has, never to a fabricated
    past.
    """
    return np.concatenate(
        [np.repeat(z[:1], min(lag, len(z)), axis=0), z[:-lag]]
    )[: len(z)]


# --- Assembly ----------------------------------------------------------------


def channel(z, name):
    """One unnormalized channel from the complex decluttered CIR `z`.

    `z` is (T, R, 3, 47) complex; the return is (T, R, 3, 47) float32.
    """
    if name == "mag":
        # Same sqrt form as `preprocessing.magnitude` -- `np.abs` differs in the
        # last bits and the deployed baseline was trained on this one.
        return np.sqrt(z.real**2 + z.imag**2).astype(np.float32)
    if name == "re":
        return z.real.astype(np.float32)
    if name == "im":
        return z.imag.astype(np.float32)
    if name == "breath":
        return breath_envelope(z)
    if name == "perio":
        return perio_coherence(z)
    if name == "aoa":
        # Phase difference between each antenna and the next on the same radar,
        # cyclic over the 3 elements. The conjugate product differences the
        # phases without ever unwrapping.
        return np.angle(z * np.conj(np.roll(z, -1, axis=-3))).astype(np.float32)
    if name.startswith("maglag"):
        past = lagged(z, int(name[6:]))
        return np.sqrt(past.real**2 + past.imag**2).astype(np.float32)
    if name.startswith("delta"):
        # |z_t - z_{t-lag}|: a complex difference, so a sub-millimetre chest
        # displacement shows up as a phase rotation rather than being lost
        # inside a 0.15 m range bin.
        return np.abs(z - lagged(z, int(name[5:]))).astype(np.float32)
    raise ValueError(f"Unknown channel {name!r}")


def build_channels(z, channel_set="mag", radar_indices=None):
    """Stack a channel set into the model input layout (T, R, 3, 47, C).

    `radar_indices` is applied HERE -- downstream of the disk cache, which always
    stores all 6 radars so it stays reusable across subsets, and BEFORE the
    channels are derived, because that is what a 4-radar deployment physically
    is. Passing the full index tuple is a no-op, byte-identically.
    """
    if radar_indices is not None:
        z = np.take(z, radar_indices, axis=-3)
    return np.stack([channel(z, name) for name in CHANNEL_SETS[channel_set]], axis=-1)


def channel_p99(x, channel_set="mag"):
    """Per-channel 99th percentile, on |x| for signed channels so the range stays
    symmetric rather than biased by the negative half."""
    out = []
    for i, name in enumerate(CHANNEL_SETS[channel_set]):
        v = np.percentile(
            np.abs(x[..., i]) if name in SIGNED_CHANNELS else x[..., i], 99
        )
        out.append(1e-6 if v == 0 else float(v))
    return np.array(out, dtype=np.float64)

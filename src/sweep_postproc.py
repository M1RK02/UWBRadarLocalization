"""Sweep the post-processing hyperparameters on the cached out-of-fold heatmaps.

`src/train.py` writes every fold's held-out heatmaps to `logs/<stamp>/oof/`
(see `accumulate_oof_counts`). Each window there was predicted by a model that
never trained on it, so replaying them is an honest generalization estimate --
and it needs neither the model nor the raw dataset.

Five knobs sit between the heatmap and the submitted `.jsonl`:

    heatmap -> extract_peaks_from_grid(threshold, min_distance)
            -> Tracker(alpha, max_distance, max_coast) -> [[x, y], ...]

Three numbers are reported for every config, and they are deliberately kept
apart from each other:

  1. pooled micro-averaged F1 / precision / recall over every window -- the
     graded metric;
  2. the **empty-room phantom rate**, computed ONLY over windows whose ground
     truth is empty in every frame, never folded into the pool (see below);
  3. localization RMSE / MAE / median / P90 on the matched TP pairs, mirroring
     the informational metrics `evaluation/evaluate_performance.py` prints.

Why (2) exists at all
---------------------
Pooled F1 aggregates TP/FP/FN across all windows into one scalar, and only a
couple of the windows are a genuinely empty room. `project_spec.md` requires
handling "the empty room (0 people) as well as the full room", but an
empty-room regression moves the pooled scalar by almost nothing next to the
recall gains from the occupied windows -- so a search that watches pooled F1
alone will trade empty-room correctness away and never show it. That is not a
hypothetical: a previous run of this tool recommended `max_coast=25` on pooled
F1 alone while pushing the phantom rate on a genuinely empty room from an
already-bad 18.3% of frames toward 3x that, with pooled F1 barely moving.

Usage:
    .venv/bin/python src/sweep_postproc.py verify
    .venv/bin/python src/sweep_postproc.py characterize
    .venv/bin/python src/sweep_postproc.py sweep --thresholds 0.30 0.40 0.50
    .venv/bin/python src/sweep_postproc.py compare
"""

import argparse
import csv
import math
import os
import re
import sys
import time
from collections import namedtuple
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.ndimage as ndimage
import scipy.signal

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_performance import MATCH_THRESHOLD, match_hungarian
from src.metrics import prf
from src.preprocessing import FRAME_RATE_HZ, extract_peaks_from_grid
from src.tracker import (
    MAX_PEOPLE,
    PERSIST_T_LO,
    PERSIST_TAU_S,
    Persistence,
    Tracker,
)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = "report"
ROOM_W, ROOM_H = 4.8, 7.2
FRAME_INTERVAL_S = 1.0 / FRAME_RATE_HZ

# Frames per chunk in the sub-pixel precompute. Bounds peak RSS (the 9 shifted
# float64 neighbour arrays) without making the vectorized ops too small to pay.
CHUNK = 2048

# Matched-distance histogram: median and P90 need the distribution, not just
# sums, and keeping every raw distance for every config would be gigabytes.
# Distances that count are in [0, MATCH_THRESHOLD] by construction, so a fixed
# 512-bin histogram over that range costs 2 KB per config and resolves ~2 mm;
# the percentiles below interpolate inside the bin. RMSE/MAE stay exact (they
# come from running sums, not from the histogram).
DIST_BINS = 512


def default_oof_dir():
    """Newest `logs/<stamp>/oof` that actually holds windows.

    Resolved at run time rather than hardcoded: every retrain writes a new
    timestamped directory, and this tool is meant to be pointed at whichever
    model survives the training phases still to come.
    """
    candidates = sorted(
        (p for p in REPO.glob("logs/*/oof") if any(p.glob("*.oof.npz"))),
        key=lambda p: p.parent.name,
    )
    return str(candidates[-1]) if candidates else "logs/*/oof"


# ── The deployed operating point ──────────────────────────────────────────────
#
# Mirrors `submission/code.py`: THRESHOLD, the `extract_peaks_from_grid` default
# `min_distance`, and the `Tracker(...)` arguments. It is the control every
# report compares against, so `check_deployed_in_sync` re-reads code.py's
# THRESHOLD from source and warns if the mirror has drifted.
DEPLOYED = dict(threshold=0.4, min_distance=0.6, alpha=0.4, max_distance=1.0, max_coast=2)

RCFG_FIELDS = ["threshold", "min_distance", "alpha", "max_distance", "max_coast"]

# The two stages that sit BEFORE the five swept knobs, and therefore change what
# every one of them is measured on. Both ship, so the sweep must model them or it
# is tuning a pipeline that does not exist. Defaults mirror src/tracker.py and
# the training run's best_energy_gate.npy; `Surface` below overrides them.
#
# `gate=None` means "read the value the OOF cache was written with", so a sweep
# reproduces the run it is replaying unless told otherwise.
SURFACE = dict(gate=None, tau_s=PERSIST_TAU_S, t_lo=PERSIST_T_LO)


def deployed_config():
    return tuple(DEPLOYED[f] for f in RCFG_FIELDS)


def check_deployed_in_sync():
    """Warn (do not fail) if `DEPLOYED['threshold']` no longer matches code.py.

    Grepped rather than imported: importing `submission/code.py` drags in
    TensorFlow for a single float.
    """
    try:
        src = (REPO / "submission" / "code.py").read_text()
    except OSError:
        return
    m = re.search(r"^THRESHOLD\s*=\s*([0-9.eE+-]+)", src, re.M)
    if m and abs(float(m.group(1)) - DEPLOYED["threshold"]) > 1e-12:
        print(f"  [warn] DEPLOYED['threshold']={DEPLOYED['threshold']} but "
              f"submission/code.py has THRESHOLD={m.group(1)} -- update DEPLOYED.")


def _tree9(a):
    """numpy's pairwise reduction for 8 <= n <= 128, specialised to n == 9."""
    return ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7])) + a[8]


def _seq9(a):
    """numpy's sequential reduction for n < 8.

    The padded neighbours are exactly 0, and `x + 0.0 == x`, so summing all nine
    in row-major order matches summing only the in-bounds ones.
    """
    out = np.zeros_like(a[0])
    for x in a:
        out = out + x
    return out


def _decode_all_cells(grid_stack):
    """Local-maxima mask and sub-pixel metre coordinates for *every* cell.

    Both are threshold-independent -- the mask because `maximum_filter(g, 3) == g`
    never looks at the threshold, the centroid because it is a property of the
    grid alone. So this runs once per window and every config reuses it.

    `grid_stack` is (T, H, W) float32. Returns (local_max, x_m, y_m).
    """
    T, H, W = grid_stack.shape
    cell_w = ROOM_W / W
    cell_h = ROOM_H / H

    # size=(1, 3, 3) is a no-op on the frame axis, so this is the per-frame
    # size=3 filter batched over the whole stack in one call.
    local_max = ndimage.maximum_filter(grid_stack, size=(1, 3, 3)) == grid_stack

    w_all = np.maximum(0, grid_stack)
    rows = np.arange(H, dtype=np.float64).reshape(1, H, 1)
    cols = np.arange(W, dtype=np.float64).reshape(1, 1, W)
    interior = np.zeros((H, W), dtype=bool)
    interior[1:H - 1, 1:W - 1] = True  # cells whose 3x3 window is not truncated

    x_m = np.empty((T, H, W), dtype=np.float64)
    y_m = np.empty((T, H, W), dtype=np.float64)

    for s in range(0, T, CHUNK):
        e = min(T, s + CHUNK)
        padded = np.zeros((e - s, H + 2, W + 2), dtype=np.float32)
        padded[:, 1:H + 1, 1:W + 1] = w_all[s:e]
        # Row-major order of the 3x3 window -- the order np.sum sees.
        shifted = [padded[:, dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)]

        # int64 index * float32 window promotes to float64 in the original, so
        # the numerators are float64 while total_mass stays float32.
        num_y = [shifted[k] * (rows + (k // 3) - 1) for k in range(9)]
        num_x = [shifted[k] * (cols + (k % 3) - 1) for k in range(9)]

        mass = np.where(interior, _tree9(shifted), _seq9(shifted))
        sum_y = np.where(interior, _tree9(num_y), _seq9(num_y))
        sum_x = np.where(interior, _tree9(num_x), _seq9(num_x))

        pos = mass > 0
        y_sub = np.divide(sum_y, mass, out=np.broadcast_to(rows, sum_y.shape).copy(), where=pos)
        x_sub = np.divide(sum_x, mass, out=np.broadcast_to(cols, sum_x.shape).copy(), where=pos)

        y_m[s:e] = (y_sub + 0.5) * cell_h
        x_m[s:e] = (x_sub + 0.5) * cell_w

    return local_max, x_m, y_m


class WindowCandidates:
    """Every local-maximum cell of one window, pre-sorted the way NMS wants it.

    `extract_peaks_from_grid` sorts the above-threshold peaks by confidence
    descending with Python's *stable* sort, over candidates produced in
    row-major order. Filtering a stably-sorted list preserves the relative order
    of the survivors, so sorting once over all local maxima and then masking by
    threshold gives exactly the order the per-frame sort would have produced.
    """

    __slots__ = ("n_frames", "conf", "x", "y", "start")

    def __init__(self, grid_stack):
        T = grid_stack.shape[0]
        local_max, x_m, y_m = _decode_all_cells(grid_stack)

        # Every usable threshold is > 0, so cells with conf <= 0 can never be
        # candidates; dropping them also discards the flat all-zero plateaus
        # that `maximum_filter` otherwise flags wholesale.
        sel = local_max & (grid_stack > 0)
        flat = np.flatnonzero(sel)  # C order == frame-major, then row-major
        frame = (flat // (grid_stack.shape[1] * grid_stack.shape[2])).astype(np.int64)
        conf = grid_stack.reshape(-1)[flat]

        # lexsort is stable: primary key frame, secondary -conf, ties keep
        # row-major order.
        order = np.lexsort((-conf, frame))
        self.n_frames = T
        self.conf = conf[order]
        self.x = x_m.reshape(-1)[flat][order]
        self.y = y_m.reshape(-1)[flat][order]
        self.start = np.searchsorted(frame[order], np.arange(T + 1))

    def detections(self, threshold, min_distance):
        """Per-frame `[[x, y], ...]` after thresholding, NMS and the cap of 4.

        Also returns an exact identity key (the surviving candidate indices) so
        the caller can skip re-scoring a stream it has already seen: neighbouring
        `min_distance` values very often suppress exactly the same peaks.
        """
        keep = self.conf > threshold
        xs, ys, start = self.x, self.y, self.start
        out = []
        key = []
        for t in range(self.n_frames):
            a, b = start[t], start[t + 1]
            frame_out = []
            for i in range(a, b):
                if not keep[i]:
                    continue
                xi = xs[i]
                yi = ys[i]
                for fx, fy in frame_out:
                    if math.hypot(xi - fx, yi - fy) < min_distance:
                        break
                else:
                    frame_out.append([xi, yi])
                    key.append(i)
                    if len(frame_out) >= MAX_PEOPLE:
                        break
            key.append(-1)
            out.append(frame_out)
        return out, np.array(key, dtype=np.int32).tobytes()


# ── Loading ───────────────────────────────────────────────────────────────────

def oof_files(oof_dir):
    return sorted(str(p) for p in Path(oof_dir).glob("*.oof.npz"))


def short_name(path):
    return os.path.basename(path).replace(".npz.oof.npz", "")


def rel(path):
    """Repo-relative form for display, so reports do not carry a home directory."""
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def decode_surface(grid_stack, energy, gate, tau_s, t_lo, threshold):
    """The heatmap the decode actually sees: energy gate, then persistence.

    A gated frame becomes an all-zero heatmap, which is exactly what skipping
    inference produces; the EMA still integrates it, so state stays causal.

    THE EMA IS WRITTEN OUT RATHER THAN HANDED TO `lfilter`, deliberately.
    `lfilter` computes the same recursion but orders its float32 arithmetic
    differently, which shifts the smoothed heatmap by ~2e-7 -- far too small to
    flip any decode, but enough to move the sub-cell centre-of-mass and so the
    matched distances in the 10th significant digit. Acceptance gate #1 compares
    those sums exactly, and it should: a tolerance wide enough to absorb this
    would also absorb a real defect. Reproducing `Persistence.update`'s exact
    expression instead keeps the gate strict and free to fail for real reasons.
    This is `remove_clutter_streaming`'s lesson in the other direction -- there,
    matching lfilter was what preserved parity.
    """
    P = np.ascontiguousarray(grid_stack, dtype=np.float32)
    if gate and energy is not None:
        P = P.copy()
        P[np.asarray(energy) < gate] = 0.0
    if not tau_s:
        return P

    decay = math.exp(-1.0 / (FRAME_RATE_HZ * tau_s))
    smoothed = np.empty_like(P)
    state = P[0].copy()  # seeded from frame 0, as Persistence does
    smoothed[0] = state
    for t in range(1, len(P)):
        state = decay * state + (1.0 - decay) * P[t]
        smoothed[t] = state
    return np.maximum(P, smoothed * (threshold / t_lo))


def surface_params(path, override=None):
    """Resolve SURFACE against this cache's own recorded gate."""
    params = dict(SURFACE if override is None else override)
    if params.get("gate") is None:
        with np.load(path) as d:
            params["gate"] = float(d["energy_gate"]) if "energy_gate" in d else 0.0
    return params


def load_window(path):
    """(grid_stack (T, H, W) float32, gts, energy (T,) or None).

    `energy` is the per-frame input energy the 5.1 gate thresholds; it is absent
    from caches written before 5.1, and the gate is then inert.
    """
    data = np.load(path)
    preds = data["preds"]
    people_xy = data["people_xy"]
    people_mask = data["people_mask"]
    gts = [
        [list(people_xy[i, p]) for p in range(people_xy.shape[1]) if people_mask[i, p]]
        for i in range(len(preds))
    ]
    energy = data["energy"] if "energy" in data else None
    return np.ascontiguousarray(preds[..., 0]), gts, energy


# ── Ground-truth metadata: which windows are empty, and how crowded ───────────
#
# Everything here is derived from `people_mask` / `people_xy`. No window is ever
# identified by name: the composition of the cache can change (a retrain, a
# different fold layout, extra recordings) and this has to keep being right.

# A person slot whose active positions never spread further than this is treated
# as stationary -- "seated" in the assignment's scenario table. The GT is
# strongly bimodal on this statistic (measured on the current cache: seated
# slots span 0.44-1.17 m over a whole recording, walking slots 4.49-8.23 m), so
# any cut inside that gap classifies identically; 2.0 m sits well clear of both
# clusters. This is a property of how people move in a 4.8 x 7.2 m room, not of
# any model.
STATIONARY_SPREAD_M = 2.0


def window_gt_meta(path):
    """(is_empty, scenario_class) for one window, from its ground truth alone.

    `is_empty` is True iff `people_mask` is False for every slot in every frame
    -- equivalently, iff the per-frame GT list is `[]` throughout. That is the
    definition the empty-room phantom rate is isolated on.
    """
    data = np.load(path)
    mask = np.asarray(data["people_mask"]).astype(bool)
    xy = np.asarray(data["people_xy"])

    if not mask.any():
        return True, "empty"

    max_people = int(mask.sum(axis=1).max())
    seated = 0
    for p in range(mask.shape[1]):
        active = xy[mask[:, p], p]
        if len(active) == 0:
            continue
        spread = active.max(axis=0) - active.min(axis=0)
        if math.hypot(*spread) < STATIONARY_SPREAD_M:
            seated += 1
    return False, f"{max_people}p" + (f"+{seated}seated" if seated else "")


def scan_windows(files, quiet=False):
    """{window: is_empty}, {window: scenario_class} -- printed so it can be checked."""
    empty, klass = {}, {}
    for f in files:
        is_empty, k = window_gt_meta(f)
        empty[short_name(f)] = is_empty
        klass[short_name(f)] = k
    if not quiet:
        n_empty = sum(empty.values())
        print(f"  {len(files)} windows; {n_empty} with empty ground truth in every frame:")
        for name in sorted(empty):
            tag = "EMPTY-ROOM" if empty[name] else klass[name]
            print(f"    {name:16s} {tag}")
    return empty, klass


# ── Scoring ───────────────────────────────────────────────────────────────────

# Per (window, config) accumulator. `det_frames`/`frames` feed the empty-room
# phantom rate and are NEVER added into tp/fp/fn; `d_*` feed the localization
# error on matched pairs.
Counts = namedtuple(
    "Counts", "tp fp fn det_frames frames d_n d_sum d_sumsq"
)

def score(detections, gts, alpha, max_distance, max_coast):
    """(Counts, distance histogram) for one window under one tracker config.

    A fresh Tracker, frames in temporal order -- state never crosses windows.
    """
    tracker = Tracker(alpha=alpha, max_distance=max_distance, max_coast=max_coast)
    tp = fp = fn = det_frames = d_n = 0
    d_sum = d_sumsq = 0.0
    hist = np.zeros(DIST_BINS, dtype=np.int32)
    for det, gt in zip(detections, gts):
        pred = tracker.update(det)
        if pred:
            det_frames += 1
        dists, n_fn, n_fp = match_hungarian(gt, pred, MATCH_THRESHOLD)
        tp += len(dists)
        fp += n_fp
        fn += n_fn
        for d in dists:
            d_n += 1
            d_sum += d
            d_sumsq += d * d
            b = int(d / MATCH_THRESHOLD * DIST_BINS)
            hist[b if b < DIST_BINS else DIST_BINS - 1] += 1
    return Counts(tp, fp, fn, det_frames, len(gts), d_n, d_sum, d_sumsq), hist


# ── Pooled metrics for one config ─────────────────────────────────────────────

def hist_percentile(hist, q):
    """q-th percentile (0..1) of a distance histogram, interpolated in the bin."""
    total = hist.sum()
    if total == 0:
        return float("nan")
    cum = np.cumsum(hist)
    target = q * total
    b = int(np.searchsorted(cum, target, side="left"))
    b = min(b, DIST_BINS - 1)
    below = cum[b - 1] if b > 0 else 0
    frac = (target - below) / hist[b] if hist[b] > 0 else 0.0
    return float((b + min(max(frac, 0.0), 1.0)) * MATCH_THRESHOLD / DIST_BINS)


Metrics = namedtuple(
    "Metrics",
    "tp fp fn precision recall f1 "
    "empty_frames empty_det_frames phantom_rate "
    "n_matched rmse mae median p90",
)


def summarize(counts_by_window, hist, empty_windows):
    """Pooled metrics for one config.

    Three accumulators, deliberately separate:
      * tp/fp/fn over EVERY window -> the graded pooled F1;
      * det_frames/frames over ONLY the always-empty windows -> phantom rate;
      * matched distances over every window -> localization error.
    The phantom rate is never mixed into tp/fp/fn -- that mixing is exactly what
    made the previous sweep blind to the empty-room regression.
    """
    tp = fp = fn = 0
    e_frames = e_det = 0
    d_n = 0
    d_sum = d_sumsq = 0.0
    for name, c in counts_by_window.items():
        tp += c.tp
        fp += c.fp
        fn += c.fn
        d_n += c.d_n
        d_sum += c.d_sum
        d_sumsq += c.d_sumsq
        if empty_windows.get(name, False):
            e_frames += c.frames
            e_det += c.det_frames
    p, r, f = prf(tp, fp, fn)
    return Metrics(
        tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f,
        empty_frames=e_frames, empty_det_frames=e_det,
        phantom_rate=(e_det / e_frames if e_frames else float("nan")),
        n_matched=d_n,
        rmse=(math.sqrt(d_sumsq / d_n) if d_n else float("nan")),
        mae=(d_sum / d_n if d_n else float("nan")),
        median=hist_percentile(hist, 0.5),
        p90=hist_percentile(hist, 0.9),
    )


# ── Grid definition ───────────────────────────────────────────────────────────
#
# Four of the five axes are pinned here to narrow, physically-argued ranges. The
# fifth (`threshold`) is NOT: it is a property of the model's output
# distribution, which changes with every retrain -- run `characterize` and pass
# the result with `--thresholds`. Read the reasoning before widening anything;
# each range is narrow for a reason that does not depend on which model produced
# the heatmaps, so "the argmax sits on the edge" is not by itself grounds to
# re-widen.

SWEEP_AXES = {
    # Blob width vs suppression radius, i.e. geometry -- not a model property.
    # The targets are sigma = 0.3 m gaussians on 0.40 m cells, so a detection is
    # ~1.2 m wide; any suppression radius comfortably inside that width behaves
    # identically, and a previous full sweep confirmed the axis is flat from
    # 0.5 to 0.9 m. These three points re-confirm the flatness on a new model
    # without spending budget searching it wide again. Only a change to the grid
    # resolution or the target sigma would justify reopening it.
    "min_distance": [0.5, 0.6, 0.7],

    # Bounded by real human movement against the 40 ms frame interval: ~1.8
    # cm/frame at a typical measured pace, ~32 cm/frame at an implausible 8 m/s
    # sprint, against 0.40 m cells and a 1.0 m acceptance radius. A previous
    # sweep spanned 0.15 -> 1.0 and moved pooled F1 by only 0.004, so a modest
    # bracket around the deployed 0.40 is enough.
    "alpha": [0.25, 0.30, 0.35, 0.40],

    # THE TIGHTEST LEASH ON PURPOSE. Coasting is directly antagonistic to
    # empty-room correctness, and structurally so: a single one-frame noise
    # spike above threshold is not reported once, it is re-reported for
    # `max_coast` further frames. Measured on the deployed model, that
    # amplification turned ~5.7% raw phantom survival into an 18.3% observed
    # frame-level phantom rate at max_coast=2. The mechanism is a property of
    # how coasting works, not of any particular model, so it survives retraining.
    # 6 frames = 240 ms is already a generous occlusion bridge; do not test
    # higher without an explicit, written reason.
    "max_coast": [0, 1, 2, 3, 4, 5, 6],

    # Not a distance -- a *speed*, from which `max_distance` is derived below.
    # 1.5 m/s comfortable walk, 2.5 m/s brisk, 4.0 m/s jog/fast walk. A 4.8 x
    # 7.2 m room does not support more, and gates implying 60 m/s (which a blind
    # wide search happily picked last time) are not a motion model at all.
    "v_max": [1.5, 2.5, 4.0],
}

CFG_FIELDS = ["threshold", "min_distance", "alpha", "max_coast", "v_max"]

# A track's decoded position is a 3x3 centroid on 0.40 m x 0.40 m cells, so two
# consecutive detections of a perfectly stationary person can land a cell apart.
# One cell diagonal (0.57 m) is therefore a floor on the association gate that
# has nothing to do with motion; below it the gate rejects matches for
# quantization reasons.
#
# GRID_W/GRID_H are the one model-shaped assumption in this block, so
# `check_grid_assumption` re-reads the actual heatmap shape and warns if a
# retrain changed the decode resolution -- which would move this floor and, with
# it, `min_distance`'s "flat because the blob is wider than the radius" argument.
GRID_W, GRID_H = 12, 18
DETECTION_JITTER_M = round(math.hypot(ROOM_W / GRID_W, ROOM_H / GRID_H), 2)


def check_grid_assumption(files):
    with np.load(files[0]) as d:
        h, w = d["preds"].shape[1:3]
    if (w, h) != (GRID_W, GRID_H):
        print(f"  [warn] heatmaps are {h}x{w}, not {GRID_H}x{GRID_W}. "
              f"DETECTION_JITTER_M ({DETECTION_JITTER_M} m) and the "
              f"`min_distance` reasoning on SWEEP_AXES both assume the old "
              f"resolution -- re-derive them before trusting this sweep.")


def association_gate(max_coast, v_max):
    """`max_distance` implied by `max_coast` -- never swept independently.

        max_distance = DETECTION_JITTER_M + (max_coast + 1) * FRAME_INTERVAL_S * v_max

    A track that has coasted `c` frames is being matched `c + 1` frame intervals
    after its last real update, so the gate must cover how far the person could
    have walked in that time, plus the jitter floor. Sweeping the gate freely is
    what produced the previous run's physically meaningless 2.5 m association
    radius at 25 Hz.
    """
    return round(DETECTION_JITTER_M + (max_coast + 1) * FRAME_INTERVAL_S * v_max, 3)


def to_runtime(gcfg):
    """Grid coordinate -> the 5-tuple that actually parameterises the pipeline."""
    threshold, min_distance, alpha, max_coast, v_max = gcfg
    return (threshold, min_distance, alpha, association_gate(max_coast, v_max), max_coast)


def build_grid(thresholds):
    axes = dict(SWEEP_AXES)
    axes["threshold"] = list(thresholds)
    axes = {f: axes[f] for f in CFG_FIELDS}  # canonical axis order
    gcfgs = [
        (t, m, a, c, v)
        for t in axes["threshold"] for m in axes["min_distance"]
        for a in axes["alpha"] for c in axes["max_coast"] for v in axes["v_max"]
    ]
    return axes, gcfgs


# ── Parallel evaluation ───────────────────────────────────────────────────────

def _eval_window(job):
    """Worker: one window, many configs. Returns {runtime cfg: (Counts, hist)}.

    Parallelising over windows (not configs) means the expensive per-window
    precompute happens once and is amortised over every config in the job.
    """
    path, peak_configs, tracker_configs, surface = job
    grid_stack, gts, energy = load_window(path)

    result = {}
    seen = {}
    cand, cand_th = None, None
    for th, md in peak_configs:
        # Persistence's gain is threshold-dependent, so the decode surface --
        # and therefore the candidate peaks -- must be rebuilt per threshold.
        # peak_configs is grouped threshold-major, so this is ~1 rebuild each.
        if cand_th != th:
            cand = WindowCandidates(
                decode_surface(grid_stack, energy, threshold=th, **surface)
            )
            cand_th = th
        dets, key = cand.detections(th, md)
        scored = seen.get(key)
        if scored is None:
            scored = {tc: score(dets, gts, *tc) for tc in tracker_configs}
            seen[key] = scored
        for tc, c in scored.items():
            result[(th, md, tc[0], tc[1], tc[2])] = c
    return short_name(path), result


def run_configs(files, peak_configs, tracker_configs, processes=None, label="",
                group=5, surface=None):
    """Evaluate the cross product.

    Returns (per_window, hists): {rcfg: {window: Counts}} and {rcfg: int64 hist}
    pooled over windows -- pooled on arrival so the histograms never all have to
    be resident per-window.
    """
    # Grouped threshold-major so each worker rebuilds the decode surface once
    # per threshold rather than once per config.
    peak_configs = sorted(peak_configs)
    groups = [peak_configs[i:i + group] for i in range(0, len(peak_configs), group)]
    jobs = [(f, g, tracker_configs, surface_params(f, surface)) for f in files for g in groups]
    n_cfg = len(peak_configs) * len(tracker_configs)
    t0 = time.time()
    per_window, hists = {}, {}
    with Pool(processes=processes) as pool:
        for k, (name, res) in enumerate(pool.imap_unordered(_eval_window, jobs), 1):
            for cfg, (counts, hist) in res.items():
                per_window.setdefault(cfg, {})[name] = counts
                acc = hists.get(cfg)
                if acc is None:
                    hists[cfg] = hist.astype(np.int64)
                else:
                    acc += hist
            if k % max(1, len(jobs) // 20) == 0 or k == len(jobs):
                print(f"  [{label}] {k}/{len(jobs)} jobs  ({time.time() - t0:.0f}s)",
                      flush=True)
    print(f"  [{label}] done: {n_cfg} configs x {len(files)} windows "
          f"in {time.time() - t0:.0f}s", flush=True)
    return per_window, hists


def _eval_window_explicit(job):
    """Worker: one window, an explicit list of runtime configs (not a grid)."""
    path, configs, surface = job
    grid_stack, gts, energy = load_window(path)
    out = {}
    cand, cand_th = None, None
    for cfg in sorted(configs):
        th, md, al, mdist, mc = cfg
        if cand_th != th:
            cand = WindowCandidates(
                decode_surface(grid_stack, energy, threshold=th, **surface)
            )
            cand_th = th
        dets, _ = cand.detections(th, md)
        out[cfg] = score(dets, gts, al, mdist, mc)
    return short_name(path), out


def run_explicit(files, configs, processes=None, surface=None):
    per_window, hists = {}, {}
    with Pool(processes=processes) as pool:
        for name, res in pool.imap_unordered(
            _eval_window_explicit, [(f, configs, surface_params(f, surface)) for f in files]
        ):
            for cfg, (counts, hist) in res.items():
                per_window.setdefault(cfg, {})[name] = counts
                acc = hists.get(cfg)
                if acc is None:
                    hists[cfg] = hist.astype(np.int64)
                else:
                    acc += hist
    return per_window, hists


# ── The configs both acceptance gates exercise ────────────────────────────────
#
# Derived from the data, not written down: a threshold literal that separates
# signal from noise on today's model can be meaningless on the next one (this
# repo has already seen the output ceiling move from ~0.25 to ~1.0), and a gate
# whose configs decode to nothing would "pass" vacuously. Quantiles of the
# observed local-maximum confidence give a permissive / middling / strict decode
# on any distribution.
DECODE_QUANTILES = (0.75, 0.85, 0.95)


def acceptance_configs(files, n_windows=4):
    """A few representative runtime configs, spanning the narrowed axes."""
    picks = [files[i] for i in
             np.unique(np.linspace(0, len(files) - 1, n_windows).astype(int))]
    conf = np.concatenate([WindowCandidates(load_window(p)[0]).conf for p in picks])
    ths = [round(float(np.quantile(conf, q)), 4) for q in DECODE_QUANTILES]
    mds = SWEEP_AXES["min_distance"]
    # Tracker settings chosen to span the Tracker's distinct code paths: the
    # deployed point, no coasting at all, and coasting at the narrowed cap.
    trackers = [
        (DEPLOYED["alpha"], DEPLOYED["max_distance"], DEPLOYED["max_coast"]),
        (SWEEP_AXES["alpha"][0], association_gate(0, SWEEP_AXES["v_max"][0]), 0),
        (SWEEP_AXES["alpha"][-1], association_gate(6, SWEEP_AXES["v_max"][-1]), 6),
    ]
    return [(t, m) + tr for t, m, tr in zip(ths, mds, trackers)]


# ── Acceptance gate #1: the fast path == a naive per-frame reference ──────────
#
# What this replaces, and why. This gate used to compare the replay against ten
# F1/P/R literals copied out of one `src/train.py` run. Those literals rot the
# moment the shipped model changes: either they keep "passing" numbers nobody
# re-derived, or they fail for the boring reason that the model moved, which is
# indistinguishable from a real bug. The property actually worth guarding is
# that the vectorized scoring path is correct -- and that can be checked against
# a deliberately naive reference on whatever `--oof-dir` is in front of it, with
# no dependency on any particular model.

def _naive_window(job):
    """Reference worker: one window, frame by frame, no vectorization tricks.

    Deliberately duplicates the accounting in `score()` rather than calling it.
    A reference implementation that reuses the code under test cannot catch a
    bug in that code. The loop is written the way `submission/code.py` runs:
    extract frame t, feed the tracker, take what the tracker reports.
    """
    path, configs, surface = job
    grid_stack, gts, energy = load_window(path)
    gate = surface["gate"]
    out = {}
    for cfg in configs:
        th, md, al, mdist, mc = cfg
        tracker = Tracker(alpha=al, max_distance=mdist, max_coast=mc)
        # The deployed classes, one frame at a time, exactly as code.py runs
        # them -- never the vectorized surface under test.
        persistence = Persistence(th, tau_s=surface["tau_s"], t_lo=surface["t_lo"])
        tp = fp = fn = det_frames = d_n = 0
        d_sum = d_sumsq = 0.0
        hist = np.zeros(DIST_BINS, dtype=np.int32)
        for t in range(len(gts)):
            frame = grid_stack[t]
            if gate and energy is not None and energy[t] < gate:
                frame = np.zeros_like(frame)
            det = extract_peaks_from_grid(
                persistence.update(frame), threshold=th, min_distance=md
            )
            pred = tracker.update(det)
            if pred:
                det_frames += 1
            dists, n_fn, n_fp = match_hungarian(gts[t], pred, MATCH_THRESHOLD)
            tp += len(dists)
            fp += n_fp
            fn += n_fn
            for d in dists:
                d_n += 1
                d_sum += d
                d_sumsq += d * d
                b = int(d / MATCH_THRESHOLD * DIST_BINS)
                hist[b if b < DIST_BINS else DIST_BINS - 1] += 1
        out[cfg] = (Counts(tp, fp, fn, det_frames, len(gts), d_n, d_sum, d_sumsq), hist)
    return short_name(path), out


def acceptance_naive_reference(files, configs, processes=None, surface=None):
    print("=" * 78)
    print("  ACCEPTANCE GATE #1 -- fast path == naive per-frame reference")
    print("=" * 78)
    print(f"  {len(configs)} configs x {len(files)} windows, both paths, "
          f"on the OOF cache in front of us.")

    t0 = time.time()
    fast_pw, fast_h = run_explicit(files, configs, processes, surface=surface)
    print(f"  fast path   {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    naive_pw, naive_h = {}, {}
    with Pool(processes=processes) as pool:
        for name, res in pool.imap_unordered(
            _naive_window, [(f, configs, surface_params(f, surface)) for f in files]
        ):
            for cfg, (counts, hist) in res.items():
                naive_pw.setdefault(cfg, {})[name] = counts
                acc = naive_h.get(cfg)
                if acc is None:
                    naive_h[cfg] = hist.astype(np.int64)
                else:
                    acc += hist
    print(f"  naive path  {time.time() - t0:.0f}s", flush=True)

    print(f"\n  {'threshold':>9} {'min_d':>6} {'alpha':>6} {'max_d':>6} {'coast':>6} | "
          f"{'TP':>7} {'FP':>7} {'FN':>7} | {'F1':>7} | ok")
    ok_all = True
    for cfg in configs:
        f_tp = sum(c.tp for c in fast_pw[cfg].values())
        f_fp = sum(c.fp for c in fast_pw[cfg].values())
        f_fn = sum(c.fn for c in fast_pw[cfg].values())
        same = fast_pw[cfg] == naive_pw[cfg]
        same &= bool((fast_h[cfg] == naive_h[cfg]).all())
        # A config that decodes nothing would match trivially; refuse to call
        # that a pass.
        nontrivial = f_tp > 0 and f_fp > 0
        ok = same and nontrivial
        ok_all &= ok
        _, _, f1 = prf(f_tp, f_fp, f_fn)
        note = "OK" if ok else ("MISMATCH" if not same else "VACUOUS (no detections)")
        print(f"  {cfg[0]:9.4f} {cfg[1]:6.2f} {cfg[2]:6.2f} {cfg[3]:6.2f} {cfg[4]:6d} | "
              f"{f_tp:7d} {f_fp:7d} {f_fn:7d} | {f1:7.4f} | {note}")
        if not same:
            for name in sorted(fast_pw[cfg]):
                if fast_pw[cfg][name] != naive_pw[cfg][name]:
                    print(f"      {name}: fast={fast_pw[cfg][name]}")
                    print(f"      {name}: naive={naive_pw[cfg][name]}")

    print("\n  Compared per window: TP/FP/FN, frames-with-a-detection, matched-pair "
          "count/sum/sum-of-squares, and the full distance histogram.")
    print(f"  ACCEPTANCE GATE #1: {'PASS' if ok_all else 'FAIL'}\n")
    return ok_all


# ── Acceptance gate #2: vectorized extractor == deployed extractor ────────────
#
# Unchanged in substance: it asserts code correctness against whatever heatmap
# arrays it is handed, so it was already model-agnostic.

def acceptance_identity(files, configs, n_frames=4200, n_windows=5):
    print("=" * 78)
    print("  ACCEPTANCE GATE #2 -- vectorized extractor is byte-identical")
    print("=" * 78)
    idx = np.unique(np.linspace(0, len(files) - 1, n_windows).astype(int))
    picks = [files[i] for i in idx]
    peak_configs = [(c[0], c[1]) for c in configs]
    total_frames = 0
    total_comparisons = 0
    for path in picks:
        grid_stack, _, _ = load_window(path)
        grid_stack = grid_stack[:n_frames]
        cand = WindowCandidates(grid_stack)
        assert cand.n_frames == len(grid_stack)
        for th, md in peak_configs:
            fast, _ = cand.detections(th, md)
            for t in range(len(grid_stack)):
                ref = extract_peaks_from_grid(grid_stack[t], threshold=th, min_distance=md)
                got = [[float(x), float(y)] for x, y in fast[t]]
                assert got == ref, (
                    f"MISMATCH {short_name(path)} frame {t} "
                    f"threshold={th} min_distance={md}\n  fast={got}\n  ref ={ref}"
                )
                total_comparisons += 1
        total_frames += len(grid_stack)
        print(f"  {short_name(path):20s} {len(grid_stack)} frames x "
              f"{len(peak_configs)} configs -- identical", flush=True)
    print(f"\n  ACCEPTANCE GATE #2: PASS -- {total_frames} distinct frames, "
          f"{total_comparisons} frame-config comparisons, all byte-identical\n")
    return total_frames, total_comparisons


# ── Landscape analysis ────────────────────────────────────────────────────────

COUNT_IDX = {f: i for i, f in enumerate(Counts._fields)}


class Landscape:
    """Dense (n_config, n_window, len(Counts)) tensor + the axis grids behind it.

    Indexed by *grid* coordinate (threshold, min_distance, alpha, max_coast,
    v_max); `max_distance` is derived from the last two, so the cube stays a
    clean cartesian product and the neighbourhood metric below still means
    something.
    """

    def __init__(self, per_window, hists, window_names, axes, gcfgs, empty_windows):
        self.axes = axes
        self.configs = list(gcfgs)
        self.runtime = [to_runtime(g) for g in self.configs]
        self.windows = list(window_names)
        self.empty_mask = np.array([empty_windows[w] for w in self.windows])
        self.counts = np.array(
            [[list(per_window[r][w]) for w in self.windows] for r in self.runtime],
            dtype=np.float64,
        )
        self.hists = np.array([hists[r] for r in self.runtime], dtype=np.int64)
        self.index = {c: i for i, c in enumerate(self.configs)}

    def _prf_arrays(self, totals):
        """Vectorized `metrics.prf` over a (..., 3) array of tp/fp/fn counts."""
        tp, fp, fn = totals[..., 0], totals[..., 1], totals[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
            r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
            f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
        return p, r, f

    def pooled_prf(self):
        return self._prf_arrays(self.counts.sum(axis=1))

    def lowo_prf(self):
        """(n_config, n_window) F1 with each window held out of the pool."""
        total = self.counts.sum(axis=1, keepdims=True)
        return self._prf_arrays(total - self.counts)[2]

    def per_window_prf(self, cfg):
        return self._prf_arrays(self.counts[self.index[cfg]])

    def phantom_rate(self):
        """(n_config,) fraction of always-empty-window frames with >=1 detection.

        Isolated on purpose: only the always-empty windows contribute, and this
        never touches tp/fp/fn.
        """
        if not self.empty_mask.any():
            return np.full(len(self.configs), np.nan)
        sub = self.counts[:, self.empty_mask, :]
        det = sub[..., COUNT_IDX["det_frames"]].sum(axis=1)
        frames = sub[..., COUNT_IDX["frames"]].sum(axis=1)
        return det / np.maximum(frames, 1)

    def empty_frames(self):
        if not self.empty_mask.any():
            return 0
        return int(self.counts[0, self.empty_mask, COUNT_IDX["frames"]].sum())

    def loc_error(self):
        """(rmse, mae, median, p90, n) per config, on matched TP pairs only."""
        n = self.counts[..., COUNT_IDX["d_n"]].sum(axis=1)
        s = self.counts[..., COUNT_IDX["d_sum"]].sum(axis=1)
        sq = self.counts[..., COUNT_IDX["d_sumsq"]].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rmse = np.where(n > 0, np.sqrt(sq / np.maximum(n, 1)), np.nan)
            mae = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        med = np.array([hist_percentile(h, 0.5) for h in self.hists])
        p90 = np.array([hist_percentile(h, 0.9) for h in self.hists])
        return rmse, mae, med, p90, n

    def class_f1(self, cfg, klass):
        """{scenario class: pooled F1} for one config -- see `window_gt_meta`."""
        row = self.counts[self.index[cfg]]
        out = {}
        for k in sorted(set(klass.values())):
            sel = np.array([klass[w] == k for w in self.windows])
            tp, fp, fn = row[sel, 0].sum(), row[sel, 1].sum(), row[sel, 2].sum()
            out[k] = (prf(tp, fp, fn)[2], int(tp), int(fp), int(fn))
        return out

    def neighbourhood(self, radius=1):
        """Mean and min F1 over each config's 1-step neighbours in the 5-D grid.

        A winner that is a lone spike has a much lower neighbourhood min than a
        winner sitting on a plateau; that difference is the whole point of
        looking, because only the plateau is likely to survive a new session.
        """
        _, _, f1 = self.pooled_prf()
        pos = {f: {v: k for k, v in enumerate(vals)} for f, vals in self.axes.items()}
        shape = tuple(len(self.axes[f]) for f in CFG_FIELDS)
        cube = np.full(shape, np.nan)
        for cfg, f in zip(self.configs, f1):
            cube[tuple(pos[fld][val] for fld, val in zip(CFG_FIELDS, cfg))] = f

        means, mins = np.empty_like(f1), np.empty_like(f1)
        for n, cfg in enumerate(self.configs):
            idx = [pos[fld][val] for fld, val in zip(CFG_FIELDS, cfg)]
            vals = [cube[tuple(idx)]]
            for d in range(len(CFG_FIELDS)):
                for step in (-radius, radius):
                    j = idx[d] + step
                    if 0 <= j < shape[d]:
                        nb = list(idx)
                        nb[d] = j
                        vals.append(cube[tuple(nb)])
            means[n] = float(np.mean(vals))
            mins[n] = float(np.min(vals))
        return means, mins

    def on_boundary(self, cfg):
        return [
            fld for fld, val in zip(CFG_FIELDS, cfg)
            if self.axes[fld].index(val) in (0, len(self.axes[fld]) - 1)
        ]


# ── Reporting ─────────────────────────────────────────────────────────────────

CSV_FIELDS = CFG_FIELDS + [
    "max_distance",
    "f1", "precision", "recall",
    "phantom_rate", "empty_det_frames", "empty_frames",
    "rmse", "mae", "median_err", "p90_err",
    "nb_mean_f1", "nb_min_f1", "tp", "fp", "fn", "n_matched",
]


def rank_rows(land):
    p, r, f = land.pooled_prf()
    nmean, nmin = land.neighbourhood()
    phantom = land.phantom_rate()
    rmse, mae, med, p90, nmatch = land.loc_error()
    totals = land.counts.sum(axis=1)
    e_det = (land.counts[:, land.empty_mask, COUNT_IDX["det_frames"]].sum(axis=1)
             if land.empty_mask.any() else np.zeros(len(land.configs)))
    e_frames = land.empty_frames()
    rows = []
    for i, cfg in enumerate(land.configs):
        rows.append(dict(
            zip(CFG_FIELDS, cfg),
            max_distance=land.runtime[i][3],
            f1=f[i], precision=p[i], recall=r[i],
            phantom_rate=phantom[i], empty_det_frames=int(e_det[i]),
            empty_frames=e_frames,
            rmse=rmse[i], mae=mae[i], median_err=med[i], p90_err=p90[i],
            nb_mean_f1=nmean[i], nb_min_f1=nmin[i],
            tp=int(totals[i, 0]), fp=int(totals[i, 1]), fn=int(totals[i, 2]),
            n_matched=int(nmatch[i]),
        ))
    rows.sort(key=lambda d: -d["f1"])
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{row[k]:.6f}" if isinstance(row[k], float) else row[k])
                        for k in CSV_FIELDS})
    print(f"  wrote {path} ({len(rows)} configs)")


MD_HEAD = ("| # | thr | min_d | alpha | coast | v_max | max_d | F1 | Prec | Rec "
           "| empty-room phantom | RMSE | MAE | median | P90 |")
MD_SEP = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"


def md_table(rows, n=10, plateau=True):
    head, sep = MD_HEAD, MD_SEP
    if plateau:
        head += " nb-mean F1 | nb-min F1 |"
        sep += "---|---|"
    out = [head, sep]
    for i, d in enumerate(rows[:n], 1):
        line = (f"| {i} | {d['threshold']:.3f} | {d['min_distance']:.2f} | "
                f"{d['alpha']:.2f} | {d['max_coast']} | {d['v_max']:.1f} | "
                f"{d['max_distance']:.2f} | **{d['f1']:.4f}** | {d['precision']:.4f} | "
                f"{d['recall']:.4f} | {d['phantom_rate']:.2%} | {d['rmse']:.3f} | "
                f"{d['mae']:.3f} | {d['median_err']:.3f} | {d['p90_err']:.3f} |")
        if plateau:
            line += f" {d['nb_mean_f1']:.4f} | {d['nb_min_f1']:.4f} |"
        out.append(line)
    return "\n".join(out)


def txt_table(rows, n=10):
    """The same columns, for the terminal."""
    head = (f"{'thr':>6} {'min_d':>5} {'alpha':>5} {'coast':>5} {'v_max':>5} {'max_d':>5} | "
            f"{'F1':>7} {'P':>7} {'R':>7} | {'phantom':>8} | "
            f"{'RMSE':>6} {'MAE':>6} {'med':>6} {'P90':>6}")
    out = [head, "-" * len(head)]
    for d in rows[:n]:
        out.append(
            f"{d['threshold']:6.3f} {d['min_distance']:5.2f} {d['alpha']:5.2f} "
            f"{d['max_coast']:5d} {d['v_max']:5.1f} {d['max_distance']:5.2f} | "
            f"{d['f1']:7.4f} {d['precision']:7.4f} {d['recall']:7.4f} | "
            f"{d['phantom_rate']:7.2%} | {d['rmse']:6.3f} {d['mae']:6.3f} "
            f"{d['median_err']:6.3f} {d['p90_err']:6.3f}"
        )
    return "\n".join(out)


def fmt_rcfg(cfg):
    return ", ".join(f"{f}={v}" for f, v in zip(RCFG_FIELDS, cfg))


def fmt_metrics(m):
    return (f"F1 {m.f1:.4f} (P {m.precision:.4f}, R {m.recall:.4f})  |  "
            f"empty-room phantom {m.phantom_rate:.2%} "
            f"({m.empty_det_frames}/{m.empty_frames} frames)  |  "
            f"RMSE {m.rmse:.3f} MAE {m.mae:.3f} med {m.median:.3f} P90 {m.p90:.3f} m")


# ── `characterize`: pick the threshold range from the model, not from memory ──
#
# `threshold` is the one axis this tool refuses to hardcode. Its useful range is
# a property of the model's output distribution, and that distribution moves:
# an earlier training fix in this repo shifted the heatmap ceiling from ~0.25 to
# ~1.0, which would have silently invalidated any range written down beforehand.
#
# It is also the primary lever on empty-room phantoms. Measured on the deployed
# model, phantom local maxima and true-positive local maxima separate cleanly by
# confidence (phantom median 0.10 / p90 0.31 vs true-positive median 0.68 /
# p10 0.40) -- so where the threshold sits decides how many phantoms survive,
# at a real and measurable recall cost. This mode re-measures that separation on
# whatever cache it is given and prints the trade-off table, so the range for
# the eventual sweep is chosen by looking rather than by guessing.

def _characterize_window(path):
    """Confidence pools for one window (see `cmd_characterize`).

    Deliberately on the RAW heatmap, before the gate and before persistence:
    this mode exists to pick the threshold range from the model's own output
    distribution, and persistence's gain is itself threshold-dependent.
    """
    grid_stack, gts, _ = load_window(path)
    is_empty = not any(gts)
    cand = WindowCandidates(grid_stack)
    del grid_stack
    conf, xs, ys, start = cand.conf, cand.x, cand.y, cand.start

    if is_empty:
        # Candidates are sorted confidence-descending inside each frame, so the
        # first entry of a frame's slice is that frame's strongest local maximum.
        frame_max = np.zeros(cand.n_frames, dtype=np.float32)
        nz = start[:-1] < start[1:]
        frame_max[nz] = conf[start[:-1][nz]]
        return short_name(path), dict(
            phantom=conf.astype(np.float32), frame_max=frame_max,
            gt_best=np.empty(0, np.float32), stray=np.empty(0, np.float32),
        )

    gt_best, stray = [], []
    for t in range(cand.n_frames):
        a, b = start[t], start[t + 1]
        g = gts[t]
        if b <= a:
            gt_best.extend([0.0] * len(g))
            continue
        cc = conf[a:b]
        if not g:
            stray.append(cc)
            continue
        G = np.asarray(g, dtype=np.float64)
        d = np.hypot(xs[a:b][None, :] - G[:, 0, None], ys[a:b][None, :] - G[:, 1, None])
        near = d <= MATCH_THRESHOLD
        for i in range(len(g)):
            # The strongest local maximum that would land within the grader's
            # match radius of this person: the confidence that decides whether
            # this person is still detected at a given threshold.
            gt_best.append(float(cc[near[i]].max()) if near[i].any() else 0.0)
        stray.append(cc[~near.any(axis=0)])

    return short_name(path), dict(
        phantom=np.empty(0, np.float32),
        frame_max=np.empty(0, np.float32),
        gt_best=np.asarray(gt_best, dtype=np.float32),
        stray=(np.concatenate(stray).astype(np.float32) if stray
               else np.empty(0, np.float32)),
    )


def describe(name, a, extra=""):
    if len(a) == 0:
        return f"  {name:34s} (no samples)"
    q = np.quantile(a, [0.10, 0.25, 0.50, 0.75, 0.90])
    return (f"  {name:34s} n={len(a):>9d}  min {a.min():.4f}  p10 {q[0]:.4f}  "
            f"p25 {q[1]:.4f}  median {q[2]:.4f}  p75 {q[3]:.4f}  p90 {q[4]:.4f}  "
            f"max {a.max():.4f}{extra}")


def cmd_characterize(args):
    files = oof_files(args.oof_dir)
    assert files, f"no *.oof.npz under {args.oof_dir}"
    print("=" * 78)
    print("  CHARACTERIZE -- phantom vs true-positive confidence separation")
    print("=" * 78)
    print(f"  {rel(args.oof_dir)}\n")
    empty, klass = scan_windows(files)
    if not any(empty.values()):
        print("\n  [warn] no always-empty window in this cache -- the phantom "
              "columns below will be empty and the empty-room phantom rate "
              "cannot be measured at all.")

    pools = {k: [] for k in ("phantom", "frame_max", "gt_best", "stray")}
    with Pool(processes=args.processes) as pool:
        for _, res in pool.imap_unordered(_characterize_window, files):
            for k, v in res.items():
                pools[k].append(v)
    pools = {k: np.concatenate(v) if v else np.empty(0, np.float32)
             for k, v in pools.items()}
    phantom, frame_max, gt_best, stray = (
        pools["phantom"], pools["frame_max"], pools["gt_best"], pools["stray"])

    print("\n  Local-maximum confidence distributions")
    print("  " + "-" * 74)
    print(describe("phantom (always-empty windows)", phantom))
    # The whole-population phantom row above is dominated by near-zero noise
    # cells that no usable threshold would ever keep. The strongest phantom in
    # each empty frame is the one that decides whether that frame reports a
    # detection, so it is the operationally meaningful distribution.
    print(describe("  strongest phantom per empty frame", frame_max))
    n_blind = int((gt_best == 0).sum())
    print(describe("true positive (best peak <=1.0 m", gt_best,
                   f"\n  {'  of a GT person)':34s} {n_blind} of {len(gt_best)} "
                   f"GT persons ({n_blind / max(len(gt_best), 1):.2%}) have no peak "
                   f"at all within 1.0 m"))
    print(describe("stray (occupied windows, >1.0 m", stray))
    print(f"  {'  from every GT person)':34s}")

    if len(gt_best) == 0:
        print("\n  No occupied windows -- nothing to separate. Stopping.")
        return

    # Threshold levels come from the observed distribution, never from a
    # literal: the top of the true-positive range is whatever this model
    # produces.
    hi = float(max(np.quantile(phantom, 0.999) if len(phantom) else 0.0,
                   np.quantile(gt_best, 0.95)))
    levels = np.round(np.linspace(0.0, hi, 21), 4)

    print(f"\n  Threshold -> survival  (levels span 0 .. {hi:.4f}, derived from the "
          f"distributions above)")
    print("  " + "-" * 74)
    print(f"  {'threshold':>9} | {'phantom peaks':>14} {'empty frames':>13} | "
          f"{'GT persons':>11} {'stray peaks':>12} | {'separation':>10}")
    print(f"  {'':>9} | {'kept':>14} {'>=1 peak':>13} | {'still peaked':>11} "
          f"{'kept':>12} | {'(GT - empty)':>10}")
    rows = []
    for t in levels:
        pk = float((phantom > t).mean()) if len(phantom) else float("nan")
        ef = float((frame_max > t).mean()) if len(frame_max) else float("nan")
        gp = float((gt_best > t).mean())
        st = float((stray > t).mean()) if len(stray) else float("nan")
        # Youden-style separation: how much better this threshold keeps real
        # people than it keeps empty-room frames reporting. Its argmax is the
        # single most defensible point to read the range around, and it needs no
        # arbitrary target rate on either side.
        sep = gp - (ef if ef == ef else 0.0)
        rows.append((float(t), pk, ef, gp, st, sep))
        print(f"  {t:9.4f} | {pk:13.2%} {ef:12.2%} | {gp:10.2%} {st:11.2%} | "
              f"{sep:9.2%}")

    # Where to start reading -- not a recommendation, and deliberately not fed
    # into the sweep automatically. Three reference points, then a grid that
    # brackets them.
    lo = next((r[0] for r in rows if r[2] <= 0.10), None)      # phantom side
    hi_b = next((r[0] for r in reversed(rows) if r[3] >= 0.90), None)  # recall side
    j = max(rows, key=lambda r: r[5])
    plateau = [r[0] for r in rows if r[5] >= j[5] - 0.01]
    print("\n  Reading guide (where to start reading, not a recommendation):")
    print(f"    empty-room frames with a surviving peak first drop below 10% at "
          f"threshold {lo if lo is not None else 'n/a'}")
    print(f"    90% of GT persons still have a peak up to threshold "
          f"{hi_b if hi_b is not None else 'n/a'}")
    print(f"    separation peaks at threshold {j[0]:g} "
          f"(GT {j[3]:.2%} vs empty frames {j[2]:.2%}); within 1 point of that "
          f"peak over {min(plateau):g}..{max(plateau):g}")
    if lo is not None and hi_b is not None and hi_b >= lo:
        span = (lo, hi_b)
    else:
        # The two rate targets do not overlap on this model: threshold alone
        # cannot both suppress phantoms and hold recall. Fall back to the
        # separation plateau, which is defined on any distribution.
        print("    the 10%/90% targets do NOT overlap here -- threshold alone "
              "cannot both suppress phantoms and hold recall on this model, so "
              "the grid below brackets the separation plateau instead.")
        span = (min(plateau), max(plateau))
    grid = np.round(np.linspace(*span, 5), 3)
    print(f"    a sweep grid over that span:  "
          f"--thresholds {' '.join(f'{v:g}' for v in grid)}")
    print("\n  Note: `extract_peaks_from_grid` compares with a strict `>`, so "
          "these columns are exactly what the decode would keep.")


# ── Entry points ──────────────────────────────────────────────────────────────

def cmd_verify(args):
    files = oof_files(args.oof_dir)
    assert files, f"no *.oof.npz under {args.oof_dir}"
    print(f"  {rel(args.oof_dir)}")
    check_grid_assumption(files)
    scan_windows(files)
    print()
    configs = acceptance_configs(files)
    acceptance_identity(files, configs)
    ok = acceptance_naive_reference(files, configs, args.processes)
    if not ok:
        print("Acceptance gate #1 FAILED -- stopping. Do not tune on this replay.")
        sys.exit(1)


def cmd_sweep(args):
    check_deployed_in_sync()
    files = oof_files(args.oof_dir)
    assert files, f"no *.oof.npz under {args.oof_dir}"
    check_grid_assumption(files)
    empty, klass = scan_windows(files)
    if not any(empty.values()):
        print("\n  [warn] no always-empty window in this cache -- the empty-room "
              "phantom rate cannot be measured and will report as nan.")

    axes, gcfgs = build_grid(args.thresholds)
    peak_configs = [(t, m) for t in axes["threshold"] for m in axes["min_distance"]]
    tracker_configs = sorted({(a, association_gate(c, v), c) for a in axes["alpha"]
                              for c in axes["max_coast"] for v in axes["v_max"]})
    print("=" * 78)
    print(f"  5-D SWEEP -- {len(peak_configs)} peak x {len(tracker_configs)} tracker "
          f"-> {len(gcfgs)} grid points x {len(files)} windows")
    print("=" * 78)
    t0 = time.time()
    per_window, hists = run_configs(files, peak_configs, tracker_configs,
                                    args.processes, "sweep")
    elapsed = time.time() - t0

    # The deployed operating point is scored separately: it is the control, and
    # it is not in general a member of the grid (its `max_distance` is whatever
    # code.py ships, not the gate formula's).
    base_cfg = deployed_config()
    base_pw, base_h = run_explicit(files, [base_cfg], args.processes)
    base = summarize(base_pw[base_cfg], base_h[base_cfg], empty)

    names = sorted(per_window[next(iter(per_window))])
    land = Landscape(per_window, hists, names, axes, gcfgs, empty)
    rows = rank_rows(land)
    _, _, pooled_f1 = land.pooled_prf()
    phantom = land.phantom_rate()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    # Raw per-window counts first: the CSV keeps only pooled totals, so without
    # this a crash anywhere in the reporting below would throw away the sweep.
    # Goes to logs/ (gitignored) -- intermediate, not a deliverable.
    np.savez(
        os.path.join("logs", f"postproc_sweep{args.tag or '_main'}_counts.npz"),
        counts=land.counts, hists=land.hists,
        configs=np.array(land.configs, dtype=np.float64),
        windows=np.array(land.windows), empty=land.empty_mask,
    )
    write_csv(rows, os.path.join(out_dir, f"postproc_sweep{args.tag}.csv"))

    # Configs that do not make the empty room worse than what is deployed. The
    # pooled-F1 argmax is reported too, but on its own it is exactly the number
    # that hid the empty-room regression last time.
    guarded = [r for r in rows if not (r["phantom_rate"] > base.phantom_rate)]

    lowo = land.lowo_prf()
    winners = [land.configs[i] for i in lowo.argmax(axis=0)]
    best = tuple(rows[0][f] for f in CFG_FIELDS)
    n_hold = sum(1 for w in winners if w == best)

    lines = []
    w = lines.append
    w("# Post-processing hyperparameter sweep\n")
    w(f"Replay of `{rel(args.oof_dir)}` -- {len(files)} out-of-fold windows, "
      f"{int(land.counts[0, :, COUNT_IDX['frames']].sum())} frames, no model and "
      f"no raw dataset involved.  ")
    w(f"Grid: {' x '.join(f'{len(axes[f])} {f}' for f in CFG_FIELDS)} = "
      f"{len(gcfgs)} configs, scored in {elapsed / 60:.1f} min on "
      f"{args.processes or os.cpu_count()} processes.\n")
    w("Pooled F1/P/R aggregate TP/FP/FN over every window, exactly as the graders "
      "do. **The empty-room phantom rate is not part of that pool** -- it is the "
      "fraction of frames with at least one reported detection, over the windows "
      "whose ground truth is empty in every frame, and nothing else. Localization "
      "error is on matched TP pairs only (RMSE/MAE exact; median/P90 read off a "
      f"{DIST_BINS}-bin histogram, ~{MATCH_THRESHOLD / DIST_BINS * 1000:.0f} mm "
      "resolution).\n")

    w("\n## Windows\n")
    w("Empty-room windows are detected from the ground truth (`people_mask` "
      "False for every slot in every frame), never by name.\n")
    w("| window | scenario class (derived) | empty-room |")
    w("|---|---|---|")
    for n in names:
        w(f"| {n} | {klass[n]} | {'**yes**' if empty[n] else 'no'} |")

    w("\n## Grid\n")
    for f in CFG_FIELDS:
        w(f"- `{f}`: {axes[f]}")
    w(f"- `max_distance`: derived, not swept -- "
      f"`{DETECTION_JITTER_M} + (max_coast + 1) * {FRAME_INTERVAL_S:g} * v_max` "
      f"-> {sorted({association_gate(c, v) for c in axes['max_coast'] for v in axes['v_max']})}")

    w("\n## Top 10 by pooled F1\n")
    w(md_table(rows, 10))
    w("\n`nb-mean F1` / `nb-min F1` are the mean and minimum pooled F1 over the "
      "config's immediate neighbours (+-1 step on each of the five swept axes). A "
      "lone spike has a much lower `nb-min` than a plateau.\n")

    w("\n## Top 10 by pooled F1, among configs that do not worsen the empty room\n")
    w(f"Filter: empty-room phantom rate <= the deployed baseline's "
      f"**{base.phantom_rate:.2%}**. {len(guarded)}/{len(rows)} configs qualify.\n")
    w(md_table(guarded, 10) if guarded else
      "_No config in this grid holds the empty-room phantom rate at or below the "
      "deployed baseline._")

    w("\n## Baseline (currently deployed)\n")
    w(f"`{fmt_rcfg(base_cfg)}`  ")
    w(f"-> F1 **{base.f1:.4f}** (P {base.precision:.4f}, R {base.recall:.4f}), "
      f"empty-room phantom **{base.phantom_rate:.2%}** "
      f"({base.empty_det_frames}/{base.empty_frames} frames), "
      f"RMSE {base.rmse:.3f} m, MAE {base.mae:.3f} m, median {base.median:.3f} m, "
      f"P90 {base.p90:.3f} m.\n")
    w(f"Best pooled F1 in the grid: **{rows[0]['f1']:.4f}** "
      f"(delta **{rows[0]['f1'] - base.f1:+.4f}**) at empty-room phantom "
      f"**{rows[0]['phantom_rate']:.2%}** "
      f"(delta **{rows[0]['phantom_rate'] - base.phantom_rate:+.2%}**).\n")

    w("\n## Per-axis marginals\n")
    w("Best pooled F1 and best/worst empty-room phantom rate at each value of "
      "each axis, with the other axes free. Reading the two side by side is the "
      "point: an axis that buys F1 by trading the empty room shows it here.\n")
    w("| axis | value | best F1 | best phantom | worst phantom |")
    w("|---|---|---|---|---|")
    for f in CFG_FIELDS:
        k = CFG_FIELDS.index(f)
        for v in axes[f]:
            sel = [i for i, c in enumerate(land.configs) if c[k] == v]
            w(f"| `{f}` | {v} | {max(pooled_f1[i] for i in sel):.4f} | "
              f"{min(phantom[i] for i in sel):.2%} | "
              f"{max(phantom[i] for i in sel):.2%} |")

    w("\n## Scenario-class breakdown\n")
    w("Classes are derived from the ground truth (max concurrent persons, plus "
      "`+Nseated` counting person slots that never move more than "
      f"{STATIONARY_SPREAD_M:g} m over the whole recording), so pooling artifacts "
      "are visible per class rather than only in the aggregate.\n")
    w("| class | baseline F1 | best-config F1 | delta |")
    w("|---|---|---|---|")
    cls_best = land.class_f1(best, klass)
    for k in sorted(cls_best):
        sel = [n for n in names if klass[n] == k]
        b_tp = sum(base_pw[base_cfg][n].tp for n in sel)
        b_fp = sum(base_pw[base_cfg][n].fp for n in sel)
        b_fn = sum(base_pw[base_cfg][n].fn for n in sel)
        bf = prf(b_tp, b_fp, b_fn)[2]
        w(f"| {k} | {bf:.4f} | {cls_best[k][0]:.4f} | {cls_best[k][0] - bf:+.4f} |")
    w("\nThe `empty` class scores F1 0 under every config (no TP is possible with "
      "no ground truth) -- that is exactly why the empty-room phantom rate is "
      "tracked separately instead of being read off F1.\n")

    w("\n## Leave-one-window-out stability of the sweep\n")
    w(f"For each window, the best config is recomputed with that window excluded "
      f"from the pool. **The overall winner `{fmt_rcfg(to_runtime(best))}` also "
      f"wins {n_hold}/{len(names)} hold-outs.**\n")

    w("\n## Per-window F1\n")
    pb = land.per_window_prf(best)[2]
    w("| window | class | baseline F1 | best-config F1 | delta |")
    w("|---|---|---|---|---|")
    for k, n in enumerate(names):
        c = base_pw[base_cfg][n]
        f0 = prf(c.tp, c.fp, c.fn)[2]
        w(f"| {n} | {klass[n]} | {f0:.4f} | {pb[k]:.4f} | {pb[k] - f0:+.4f} |")

    boundary = land.on_boundary(best)
    if boundary:
        w(f"\n> **Boundary note:** the winner sits on the edge of the grid for "
          f"{boundary}. For `threshold` that means re-run `characterize` and "
          f"widen. For the four pinned axes it does **not** automatically mean "
          f"widen -- read the reasoning comments on `SWEEP_AXES` first; each is "
          f"narrow for a physical reason, and `max_coast` in particular is capped "
          f"because coasting amplifies empty-room phantoms.\n")

    path = os.path.join(out_dir, f"postproc_sweep{args.tag}.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")

    print("\nTop 10 by pooled F1")
    print(txt_table(rows, 10))
    if guarded:
        print(f"\nTop 5 that do not worsen the empty room (phantom <= "
              f"{base.phantom_rate:.2%})")
        print(txt_table(guarded, 5))
    print(f"\nbaseline {fmt_rcfg(base_cfg)}")
    print(f"         {fmt_metrics(base)}")
    print(f"LOWO: pooled-F1 winner holds {n_hold}/{len(names)}")
    if boundary:
        print(f"BOUNDARY: winner on grid edge for {boundary} "
              f"(read the SWEEP_AXES comments before widening)")
    return land, rows


# ── Head-to-head comparison of a few named configs ────────────────────────────
#
# Every entry is derived from the deployed operating point rather than written
# down, so nothing here encodes a property of today's model. The two variants
# move only `max_coast`, the axis directly implicated in the empty-room
# regression, and hold everything else at the deployed value so the comparison
# has one moving part. Add arbitrary configs with `--config`.

def named_configs():
    th, md, al, dist, _ = deployed_config()
    return {
        "deployed baseline": deployed_config(),
        "deployed, no coasting": (th, md, al, dist, 0),
        "deployed, coast at cap": (th, md, al, dist, max(SWEEP_AXES["max_coast"])),
    }


def cmd_compare(args):
    check_deployed_in_sync()
    files = oof_files(args.oof_dir)
    assert files, f"no *.oof.npz under {args.oof_dir}"
    empty, klass = scan_windows(files)

    configs = dict(named_configs())
    for spec in args.config or []:
        vals = [float(v) for v in spec.split(",")]
        assert len(vals) == 5, f"--config wants 5 comma-separated values, got {spec!r}"
        configs[f"custom {spec}"] = tuple(vals[:4] + [int(vals[4])])

    per_window, hists = run_explicit(files, list(configs.values()), args.processes)
    names = sorted(per_window[next(iter(per_window))])
    mets = {label: summarize(per_window[cfg], hists[cfg], empty)
            for label, cfg in configs.items()}

    lines = []
    w = lines.append
    w("# Post-processing configurations, head to head\n")
    w(f"Same {len(files)} out-of-fold windows for every row. Pooled F1/P/R are "
      "micro-averaged over pooled TP/FP/FN exactly as the graders aggregate; the "
      "empty-room phantom rate is computed only over the always-empty windows and "
      "is never folded into that pool.\n")
    w("\n## Pooled\n")
    w("| config | thr | min_d | alpha | max_d | coast | F1 | Prec | Rec "
      "| empty-room phantom | RMSE | MAE | median | P90 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, cfg in configs.items():
        m = mets[label]
        w(f"| {label} | {cfg[0]:.3f} | {cfg[1]:.2f} | {cfg[2]:.2f} | {cfg[3]:.2f} | "
          f"{cfg[4]} | **{m.f1:.4f}** | {m.precision:.4f} | {m.recall:.4f} | "
          f"{m.phantom_rate:.2%} | {m.rmse:.3f} | {m.mae:.3f} | {m.median:.3f} | "
          f"{m.p90:.3f} |")

    base_label = "deployed baseline"
    w(f"\n## Empty-room detail (isolated, {mets[base_label].empty_frames} frames)\n")
    w("| config | frames with >=1 detection | phantom rate | FPs on empty windows |")
    w("|---|---|---|---|")
    for label, cfg in configs.items():
        m = mets[label]
        fp_empty = sum(per_window[cfg][n].fp for n in names if empty[n])
        w(f"| {label} | {m.empty_det_frames} | {m.phantom_rate:.2%} | {fp_empty} |")

    w("\n## Per-window F1\n")
    w("| window | class | " + " | ".join(configs) + " |")
    w("|---|---|" + "---|" * len(configs))
    for n in names:
        cells = " | ".join(f"{prf(*per_window[cfg][n][:3])[2]:.4f}"
                           for cfg in configs.values())
        w(f"| {n} | {klass[n]} | {cells} |")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "postproc_compare.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print()
    for label in configs:
        print(f"  {label:26s} {fmt_metrics(mets[label])}")
    print(f"\n  wrote {path}")


def cmd_surface(args):
    """Sweep the two stages that sit BEFORE the five tracker/decode knobs.

    The energy gate (5.1) and persistence (4.3) both change the heatmap every
    other knob is measured on, and both were adopted on parameters chosen from
    the same 24 windows they were scored on. This re-measures that surface on
    whatever cache it is given, at the deployed tracker config, so the numbers
    are re-derived on the final model rather than carried over.

    Reported apart, never blended: pooled F1 (the graded metric), the isolated
    empty-room phantom rate, and the fraction of frames the gate skips -- which
    is the compute saving 5.1 exists for, and is not a quality metric at all.
    """
    check_deployed_in_sync()
    files = oof_files(args.oof_dir)
    assert files, f"no *.oof.npz under {args.oof_dir}"
    empty, _ = scan_windows(files)
    cfg = deployed_config()

    native = surface_params(files[0])["gate"]
    gates = args.gates if args.gates else [0.0, native * 0.5, native, native * 1.5]
    taus = args.taus if args.taus else [0.0, 2.0, 5.0, 10.0, 20.0]
    t_los = args.t_los if args.t_los else [0.25, 0.30, 0.35]

    skip = {}
    for f in files:
        with np.load(f) as d:
            skip[short_name(f)] = (
                np.asarray(d["energy"]) if "energy" in d else None
            )

    def skip_rate(gate):
        vals = [v for v in skip.values() if v is not None]
        if not vals or not gate:
            return 0.0
        return float(np.mean([(v < gate).mean() for v in vals]))

    print(f"\n  deployed tracker config: {fmt_rcfg(cfg)}")
    print(f"  cache's own gate: {native:.6f}\n")
    print(f"  {'gate':>10}{'tau_s':>8}{'t_lo':>7}{'skip%':>8}"
          f"{'F1':>9}{'Prec':>9}{'Rec':>9}{'phantom':>10}")

    rows = []
    for gate in gates:
        for tau in taus:
            # t_lo only means anything once persistence is on.
            for t_lo in (t_los if tau else t_los[:1]):
                surface = dict(gate=gate, tau_s=tau, t_lo=t_lo)
                per_window, hists = run_explicit(
                    files, [cfg], args.processes, surface=surface
                )
                m = summarize(per_window[cfg], hists[cfg], empty)
                rows.append((gate, tau, t_lo, skip_rate(gate), m))
                print(f"  {gate:>10.6f}{tau:>8.1f}{t_lo:>7.2f}"
                      f"{100 * skip_rate(gate):>7.1f}%{m.f1:>9.4f}"
                      f"{m.precision:>9.4f}{m.recall:>9.4f}"
                      f"{m.phantom_rate:>9.2%}", flush=True)

    best = max(rows, key=lambda r: r[4].f1)
    print(f"\n  best pooled F1: gate={best[0]:.6f} tau_s={best[1]:.1f} "
          f"t_lo={best[2]:.2f} -> F1 {best[4].f1:.4f} "
          f"(skips {100 * best[3]:.1f}% of frames, "
          f"phantom {best[4].phantom_rate:.2%})")
    print("  Adopt nothing on less than 1 PAIRED SE (discipline #6), and check "
          "the seated split with oof_breakdown.py before believing a pooled gain.")

    path = os.path.join(args.out_dir, f"surface_sweep{args.tag}.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gate", "tau_s", "t_lo", "skip_rate", "f1", "precision",
                         "recall", "phantom_rate", "rmse", "mae"])
        for gate, tau, t_lo, sk, m in rows:
            writer.writerow([gate, tau, t_lo, f"{sk:.6f}", f"{m.f1:.6f}",
                             f"{m.precision:.6f}", f"{m.recall:.6f}",
                             f"{m.phantom_rate:.6f}", f"{m.rmse:.6f}",
                             f"{m.mae:.6f}"])
    print(f"  wrote {rel(path)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "mode", choices=["verify", "characterize", "sweep", "compare", "surface"]
    )
    ap.add_argument("--gates", type=float, nargs="+", default=None,
                    help="surface mode: energy gates to sweep (0 disables it)")
    ap.add_argument("--taus", type=float, nargs="+", default=None,
                    help="surface mode: persistence tau in seconds (0 disables it)")
    ap.add_argument("--t-los", type=float, nargs="+", default=None,
                    dest="t_los", help="surface mode: persistence t_lo")
    ap.add_argument("--oof-dir", default=None,
                    help="default: the newest logs/<stamp>/oof holding windows")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--tag", default="", help="suffix for the sweep output files")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    help="decode thresholds to sweep. REQUIRED for `sweep`: this "
                         "axis is a property of the model's output distribution, "
                         "so run `characterize` first and pick from its table.")
    ap.add_argument("--config", action="append",
                    help="compare mode: extra config as "
                         "threshold,min_distance,alpha,max_distance,max_coast")
    args = ap.parse_args()
    if args.oof_dir is None:
        args.oof_dir = default_oof_dir()

    if args.mode == "verify":
        cmd_verify(args)
    elif args.mode == "characterize":
        cmd_characterize(args)
    elif args.mode == "compare":
        cmd_compare(args)
    elif args.mode == "surface":
        cmd_surface(args)
    else:
        if not args.thresholds:
            ap.error("sweep needs --thresholds; run `characterize` first to see "
                     "where this model's phantom and true-positive confidences "
                     "separate, then pick the range from that table.")
        cmd_sweep(args)


if __name__ == "__main__":
    main()

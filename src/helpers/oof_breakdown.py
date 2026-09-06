"""Per-window out-of-fold F1 from a cached OOF run, a seated-subset score, and
PER-PERSON recall split by whether that person is walking or seated.

Phase 4 of the task board requires scoring the seated-containing windows as
their own subset rather than watching pooled F1 alone: the seated error mode is
worth ~0.06 pooled F1, but pooled F1 averages it across the 22 windows that do
not have the problem, so a real fix can read as noise. This prints both.

The per-person section exists because that was still not enough. Measured
2026-08-27, the seated-*subset* F1 also hid the real failure: in window 004 the
walking subject is recalled at 99.8% and the seated one at 8.7%, and the
window's pooled F1 looks unremarkable because the walker dominates the average.
Any claim about seated subjects has to be made at the identity level.

Scoring is the deployed path -- peak extraction then `Tracker`, frames in
temporal order, one fresh tracker per window -- so the numbers are comparable to
the ones `train.py` reports. The F1 section calls `extract_peaks_from_grid`
directly; the per-person section reuses `sweep_postproc.WindowCandidates`, the
vectorized equivalent that tool already gates against a naive per-frame
reference, because it needs the whole window's peaks at once.

Run:  .venv/bin/python src/helpers/oof_breakdown.py logs/<stamp>/oof [--threshold 0.4]
      .venv/bin/python src/helpers/oof_breakdown.py logs/<new>/oof --compare logs/<old>/oof
"""

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from evaluation.evaluate_performance import MATCH_THRESHOLD, match_hungarian
from src.metrics import prf
from src.preprocessing import extract_peaks_from_grid
from src.sweep_postproc import STATIONARY_SPREAD_M, WindowCandidates
from src.tracker import Tracker

# Windows whose metadata description mentions a seated subject. These hold the
# bulk of pooled FN; see src/helpers/seated_diagnostics.py for why.
SEATED_WINDOWS = {4, 8, 9, 10, 11, 14, 15, 21}
# Genuinely empty rooms. Pooled F1 is 0 by construction here (no true positives
# are possible), so they are reported apart as a phantom-detection count.
EMPTY_WINDOWS = {22, 23}

# `extract_peaks_from_grid`'s default, mirrored by the deployed `code.py`.
MIN_DISTANCE = 0.6


def score_window(path, threshold):
    data = np.load(path)
    preds, people_xy, people_mask = data["preds"], data["people_xy"], data["people_mask"]
    tracker = Tracker()
    tp = fp = fn = 0
    for i in range(len(preds)):
        truth = [
            list(people_xy[i, p])
            for p in range(people_xy.shape[1])
            if people_mask[i, p]
        ]
        pred = tracker.update(extract_peaks_from_grid(preds[i], threshold=threshold))
        dists, n_fn, n_fp = match_hungarian(truth, pred, MATCH_THRESHOLD)
        tp += len(dists)
        fp += n_fp
        fn += n_fn
    return tp, fp, fn, len(preds)


def person_kinds(people_xy, people_mask):
    """('walking' | 'seated') per ground-truth person slot, or None if unused.

    Classified by how far the slot's own position travels over the frames it is
    active, with the same `STATIONARY_SPREAD_M = 2.0` cut `sweep_postproc` uses.
    The ground truth is strongly bimodal on this statistic (seated slots span
    0.44-1.17 m over a whole recording, walking slots 4.49-8.23 m), so the exact
    cut inside that gap does not matter. Derived from the GT alone -- no window
    is identified by name, so this keeps working if the recordings change.
    """
    kinds = []
    for p in range(people_mask.shape[1]):
        active = people_xy[people_mask[:, p], p]
        if len(active) == 0:
            kinds.append(None)
            continue
        spread = active.max(axis=0) - active.min(axis=0)
        kinds.append("seated" if math.hypot(*spread) < STATIONARY_SPREAD_M else "walking")
    return kinds


def person_recall(path, threshold, min_distance=MIN_DISTANCE):
    """Per-identity recall for one window: {slot: (kind, hits, active_frames)}.

    For every frame in which a ground-truth slot is active, that slot counts as
    recalled if ANY emitted detection lands within `MATCH_THRESHOLD`. This is
    deliberately NOT the Hungarian assignment the grader uses: two GT people may
    both be credited to the same detection, so the number is an upper bound on
    what the grader would award. That is the point -- the question this answers
    is "is the seated subject visible to the model at all", and an upper bound
    that still reads 9% is a far stronger statement than a matched one would be.

    Detections are the deployed stream: peaks (threshold + NMS + cap of 4) run
    through one fresh `Tracker` over the window in temporal order.
    """
    data = np.load(path)
    preds = np.ascontiguousarray(data["preds"][..., 0])
    people_xy = np.asarray(data["people_xy"])
    people_mask = np.asarray(data["people_mask"]).astype(bool)

    kinds = person_kinds(people_xy, people_mask)
    detections, _ = WindowCandidates(preds).detections(threshold, min_distance)

    tracker = Tracker()
    hits = np.zeros(people_mask.shape[1], dtype=np.int64)
    active = np.zeros(people_mask.shape[1], dtype=np.int64)
    for t, det in enumerate(detections):
        pred = tracker.update(det)
        live = np.flatnonzero(people_mask[t])
        if not len(live):
            continue
        if pred:
            pred_xy = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
            for p in live:
                active[p] += 1
                d = np.hypot(*(pred_xy - people_xy[t, p]).T)
                if d.min() <= MATCH_THRESHOLD:
                    hits[p] += 1
        else:
            active[live] += 1

    return {
        p: (kinds[p], int(hits[p]), int(active[p]))
        for p in range(people_mask.shape[1])
        if kinds[p] is not None
    }


def scenario_label(people):
    """"2 walking + 1 seated", matching REPORT_NOTES.md's table."""
    parts = [
        f"{sum(1 for k, _, _ in people.values() if k == kind)} {kind}"
        for kind in ("walking", "seated")
        if any(k == kind for k, _, _ in people.values())
    ]
    return " + ".join(parts) or "-"


def per_person_table(rows, label="per-person OOF recall"):
    """Print the per-window walking/seated recall table and return the totals."""
    print(f"\n{label} -- a GT identity counts as recalled in a frame if ANY")
    print("detection lands within 1.0 m (upper bound: not Hungarian-matched).\n")
    print(f"{'win':>4}  {'scenario':<22} {'walking':>18} {'seated':>18}")
    totals = {"walking": [0, 0], "seated": [0, 0]}
    for window, people in rows:
        cells = {}
        for kind in ("walking", "seated"):
            hit = sum(h for k, h, _ in people.values() if k == kind)
            act = sum(a for k, _, a in people.values() if k == kind)
            totals[kind][0] += hit
            totals[kind][1] += act
            cells[kind] = f"{100 * hit / act:5.1f}%  ({act:>6})" if act else " " * 15
        print(
            f"{window:>4}  {scenario_label(people):<22} "
            f"{cells['walking']:>18} {cells['seated']:>18}"
        )
    print()
    for kind in ("walking", "seated"):
        hit, act = totals[kind]
        if act:
            print(f"{'ALL ' + kind:>22}: {100 * hit / act:6.2f}%  ({hit}/{act} person-frames)")
    return totals


def _window_id(path):
    return int(os.path.basename(path).split("_")[1][:6])


def scan(oof_dir, threshold):
    """(f1_rows, person_rows) for one cached OOF run."""
    paths = sorted(glob.glob(os.path.join(oof_dir, "*.oof.npz")))
    if not paths:
        raise SystemExit(f"No *.oof.npz in {oof_dir}")
    f1_rows = [(_window_id(p), *score_window(p, threshold)) for p in paths]
    person_rows = [(_window_id(p), person_recall(p, threshold)) for p in paths]
    return f1_rows, person_rows


def _jackknife_se(values_a, values_b, metric):
    """Delete-one jackknife SE of the DIFFERENCE metric(A) - metric(B).

    Two models scored on the same windows are a PAIRED comparison: per-window
    difficulty is common to both and cancels in the difference, so the SE of the
    delta is roughly half the single-model jackknife SE (measured over the Phase
    4 ablation: pooled +-0.0089 against a single-model +-0.018). Using the
    unpaired number as the adoption bar is too strict and would hide real
    effects -- see REPORT_NOTES discipline #6. `values_*` are per-window
    count tuples; `metric` maps a list of them to a scalar.
    """
    n = len(values_a)
    deltas = np.array(
        [
            metric([v for j, v in enumerate(values_a) if j != i])
            - metric([v for j, v in enumerate(values_b) if j != i])
            for i in range(n)
        ]
    )
    return float(np.sqrt((n - 1) / n * np.sum((deltas - deltas.mean()) ** 2)))


def _pooled_f1(counts):
    tp = sum(c[0] for c in counts)
    fp = sum(c[1] for c in counts)
    fn = sum(c[2] for c in counts)
    return prf(tp, fp, fn)[2]


def _recall(counts):
    hit = sum(c[0] for c in counts)
    act = sum(c[1] for c in counts)
    return hit / act if act else 0.0


def compare(oof_dir, other_dir, threshold):
    """Paired A/B of two cached OOF runs: pooled F1 and per-kind recall."""
    a_f1, a_pp = scan(oof_dir, threshold)
    b_f1, b_pp = scan(other_dir, threshold)
    if [w for w, *_ in a_f1] != [w for w, *_ in b_f1]:
        raise SystemExit("The two OOF caches cover different windows -- not paired.")

    print(f"\nPaired comparison at threshold {threshold}")
    print(f"  A (this run) {oof_dir}")
    print(f"  B (baseline) {other_dir}\n")
    print(f"{'metric':>24} {'A':>9} {'B':>9} {'delta':>9} {'paired SE':>10} {'SEs':>7}")

    def report(label, a_counts, b_counts, metric):
        va, vb = metric(a_counts), metric(b_counts)
        se = _jackknife_se(a_counts, b_counts, metric)
        n_se = (va - vb) / se if se else float("nan")
        print(
            f"{label:>24} {va:9.4f} {vb:9.4f} {va - vb:+9.4f} "
            f"{se:10.4f} {n_se:+7.1f}"
        )

    report(
        "pooled F1",
        [row[1:4] for row in a_f1],
        [row[1:4] for row in b_f1],
        _pooled_f1,
    )
    for kind in ("walking", "seated"):
        def kind_counts(rows, kind=kind):
            return [
                (
                    sum(h for k, h, _ in people.values() if k == kind),
                    sum(a for k, _, a in people.values() if k == kind),
                )
                for _, people in rows
            ]
        report(f"{kind} recall", kind_counts(a_pp), kind_counts(b_pp), _recall)

    # Per-window deltas. An aggregate can hide a change that is real in a few
    # windows and absent elsewhere -- which is the shape the seated error mode
    # has taken every time it has been measured -- so print where it moved.
    print("\nPer-window per-person recall, A vs B (percentage points)\n")
    print(
        f"{'win':>4}  {'scenario':<22} "
        f"{'walk A':>8} {'walk B':>8} {'d':>7}   {'seat A':>8} {'seat B':>8} {'d':>7}"
    )
    b_by_window = dict(b_pp)
    for window, people_a in a_pp:
        people_b = b_by_window[window]
        cells = []
        for kind in ("walking", "seated"):
            def pct(people, kind=kind):
                hit = sum(h for k, h, _ in people.values() if k == kind)
                act = sum(a for k, _, a in people.values() if k == kind)
                return 100 * hit / act if act else None
            va, vb = pct(people_a), pct(people_b)
            cells.append(
                f"{va:8.1f} {vb:8.1f} {va - vb:+7.1f}"
                if va is not None and vb is not None
                else " " * 24
            )
        print(f"{window:>4}  {scenario_label(people_a):<22} {cells[0]}   {cells[1]}")


def main(oof_dir, threshold):
    rows, person_rows = scan(oof_dir, threshold)

    subsets = {
        "POOLED (all windows)": lambda w: True,
        "seated-containing": lambda w: w in SEATED_WINDOWS,
        "walking-only": lambda w: w not in SEATED_WINDOWS and w not in EMPTY_WINDOWS,
    }

    print(f"\nOOF breakdown at threshold {threshold} -- {oof_dir}\n")
    print(f"{'win':>4} {'kind':>8} {'tp':>7} {'fp':>7} {'fn':>7} {'P':>7} {'R':>7} {'F1':>7}")
    for window, tp, fp, fn, _ in rows:
        kind = (
            "seated" if window in SEATED_WINDOWS
            else "empty" if window in EMPTY_WINDOWS
            else ""
        )
        p, r, f1 = prf(tp, fp, fn)
        print(f"{window:>4} {kind:>8} {tp:>7} {fp:>7} {fn:>7} {p:>7.3f} {r:>7.3f} {f1:>7.3f}")

    total_fn = sum(row[3] for row in rows)
    print()
    for label, keep in subsets.items():
        tp = sum(row[1] for row in rows if keep(row[0]))
        fp = sum(row[2] for row in rows if keep(row[0]))
        fn = sum(row[3] for row in rows if keep(row[0]))
        p, r, f1 = prf(tp, fp, fn)
        share = f"  ({100 * fn / total_fn:.1f}% of all FN)" if total_fn else ""
        print(f"{label:>22}:  F1 {f1:.4f}   P {p:.4f}   R {r:.4f}   fn {fn}{share}")

    phantom_fp = sum(row[2] for row in rows if row[0] in EMPTY_WINDOWS)
    phantom_frames = sum(row[4] for row in rows if row[0] in EMPTY_WINDOWS)
    print(
        f"{'empty-room phantoms':>22}:  {phantom_fp} false detections over "
        f"{phantom_frames} frames (reported apart -- F1 is undefined with no positives)"
    )

    per_person_table(person_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oof_dir", help="logs/<stamp>/oof directory to score")
    parser.add_argument("--threshold", type=float, default=0.4, help="decode threshold")
    parser.add_argument(
        "--compare",
        metavar="BASELINE_OOF_DIR",
        help=(
            "Score a second cached OOF run and print the PAIRED comparison "
            "(pooled F1 and per-kind recall, each with its delete-one jackknife "
            "SE over windows). See REPORT_NOTES discipline #6."
        ),
    )
    args = parser.parse_args()
    if args.compare:
        compare(args.oof_dir, args.compare, args.threshold)
    else:
        main(args.oof_dir, args.threshold)

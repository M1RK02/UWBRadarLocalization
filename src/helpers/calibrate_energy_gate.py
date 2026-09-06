"""Choose the conditional-inference gate, and prove which windows it can reach.

Phase 5.1: skip the model entirely on frames whose decluttered energy says
nothing is moving. A gated frame costs one pass over 846 values instead of
~330k MACs, and it also cannot emit a phantom -- the empty windows are where
this model's false positives live.

THE CONSTRAINT, and it is the same wall Phase 4 hit eight times. w021 is a
single seated subject with nobody walking, and its occupied frames are DIMMER
THAN AN EMPTY ROOM: median 0.01693 against 0.01627 and 0.01632 for w022/w023,
a ratio of 1.04. So "never skip an occupied frame" is not a calibratable rule
here -- honouring it for w021 puts the gate below every empty room too and buys
a 0.06% skip rate, i.e. nothing.

The gate is therefore calibrated against the windows where a person is
SEPARABLE from an empty room at all, and w021 is excluded BY MEASUREMENT rather
than by name (`separable_windows` below): its 1st-percentile occupied energy
falls below the empty-room median, so no threshold can distinguish it. Windows
that fail that test are reported as a known limitation, not silently dropped.

FRAME 0 IS ALWAYS GATED, in every window, and that is correct. `remove_clutter`
seeds its EMA from frame 0, so the decluttered frame 0 is identically zero
(~1e-15 after normalization) and the model has nothing to see there either. It
is excluded from every statistic below because it is a cold-start artifact, not
a measurement.

STATISTIC. `mean` and `max` are both reported. `max` has more than twice the
pooled d' (5.53 vs 2.76) and is the WORSE gate: at a matched safety factor it
skips 77.0% of empty frames against `mean`'s 95.1%. The pooled d' is inflated by
walking frames saturating `max` at 1.0, which says nothing about the
empty-vs-quiet boundary the gate actually sits on. Measurement discipline #9,
in miniature.

Run:  .venv/bin/python src/helpers/calibrate_energy_gate.py
      .venv/bin/python src/helpers/calibrate_energy_gate.py --gate 0.0182
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.cache import cached_window
from src.channels import build_channels
from src.data_loader import list_windows, load_seated_windows
from src.preprocessing import frame_energy, normalize_channels

# How far below the dimmest separable occupied frame the gate sits. The margin
# exists because the final test is a DIFFERENT acquisition session, so the
# dimmest frame in these 24 windows is a sample, not a bound.
GATE_SAFETY_FACTOR = 0.9


def window_stats(data_dir, p99):
    """Per-window (name, {stat: (T-1,) values}, occupied (T-1,)), frame 0 dropped."""
    rows = []
    for filepath in list_windows(data_dir):
        z, _ = cached_window(filepath)
        x = normalize_channels(build_channels(z), p99)
        flat = x[..., 0].reshape(len(x), -1)
        occupied = np.load(filepath)["people_mask"][: len(x)].any(axis=1)
        rows.append(
            (
                os.path.basename(filepath),
                {"mean": flat.mean(axis=1)[1:], "max": flat.max(axis=1)[1:]},
                occupied[1:],
            )
        )
        del z, x, flat
    return rows


def empty_reference(rows, stat):
    """Pooled energies of the windows whose ground truth is empty throughout."""
    empty = [s[stat] for _, s, m in rows if not m.any()]
    return np.concatenate(empty) if empty else None


def separable_windows(rows, stat):
    """Windows where an occupied frame is distinguishable from an empty room.

    The test: this window's 1st-percentile occupied energy must exceed the
    pooled empty-room median. A window that fails it cannot be protected by ANY
    gate, so including it in the calibration would only push the gate below
    every empty room and buy nothing.
    """
    reference = empty_reference(rows, stat)
    if reference is None:
        return [r[0] for r in rows if r[2].any()], []
    floor = float(np.median(reference))
    ok, blind = [], []
    for name, stats, occupied in rows:
        if not occupied.any():
            continue
        separable = float(np.quantile(stats[stat][occupied], 0.01)) > floor
        (ok if separable else blind).append(name)
    return ok, blind


def derive_gate(rows, stat, safety_factor=GATE_SAFETY_FACTOR):
    """The deployed gate: a margin below the dimmest separable occupied frame."""
    ok, _ = separable_windows(rows, stat)
    dimmest = min(
        float(s[stat][m].min()) for n, s, m in rows if n in ok and m.any()
    )
    return dimmest * safety_factor, dimmest


def main():
    ap = argparse.ArgumentParser(description="Calibrate the 5.1 energy gate.")
    ap.add_argument("--data-dir", default="multi-person-localization/data")
    ap.add_argument("--stat", default="mean", choices=["mean", "max"])
    ap.add_argument("--gate", type=float, default=None,
                    help="check this gate instead of deriving one")
    ap.add_argument("--p99", type=float, default=None,
                    help="override p99 (default: src/models/best_p99.npy)")
    args = ap.parse_args()

    seated = load_seated_windows(args.data_dir)

    p99_path = Path("src/models/best_p99.npy")
    if args.p99 is not None:
        p99 = np.array([args.p99])
    elif p99_path.exists():
        p99 = np.atleast_1d(np.load(p99_path))
    else:
        sys.exit("no --p99 given and src/models/best_p99.npy is missing")
    print(f"p99 = {p99}")

    rows = window_stats(args.data_dir, p99)

    print("\nStatistic choice -- occupied vs empty frames, pooled")
    print(f"  {'stat':<6}{'occ mean':>11}{'emp mean':>11}{'d-prime':>10}"
          f"{'empty skipped at its own safe gate':>38}")
    for stat in ("mean", "max"):
        occ = np.concatenate([s[stat][m] for _, s, m in rows if m.any()])
        emp = empty_reference(rows, stat)
        spread = np.sqrt(0.5 * (occ.std() ** 2 + emp.std() ** 2))
        g, _ = derive_gate(rows, stat)
        print(f"  {stat:<6}{occ.mean():>11.5f}{emp.mean():>11.5f}"
              f"{(occ.mean() - emp.mean()) / spread:>10.3f}"
              f"{100 * (emp < g).mean():>37.1f}%")

    stat = args.stat
    ok, blind = separable_windows(rows, stat)
    derived, dimmest = derive_gate(rows, stat)
    gate = args.gate if args.gate is not None else derived

    n_occupied = sum(1 for r in rows if r[2].any())
    print(f"\nSeparable windows: {len(ok)} of {n_occupied} occupied")
    if blind:
        print("  NOT separable from an empty room -- no gate can protect these:")
        for name in blind:
            print(f"      {name}{'  (seated)' if name in seated else ''}")
    print(f"  dimmest separable occupied frame : {dimmest:.5f}")
    print(f"  gate = {gate:.6f}" + ("" if args.gate is not None
                                    else f"  ({GATE_SAFETY_FACTOR} x dimmest)"))

    print(f"\n  {'window':<20}{'kind':<10}{'occ':>8}{'skip%':>8}{'occ skipped':>13}")
    violations = []
    for name, stats, occupied in rows:
        v = stats[stat]
        gated = v < gate
        occ_gated = int((gated & occupied).sum())
        kind = ("empty" if not occupied.any()
                else "seated" if name in seated else "walking")
        flag = ""
        if occ_gated and name in ok:
            violations.append((name, occ_gated, int(occupied.sum())))
            flag = "  <-- VIOLATION"
        elif occ_gated:
            flag = "  (not separable)"
        print(f"  {name:<20}{kind:<10}{int(occupied.sum()):>8}"
              f"{100 * gated.mean():>7.1f}%{occ_gated:>13}{flag}")

    all_v = np.concatenate([s[stat] for _, s, _ in rows])
    all_occ = np.concatenate([m for _, _, m in rows])
    emp = empty_reference(rows, stat)
    print(f"\n  frames gated overall     : {(all_v < gate).sum():,} / {len(all_v):,} "
          f"({100 * (all_v < gate).mean():.1f}%)")
    print(f"  of empty-room frames     : {100 * (emp < gate).mean():.1f}%")
    # Built outside the f-string on purpose: a multi-line expression *inside*
    # one is PEP 701, i.e. Python 3.12+, and this repo runs on 3.10.
    occ_gated_rates = [(s[stat][m] < gate).mean() for n, s, m in rows if n in ok]
    print(f"  of separable occupied    : {100 * np.mean(occ_gated_rates):.4f}%")

    print()
    if violations:
        print("*** GATE REACHES A SEPARABLE WINDOW'S OCCUPANT -- do not deploy it:")
        for name, n, total in violations:
            print(f"      {name}: {n}/{total} occupied frames gated")
        return 1
    print("GATE SAFE -- no occupied frame gated in any of the "
          f"{len(ok)} separable windows")
    if blind:
        print(f"           {len(blind)} window(s) are unprotectable and ARE gated; "
              "the sweep decides whether that trade is worth taking")
    return 0


if __name__ == "__main__":
    sys.exit(main())

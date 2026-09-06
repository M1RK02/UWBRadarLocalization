"""Prove that `--seated-oversample K` duplicates ONLY training frames.

This exists because one specific bug would silently invalidate the whole 4.1b
experiment without changing a single visible number: applying the oversampling
before the per-fold split, or from a global window list, would copy a held-out
window's frames into its own fold's training set. The CV score would then rise
for a reason that has nothing to do with the hypothesis, and nothing in the run
output would look wrong.

So the check is run against the real fold construction, not a re-implementation
of it, and it is run at FRAME granularity. `load_fold_data` fills its buffer
window by window in list order, so repeating each window's basename by its
cached frame count reconstructs, row for row, the training index array the loss
actually iterates over. Every assertion below is made on that array.

Run:  .venv/bin/python src/helpers/verify_oversample.py            # K=3
      .venv/bin/python src/helpers/verify_oversample.py -k 5 --load-fold 1
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.cache import cached_frame_count
from src.data_loader import (
    get_kfold_splits,
    list_windows,
    load_fold_data,
    load_seated_windows,
    oversample_seated,
)

SEED = 42
# The 8 windows Phase 4.1b names. Hardcoded HERE on purpose, as an
# independent cross-check of `load_seated_windows`, which derives the same set
# from metadata.jsonl -- a literal and a derivation that agree are evidence;
# either one alone is an assumption.
DOCUMENTED_SEATED = {4, 8, 9, 10, 11, 14, 15, 21}


def window_id(path):
    return int(os.path.basename(path).split("_")[1][:6])


def frame_provenance(filepaths):
    """The training index array, labelled by source window.

    One entry per row `load_fold_data` will build, in the same order, holding
    the basename of the window that row came from.
    """
    return np.concatenate(
        [np.full(cached_frame_count(f), os.path.basename(f)) for f in filepaths]
    )


def main(data_dir, k, factor, load_fold):
    all_files = list_windows(data_dir)
    seated = load_seated_windows(data_dir)
    frames = {os.path.basename(f): cached_frame_count(f) for f in all_files}

    print(f"Dataset: {len(all_files)} windows, {sum(frames.values())} frames")
    print(f"Seated windows from metadata: {sorted(window_id(w) for w in seated)}")
    assert {window_id(w) for w in seated} == DOCUMENTED_SEATED, (
        "load_seated_windows disagrees with the 8 windows Phase 4.1b names"
    )
    print(f"  matches the documented set {sorted(DOCUMENTED_SEATED)}  OK\n")

    # Two independent fold constructions, each from a freshly seeded global RNG
    # so `calculate_p99_from_files` draws the same sample in both -- exactly what
    # a fresh `train.py` process does at import time.
    np.random.seed(SEED)
    base = get_kfold_splits(data_dir, k=k, seed=SEED, seated_oversample=1)
    np.random.seed(SEED)
    over = get_kfold_splits(data_dir, k=k, seed=SEED, seated_oversample=factor)

    print(f"\n{'=' * 78}\nPer-fold checks at K={factor}\n{'=' * 78}")
    header = f"{'fold':>4} {'val windows':<22} {'train rows':>11} {'oversampled':>12} {'x':>6}"
    print(header)

    for (i, base_train, val_files, p99_base), (_, over_train, over_val, p99_over) in zip(
        base, over
    ):
        val_names = {os.path.basename(f) for f in val_files}
        provenance = frame_provenance(over_train)
        present, counts = np.unique(provenance, return_counts=True)
        seen = dict(zip(present.tolist(), counts.tolist()))

        # 1. THE check: no held-out window contributes a single training row.
        leaked = val_names & set(seen)
        assert not leaked, f"fold {i + 1}: held-out windows in training rows: {leaked}"

        # 2. The val split itself is untouched by the knob.
        assert [os.path.basename(f) for f in over_val] == [
            os.path.basename(f) for f in val_files
        ], f"fold {i + 1}: oversampling changed the validation set"

        # 3. Exact multiplicity: seated train windows K times, everything else once.
        for name in {os.path.basename(f) for f in base_train}:
            want = frames[name] * (factor if name in seated else 1)
            assert seen.get(name) == want, (
                f"fold {i + 1}: {name} contributes {seen.get(name)} rows, expected {want}"
            )

        # 4. The base training list survives as a prefix -- nothing was dropped
        #    or reordered, only appended.
        assert over_train[: len(base_train)] == base_train, (
            f"fold {i + 1}: oversampling perturbed the base training list"
        )

        # 5. p99 is bit-identical: it is computed on the DISTINCT windows before
        #    oversampling, so the knob cannot move the input normalization.
        assert np.array_equal(p99_base, p99_over), (
            f"fold {i + 1}: p99 changed with the oversample factor "
            f"({p99_base} -> {p99_over}) -- it must be computed pre-oversampling"
        )

        n_base = sum(frames[os.path.basename(f)] for f in base_train)
        n_over = len(provenance)
        print(
            f"{i + 1:>4} {','.join(f'{window_id(f):03d}' for f in val_files):<22} "
            f"{n_base:>11,} {n_over:>12,} {n_over / n_base:>6.2f}"
        )

    print("\nAll folds: no held-out frames in training, exact K multiplicity,")
    print("val split unchanged, base list preserved, p99 bit-identical.  OK")

    # 6. K=1 is a true no-op, checked on the fold lists themselves.
    np.random.seed(SEED)
    noop = get_kfold_splits(data_dir, k=k, seed=SEED, seated_oversample=1)
    assert [f[1] for f in noop] == [f[1] for f in base], "K=1 changed the training lists"
    assert all(
        oversample_seated(t, seated, 1) == list(t) for _, t, _, _ in base
    ), "oversample_seated(.., 1) is not the identity"
    print("K=1 is a bit-exact no-op on every fold's training list.  OK")

    # 7. Optional: materialize a real fold and confirm the array the model is
    #    actually fed has the row count the provenance array predicts.
    if load_fold:
        i, train_files, _, p99 = over[load_fold - 1]
        expected = len(frame_provenance(train_files))
        X, y = load_fold_data(train_files, p99)
        assert len(X) == len(y) == expected, (
            f"fold {load_fold}: load_fold_data built {len(X)} rows, expected {expected}"
        )
        print(
            f"\nFold {load_fold} materialized: X{X.shape} y{y.shape} "
            f"= {X.nbytes / 2**30:.2f} + {y.nbytes / 2**30:.2f} GiB, "
            f"{len(X):,} rows as predicted.  OK"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="multi-person-localization/data")
    parser.add_argument("-k", "--factor", type=int, default=3, help="oversample factor")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument(
        "--load-fold",
        type=int,
        default=0,
        metavar="N",
        help="Also build fold N's real arrays and check the row count (slow, ~1 GB)",
    )
    args = parser.parse_args()
    main(args.data_dir, args.folds, args.factor, args.load_fold)

"""Windows in, fold-shaped arrays out: metadata, splits, calibration, tf.data."""

import glob
import json
import os
import re

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from src.cache import cached_frame_count, cached_window
from src.channels import ALL_RADARS, CHANNEL_SETS, build_channels, channel_p99
from src.preprocessing import frame_energy, normalize_channels


def window_channels(filepath, channel_set="mag", radar_indices=None):
    """(x, y) for one window, x as (T, R, 3, 47, C) unnormalized channels.

    The single entry point training, OOF scoring and quantization all go
    through, so those three cannot drift apart on the channel set or the radar
    subset. The cache underneath always holds all 6 radars, which is what keeps
    one cache valid for every subset.
    """
    z, y = cached_window(filepath)
    return build_channels(z, channel_set, radar_indices), y


def n_radars(radar_indices):
    """How many radars reach the model. `None` means all of them."""
    return len(ALL_RADARS if radar_indices is None else radar_indices)


def list_windows(data_dir):
    """All window .npz paths in a directory, sorted (stable ordering)."""
    return sorted(glob.glob(os.path.join(data_dir, "*.npz")))


def _count_people(description):
    """Total people *present* in a window (walking + seated), parsed from the free-text
    metadata description. Returns None if nothing parses (caller asserts on this)."""
    if "0 subjects" in description:
        return 0
    n = sum(int(x) for x in re.findall(r"(\d+)\s+subject", description))
    n += sum(int(x) for x in re.findall(r"(\d+)\s+seated", description))
    # e.g. "1 subject while the other is seated" -> the seated one isn't counted above
    if "the other is seated" in description:
        n += 1
    return n if n else None


def _metadata_rows(data_dir):
    """Yield (window_basename.npz, description) from metadata.jsonl (sibling of data_dir)."""
    metadata_path = os.path.join(
        os.path.dirname(os.path.normpath(data_dir)), "metadata.jsonl"
    )
    with open(metadata_path) as f:
        for line in f:
            row = json.loads(line)
            yield os.path.basename(row["path"]), row["description"]


def load_people_counts(data_dir):
    """Map {window_basename.npz: people_count} from metadata.jsonl (sibling of data_dir)."""
    return {
        basename: _count_people(description)
        for basename, description in _metadata_rows(data_dir)
    }


def load_seated_windows(data_dir):
    """Basenames of the windows whose description mentions a seated subject.

    Derived from `metadata.jsonl`, never hardcoded: a literal list of window
    numbers rots silently if the dataset changes. On the shipped dataset this is
    windows 004, 008-011, 014, 015 and 021 -- the 8 holding ~74% of pooled FN.
    """
    return {
        basename
        for basename, description in _metadata_rows(data_dir)
        if "seat" in description.lower()
    }


def oversample_seated(train_files, seated_windows, factor):
    """Repeat one fold's seated-containing TRAINING windows `factor` times.

    The loss then sees a seated frame `factor` times per epoch and everything
    else once; `make_dataset` permutes globally afterwards, so the copies scatter
    through the epoch. `factor=1` is a verified bit-exact no-op. Negative result,
    REPORT_NOTES Step 6.

    LEAK SAFETY: the copies are drawn from `train_files` itself, never from a
    global window list, so a held-out window structurally cannot enter its own
    fold's training set. Appended rather than interleaved so the original list
    stays a prefix and both properties are visible by inspection. The caller
    asserts it anyway, and `verify_oversample.py` proves it at frame granularity.
    """
    if factor < 1:
        raise ValueError(f"seated oversample factor must be >= 1, got {factor}")
    if factor == 1:
        return list(train_files)
    extra = [f for f in train_files if os.path.basename(f) in seated_windows]
    return list(train_files) + extra * (factor - 1)


def calculate_p99_from_files(
    filepaths, sample_ratio=0.1, channel_set="mag", radar_indices=None
):
    """99th percentile of each input channel, from a 10% frame sample.

    One value PER CHANNEL: a single shared scale stops working the moment
    channels carry different units, and signed channels take their percentile on
    |x| so the range stays symmetric.
    """
    sampled = []
    for filepath in filepaths:
        # Cache-backed decluttered channels (p99-independent), (T, R, 3, 47, C).
        x, _ = window_channels(filepath, channel_set, radar_indices)

        # Sample frames randomly (same draw as before -> identical p99).
        T = x.shape[0]
        num_samples = max(1, int(T * sample_ratio))
        indices = np.random.choice(T, num_samples, replace=False)
        sampled.append(x[indices].reshape(-1, x.shape[-1]))
        del x

    if not sampled:
        return np.ones(len(CHANNEL_SETS[channel_set]))

    return channel_p99(np.concatenate(sampled), channel_set)


ENERGY_GATE_SAFETY_FACTOR = 0.9


def calculate_energy_gate_from_files(
    filepaths, p99, safety_factor=ENERGY_GATE_SAFETY_FACTOR,
    channel_set="mag", radar_indices=None,
):
    """The 5.1 conditional-inference gate, from TRAINING windows only.

    Below this frame energy the model is skipped and a zero heatmap substituted.
    Calibrated exactly as `p99` is -- on the fold's own training files, never on
    the held-out ones -- so the OOF score stays honest.

    THE RULE, and why it is not "never skip an occupied frame". One window's
    occupant (w021: seated, nobody walking) is DIMMER than an empty room, so a
    gate that protects it sits below every empty room too and skips nothing at
    all. Windows are therefore split by whether a person is separable from an
    empty room at all -- 1st-percentile occupied energy above the pooled
    empty-room median -- and the gate is a margin below the dimmest frame of the
    separable ones. The unprotectable windows are excluded BY MEASUREMENT, not
    by name, so this rule survives a different dataset.

    Degrades safe in both directions: with no empty window to reference, every
    occupied window counts as separable and the dimmest of them holds the gate
    down; with no separable window at all, it returns 0.0 and nothing is ever
    gated. See `src/helpers/calibrate_energy_gate.py` for the full table.

    Frame 0 is excluded throughout: `remove_clutter` seeds its EMA from frame 0,
    so the decluttered frame 0 is identically zero in every window. It is always
    gated, and correctly so -- the model has nothing to see there either.
    """
    energies, occupancies = [], []
    for filepath in filepaths:
        x, _ = window_channels(filepath, channel_set, radar_indices)
        energies.append(frame_energy(normalize_channels(x, p99))[1:])
        occupied = np.load(filepath)["people_mask"][: len(x)].any(axis=1)
        occupancies.append(occupied[1:])
        del x

    empty = [e for e, o in zip(energies, occupancies) if not o.any()]
    floor = float(np.median(np.concatenate(empty))) if empty else -np.inf

    dimmest = [
        float(e[o].min())
        for e, o in zip(energies, occupancies)
        if o.any() and float(np.quantile(e[o], 0.01)) > floor
    ]
    return min(dimmest) * safety_factor if dimmest else 0.0


def load_fold_data(filepaths, p99, channel_set="mag", radar_indices=None):
    """All windows of a fold as (X_all, y_all), in the model's input layout.

    Fills a preallocated buffer window by window rather than concatenating a
    list: concatenating holds the whole fold twice at the moment of the copy,
    ~2 GB for a 2-channel fold, on top of the copy TensorFlow takes downstream.
    """
    total = sum(cached_frame_count(f) for f in filepaths)
    n_channels = len(CHANNEL_SETS[channel_set])
    X_all = np.empty(
        (total, n_radars(radar_indices), 3, 47, n_channels), dtype=np.float32
    )
    y_all = None

    at = 0
    for filepath in filepaths:
        # (T,R,3,47,C), (T,H,W,1)
        x, y = window_channels(filepath, channel_set, radar_indices)
        if y_all is None:
            y_all = np.empty((total, *y.shape[1:]), dtype=np.float32)
        X_all[at : at + len(x)] = normalize_channels(x, p99)
        y_all[at : at + len(y)] = y
        at += len(x)
        del x, y

    return X_all, y_all


def _permute_rows_inplace(a, perm):
    """In-place `a[:] = a[perm]`, without allocating a second copy of `a`.

    `a[perm]` allocates a whole new fold (~1.2 GB for 2 channels); measured peak
    for the 24-window refit that way was 5.8 GB on a 7 GB machine, at the very
    END of a ~22 minute run. Rewriting each permutation cycle through a
    single-row scratch buffer costs one row instead.
    """
    scratch = np.empty_like(a[0])
    visited = np.zeros(len(perm), dtype=bool)
    for start in range(len(perm)):
        if visited[start]:
            continue
        scratch[...] = a[start]
        current = start
        while True:
            visited[current] = True
            source = perm[current]
            if source == start:
                a[current] = scratch
                break
            a[current] = a[source]
            current = source
    return a


def make_dataset(
    X, y, batch_size=32, is_training=False, radar_dropout_prob=0.1, shuffle_seed=42
):
    """Build the tf.data pipeline for one fold's in-memory arrays.

    Training path: global permutation -> reservoir shuffle 4000 -> batch ->
    per-batch radar dropout -> prefetch.
    """
    if is_training:
        # `load_fold_data` concatenates windows in file order and each window is
        # a single scenario for its full 300 s, so the stream is a sequence of
        # scenario-homogeneous blocks. tf.data's shuffle is a *streaming
        # reservoir*: with buffer 4000 an element can only move ~4000 positions,
        # i.e. 2.7% of a ~150k-frame fold, so one recording still supplies ~68%
        # of every batch. That biases the four BatchNormalization layers, which
        # normalize on per-batch statistics -- the same input is normalized
        # differently depending on which scenario happened to dominate its batch.
        # Permuting the arrays once up front makes the shuffle global (any frame
        # can land in any batch); the reservoir below still varies the order
        # across epochs. Costs one transient copy of the fold (~0.5 GB per
        # input channel).
        #
        # Uses a DEDICATED generator, not the seeded global stream: p99 is drawn
        # from that global stream (calculate_p99_from_files -> np.random.choice)
        # and is computed after the fold loop, so permuting from it would shift
        # p99 as a side effect. That is still deterministic, but it would
        # confound a baseline-vs-fixed comparison of this very change.
        # Permuted in place: `X` and `y` are freshly built by `load_fold_data`
        # for this one call and the caller drops them immediately after, so
        # mutating them is safe and saves a full extra copy of the fold.
        perm = np.random.default_rng(shuffle_seed).permutation(len(X))
        X = _permute_rows_inplace(X, perm)
        y = _permute_rows_inplace(y, perm)

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if is_training:
        dataset = dataset.shuffle(buffer_size=4000)

    # Batch FIRST to drastically reduce map call overhead
    dataset = dataset.batch(batch_size)

    if is_training and radar_dropout_prob > 0:

        def apply_dropout_batched(x, y):
            batch_sz = tf.shape(x)[0]
            # One mask value per (sample, radar), broadcast over the remaining
            # axes. Built from the input's own SHAPE, not from constants, so the
            # same code serves the 1-channel baseline, the multi-channel Phase 4
            # inputs and a reduced radar subset. On the 6-radar path the mask
            # still has B*6 elements and draws exactly the same random numbers,
            # keeping the baseline bit-reproducible; a 4-radar run necessarily
            # draws B*4 instead, which is inherent to the config, not a defect.
            trailing = tf.ones([tf.rank(x) - 2], dtype=tf.int32)
            shape = tf.concat([tf.stack([batch_sz, tf.shape(x)[1]]), trailing], axis=0)
            mask = tf.cast(
                tf.random.uniform(shape) > radar_dropout_prob, tf.float32
            )
            return x * mask, y

        dataset = dataset.map(
            apply_dropout_batched, num_parallel_calls=tf.data.AUTOTUNE
        )

    return dataset.prefetch(tf.data.AUTOTUNE)


def get_kfold_splits(
    data_dir, k=6, seed=42, channel_set="mag", radar_indices=None, seated_oversample=1
):
    """
    Splits the .npz files into k folds for cross-validation, stratified by the number of
    people present per window (0-4). This guarantees every training fold has seen all
    scenario classes. Yields (fold_idx, train_files, val_files, p99) for each fold,
    with p99 computed on that fold's training files only (no leakage).

    `seated_oversample` > 1 repeats each seated-containing window of a fold's
    TRAINING list that many times (see `oversample_seated`). It is applied per
    fold, after the split and after p99 -- never to `all_files` -- so it cannot
    move a held-out window into its own training set.
    """
    all_files = list_windows(data_dir)
    seated = load_seated_windows(data_dir)

    # Stratify on people-count so no scenario class can be fully swallowed by one val fold.
    count_map = load_people_counts(data_dir)
    labels = [count_map.get(os.path.basename(f)) for f in all_files]
    assert None not in labels, (
        "Some windows have no people-count label (metadata parse gap): "
        f"{[f for f, y in zip(all_files, labels) if y is None]}"
    )

    # NOTE: sklearn warns "least populated class < n_splits" when a class has fewer members
    # than k (e.g. the 2 empty-room windows). That is expected and harmless here: it only
    # means that class can't appear in every *validation* fold, which is irrelevant to the
    # training-side coverage guarantee.
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    splits = list(skf.split(all_files, labels))

    folds = []
    for i, (train_idx, val_idx) in enumerate(splits):
        train_files = [all_files[j] for j in train_idx]
        val_files = [all_files[j] for j in val_idx]

        print(f"--- Preparing Fold {i + 1}/{k} ---")
        print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

        # Calculate p99 only on the training files for this fold.
        # ORDER MATTERS: p99 is computed on the DISTINCT training windows, before
        # any oversampling. It is a calibration scale, not a gradient budget --
        # weighting the seated windows 3x inside it would change the input
        # normalization as a side effect and stop this being a one-variable test.
        # It also keeps the draw `calculate_p99_from_files` takes from the global
        # RNG identical across K, so p99 is bit-identical between the runs.
        p99 = calculate_p99_from_files(
            train_files, channel_set=channel_set, radar_indices=radar_indices
        )

        train_files = oversample_seated(train_files, seated, seated_oversample)
        assert not (set(train_files) & set(val_files)), (
            f"Fold {i + 1}: oversampling leaked held-out windows into training: "
            f"{sorted(os.path.basename(f) for f in set(train_files) & set(val_files))}"
        )
        if seated_oversample > 1:
            print(
                f"Seated oversample x{seated_oversample}: "
                f"{len(train_idx)} -> {len(train_files)} training windows "
                f"({sum(1 for f in train_files if os.path.basename(f) in seated)} "
                f"seated-containing entries)"
            )

        folds.append((i, train_files, val_files, p99))

    return folds

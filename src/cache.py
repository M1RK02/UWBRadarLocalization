"""Disk cache for the p99- and channel-independent preprocessing artifact.

Declutter is the slow part of the pipeline and the raw `.npz` set is 1.5 GB, so
the heavy normalization-free result is cached and every CV fold reuses it.

The cached artifact is the **complex** decluttered CIR, one step before
`magnitude()` collapses the phase. That is independent of both p99 (which is
per-fold) and the channel set, so switching channel sets is a cheap
re-derivation rather than a full re-declutter. Targets `y` are cached too.

`PREPROC_VERSION` hashes the knobs that define these artifacts, so editing any of
them rebuilds the cache transparently -- no manual invalidation.
"""

import hashlib
import inspect
import os

import numpy as np

from src.channels import build_channels, channel
from src.preprocessing import (
    CROP_END,
    CROP_START,
    decluttered_complex,
    format_targets_grid,
    remove_clutter,
)

CACHE_DIR = os.environ.get("EEAI_CACHE_DIR", "cache")

# Derived from preprocessing rather than restated, so the key cannot drift from
# the code it describes. This is the one knob `_preproc_version` cannot reach by
# introspection.
_CROP = f"{CROP_START}:{CROP_END}"


def _preproc_version():
    """Short hash of every parameter the cached artifact depends on."""
    alpha = inspect.signature(remove_clutter).parameters["alpha"].default
    sig = inspect.signature(format_targets_grid).parameters
    key = (
        f"v2|alpha={alpha}|crop={_CROP}"
        f"|grid={sig['grid_h'].default}x{sig['grid_w'].default}"
        f"|sigma={sig['sigma'].default}"
    )
    return hashlib.sha1(key.encode()).hexdigest()[:12]


PREPROC_VERSION = _preproc_version()


def _build(filepath):
    """Compute (z, y) for one window from raw data."""
    data = np.load(filepath)
    z = decluttered_complex(remove_clutter(data["radar_cir_iq"]))  # (T, 6, 3, 47)
    y = format_targets_grid(data["people_xy"], data["people_mask"])  # (T, H, W, 1)
    return z.astype(np.complex64), y.astype(np.float32)


def cached_window(filepath, cache_dir=CACHE_DIR):
    """(z, y) for a window, from disk when the version matches.

    z : (T, 6, 3, 47) complex64 decluttered, cropped CIR.
    y : (T, grid_h, grid_w, 1) Gaussian occupancy target.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, os.path.basename(filepath) + ".cplx.npz")

    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if str(cached["version"]) == PREPROC_VERSION:
            return cached["z"], cached["y"]

    z, y = _build(filepath)
    np.savez(cache_path, z=z, y=y, n_frames=len(z), version=PREPROC_VERSION)
    return z, y


def cached_frame_count(filepath, cache_dir=CACHE_DIR):
    """Frame count for a window without materializing it.

    An .npz holds each array as its own zip member, so reading the one-element
    `n_frames` entry costs nothing. Lets a caller size a fold buffer up front
    instead of concatenating a list, which would briefly hold two copies of a
    ~1 GB fold.
    """
    cache_path = os.path.join(cache_dir, os.path.basename(filepath) + ".cplx.npz")
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if str(cached["version"]) == PREPROC_VERSION and "n_frames" in cached:
            return int(cached["n_frames"])
    return len(cached_complex(filepath, cache_dir=cache_dir))


def cached_complex(filepath, cache_dir=CACHE_DIR):
    """(T, 6, 3, 47) complex64 decluttered CIR only (skips the targets)."""
    return cached_window(filepath, cache_dir=cache_dir)[0]


def cached_magnitude(filepath, cache_dir=CACHE_DIR):
    """(T, 6, 3, 47) decluttered magnitude -- the deployed model's input.

    Derived from the complex cache rather than stored: `channel(z, "mag")` is
    byte-identical to the magnitude the old cache held.
    """
    return channel(cached_complex(filepath, cache_dir=cache_dir), "mag")


def cached_channels(filepath, channel_set="mag", cache_dir=CACHE_DIR):
    """(x, y) for a window with x as (T, 6, 3, 47, C) unnormalized channels."""
    z, y = cached_window(filepath, cache_dir=cache_dir)
    return build_channels(z, channel_set), y

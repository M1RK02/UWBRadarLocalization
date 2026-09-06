"""The deployed signal path: raw CIR in, model input and decoded positions out.

Everything here runs on the ESP32 side of the problem, so `submission/code.py`
carries a verbatim copy of `remove_clutter`, `magnitude`, `normalize_channels`,
`preprocess_radar` and `extract_peaks_from_grid`. Change one, change both, then
run `src/helpers/verify_parity.py`.

The Phase 4 input-representation ablation lives in `src/channels.py`; none of it
was adopted, and nothing here depends on it.
"""

import math

import numpy as np
import scipy.ndimage as ndimage
import scipy.signal

# Range-bin crop. Bins 0-4 carry TX->RX coupling inside the sensor itself; bin 52
# sits at ~7.8 m, just short of the room's ~8.65 m diagonal, so the crop keeps
# essentially the full room while excluding bins that can only be multipath.
# `src/cache.py` derives its invalidation key from these, so editing them
# rebuilds the cache automatically.
CROP_START, CROP_END = 5, 52

# Capture rate, project_spec.md §3. The one definition in the project: `tracker`
# and `sweep_postproc` import it rather than restating 25.
FRAME_RATE_HZ = 25.0

ROOM_WIDTH_M, ROOM_HEIGHT_M = 4.8, 7.2
GRID_H, GRID_W = 18, 12


def remove_clutter(radar_cir_iq, alpha=0.99):
    """Subtract an EMA background from (T, 6, 3, 120, 2) raw CIR.

    The EMA is a 0.04 Hz high-pass, so a static room cancels and a moving one
    does not. `zi` is seeded from frame 0 rather than from zero: without it the
    filter assumes the background starts at 0 and reads the whole room as one
    huge fake movement for the first few seconds.
    """
    b_ema = [1.0 - alpha]
    a_ema = [1.0, -alpha]

    zi = scipy.signal.lfilter_zi(b_ema, a_ema)
    zi_broadcast = zi.reshape((1, 1, 1, 1, 1)) * radar_cir_iq[0:1]

    background, _ = scipy.signal.lfilter(
        b_ema, a_ema, radar_cir_iq, axis=0, zi=zi_broadcast
    )
    return (radar_cir_iq - background).astype(np.float32)


def remove_clutter_streaming(radar_cir_iq, alpha=0.99):
    """`remove_clutter`, one frame at a time, so the recording is never resident.

    Feeding `lfilter` a single frame per call while carrying its own `zi` is the
    same code path as the batch call, so the output is bit-for-bit identical.
    Hand-rolling `bg = alpha*bg + (1-alpha)*x` is NOT identical -- the arithmetic
    order differs and the result drifts by ~4e-2.
    """
    b_ema = [1.0 - alpha]
    a_ema = [1.0, -alpha]

    zi = scipy.signal.lfilter_zi(b_ema, a_ema)
    zi = zi.reshape((1, 1, 1, 1, 1)) * radar_cir_iq[0:1]

    for t in range(radar_cir_iq.shape[0]):
        frame = np.asarray(radar_cir_iq[t : t + 1])
        background, zi = scipy.signal.lfilter(b_ema, a_ema, frame, axis=0, zi=zi)
        yield (frame - background).astype(np.float32)


def decluttered_complex(radar_cir_iq):
    """Cropped complex CIR, (..., 6, 3, 47). What `src/cache.py` stores.

    One step upstream of `magnitude`: same crop, phase still intact, and
    independent of both p99 and the channel set.
    """
    sliced = radar_cir_iq[..., CROP_START:CROP_END, :]
    return sliced[..., 0] + 1j * sliced[..., 1]


def magnitude(radar_cir_iq):
    """Cropped |I + jQ|, (..., 6, 3, 47). p99-independent.

    NOT `np.abs` on the complex form: `np.abs` uses a hypot-style algorithm whose
    result differs in the last bits from `sqrt(re**2 + im**2)`. The cache and the
    deployed model were both built on the sqrt form.
    """
    sliced = radar_cir_iq[..., CROP_START:CROP_END, :]
    return np.sqrt(sliced[..., 0] ** 2 + sliced[..., 1] ** 2)


def normalize_channels(x, p99):
    """Scale each channel by its own p99 into [0, 1] (or [-1, 1] if signed).

    `p99` is one value per channel, in channel-set order -- a scalar range stops
    working once channels carry different physical units.

    The divisor is cast to float32 so the arithmetic stays float32 end to end.
    Leaving it float64 would silently promote the whole fold: different in the
    last ulp from the deployed baseline, and twice the resident size.
    """
    p99 = np.asarray(p99, dtype=np.float32).reshape((1,) * (x.ndim - 1) + (-1,))
    return (np.clip(x, -p99, p99) / p99).astype(np.float32)


def preprocess_radar(radar_cir_iq, p99):
    """Decluttered CIR -> model input, (..., 6, 3, 47, 1).

    The canonical deployed transform, and the reason the trailing channel axis is
    added here rather than at the call site: `build_grid_model`'s input is
    (R, 3, 47, C), so a 4-D input is a hard `set_tensor` failure at deploy time.
    """
    return normalize_channels(magnitude(radar_cir_iq)[..., np.newaxis], p99)


def frame_energy(x):
    """Per-frame scalar for the conditional-inference gate (Phase 5.1).

    `x` is the normalized model input, (T, R, 3, 47, C); the return is (T,).
    Reduced over the `mag` channel only, so the statistic means the same thing
    whatever channel set is in front of it.

    Declutter has already removed the static room, so this is "how much moving
    energy is in this frame" -- near zero for an empty room, and the model can be
    skipped entirely. One pass over 846 values against the model's ~330k MACs,
    so a gated frame costs ~0.3% of an inference.

    Mean rather than max: chosen on measured occupied-vs-empty separation, not
    on argument. `src/helpers/calibrate_energy_gate.py` reports both.
    """
    return x[..., 0].reshape(len(x), -1).mean(axis=1)


def format_targets_grid(
    people_xy,
    people_mask,
    room_width=ROOM_WIDTH_M,
    room_height=ROOM_HEIGHT_M,
    grid_w=GRID_W,
    grid_h=GRID_H,
    sigma=0.3,
):
    """(people_xy, people_mask) -> Gaussian occupancy grid, (T, grid_h, grid_w, 1).

    Row = y, column = x (project_spec.md §3: origin bottom-left, x right, y up).
    """
    T = people_xy.shape[0]
    grid = np.zeros((T, grid_h, grid_w), dtype=np.float32)

    cell_w = room_width / grid_w
    cell_h = room_height / grid_h

    x_coords = (np.arange(grid_w) + 0.5) * cell_w
    y_coords = (np.arange(grid_h) + 0.5) * cell_h
    xv, yv = np.meshgrid(x_coords, y_coords)

    for t in range(T):
        for p in range(people_xy.shape[1]):
            if people_mask[t, p]:
                px, py = people_xy[t, p]
                dist_sq = (xv - px) ** 2 + (yv - py) ** 2
                gaussian = np.exp(-dist_sq / (2 * sigma**2))
                grid[t] = np.maximum(grid[t], gaussian)

    return np.expand_dims(grid, axis=-1)


def extract_peaks_from_grid(
    grid,
    threshold=0.2,
    min_distance=0.6,
    room_width=ROOM_WIDTH_M,
    room_height=ROOM_HEIGHT_M,
):
    """Up to 4 peaks from an occupancy grid, in metres, NMS'd by `min_distance`.

    Peaks are refined to sub-cell precision by the centre of mass of their 3x3
    neighbourhood, then suppressed nearest-first so two detections cannot sit
    closer together than one person physically can.
    """
    grid = np.squeeze(grid)
    grid_h, grid_w = grid.shape

    cell_w = room_width / grid_w
    cell_h = room_height / grid_h

    local_max = ndimage.maximum_filter(grid, size=3) == grid
    local_max = local_max & (grid > threshold)

    y_indices, x_indices = np.where(local_max)

    peaks = []
    for y, x in zip(y_indices, x_indices):
        conf = grid[y, x]

        y_min, y_max = max(0, y - 1), min(grid_h, y + 2)
        x_min, x_max = max(0, x - 1), min(grid_w, x + 2)

        window = np.maximum(0, grid[y_min:y_max, x_min:x_max])
        y_grid, x_grid = np.meshgrid(
            np.arange(y_min, y_max), np.arange(x_min, x_max), indexing="ij"
        )

        total_mass = np.sum(window)
        if total_mass > 0:
            y_sub = np.sum(y_grid * window) / total_mass
            x_sub = np.sum(x_grid * window) / total_mass
        else:
            y_sub, x_sub = float(y), float(x)

        peaks.append(
            (float((x_sub + 0.5) * cell_w), float((y_sub + 0.5) * cell_h), conf)
        )

    peaks.sort(key=lambda p: p[2], reverse=True)

    kept = []
    for x_m, y_m, _ in peaks:
        if any(math.hypot(x_m - fx, y_m - fy) < min_distance for fx, fy in kept):
            continue
        kept.append([x_m, y_m])
        if len(kept) >= 4:
            break

    return kept

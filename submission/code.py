"""Inference for multi-person UWB radar localization on an ESP32-S3 TFLite model.

Input:  .npy of shape (T, 6, 3, 120, 2) -- T frames x 6 radars x 3 antennas
        x 120 range bins x [I, Q], sampled at 25 Hz.
Output: .jsonl, one record per frame:
        {"frame": t, "localizations": [[x, y], ...]}  (<= 4, [] if none)

Usage:
    python submission/code.py --input-path <input.npy> --output-path <output.jsonl>

EVERYTHING BELOW THE CONSTANTS IS A VERBATIM COPY of src/preprocessing.py and
src/tracker.py -- this file cannot import from src/, so the two are kept in sync
by hand. Change one, change the other, then run src/helpers/verify_parity.py.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndimage
import scipy.signal
import scipy.spatial.distance as distance
import tensorflow as tf

# Calibration written by the training run that produced model.tflite
# (src/models/best_p99.npy, best_threshold.npy, best_energy_gate.npy). Never
# edit these by hand without re-running verify_parity.py, which asserts they
# match the artifacts.
P99_VALUE = 321.8782666015625
THRESHOLD = 0.5
ENERGY_GATE = 0.018212279677391054

# --- Verbatim from src/preprocessing.py --------------------------------------

CROP_START, CROP_END = 5, 52
FRAME_RATE_HZ = 25.0
ROOM_WIDTH_M, ROOM_HEIGHT_M = 4.8, 7.2


def remove_clutter_streaming(radar_cir_iq, alpha=0.99):
    """Yield one decluttered frame at a time, so the recording is never resident.

    Feeding `lfilter` a single frame per call while carrying its own `zi` is the
    same code path as the batch call in src/, so the output is bit-for-bit
    identical. Hand-rolling `bg = alpha*bg + (1-alpha)*x` is NOT identical --
    the arithmetic order differs and the result drifts by ~4e-2. `zi` is seeded
    from frame 0, not zero, or the filter reads the whole room as one huge fake
    movement for the first few seconds.
    """
    b_ema = [1.0 - alpha]
    a_ema = [1.0, -alpha]

    zi = scipy.signal.lfilter_zi(b_ema, a_ema)
    zi = zi.reshape((1, 1, 1, 1, 1)) * radar_cir_iq[0:1]

    for t in range(radar_cir_iq.shape[0]):
        frame = np.asarray(radar_cir_iq[t : t + 1])
        background, zi = scipy.signal.lfilter(b_ema, a_ema, frame, axis=0, zi=zi)
        yield (frame - background).astype(np.float32)


def magnitude(radar_cir_iq):
    """Cropped |I + jQ|, (..., 6, 3, 47).

    NOT `np.abs` on a complex array: that uses a hypot-style algorithm whose
    result differs in the last bits from `sqrt(re**2 + im**2)`, and the model
    was trained on this form.
    """
    sliced = radar_cir_iq[..., CROP_START:CROP_END, :]
    return np.sqrt(sliced[..., 0] ** 2 + sliced[..., 1] ** 2)


def normalize_channels(x, p99):
    """Scale each channel by its own p99 into [0, 1].

    The divisor is cast to float32 so the arithmetic stays float32 end to end,
    matching training exactly.
    """
    p99 = np.asarray(p99, dtype=np.float32).reshape((1,) * (x.ndim - 1) + (-1,))
    return (np.clip(x, -p99, p99) / p99).astype(np.float32)


def preprocess_radar(radar_cir_iq, p99=P99_VALUE):
    """Decluttered CIR -> model input, (..., 6, 3, 47, 1).

    The trailing channel axis is not optional: the model's input is
    (R, 3, 47, C), so a 4-D array is a hard `set_tensor` failure.
    """
    return normalize_channels(magnitude(radar_cir_iq)[..., np.newaxis], p99)


def frame_energy(x):
    """Per-frame scalar for the conditional-inference gate.

    `x` is the normalized model input, (T, R, 3, 47, C); the return is (T,).
    Declutter has already removed the static room, so this is "how much moving
    energy is in this frame". One pass over 846 values against the model's
    ~330k MACs, so a gated frame costs ~0.3% of an inference.
    """
    return x[..., 0].reshape(len(x), -1).mean(axis=1)


def extract_peaks_from_grid(
    grid,
    threshold=0.2,
    min_distance=0.6,
    room_width=ROOM_WIDTH_M,
    room_height=ROOM_HEIGHT_M,
):
    """Up to 4 peaks from an occupancy grid, in metres, NMS'd by `min_distance`.

    Row = y, column = x (project_spec.md §3: origin bottom-left, x right, y up).
    Peaks are refined by the centre of mass of their 3x3 neighbourhood, then
    suppressed nearest-first so two detections cannot sit closer together than
    one person physically can.
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


# --- Verbatim from src/tracker.py --------------------------------------------

MAX_PEOPLE = 4
PERSIST_TAU_S = 10.0
PERSIST_T_LO = 0.30


class Persistence:
    """Promote grid cells that hold sub-threshold confidence for several seconds.

    A seated subject occupies one grid cell for a whole recording and a phantom
    does not, so integrating over ~10 s recovers evidence a per-frame threshold
    throws away. `update` returns `max(heatmap, smoothed * threshold / t_lo)`:
    the rescale lets one unchanged threshold decode both the bright-now and the
    persistently-dim cases, on a single surface so NMS and the centre-of-mass
    refinement still see a coherent peak.

    Costs one float32 state array of the grid's shape (864 bytes at 18x12) and
    is causal, so it streams.
    """

    def __init__(self, threshold, tau_s=PERSIST_TAU_S, t_lo=PERSIST_T_LO):
        self.decay = math.exp(-1.0 / (FRAME_RATE_HZ * tau_s))
        self.gain = threshold / t_lo
        self.smoothed = None

    def update(self, heatmap):
        heatmap = np.asarray(heatmap, dtype=np.float32)
        if self.smoothed is None:
            # Seeded from frame 0, not zero, so a cell occupied from the start is
            # not suppressed while the filter charges.
            self.smoothed = heatmap.copy()
        else:
            self.smoothed = self.decay * self.smoothed + (1.0 - self.decay) * heatmap
        return np.maximum(heatmap, self.smoothed * self.gain)


class Tracker:
    """Greedy nearest-neighbour association with coasting and position smoothing."""

    def __init__(self, alpha=0.4, max_distance=1.0, max_coast=2):
        self.alpha = alpha
        self.max_distance = max_distance
        self.max_coast = max_coast
        self.tracks = []

    def update(self, detections):
        if not self.tracks:
            self.tracks = [{"pos": d, "coast": 0} for d in detections]
            return [t["pos"] for t in self.tracks][:MAX_PEOPLE]

        if not detections:
            for t in self.tracks:
                t["coast"] += 1
            self.tracks = [t for t in self.tracks if t["coast"] <= self.max_coast]
            # Prefer recently-matched tracks (low coast) when over the cap.
            positions = [
                t["pos"] for t in sorted(self.tracks, key=lambda t: t["coast"])
            ]
            return positions[:MAX_PEOPLE]

        track_pos = np.array([t["pos"] for t in self.tracks])
        det_pos = np.array(detections)
        cost_matrix = distance.cdist(track_pos, det_pos)

        matched_tracks = set()
        matched_dets = set()
        active_positions = []

        while len(matched_tracks) < len(self.tracks) and len(matched_dets) < len(
            detections
        ):
            r, c = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            if cost_matrix[r, c] > self.max_distance:
                break

            old_pos = np.array(self.tracks[r]["pos"])
            new_pos = np.array(detections[c])
            self.tracks[r]["pos"] = (
                self.alpha * new_pos + (1 - self.alpha) * old_pos
            ).tolist()
            self.tracks[r]["coast"] = 0
            active_positions.append(self.tracks[r]["pos"])

            matched_tracks.add(r)
            matched_dets.add(c)

            cost_matrix[r, :] = np.inf
            cost_matrix[:, c] = np.inf

        for r in range(len(self.tracks)):
            if r not in matched_tracks:
                self.tracks[r]["coast"] += 1
                active_positions.append(self.tracks[r]["pos"])

        for c in range(len(detections)):
            if c not in matched_dets:
                self.tracks.append({"pos": detections[c], "coast": 0})
                active_positions.append(detections[c])

        self.tracks = [t for t in self.tracks if t["coast"] <= self.max_coast]

        # Matched tracks were appended first, so the cap keeps them over
        # coasting/new ones. The room holds at most MAX_PEOPLE people.
        return active_positions[:MAX_PEOPLE]


# --- Entry point -------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Inference for multi-person UWB radar localization"
    )
    p.add_argument(
        "--input-path",
        required=True,
        help="Path to input .npy file of shape (T, 6, 3, 120, 2)",
    )
    p.add_argument(
        "--output-path", required=True, help="Path to write output .jsonl file"
    )
    return p.parse_args()


def quantize_input(x, details):
    """Float input -> the interpreter's int8 domain.

    Rounds to nearest like the TFLite converter, then clamps: a bare `astype`
    truncates toward zero and costs 1 LSB of bias.
    """
    scale, zero_point = details["quantization"]
    if scale <= 0:
        return x.astype(details["dtype"])
    info = np.iinfo(details["dtype"])
    return np.clip(np.round(x / scale + zero_point), info.min, info.max).astype(
        details["dtype"]
    )


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # mmap so the raw CIR is paged in frame by frame rather than made resident.
    raw = np.load(input_path, mmap_mode="r")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    interpreter = tf.lite.Interpreter(
        model_path=str(Path(__file__).parent / "model.tflite")
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    output_scale, output_zero_point = output_details["quantization"]

    tracker = Tracker(alpha=0.4, max_distance=1.0, max_coast=2)
    persistence = Persistence(THRESHOLD)
    empty_heatmap = np.zeros(output_details["shape"][1:], dtype=np.float32)

    with open(output_path, "w") as out_f:
        for t, frame in enumerate(remove_clutter_streaming(raw)):
            X = preprocess_radar(frame)

            # Conditional inference: below the calibrated gate nothing is moving,
            # so the model is skipped outright and the all-zero heatmap it would
            # have produced is substituted. Persistence and the tracker still run
            # on that frame, so their state stays causal.
            if frame_energy(X)[0] < ENERGY_GATE:
                y = empty_heatmap
            else:
                interpreter.set_tensor(
                    input_details["index"], quantize_input(X, input_details)
                )
                interpreter.invoke()
                y_quant = interpreter.get_tensor(output_details["index"])
                if output_scale > 0:
                    y = (
                        y_quant.astype(np.float32) - output_zero_point
                    ) * output_scale
                else:
                    y = y_quant.astype(np.float32)
                y = y[0]

            localizations = tracker.update(
                extract_peaks_from_grid(persistence.update(y), threshold=THRESHOLD)
            )

            out_f.write(
                json.dumps({"frame": t, "localizations": localizations}) + "\n"
            )


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import json
import os


def load_jsonl(path):
    frames = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                frames.append(json.loads(line))
    return frames


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    radar_cir_iq_path = os.path.join(project_root, "evaluation/example/input_test.npy")
    gt_path = os.path.join(project_root, "evaluation/example/output_test.jsonl")

    print("Loading data...")
    radar_cir_iq = np.load(radar_cir_iq_path)  # shape: (T, 6, 3, 120, 2)
    frames_data = load_jsonl(gt_path)

    num_frames = len(frames_data)
    num_frames = min(num_frames, radar_cir_iq.shape[0])

    # Ground truth
    max_people = 4
    people_xy = np.zeros((num_frames, max_people, 2), dtype=np.float32)
    people_mask = np.zeros((num_frames, max_people), dtype=bool)

    for t in range(num_frames):
        locs = frames_data[t].get("localizations", [])
        for i, loc in enumerate(locs):
            if i < max_people:
                people_xy[t, i] = loc
                people_mask[t, i] = True

    # Timestamps assuming 25 Hz
    timestamps = np.arange(num_frames) / 25.0

    # Radar positions
    radar_positions = [
        (2.4, 0.0),  # SR250_1
        (4.8, 1.8),  # SR250_2
        (4.8, 5.4),  # SR250_3
        (2.4, 7.2),  # SR250_4
        (0.0, 5.4),  # SR250_5
        (0.0, 1.8),  # SR250_6
    ]

    # Compute complex CIR
    cir_complex = radar_cir_iq[..., 0] + 1j * radar_cir_iq[..., 1]

    RADAR_IDX = 0
    ANTENNA_IDX = 0
    FRAME_LIMIT = min(1200, num_frames)
    ALPHA = 0.8
    BIN_SIZE_M = 0.15
    MAX_RANGE_M = 8.0
    MAGNITUDE_CLIP_MAX = 400
    BIN_OFFSET = 5

    raw_signal = cir_complex[:FRAME_LIMIT, RADAR_IDX, ANTENNA_IDX, :]

    print("Computing decluttered signal...")
    background = raw_signal[0].copy()
    decluttered = np.empty_like(raw_signal)

    for t in range(raw_signal.shape[0]):
        background = ALPHA * background + (1.0 - ALPHA) * raw_signal[t]
        decluttered[t] = raw_signal[t] - background

    radar_pos_xy = np.array(radar_positions[RADAR_IDX], dtype=np.float32)
    people_distance_m = np.full((FRAME_LIMIT, max_people), np.nan, dtype=np.float32)

    for t in range(FRAME_LIMIT):
        valid = people_mask[t]
        if not np.any(valid):
            continue
        deltas = people_xy[t, valid] - radar_pos_xy
        distances = np.linalg.norm(deltas, axis=1)
        people_distance_m[t, np.where(valid)[0]] = distances

    offset_signal = decluttered[:, BIN_OFFSET:]
    offset_magnitude = np.abs(offset_signal)
    offset_magnitude_clipped = np.clip(offset_magnitude, 0, MAGNITUDE_CLIP_MAX)
    offset_range_axis_m = np.arange(offset_signal.shape[1]) * BIN_SIZE_M
    time_axis_plot = timestamps[:FRAME_LIMIT]

    plt.figure(figsize=(14, 5))
    plt.imshow(
        offset_magnitude_clipped.T,
        aspect="auto",
        origin="lower",
        extent=[
            time_axis_plot[0],
            time_axis_plot[-1],
            offset_range_axis_m[0],
            offset_range_axis_m[-1],
        ],
        cmap="viridis",
    )

    for person_idx in range(max_people):
        valid = people_mask[:FRAME_LIMIT, person_idx]
        if np.any(valid):
            plt.plot(
                time_axis_plot[valid],
                people_distance_m[valid, person_idx],
                linewidth=1.5,
                label=f"Person slot {person_idx}",
            )

    plt.colorbar(label=f"Magnitude (clipped to {MAGNITUDE_CLIP_MAX})")
    plt.xlabel("Time [s]")
    plt.ylabel("Aligned range [m]")
    plt.ylim(0, MAX_RANGE_M)
    plt.title(
        f"Decluttered CIR after removing the first {BIN_OFFSET} bins - Radar {RADAR_IDX + 1}"
    )
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_dir = os.path.join(project_root, "report/images")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "radar_signal_plot.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()

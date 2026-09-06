import json
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def load_jsonl(path):
    frames = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                frames.append(json.loads(line))
    return frames


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    gt_path = os.path.join(project_root, "evaluation/example/output_test.jsonl")
    pred_path = os.path.join(project_root, "evaluation/example/my_output.jsonl")

    gt_frames = load_jsonl(gt_path)
    pred_frames = load_jsonl(pred_path)

    # Animate 500 frames (20 seconds at 25 fps) to keep generation fast
    num_frames = min(500, len(gt_frames))

    fig, ax = plt.subplots(figsize=(6, 9))
    ax.set_xlim(-0.5, 5.3)
    ax.set_ylim(-0.5, 7.7)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.plot(
        [0, 4.8, 4.8, 0, 0],
        [0, 0, 7.2, 7.2, 0],
        "k--",
        linewidth=2,
        label="Room bounds",
    )
    ax.grid(True, linestyle=":", alpha=0.7)

    # Active points
    gt_scatter = ax.scatter(
        [],
        [],
        c="blue",
        s=60,
        label="Ground Truth",
        marker="o",
        edgecolors="white",
        zorder=5,
    )
    pred_scatter = ax.scatter(
        [],
        [],
        c="red",
        marker="X",
        s=80,
        label="Predictions",
        edgecolors="white",
        zorder=5,
    )
    ax.legend(loc="upper right")

    # Fading trails
    gt_x_history, gt_y_history = [], []
    pred_x_history, pred_y_history = [], []

    gt_tail = ax.scatter([], [], c="blue", s=15, alpha=0.3, zorder=4)
    pred_tail = ax.scatter([], [], c="red", marker="x", s=25, alpha=0.3, zorder=4)

    def init():
        gt_scatter.set_offsets(np.empty((0, 2)))
        pred_scatter.set_offsets(np.empty((0, 2)))
        gt_tail.set_offsets(np.empty((0, 2)))
        pred_tail.set_offsets(np.empty((0, 2)))
        return gt_scatter, pred_scatter, gt_tail, pred_tail

    def update(frame_idx):
        gt = gt_frames[frame_idx].get("localizations", [])
        pred = pred_frames[frame_idx].get("localizations", [])

        if gt:
            gt_scatter.set_offsets(np.array(gt))
            for loc in gt:
                gt_x_history.append(loc[0])
                gt_y_history.append(loc[1])
        else:
            gt_scatter.set_offsets(np.empty((0, 2)))

        if pred:
            pred_scatter.set_offsets(np.array(pred))
            for loc in pred:
                pred_x_history.append(loc[0])
                pred_y_history.append(loc[1])
        else:
            pred_scatter.set_offsets(np.empty((0, 2)))

        # Keep only last 100 trail points
        tail_len = 100
        if len(gt_x_history) > tail_len:
            del gt_x_history[:-tail_len]
            del gt_y_history[:-tail_len]
        if len(pred_x_history) > tail_len:
            del pred_x_history[:-tail_len]
            del pred_y_history[:-tail_len]

        if gt_x_history:
            gt_tail.set_offsets(np.c_[gt_x_history, gt_y_history])
        else:
            gt_tail.set_offsets(np.empty((0, 2)))

        if pred_x_history:
            pred_tail.set_offsets(np.c_[pred_x_history, pred_y_history])
        else:
            pred_tail.set_offsets(np.empty((0, 2)))

        ax.set_title(f"Tracking Animation - Frame {frame_idx} / {num_frames}")
        return gt_scatter, pred_scatter, gt_tail, pred_tail

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, blit=True
    )

    out_dir = os.path.join(project_root, "report/images")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tracking_animation.gif")

    print(f"Saving animation to {out_path} ... (this takes a moment)")
    writer = animation.PillowWriter(fps=25)
    ani.save(out_path, writer=writer)
    print("Done!")


if __name__ == "__main__":
    main()

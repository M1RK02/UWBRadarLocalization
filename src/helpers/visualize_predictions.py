import json
import matplotlib.pyplot as plt
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
    gt_path = os.path.join(project_root, "evaluation/example/output_test.jsonl")
    pred_path = os.path.join(project_root, "evaluation/example/my_output.jsonl")

    gt_frames = load_jsonl(gt_path)
    if not os.path.exists(pred_path):
        print(f"Prediction file {pred_path} not found. Ensure you run inference first.")
        return

    pred_frames = load_jsonl(pred_path)

    # Collect all points
    gt_x, gt_y = [], []
    for frame in gt_frames:
        for loc in frame.get("localizations", []):
            gt_x.append(loc[0])
            gt_y.append(loc[1])

    pred_x, pred_y = [], []
    for frame in pred_frames:
        for loc in frame.get("localizations", []):
            pred_x.append(loc[0])
            pred_y.append(loc[1])

    plt.figure(figsize=(8, 12))

    # Draw room bounds
    plt.plot(
        [0, 4.8, 4.8, 0, 0],
        [0, 0, 7.2, 7.2, 0],
        "k--",
        linewidth=2,
        label="Room bounds",
    )

    # Plot ground truth
    plt.scatter(gt_x, gt_y, c="blue", s=10, alpha=0.5, label="Ground Truth")

    # Plot predictions
    plt.scatter(
        pred_x, pred_y, c="red", marker="x", s=20, alpha=0.5, label="Predictions"
    )

    plt.xlim(-0.5, 5.3)
    plt.ylim(-0.5, 7.7)
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("Predictions vs Ground Truth Trajectories")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)

    out_dir = os.path.join(project_root, "report/images")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions_vs_gt.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {out_path}")


if __name__ == "__main__":
    main()

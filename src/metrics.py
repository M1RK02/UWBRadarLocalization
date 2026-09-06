"""Scoring primitives shared by training, the sweep tool and the OOF breakdown.

Deliberately numpy-free and TensorFlow-free: `sweep_postproc.py` and the helpers
replay cached heatmaps and must not pull TensorFlow in to do it.
"""


def prf(tp, fp, fn):
    """Precision, recall, F1 from raw detection counts (micro-averaged).

    The graded metric is global micro-averaged F1, so counts are pooled across
    every window BEFORE this is called -- never averaged per window.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1

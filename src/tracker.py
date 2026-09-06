"""Temporal post-processing between the model output and the emitted positions.

Two stages, applied in this order: `Persistence` integrates the heatmap so a
motionless subject survives thresholding, then `Tracker` associates the decoded
peaks across frames.

`submission/code.py` carries a verbatim copy of both (it cannot import from
`src/`). Change them here -> copy them there -> re-run the parity check.
"""

import math

import numpy as np
import scipy.spatial.distance as distance

from src.preprocessing import FRAME_RATE_HZ

# project_spec.md §3: the room holds at most 4 people.
MAX_PEOPLE = 4

# Persistence constants, calibrated on the pooled OOF heatmaps of the 24-window
# CV run. Phase 4.3 has the table: this is a TRADE, not a free win -- it buys
# seated recall and costs precision where nobody is seated.
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

    Costs one float32 state array of the grid's shape (864 bytes at 18x12) and is
    causal, so it streams. See REPORT_NOTES Step 10.
    """

    def __init__(self, threshold, tau_s=PERSIST_TAU_S, t_lo=PERSIST_T_LO):
        self.decay = math.exp(-1.0 / (FRAME_RATE_HZ * tau_s))
        self.gain = threshold / t_lo
        self.smoothed = None

    def update(self, heatmap):
        heatmap = np.asarray(heatmap, dtype=np.float32)
        if self.smoothed is None:
            # Seeded from frame 0, not from zero, so a cell that is occupied from
            # the start is not suppressed while the filter charges. Same reason
            # `remove_clutter` seeds its own EMA that way.
            self.smoothed = heatmap.copy()
        else:
            self.smoothed = self.decay * self.smoothed + (1.0 - self.decay) * heatmap
        return np.maximum(heatmap, self.smoothed * self.gain)


class Tracker:
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
            min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            r, c = min_idx
            if cost_matrix[r, c] > self.max_distance:
                break

            old_pos = np.array(self.tracks[r]["pos"])
            new_pos = np.array(detections[c])
            smoothed_pos = self.alpha * new_pos + (1 - self.alpha) * old_pos
            self.tracks[r]["pos"] = smoothed_pos.tolist()
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
        # coasting/new ones. Room holds at most MAX_PEOPLE people.
        return active_positions[:MAX_PEOPLE]

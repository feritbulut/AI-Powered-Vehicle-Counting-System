"""
Lightweight SORT tracker implementation with an internal Kalman filter (no external deps).
Adapted from the original SORT algorithm by Bewley et al.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.utils.types import Detection, TrackState


def iou(bb_test: Tuple[int, int, int, int], bb_gt: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bb_test
    x1g, y1g, x2g, y2g = bb_gt
    xx1 = max(x1, x1g)
    yy1 = max(y1, y1g)
    xx2 = min(x2, x2g)
    yy2 = min(y2, y2g)
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - inter
    return inter / union if union > 0 else 0.0


def convert_bbox_to_z(bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2.0
    y = y1 + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r], dtype=float).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray) -> Tuple[int, int, int, int]:
    x_c, y_c, s, r = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
    w = np.sqrt(max(0.0, s * r))
    h = max(1e-6, s / (w + 1e-6))
    x1 = x_c - w / 2.0
    y1 = y_c - h / 2.0
    x2 = x_c + w / 2.0
    y2 = y_c + h / 2.0
    return int(x1), int(y1), int(x2), int(y2)


@dataclass
class KalmanBoxTracker:
    """
    Internal Kalman box tracker.
    State: [x, y, s, r, vx, vy, vs]
    """

    id: int
    x: np.ndarray
    P: np.ndarray
    F: np.ndarray
    Q: np.ndarray
    H: np.ndarray
    R: np.ndarray
    hits: int = 0
    time_since_update: int = 0

    @classmethod
    def create(cls, det_box: Tuple[int, int, int, int], kf_id: int) -> "KalmanBoxTracker":
        x_init = np.zeros((7, 1), dtype=float)
        x_init[:4, 0:1] = convert_bbox_to_z(det_box)
        F = np.eye(7, dtype=float)
        for i in range(4, 7):
            F[i - 4, i] = 1.0
        Q = np.eye(7, dtype=float) * 0.01
        R = np.eye(4, dtype=float) * 10.0
        H = np.zeros((4, 7), dtype=float)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        P = np.eye(7, dtype=float) * 10.0
        return cls(id=kf_id, x=x_init, P=P, F=F, Q=Q, H=H, R=R)

    def predict(self) -> Tuple[int, int, int, int]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return convert_x_to_bbox(self.x)

    def update(self, det_box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        z = convert_bbox_to_z(det_box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        return convert_x_to_bbox(self.x)

    @property
    def velocity(self) -> float:
        return float(np.linalg.norm(self.x[4:6, 0]))

    @property
    def centroid(self) -> Tuple[int, int]:
        return int(self.x[0, 0]), int(self.x[1, 0])


class SortTracker:
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 15, min_hits: int = 1):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers: List[KalmanBoxTracker] = []
        self.track_id = 1

    def _associate(self, detections: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(self.trackers) == 0 or len(detections) == 0:
            return [], list(range(len(self.trackers))), list(range(len(detections)))

        iou_matrix = np.zeros((len(self.trackers), len(detections)), dtype=float)
        for t_idx, trk in enumerate(self.trackers):
            predicted_box = trk.predict()  # predict forward
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = iou(predicted_box, det.box)

        matches: List[Tuple[int, int]] = []
        while True:
            t_idx, d_idx = divmod(np.argmax(iou_matrix), iou_matrix.shape[1])
            if iou_matrix[t_idx, d_idx] < self.iou_thresh:
                break
            matches.append((t_idx, d_idx))
            iou_matrix[t_idx, :] = -1
            iou_matrix[:, d_idx] = -1

        unmatched_tracks = [i for i in range(len(self.trackers)) if all(iou_matrix[i, :] != -1)]
        unmatched_dets = [j for j in range(len(detections)) if all(iou_matrix[:, j] != -1)]
        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: List[Detection]) -> List[TrackState]:
        # First, predict all trackers
        predicted_boxes = [trk.predict() for trk in self.trackers]

        matches, unmatched_tracks, unmatched_dets = self._associate(detections)

        # Update matched trackers
        for t_idx, d_idx in matches:
            trk = self.trackers[t_idx]
            trk.update(detections[d_idx].box)

        # Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            trk = KalmanBoxTracker.create(detections[d_idx].box, self.track_id)
            trk.update(detections[d_idx].box)
            self.trackers.append(trk)
            self.track_id += 1

        # Age unmatched trackers and remove stale
        alive_trackers: List[KalmanBoxTracker] = []
        for idx, trk in enumerate(self.trackers):
            if idx in unmatched_tracks:
                trk.time_since_update += 1
            if trk.time_since_update <= self.max_age:
                alive_trackers.append(trk)
        self.trackers = alive_trackers

        # Build TrackState outputs
        track_states: List[TrackState] = []
        for trk in self.trackers:
            if trk.hits < self.min_hits and trk.time_since_update > 0:
                continue
            bbox = convert_x_to_bbox(trk.x)
            cx, cy = trk.centroid
            # prev_centroid approximated by subtracting velocity (1 frame step)
            prev_centroid = (cx - int(trk.x[4, 0]), cy - int(trk.x[5, 0]))
            det = Detection(box=bbox, score=1.0, cls="vehicle")
            track_states.append(
                TrackState(
                    track_id=trk.id,
                    detection=det,
                    centroid=(cx, cy),
                    prev_centroid=prev_centroid,
                    velocity=trk.velocity,
                )
            )
        return track_states

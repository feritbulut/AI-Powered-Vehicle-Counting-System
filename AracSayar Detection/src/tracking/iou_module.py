from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from src.utils.types import Detection, TrackState


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


class _Kalman2D:

    def __init__(self):
        self.x = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4, dtype=float) * 500.0
        self.F = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.R = np.eye(2, dtype=float) * 10.0
        self.Q = np.eye(4, dtype=float) * 0.01

    def init_state(self, x: float, y: float) -> None:
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=float)
        self.P = np.eye(4, dtype=float) * 50.0

    def predict(self) -> Tuple[int, int]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return int(self.x[0, 0]), int(self.x[1, 0])

    def update(self, z_x: float, z_y: float) -> Tuple[int, int]:
        z = np.array([[z_x], [z_y]], dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return int(self.x[0, 0]), int(self.x[1, 0])

    @property
    def velocity(self) -> float:
        return float(np.linalg.norm(self.x[2:4, 0]))


class IOUKalmanTracker:

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 20):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, TrackState] = {}
        self.kalmans: Dict[int, _Kalman2D] = {}
        self.ages: Dict[int, int] = {}
        self.class_votes: Dict[int, Counter] = {}

    def _predict_boxes(self) -> Dict[int, Tuple[int, int, int, int]]:
        predicted_boxes: Dict[int, Tuple[int, int, int, int]] = {}
        for tid, track in self.tracks.items():
            kf = self.kalmans[tid]
            pred_cx, pred_cy = kf.predict()
            x1, y1, x2, y2 = track.detection.box
            w = x2 - x1
            h = y2 - y1
            new_box = (
                int(pred_cx - w / 2),
                int(pred_cy - h / 2),
                int(pred_cx + w / 2),
                int(pred_cy + h / 2),
            )
            predicted_boxes[tid] = new_box
        return predicted_boxes

    def _match(self, track_ids: List[int], detections: List[Detection], predicted_boxes: Dict[int, Tuple[int, int, int, int]]):
        if not track_ids or not detections:
            return [], set(track_ids), set(range(len(detections)))

        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=float)
        for ti, track_id in enumerate(track_ids):
            p_box = predicted_boxes[track_id]
            for di, det in enumerate(detections):
                iou_matrix[ti, di] = _iou(p_box, det.box)

        matches = []
        while True:
            ti, di = divmod(np.argmax(iou_matrix), iou_matrix.shape[1])
            max_iou = iou_matrix[ti, di]
            if max_iou < self.iou_threshold:
                break
            matches.append((track_ids[ti], di))
            iou_matrix[ti, :] = -1
            iou_matrix[:, di] = -1

        matched_tracks = {t for t, _ in matches}
        matched_dets = {d for _, d in matches}
        unmatched_tracks = set(track_ids) - matched_tracks
        unmatched_dets = set(range(len(detections))) - matched_dets
        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: List[Detection]) -> List[TrackState]:
        track_ids = list(self.tracks.keys())
        predicted_boxes = self._predict_boxes()
        matches, unmatched_tracks, unmatched_dets = self._match(track_ids, detections, predicted_boxes)

        for track_id, det_idx in matches:
            det = detections[det_idx]
            track = self.tracks[track_id]
            kf = self.kalmans[track_id]
            prev_centroid = track.centroid
            meas_cx = int((det.box[0] + det.box[2]) / 2)
            meas_cy = int((det.box[1] + det.box[3]) / 2)
            cx, cy = kf.update(meas_cx, meas_cy)
            velocity = kf.velocity

            track.prev_centroid = prev_centroid
            track.centroid = (cx, cy)
            track.detection = det
            track.velocity = velocity

            self.class_votes.setdefault(track_id, Counter()).update([det.cls])
            det.cls = self.class_votes[track_id].most_common(1)[0][0]
            track.detection.cls = det.cls

            self.ages[track_id] = 0

        # Add new tracks
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            cx = int((det.box[0] + det.box[2]) / 2)
            cy = int((det.box[1] + det.box[3]) / 2)
            track = TrackState(
                track_id=self.next_id,
                detection=det,
                centroid=(cx, cy),
                prev_centroid=None,
                velocity=0.0,
            )
            self.tracks[self.next_id] = track
            kf = _Kalman2D()
            kf.init_state(cx, cy)
            self.kalmans[self.next_id] = kf
            self.ages[self.next_id] = 0
            self.class_votes[self.next_id] = Counter([det.cls])
            self.next_id += 1

        # Age unmatched tracks and drop stale
        for track_id in list(unmatched_tracks):
            self.ages[track_id] += 1
            self.kalmans[track_id].predict()
            if self.ages[track_id] > self.max_age:
                self.tracks.pop(track_id, None)
                self.kalmans.pop(track_id, None)
                self.ages.pop(track_id, None)
                self.class_votes.pop(track_id, None)

        return list(self.tracks.values())

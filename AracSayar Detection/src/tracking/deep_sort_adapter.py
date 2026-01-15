import sys
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

from src.utils.types import Detection, TrackState


def _ensure_deep_sort_on_path():
    root = Path(__file__).resolve().parents[2]
    ds_path = root / "deep_sort"
    if ds_path.exists():
        sys.path.append(str(ds_path))


_ensure_deep_sort_on_path()

from deep_sort import nn_matching  # type: ignore  # noqa: E402
from deep_sort.detection import Detection as DSD  # type: ignore  # noqa: E402
from deep_sort.tracker import Tracker as DSTracker  # type: ignore  # noqa: E402


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
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


class DeepSortTracker:
    """
    Wrapper around the local deep_sort repo (Kalman + appearance-aware tracker).
    Uses zero features by default, effectively behaving like SORT+Kalman with IOU gating,
    but can be swapped to real appearance embeddings later.
    """

    def __init__(
        self,
        max_cosine_distance: float = 0.2,
        nn_budget: int = 50,
        max_iou_distance: float = 0.7,
        max_age: int = 30,
        n_init: int = 3,
        feature_dim: int = 128,
    ):
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = DSTracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init
        )
        self.feature_dim = feature_dim
        self.class_votes: dict[int, Counter] = {}
        self.prev_centroids: dict[int, tuple[int, int]] = {}

    def _detections_to_ds(self, detections: List[Detection]) -> List[DSD]:
        ds_dets = []
        # Use a constant non-zero feature vector to avoid NaNs in cosine distance when no embeddings are available.
        base_feature = np.ones((self.feature_dim,), dtype=np.float32)
        base_feature /= np.linalg.norm(base_feature)
        for det in detections:
            x1, y1, x2, y2 = det.box
            tlwh = (x1, y1, x2 - x1, y2 - y1)
            ds_dets.append(DSD(tlwh, det.score, base_feature))
        return ds_dets

    def _update_class_votes(
        self, track_id: int, track_box: tuple[int, int, int, int], detections: List[Detection]
    ) -> str:
        best_cls = None
        best_iou = 0.0
        for det in detections:
            iou_val = _iou(track_box, det.box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_cls = det.cls
        if best_cls is None:
            best_cls = "car"
        self.class_votes.setdefault(track_id, Counter()).update([best_cls])
        return self.class_votes[track_id].most_common(1)[0][0]

    def update(self, detections: List[Detection]) -> List[TrackState]:
        ds_dets = self._detections_to_ds(detections)
        self.tracker.predict()
        self.tracker.update(ds_dets)

        tracks: List[TrackState] = []
        for trk in self.tracker.tracks:
            if not trk.is_confirmed() or trk.time_since_update > 0:
                continue
            tlwh = trk.to_tlwh()
            x1, y1, w, h = tlwh
            bbox = (int(x1), int(y1), int(x1 + w), int(y1 + h))
            cx = int(x1 + w / 2)
            cy = int(y1 + h / 2)
            prev = self.prev_centroids.get(trk.track_id)
            velocity = 0.0 if prev is None else float(np.linalg.norm(np.array([cx, cy]) - np.array(prev)))
            cls_name = self._update_class_votes(trk.track_id, bbox, detections)
            det = Detection(box=bbox, score=1.0, cls=cls_name)
            tracks.append(
                TrackState(
                    track_id=trk.track_id,
                    detection=det,
                    centroid=(cx, cy),
                    prev_centroid=prev,
                    velocity=velocity,
                )
            )
            self.prev_centroids[trk.track_id] = (cx, cy)
        return tracks

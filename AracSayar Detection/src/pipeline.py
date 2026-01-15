import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from src.utils.drawing import draw_tracks
from src.utils.fps_logger import FPSLogger
from src.utils.road import RoadRegionEstimator
from src.utils.types import Detection, TrackState


class Tracker(Protocol):
    def update(self, detections: list[Detection]) -> list[TrackState]: ...


DetectorCallable = Callable[[np.ndarray], list[Detection]]


class Pipeline:
    def __init__(
        self,
        source: str,
        detector: DetectorCallable,
        tracker: Tracker,
        display: bool = True,
        count_mode: str = "single",
        single_line_pos: int | None = None,
        start_line_pos: int | None = None,
        end_line_pos: int | None = None,
        line_offset_pct: float = 0.2,
        line_gap_pct: float = 0.2,
        forced_orientation: str | None = None,
        db_client=None,
    ):
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.display = display
        self.source_is_camera = self._looks_like_camera(source)
        self.cap = self._open_capture()
        self.frame_delay = self._compute_frame_delay()
        self.counter = CountingController(
            mode=count_mode,
            single_line_pos=single_line_pos,
            start_line_pos=start_line_pos,
            end_line_pos=end_line_pos,
            line_offset_pct=line_offset_pct,
            line_gap_pct=line_gap_pct,
            forced_orientation=forced_orientation,
        )
        self.db_client = db_client

        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame = {"frame": None, "seq": -1}
        self.fps_logger = FPSLogger()
        self.det_frame_idx = 0
        self.det_full_interval = 5
        self._prev_track_count = 0

    def _looks_like_camera(self, src: str) -> bool:
        if src.isdigit():
            return True
        try:
            int(src)
            return True
        except ValueError:
            return False

    def _open_capture(self) -> cv2.VideoCapture:
        if self.source_is_camera:
            cam_idx = int(self.source)
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_idx)
        else:
            path = Path(self.source)
            if path.exists():
                cap = cv2.VideoCapture(str(path))
            else:
                cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            raise RuntimeError(f"Unable to open source: {self.source}")
        return cap

    def _compute_frame_delay(self) -> float:
        if self.source_is_camera:
            return 0.0
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if fps <= 1.0 or np.isnan(fps):
            fps = 30.0
        return 1.0 / fps

    def start_reader(self):
        frame_seq = {"value": 0}

        def reader():
            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    if self.source_is_camera:
                        logging.error("Frame grab failed (camera). Stopping pipeline.")
                    else:
                        logging.info(
                            "Video stream ended or unreadable; stopping pipeline."
                        )
                    self.stop_event.set()
                    break
                with self.frame_lock:
                    frame_seq["value"] += 1
                    self.latest_frame["frame"] = frame
                    self.latest_frame["seq"] = frame_seq["value"]
                if self.frame_delay > 0:
                    time.sleep(self.frame_delay)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        return thread

    def run(self):
        reader_thread = self.start_reader()
        last_seq = -1

        try:
            while not self.stop_event.is_set():
                with self.frame_lock:
                    frame = (
                        None
                        if self.latest_frame["frame"] is None
                        else self.latest_frame["frame"].copy()
                    )
                    seq = self.latest_frame["seq"]

                if frame is None or seq == last_seq:
                    time.sleep(0.01)
                    continue
                last_seq = seq

                clean_frame = frame.copy()
                # Update shape early so ROI filters can use it during detection.
                self.counter.frame_shape = frame.shape
                detections = self._detect_with_roi(frame)
                tracks = self.tracker.update(detections)
                # Increase detection frequency when scene is busy to avoid missing fast vehicles.
                tcount = len(tracks)
                if tcount > 8:
                    self.det_full_interval = 2
                elif tcount > 3:
                    self.det_full_interval = 3
                else:
                    self.det_full_interval = 5
                events = self.counter.process(tracks, frame)
                display_tracks = self.counter.filter_tracks_for_display(tracks)
                vis_frame = draw_tracks(
                    frame,
                    display_tracks,
                    self.counter.start_line,
                    self.counter.end_line,
                    self.counter.orientation or "vertical",
                    self.counter.vehicle_counts,
                    None,
                )
                self.fps_logger.tick()

                if self.display:
                    cv2.imshow("Vehicle Detection", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                self._send_events(events, clean_frame)
        finally:
            self.stop_event.set()
            reader_thread.join(timeout=1.0)
            self.cap.release()
            cv2.destroyAllWindows()
            if self.db_client and hasattr(self.db_client, "close"):
                try:
                    self.db_client.close()
                except Exception as exc:  # pragma: no cover - best-effort cleanup
                    logging.warning("Failed to close DB client cleanly: %s", exc)

    def _detect_with_roi(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detector on full frame every Nth call; on intermediate frames,
        restrict to a band between counting lines to reduce noise and jitter.
        """
        self.det_frame_idx = (self.det_frame_idx + 1) % self.det_full_interval
        roi = (
            None if self.det_frame_idx == 0 else self.counter.detection_roi(frame.shape)
        )
        if roi is None:
            return self.counter.filter_detections(self.detector(frame))

        x1, y1, x2, y2 = roi
        if x2 <= x1 or y2 <= y1:
            return self.counter.filter_detections(self.detector(frame))
        crop = frame[y1:y2, x1:x2]
        dets = self.detector(crop)
        # Shift detections back to full-frame coordinates.
        shifted: list[Detection] = []
        for d in dets:
            bx1, by1, bx2, by2 = d.box
            shifted.append(
                Detection(
                    box=(
                        bx1 + x1,
                        by1 + y1,
                        bx2 + x1,
                        by2 + y1,
                    ),
                    score=d.score,
                    cls=d.cls,
                )
            )
        return self.counter.filter_detections(shifted)

    def _send_events(self, events: list[dict], frame: np.ndarray) -> None:
        if not events or self.db_client is None:
            return
        for ev in events:
            x1, y1, x2, y2 = ev["box"]
            crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            timestamp = datetime.now(ZoneInfo("Europe/Istanbul")).isoformat()
            vehicle_id = (
                f"track_{ev['track_id']}"
                if ev.get("track_id") is not None
                else f"event_{ev['event_id']}"
            )
            self.db_client.send(
                vehicle_id=vehicle_id,
                vehicle_type=ev["type"],
                timestamp=timestamp,
                image=crop if crop.size > 0 else None,
            )


class CountingController:
    def __init__(
        self,
        mode: str = "single",
        single_line_pos: int | None = None,
        start_line_pos: int | None = None,
        end_line_pos: int | None = None,
        line_offset_pct: float = 0.2,
        line_gap_pct: float = 0.2,
        forced_orientation: str | None = None,
    ):
        self.mode = mode
        self.single_line_pos = single_line_pos
        self.manual_start = start_line_pos
        self.manual_end = end_line_pos
        self.line_offset_pct = line_offset_pct
        self.line_gap_pct = line_gap_pct
        self.forced_axis: int | None = None
        initial_sign: int | None = None
        if forced_orientation == "vertical":
            self.forced_axis = 0
            self.orientation: str | None = "vertical"
            self.axis: int | None = 0
            initial_sign = 1
        elif forced_orientation == "horizontal":
            self.forced_axis = 1
            self.orientation = "horizontal"
            self.axis = 1
            initial_sign = 1
        else:
            self.axis = None
            self.orientation = None

        self.direction_sign: int | None = initial_sign  # +1 increasing axis, -1 decreasing axis

        self.start_line: int | None = None
        self.end_line: int | None = None
        self.lines_locked = False

        self.counted_ids: set[int] = set()
        self.zone_active: dict[int, bool] = {}
        self.vehicle_counts: dict[str, int] = {}
        self.band_half: int = 5

        self.last_dir_check = time.time()
        self.inconsistency_count = 0
        self.track_states: dict[
            int, bool
        ] = {}  # per-track inside-band state for double mode
        self.wrong_dir_count = 0
        self.frame_counter = 0
        self.motion_samples: list[tuple[float, float]] = []
        self.track_motion: dict[int, list[float]] = {}  # tid -> [dx_sum, dy_sum]
        self.track_last_seen: dict[int, int] = {}
        self.bidirectional = False
        self.last_bidirectional_frame = -1
        self.class_alias = {
            "vehicle": "car",
            "unknown": "car",
            "pickup": "car",
            "van": "car",
            "trailer": "car",
            "suv": "car",
        }
        self.min_motion_samples = 4
        self.min_motion_tracks = 2
        self.min_motion_magnitude = 0.5
        self.max_stale_frames = 30
        self.last_motion_frame = -1
        self.empty_frame_streak = 0
        self.axis_switch_ratio = 1.35
        self.sign_flip_threshold = 0.6
        self.frame_shape: tuple[int, int, int] | None = None
        self.track_labels: dict[int, str] = {}
        self.track_label_pending: dict[int, tuple[str, int]] = {}
        self.flip_cooldown_frames = 80
        self.last_flip_frame = -31
        self.road_estimator = RoadRegionEstimator()
        self.orientation_votes: list[str] = []
        self.orientation_window = 30
        self.orientation_switch_ratio = 0.6
        self.last_orientation_set_frame = -9999
        self.orientation_fixed = self.forced_axis is not None

    def _clamp(self, val: int, dim: int) -> int:
        return max(0, min(dim - 1, val))

    def _fallback_orientation_from_motion(self) -> None:
        if self.orientation_fixed:
            return
        if len(self.motion_samples) < self.min_motion_samples:
            return
        abs_dx = float(np.mean([abs(d[0]) for d in self.motion_samples]))
        abs_dy = float(np.mean([abs(d[1]) for d in self.motion_samples]))
        if abs_dx < self.min_motion_magnitude and abs_dy < self.min_motion_magnitude:
            return
        if abs_dx >= abs_dy:
            ori = "vertical"
            axis = 0
        else:
            ori = "horizontal"
            axis = 1
        self.orientation = ori
        self.axis = axis
        if self.direction_sign is None:
            self.direction_sign = 1
        self.lines_locked = False
        self.orientation_fixed = True
        self.last_orientation_set_frame = self.frame_counter

    def _infer_orientation_from_bg(self, frame: np.ndarray) -> None:
        if self.orientation_fixed:
            return
        try:
            ori, row_band, col_band = self.road_estimator.update(frame)
        except Exception:
            return
        if ori not in ("horizontal", "vertical"):
            return
        self.orientation_votes.append(ori)
        if len(self.orientation_votes) > self.orientation_window:
            self.orientation_votes = self.orientation_votes[-self.orientation_window :]
        counts = {
            "horizontal": self.orientation_votes.count("horizontal"),
            "vertical": self.orientation_votes.count("vertical"),
        }
        total = max(1, len(self.orientation_votes))
        best_ori = (
            "horizontal" if counts["horizontal"] >= counts["vertical"] else "vertical"
        )
        confidence = counts[best_ori] / total
        should_set = confidence >= self.orientation_switch_ratio or (
            self.orientation is None and len(self.orientation_votes) >= self.orientation_window // 2
        )
        if should_set and (
            (best_ori != self.orientation or self.orientation is None)
            and self.frame_counter - self.last_orientation_set_frame
            >= self.flip_cooldown_frames
        ):
            self.orientation = best_ori
            self.axis = 1 if best_ori == "horizontal" else 0
            if self.direction_sign is None:
                self.direction_sign = 1
            self.lines_locked = False
            self.last_orientation_set_frame = self.frame_counter
            self.orientation_fixed = True
            # derive a rough band size from detected motion band
            if best_ori == "horizontal" and row_band is not None:
                self.band_half = max(self.band_half, int((row_band[1] - row_band[0]) / 2))
            if best_ori == "vertical" and col_band is not None:
                self.band_half = max(self.band_half, int((col_band[1] - col_band[0]) / 2))

    def _infer_flow(self, tracks: list[TrackState]) -> None:
        motions = []
        for track in tracks:
            if track.prev_centroid is None:
                continue
            dx = track.centroid[0] - track.prev_centroid[0]
            dy = track.centroid[1] - track.prev_centroid[1]
            if abs(dx) + abs(dy) < self.min_motion_magnitude:
                continue
            motions.append((dx, dy))

        if not motions:
            return
        if self.forced_axis is None and len(motions) < self.min_motion_tracks:
            return

        # When already locked to an axis/sign (non-forced), leave reassessment to the slower hysteresis path.
        if (
            self.forced_axis is None
            and self.axis is not None
            and self.direction_sign is not None
        ):
            return

        avg_dx = float(np.mean([m[0] for m in motions]))
        avg_dy = float(np.mean([m[1] for m in motions]))
        if (
            abs(avg_dx) < self.min_motion_magnitude
            and abs(avg_dy) < self.min_motion_magnitude
        ):
            return  # not enough motion yet

        if self.forced_axis is not None:
            self.axis = self.forced_axis
            if self.forced_axis == 0:
                self.orientation = "vertical"
                self.direction_sign = 1 if avg_dx >= 0 else -1
            else:
                self.orientation = "horizontal"
                self.direction_sign = 1 if avg_dy >= 0 else -1
            return

        if abs(avg_dx) >= abs(avg_dy):
            self.axis = 0
            self.direction_sign = 1 if avg_dx >= 0 else -1
            self.orientation = "vertical"
        else:
            self.axis = 1
            self.direction_sign = 1 if avg_dy >= 0 else -1
            self.orientation = "horizontal"
        if not self.orientation_fixed:
            self.orientation_fixed = True
            self.last_orientation_set_frame = self.frame_counter

    def _auto_lines(self, shape: tuple[int, int, int]) -> None:
        if self.orientation is None:
            return
        height, width = shape[0], shape[1]
        dim = height if self.orientation == "horizontal" else width

        offset_px = int(dim * self.line_offset_pct)
        gap_px = int(dim * self.line_gap_pct)
        self.band_half = max(8, gap_px // 2 if gap_px > 0 else int(dim * 0.05))

        if self.orientation == "horizontal":
            exit_pos = height - 1
            if self.mode == "single":
                line = (
                    self.single_line_pos
                    if self.single_line_pos is not None
                    else exit_pos - offset_px
                )
                self.start_line = self._clamp(int(line), height)
                self.end_line = None
            else:
                end_line = (
                    self.manual_end
                    if self.manual_end is not None
                    else exit_pos - offset_px
                )
                start_line = (
                    self.manual_start
                    if self.manual_start is not None
                    else end_line - gap_px
                )
                self.start_line = self._clamp(int(start_line), height)
                self.end_line = self._clamp(int(end_line), height)
        else:
            exit_pos = width - 1
            if self.mode == "single":
                line = (
                    self.single_line_pos
                    if self.single_line_pos is not None
                    else exit_pos - offset_px
                )
                self.start_line = self._clamp(int(line), width)
                self.end_line = None
            else:
                end_line = (
                    self.manual_end
                    if self.manual_end is not None
                    else exit_pos - offset_px
                )
                start_line = (
                    self.manual_start
                    if self.manual_start is not None
                    else end_line - gap_px
                )
                self.start_line = self._clamp(int(start_line), width)
                self.end_line = self._clamp(int(end_line), width)
        self.lines_locked = True
        logging.info(
            "Sayim cizgisi olusturuldu | mod: %s | yon: %s | start: %s | end: %s",
            self.mode,
            self.orientation or "bilinmiyor",
            self.start_line,
            self.end_line,
        )
        # Reset per-track state when lines move
        self.zone_active.clear()
        self.track_states.clear()

    def _count_single(self, track: TrackState) -> None:
        if self.start_line is None or self.axis is None or self.direction_sign is None:
            return None
        if track.prev_centroid is None or track.track_id in self.counted_ids:
            return None
        coord = track.centroid[self.axis]
        prev_coord = track.prev_centroid[self.axis]

        if self.bidirectional:
            move_sign = self._track_sign(track)
            if move_sign == 0:
                return None
            if move_sign > 0:
                crossed = prev_coord < self.start_line <= coord
            else:
                crossed = prev_coord > self.start_line >= coord
            direction_label = self._direction_label(move_sign)
        else:
            if self.direction_sign > 0:
                crossed = prev_coord < self.start_line <= coord
            else:
                crossed = prev_coord > self.start_line >= coord
            direction_label = self._direction_label()
        if crossed:
            cls_name = self._stable_label(track)
            self.vehicle_counts[cls_name] = self.vehicle_counts.get(cls_name, 0) + 1
            self.counted_ids.add(track.track_id)
            return {
                "event_id": self._next_event_id(),
                "track_id": track.track_id,
                "type": cls_name,
                "direction": direction_label,
                "box": track.detection.box,
            }
        return None

    def _count_double(self, track: TrackState) -> None:
        if (
            self.start_line is None
            or self.end_line is None
            or self.axis is None
            or self.direction_sign is None
        ):
            return None
        if track.prev_centroid is None:
            return None

        if not self._is_in_band(track):
            self.track_states[track.track_id] = False
            return None

        coord = track.centroid[self.axis]
        prev_coord = track.prev_centroid[self.axis]
        tid = track.track_id
        in_zone = self.track_states.get(tid, False)
        move = coord - prev_coord

        low_line = min(self.start_line, self.end_line)
        high_line = max(self.start_line, self.end_line)
        if self.bidirectional:
            move_sign = self._track_sign(track)
            if move_sign == 0:
                self.track_states[tid] = in_zone
                return None
        else:
            move_sign = self.direction_sign or 0
            # Require movement along the inferred direction to avoid noise/backtracking
            if move * move_sign <= -2.0:
                return

        entry_line = low_line if move_sign > 0 else high_line
        exit_line = high_line if move_sign > 0 else low_line
        # Enter zone when crossing the entry line or when inside band.
        if (prev_coord - entry_line) * (coord - entry_line) <= 0 or (
            low_line <= coord <= high_line
        ):
            in_zone = True

        spanned_in_one_step = False
        if (move_sign > 0 and prev_coord <= entry_line and coord >= exit_line) or (
            move_sign < 0 and prev_coord >= entry_line and coord <= exit_line
        ):
            spanned_in_one_step = True

        if (
            (in_zone or spanned_in_one_step)
            and (prev_coord - exit_line) * (coord - exit_line) <= 0
            and tid not in self.counted_ids
        ):
            cls_name = self._stable_label(track)
            self.vehicle_counts[cls_name] = self.vehicle_counts.get(cls_name, 0) + 1
            self.counted_ids.add(tid)
            in_zone = False
            self.track_states[tid] = in_zone
            return {
                "event_id": self._next_event_id(),
                "track_id": tid,
                "type": cls_name,
                "direction": self._direction_label(
                    move_sign if self.bidirectional else None
                ),
                "box": track.detection.box,
            }

        # Reset when well past exit to allow recount if track re-enters.
        margin = max(5, self.band_half)
        if move_sign > 0 and coord > exit_line + margin:
            in_zone = False
        if move_sign < 0 and coord < exit_line - margin:
            in_zone = False

        self.track_states[tid] = in_zone
        return None

    def process(self, tracks: list[TrackState], frame: np.ndarray) -> list[dict]:
        if self.orientation is None:
            self._infer_orientation_from_bg(frame)
            self._fallback_orientation_from_motion()
        if self.axis is None and self.orientation is not None:
            self.axis = 1 if self.orientation == "horizontal" else 0
        if self.axis is None or self.direction_sign is None:
            self._infer_flow(tracks)
        else:
            # keep allowing flow updates to flip if traffic changes drastically
            self._infer_flow(tracks)

        if not self.lines_locked:
            self._auto_lines(frame.shape)

        self.frame_shape = frame.shape
        self.frame_counter += 1
        if tracks:
            self.empty_frame_streak = 0
        else:
            self.empty_frame_streak += 1
        self._collect_motion(tracks)

        events: list[dict] = []
        for track in tracks:
            if self.mode == "single":
                ev = self._count_single(track)
            else:
                ev = self._count_double(track)
            if ev:
                events.append(ev)

        if self.frame_counter % 10 == 0:
            self._reassess_direction(frame.shape)
        return events

    def _is_in_band(self, track: TrackState, margin: int = 2) -> bool:
        if self.axis is None:
            return True
        coord = track.centroid[self.axis]
        if self.axis == 0:
            box_min, box_max = track.detection.box[0], track.detection.box[2]
        else:
            box_min, box_max = track.detection.box[1], track.detection.box[3]
        if self.mode == "single":
            if self.start_line is None:
                return True
            box_span = box_max - box_min
            dynamic_margin = max(self.band_half, margin, int(box_span * 0.6))
            low = self.start_line - dynamic_margin
            high = self.start_line + dynamic_margin
            return (low <= coord <= high) or (box_max >= low and box_min <= high)
        if self.start_line is None or self.end_line is None:
            return True
        band_width = max(
            12,
            int(abs(self.end_line - self.start_line) * 0.7),
            int((box_max - box_min) * 1.2),
            self.band_half * 2,
        )
        low = min(self.start_line, self.end_line) - max(band_width, margin)
        high = max(self.start_line, self.end_line) + max(band_width, margin)
        return (low <= coord <= high) or (box_max >= low and box_min <= high)

    def _accept_detection(self, det: Detection) -> bool:
        if not self.lines_locked or self.axis is None or self.start_line is None:
            return True
        x1, y1, x2, y2 = det.box
        if self.axis == 0:
            coord = (x1 + x2) // 2
            box_min, box_max = x1, x2
        else:
            coord = (y1 + y2) // 2
            box_min, box_max = y1, y2
        if self.mode == "single":
            margin = max(self.band_half, int((box_max - box_min) * 0.6))
            low = self.start_line - margin
            high = self.start_line + margin
            return (low <= coord <= high) or (box_max >= low and box_min <= high)
        if self.end_line is None:
            return True
        band_width = max(
            12,
            int(abs(self.end_line - self.start_line) * 0.7),
            int((box_max - box_min) * 1.2),
            self.band_half * 2,
        )
        low = min(self.start_line, self.end_line) - band_width
        high = max(self.start_line, self.end_line) + band_width
        return (low <= coord <= high) or (box_max >= low and box_min <= high)

    def filter_detections(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return detections
        filtered = [d for d in detections if self._accept_detection(d)]
        if (
            not self.lines_locked
            or self.axis is None
            or self.start_line is None
            or self.frame_shape is None
        ):
            return filtered
        roi = self.detection_roi(self.frame_shape)
        if roi is None:
            return filtered
        rx1, ry1, rx2, ry2 = roi
        margin = 4
        strict: list[Detection] = []
        for d in filtered:
            x1, y1, x2, y2 = d.box
            if x2 < rx1 - margin or x1 > rx2 + margin:
                continue
            if y2 < ry1 - margin or y1 > ry2 + margin:
                continue
            strict.append(d)
        return strict

    def filter_tracks_for_display(self, tracks: list[TrackState]) -> list[TrackState]:
        if self.lines_locked and self.axis is not None:
            return [t for t in tracks if self._is_in_band(t, margin=3)]
        return tracks

    def _next_event_id(self) -> int:
        if not hasattr(self, "_event_counter"):
            self._event_counter = 1
        eid = self._event_counter
        self._event_counter += 1
        return eid

    def _direction_label(self, sign_override: int | None = None) -> str:
        sign = (
            self.direction_sign
            if sign_override is None
            else (1 if sign_override > 0 else -1)
        )
        if self.axis is None or sign is None:
            return "unknown"
        if self.axis == 0:
            return "right" if sign > 0 else "left"
        return "down" if sign > 0 else "up"

    def _normalize_cls(self, cls_name: str, box: tuple[int, int, int, int]) -> str:
        name = cls_name.lower()
        name = self.class_alias.get(name, name or "car")
        if self.frame_shape is None:
            return name

        h, w = self.frame_shape[0], self.frame_shape[1]
        x1, y1, x2, y2 = box
        area = max(1, (x2 - x1) * (y2 - y1))
        frame_area = max(1, h * w)
        area_ratio = area / frame_area
        height_ratio = (y2 - y1) / max(1, h)

        # Upgrade very large vehicles that YOLO labels as car/van/pickup.
        if name in {"car", "vehicle", "pickup", "van", "unknown", "trailer", "suv"}:
            if area_ratio > 0.10 or height_ratio > 0.45:
                name = "bus"
            elif area_ratio > 0.05 or height_ratio > 0.30:
                name = "truck"
        elif name == "truck" and (area_ratio > 0.12 or height_ratio > 0.5):
            name = "bus"

        return name

    def _stable_label(self, track: TrackState) -> str:
        tid = track.track_id
        normalized = self._normalize_cls(track.detection.cls, track.detection.box)
        prev = self.track_labels.get(tid)
        if prev is None:
            self.track_labels[tid] = normalized
            self.track_label_pending.pop(tid, None)
            return normalized
        if normalized == prev:
            self.track_label_pending.pop(tid, None)
            return normalized
        pending_label, pending_count = self.track_label_pending.get(
            tid, (normalized, 0)
        )
        if normalized != pending_label:
            pending_label, pending_count = normalized, 0
        pending_count += 1
        if pending_count >= 2:
            self.track_labels[tid] = normalized
            self.track_label_pending.pop(tid, None)
            return normalized
        self.track_label_pending[tid] = (pending_label, pending_count)
        return prev

    def _collect_motion(self, tracks: list[TrackState]) -> None:
        saw_motion = False
        for track in tracks:
            if track.prev_centroid is None:
                continue
            dx = track.centroid[0] - track.prev_centroid[0]
            dy = track.centroid[1] - track.prev_centroid[1]
            if abs(dx) + abs(dy) < 0.1:
                continue
            saw_motion = True
            self.last_motion_frame = self.frame_counter
            self.motion_samples.append((dx, dy))
            # accumulate per-track displacement to stabilize direction sign
            acc = self.track_motion.setdefault(track.track_id, [0.0, 0.0])
            acc[0] += dx
            acc[1] += dy
            self.track_last_seen[track.track_id] = self.frame_counter
        # keep a reasonable window
        if len(self.motion_samples) > 200:
            self.motion_samples = self.motion_samples[-200:]
        if len(self.track_motion) > 500:
            # drop oldest entries (arbitrary) to bound memory
            for tid in list(self.track_motion.keys())[:100]:
                self.track_motion.pop(tid, None)
                self.track_last_seen.pop(tid, None)
        stale_cutoff = self.frame_counter - self.max_stale_frames
        for tid, last_seen in list(self.track_last_seen.items()):
            if last_seen < stale_cutoff:
                self.track_last_seen.pop(tid, None)
                self.track_motion.pop(tid, None)
                self.track_labels.pop(tid, None)
                self.track_label_pending.pop(tid, None)
        if (
            not saw_motion
            and self.last_motion_frame >= 0
            and (self.frame_counter - self.last_motion_frame) > self.max_stale_frames
        ):
            # If we have gone too long without motion, discard stale direction hints.
            self.motion_samples.clear()
            self.track_motion.clear()
            self.track_last_seen.clear()

    def _reassess_direction(self, frame_shape: tuple[int, int, int]) -> None:
        if (
            self.empty_frame_streak > self.max_stale_frames
            or not self._has_recent_motion()
        ):
            self.motion_samples.clear()
            return
        estimate = self._compute_flow_direction()
        if estimate is None:
            return
        axis, sign, orientation = estimate
        now = self.frame_counter
        if self.orientation_fixed and self.axis is not None:
            axis = self.axis
            orientation = self.orientation or orientation

        if self.axis is None or self.direction_sign is None:
            self.axis = axis
            self.direction_sign = sign
            self.orientation = orientation
            self.lines_locked = False
            self._auto_lines(frame_shape)
            self.inconsistency_count = 0
            self.motion_samples.clear()
            self.last_flip_frame = now
            return

        if axis != self.axis or sign != self.direction_sign:
            self.inconsistency_count += 1
            if self.inconsistency_count >= 3:
                if now - self.last_flip_frame < self.flip_cooldown_frames:
                    self.inconsistency_count = 0
                    self.motion_samples.clear()
                    return
                logging.warning(
                    "Yön tutarsızlığı tespit edildi, yön %s olarak güncellendi.",
                    orientation,
                )
                self.axis = self.axis if self.orientation_fixed else axis
                self.direction_sign = sign
                self.orientation = self.orientation if self.orientation_fixed else orientation
                self.lines_locked = False
                self._auto_lines(frame_shape)
                self.inconsistency_count = 0
                self.motion_samples.clear()
                self.last_flip_frame = now
        else:
            self.inconsistency_count = 0
            self.motion_samples.clear()

    def _vote_sign(
        self, _axis: int, axis_vals: list[float], cum_vals: list[float]
    ) -> tuple[int, float]:
        """
        Returns (sign, confidence) using cumulative per-track displacement when available.
        Confidence is the normalized difference between positive and negative magnitude.
        """
        values = [v for v in cum_vals if abs(v) > 0.8]
        if not values:
            values = [v for v in axis_vals if abs(v) > 0.2]
        if not values:
            return 0, 0.0
        pos = sum(v for v in values if v > 0)
        neg = -sum(v for v in values if v < 0)
        total = pos + neg
        if total < self.min_motion_magnitude:
            return 0, 0.0
        score = pos - neg
        confidence = abs(score) / max(total, 1e-6)
        sign = 1 if score >= 0 else -1
        return sign, confidence

    def _compute_flow_direction(self) -> tuple[int, int, str] | None:
        if not self._has_recent_motion():
            return None
        if len(self.motion_samples) < self.min_motion_samples:
            return None
        # Prefer per-track cumulative displacement for sign stability.
        dx_vals = [d[0] for d in self.motion_samples]
        dy_vals = [d[1] for d in self.motion_samples]
        if not dx_vals and not dy_vals:
            return None

        abs_dx = float(np.mean([abs(v) for v in dx_vals])) if dx_vals else 0.0
        abs_dy = float(np.mean([abs(v) for v in dy_vals])) if dy_vals else 0.0
        if abs_dx < self.min_motion_magnitude and abs_dy < self.min_motion_magnitude:
            return None
        # Hysteresis and lane-change robustness: score axes by projection minus
        # perpendicular energy; stick to current axis unless clearly weaker.
        score_x = abs_dx - 0.5 * abs_dy
        score_y = abs_dy - 0.5 * abs_dx
        dominant_axis = 0 if score_x >= score_y else 1
        prefer_axis = None
        if self.axis is not None:
            cur_score = score_x if self.axis == 0 else score_y
            other_score = score_y if self.axis == 0 else score_x
            if cur_score >= other_score / self.axis_switch_ratio:
                prefer_axis = self.axis
            elif dominant_axis != self.axis and other_score > cur_score:
                prefer_axis = dominant_axis

        # Use cumulative displacement when available to decide sign.
        cum_dx = [v[0] for v in self.track_motion.values()]
        cum_dy = [v[1] for v in self.track_motion.values()]

        if self.forced_axis is not None:
            axis = self.forced_axis
            axis_vals = dx_vals if axis == 0 else dy_vals
            cum_vals = cum_dx if axis == 0 else cum_dy
            if not axis_vals and not cum_vals:
                return None
            weighted = [val * abs(val) for val in cum_vals if abs(val) > 1.0]
            if weighted:
                sign = 1 if sum(weighted) >= 0 else -1
            else:
                sign = 1 if sum(axis_vals) >= 0 else -1
            orientation = "vertical" if axis == 0 else "horizontal"
            self.bidirectional = False
            return axis, sign, orientation

        # Detect bidirectional flow along dominant axis using both sample ratios and per-track cumulative signs.
        axis_displacements = dx_vals if dominant_axis == 0 else dy_vals
        pos = [v for v in axis_displacements if v > 0.5]
        neg = [v for v in axis_displacements if v < -0.5]
        pos_ratio = len(pos) / max(1, len(axis_displacements))
        neg_ratio = len(neg) / max(1, len(axis_displacements))
        track_signs = [v[dominant_axis] for v in self.track_motion.values()]
        pos_tracks = len([v for v in track_signs if v > 1.0])
        neg_tracks = len([v for v in track_signs if v < -1.0])
        bidir_detected = (pos_ratio > 0.1 and neg_ratio > 0.1) or (
            pos_tracks >= 1 and neg_tracks >= 1
        )
        if bidir_detected:
            self.bidirectional = True
            self.last_bidirectional_frame = self.frame_counter
        elif (
            self.bidirectional
            and self.last_bidirectional_frame >= 0
            and self.frame_counter - self.last_bidirectional_frame > 60
        ):
            self.bidirectional = False

        axis = prefer_axis if prefer_axis is not None else dominant_axis
        if self.orientation_fixed and self.axis is not None:
            axis = self.axis
        if axis == 0:
            axis_vals = dx_vals
            cum_vals = cum_dx
            orientation = "vertical"
        else:
            axis_vals = dy_vals
            cum_vals = cum_dy
            orientation = "horizontal"
        sign, confidence = self._vote_sign(axis, axis_vals, cum_vals)
        if sign == 0:
            if self.direction_sign is not None and self.axis == axis:
                sign = self.direction_sign
            else:
                return None
        if (
            self.direction_sign is not None
            and self.axis == axis
            and sign != self.direction_sign
            and confidence < self.sign_flip_threshold
        ):
            sign = self.direction_sign
        return axis, sign, orientation

    def _has_recent_motion(self) -> bool:
        return (
            self.last_motion_frame >= 0
            and (self.frame_counter - self.last_motion_frame) <= self.max_stale_frames
        )

    def _track_sign(self, track: TrackState) -> int:
        """Estimate per-track travel direction along current axis."""
        if self.axis is None:
            return 0
        tid = track.track_id
        if tid in self.track_motion:
            acc = self.track_motion[tid]
            move = acc[self.axis]
            if abs(move) > 0.4:
                return 1 if move > 0 else -1
        if track.prev_centroid is None:
            return 0
        step = track.centroid[self.axis] - track.prev_centroid[self.axis]
        if abs(step) < 0.2:
            return 0
        return 1 if step > 0 else -1

    def detection_roi(
        self, shape: tuple[int, int, int]
    ) -> tuple[int, int, int, int] | None:
        """
        Return an ROI between counting lines to constrain detection.
        Coordinates: (x1, y1, x2, y2).
        """
        if (
            self.axis is None
            or self.start_line is None
            or not self.lines_locked
            or self.orientation is None
        ):
            return None
        height, width = shape[0], shape[1]
        band_margin = int(max(12, self.band_half * 2))
        max_margin_w = int(width * 0.25)
        max_margin_h = int(height * 0.25)
        if self.orientation == "vertical":
            # Band around vertical lines -> x range limited, full height.
            if self.mode == "single":
                margin = min(
                    max_margin_w,
                    int(band_margin * (1.4 if self.bidirectional else 1.0)),
                )
                low = max(0, self.start_line - margin)
                high = min(width, self.start_line + margin)
            else:
                if self.end_line is None:
                    return None
                margin = min(
                    max_margin_w,
                    int(band_margin * (1.4 if self.bidirectional else 1.0)),
                )
                low = max(
                    0,
                    min(self.start_line, self.end_line) - margin,
                )
                high = min(
                    width,
                    max(self.start_line, self.end_line) + margin,
                )
            return (low, 0, high, height)
        else:
            # Horizontal orientation: y range limited, full width.
            if self.mode == "single":
                margin = min(
                    max_margin_h,
                    int(band_margin * (1.4 if self.bidirectional else 1.0)),
                )
                low = max(0, self.start_line - margin)
                high = min(height, self.start_line + margin)
            else:
                if self.end_line is None:
                    return None
                margin = min(
                    max_margin_h,
                    int(band_margin * (1.4 if self.bidirectional else 1.0)),
                )
                low = max(
                    0,
                    min(self.start_line, self.end_line) - margin,
                )
                high = min(
                    height,
                    max(self.start_line, self.end_line) + margin,
                )
            return (0, low, width, high)

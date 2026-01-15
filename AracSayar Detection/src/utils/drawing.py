import cv2
import numpy as np

from src.utils.types import TrackState


def draw_tracks(
    frame: np.ndarray,
    tracks: list[TrackState],
    start_line: int | None,
    end_line: int | None,
    orientation: str,
    counts: dict[str, int],
    road_box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    x1 = y1 = x2 = y2 = None
    if road_box is not None:
        x1, y1, x2, y2 = road_box
        if orientation == "vertical":
            # Draw only vertical road boundaries.
            cv2.line(frame, (x1, 0), (x1, frame.shape[0]), (0, 255, 255), 2)
            cv2.line(frame, (x2, 0), (x2, frame.shape[0]), (0, 255, 255), 2)
        else:
            # Draw only horizontal road boundaries.
            cv2.line(frame, (0, y1), (frame.shape[1], y1), (0, 255, 255), 2)
            cv2.line(frame, (0, y2), (frame.shape[1], y2), (0, 255, 255), 2)

    # Counting lines (limited to road width/height when known)
    if orientation == "vertical":
        # Vehicles move left-right -> counting lines are vertical (x axis)
        if start_line is not None:
            cv2.line(
                frame,
                (start_line, 0 if y1 is None else y1),
                (start_line, frame.shape[0] if y2 is None else y2),
                (0, 200, 255),
                2,
            )
        if end_line is not None:
            cv2.line(
                frame,
                (end_line, 0 if y1 is None else y1),
                (end_line, frame.shape[0] if y2 is None else y2),
                (0, 200, 255),
                2,
            )
    else:
        # Vehicles move top-bottom -> counting lines are horizontal (y axis)
        if start_line is not None:
            cv2.line(
                frame,
                (0 if x1 is None else x1, start_line),
                (frame.shape[1] if x2 is None else x2, start_line),
                (0, 200, 255),
                2,
            )
        if end_line is not None:
            cv2.line(
                frame,
                (0 if x1 is None else x1, end_line),
                (frame.shape[1] if x2 is None else x2, end_line),
                (0, 200, 255),
                2,
            )

    # Tracks
    for track in tracks:
        det = track.detection
        x1, y1, x2, y2 = det.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.cls} {det.score:.2f} ID:{track.track_id}"
        cv2.putText(
            frame,
            label,
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Counts summary
    y0 = 20
    for cls_name, cnt in sorted(counts.items()):
        cv2.putText(
            frame,
            f"{cls_name}: {cnt}",
            (10, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y0 += 22

    return frame

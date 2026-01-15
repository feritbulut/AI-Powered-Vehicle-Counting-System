from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float
    cls: str


@dataclass
class TrackState:
    track_id: int
    detection: Detection
    centroid: Tuple[int, int]
    prev_centroid: Tuple[int, int] | None
    velocity: float
    crossed_start: bool = False
    crossed_end: bool = False
    counted: bool = False
    in_area: bool = False

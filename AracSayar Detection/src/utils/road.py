import cv2
import numpy as np


class RoadRegionEstimator:
    """
    Estimates a dominant motion band and orientation (horizontal/vertical).
    Uses background subtraction + projections on rows and columns with smoothing.
    """

    def __init__(self, smooth: float = 0.7, min_band_ratio: float = 0.15):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.smooth = smooth
        self.min_band_ratio = min_band_ratio
        self.row_band: tuple[int, int] | None = None
        self.col_band: tuple[int, int] | None = None
        self.orientation: str = "horizontal"

    def update(self, frame: np.ndarray) -> tuple[str, tuple[int, int], tuple[int, int]]:
        mask = self.bg.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        row_energy = mask.sum(axis=1)
        col_energy = mask.sum(axis=0)

        row_band_raw = self._band_from_energy(row_energy, frame.shape[0])
        col_band_raw = self._band_from_energy(col_energy, frame.shape[1])

        # Smooth bands independently to keep both vertical and horizontal limits stable.
        self.row_band = self._smooth_band(self.row_band, row_band_raw, frame.shape[0], axis="row")
        self.col_band = self._smooth_band(self.col_band, col_band_raw, frame.shape[1], axis="col")

        row_span = self.row_band[1] - self.row_band[0] if self.row_band else 0
        col_span = self.col_band[1] - self.col_band[0] if self.col_band else 0
        row_norm = row_span / max(1, frame.shape[0])
        col_norm = col_span / max(1, frame.shape[1])

        # Orientation hysteresis to reduce flipping
        if self.orientation == "vertical":
            if row_norm > col_norm * 1.15:
                self.orientation = "horizontal"
        else:
            if col_norm > row_norm * 1.15:
                self.orientation = "vertical"
        return self.orientation, (self.row_band or (0, frame.shape[0] - 1)), (self.col_band or (0, frame.shape[1] - 1))

    def _smooth_band(
        self,
        prev_band: tuple[int, int] | None,
        new_band: tuple[int, int] | None,
        length: int,
        axis: str,
    ) -> tuple[int, int] | None:
        if new_band is None:
            return prev_band
        if prev_band is None:
            return new_band
        b1 = int(self.smooth * prev_band[0] + (1 - self.smooth) * new_band[0])
        b2 = int(self.smooth * prev_band[1] + (1 - self.smooth) * new_band[1])
        b1 = max(0, min(length - 1, b1))
        b2 = max(b1 + 1, min(length - 1, b2))
        return b1, b2

    def _band_from_energy(self, energy: np.ndarray, length: int) -> tuple[int, int] | None:
        if energy.max() == 0:
            return None
        threshold = 0.2 * float(energy.max())
        idx = np.where(energy > threshold)[0]
        if idx.size == 0:
            return None
        start, end = int(idx[0]), int(idx[-1])
        min_band = int(length * self.min_band_ratio)
        if end - start < min_band:
            center = (start + end) // 2
            start = max(0, center - min_band // 2)
            end = min(length - 1, start + min_band)
        return start, end

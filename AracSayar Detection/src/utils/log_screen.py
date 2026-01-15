import cv2
import numpy as np


class LogScreen:
    """
    Lightweight side log window for counted vehicles.
    """

    def __init__(self, width: int = 600, height: int = 720, max_rows: int = 7):
        self.width = width
        self.height = height
        self.background = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.logs: list[dict] = []
        self.max_rows = max_rows

    def add(self, entry_id: int, vehicle_type: str, thumb: np.ndarray, timestamp: str, direction: str) -> None:
        color = (0, 255, 0) if direction == "Gelen" else (0, 0, 255)
        self.logs.insert(
            0,
            {
                "id": entry_id,
                "type": vehicle_type,
                "img": thumb,
                "time": timestamp,
                "color": color,
                "dir": direction,
            },
        )
        if len(self.logs) > self.max_rows:
            self.logs.pop()

    def show(self) -> None:
        canvas = self.background.copy()
        cv2.rectangle(canvas, (0, 0), (self.width, 80), (30, 30, 30), -1)
        cv2.putText(canvas, "ARAC TAKIP LOGU", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        y_pos = 120
        for log in self.logs:
            cv2.putText(canvas, f"#{log['id']}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, log["color"], 2)
            cv2.putText(canvas, f"{log['type']}", (100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            cv2.putText(canvas, f"{log['dir']}", (100, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, log["color"], 1)

            thumb = log["img"]
            h, w = thumb.shape[:2]
            if y_pos - 40 + h < self.height:
                canvas[y_pos - 40 : y_pos - 40 + h, 350 : 350 + w] = thumb

            cv2.putText(canvas, log["time"], (510, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            cv2.line(canvas, (0, y_pos + 50), (self.width, y_pos + 50), (50, 50, 50), 1)
            y_pos += 100

        cv2.imshow("Log Ekrani", canvas)

import base64
import logging
import queue
import threading
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

DEFAULT_DB_ENDPOINT = "https://ai-vehicle-detection-backend-api.onrender.com/api/detect"


class DatabaseClient:
    def __init__(self, endpoint: str, timeout: float = 5.0):
        self.endpoint = endpoint
        self.timeout = timeout
        self.last_warn_code = None
        self.warn_count = 0
        self._queue: queue.Queue | None = None
        self._worker: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        try:
            import requests  # type: ignore
        except ImportError:
            self.requests = None
            logging.error(
                "`requests` is required for database posting; install it to enable uploads."
            )
        else:
            self.requests = requests
            self._queue = queue.Queue(maxsize=128)
            self._stop_event = threading.Event()
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()
        if "3306" in endpoint and endpoint.startswith("http"):
            logging.warning(
                "DB endpoint %s looks like a MySQL port. Ensure this points to an HTTP endpoint (e.g., http://localhost/your_api.php).",
                endpoint,
            )

    def send(
        self,
        vehicle_id: str,
        vehicle_type: str,
        timestamp: Optional[str],
        image: Optional[np.ndarray] = None,
    ) -> None:
        if self.requests is None or self._queue is None:
            return

        payload_time = timestamp or datetime.now(timezone.utc).isoformat()
        try:
            self._queue.put_nowait((vehicle_id, vehicle_type, payload_time, image))
        except queue.Full:
            if self.warn_count % 10 == 0:
                logging.warning("DB queue is full; dropping event %s to keep loop responsive.", vehicle_id)
            self.warn_count += 1

    def _worker_loop(self) -> None:
        if self._queue is None or self.requests is None or self._stop_event is None:
            return
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                self._queue.task_done()
                break
            vehicle_id, vehicle_type, payload_time, image = item
            try:
                self._post_payload(vehicle_id, vehicle_type, payload_time, image)
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning("DB worker exception: %s", exc)
            finally:
                self._queue.task_done()

    def _post_payload(
        self,
        vehicle_id: str,
        vehicle_type: str,
        payload_time: str,
        image: Optional[np.ndarray],
    ) -> None:
        img_str = ""
        if image is not None and image.size > 0:
            ok, buf = cv2.imencode(".jpg", image)
            if ok:
                img_str = base64.b64encode(buf.tobytes()).decode("ascii")

        payload = {
            "id": str(vehicle_id),
            "vehicle_type": vehicle_type,
            "vehicle_img": img_str,
            "time": payload_time,
        }
        try:
            resp = self.requests.post(self.endpoint, json=payload, timeout=self.timeout)
            if resp.status_code >= 400:
                if resp.status_code != self.last_warn_code or self.warn_count % 5 == 0:
                    logging.warning(
                        "DB post failed (%s): %s", resp.status_code, resp.text[:200]
                    )
                self.last_warn_code = resp.status_code
                self.warn_count += 1
        except Exception as exc:
            logging.warning("DB post exception: %s", exc)

    def close(self, timeout: float = 1.0) -> None:
        if self._queue is None or self._stop_event is None:
            return
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=timeout)

import time
from collections import deque
from typing import Deque


class FPSLogger:
    def __init__(self, window: int = 60, log_interval: float = 2.0):
        self.times: Deque[float] = deque(maxlen=window)
        self.last_log = time.time()
        self.log_interval = log_interval

    def tick(self):
        now = time.time()
        self.times.append(now)
        if len(self.times) > 1 and now - self.last_log >= self.log_interval:
            fps = (len(self.times) - 1) / (self.times[-1] - self.times[0])
            print(f"[FPS] {fps:.2f}")
            self.last_log = now

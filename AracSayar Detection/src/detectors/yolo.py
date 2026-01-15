from typing import List

import numpy as np

from src.utils.types import Detection

VEHICLE_CLASSES = {
    "bus",
    "car",
    "motorbike",
    "motorcycle",
    "truck",
    "bicycle",
    "van",
    "pickup",
    "trailer",
    "vehicle",
    "suv",
}

# Map YOLO class names to canonical labels used in counting.
CANONICAL_LABEL = {
    "motorbike": "motorcycle",
    "motorcycle": "motorcycle",
    "pickup": "car",
    "trailer": "car",
    "vehicle": "car",
    "suv": "car",
}


class YOLODetector:
    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        device: str | None = None,
        vehicle_classes: set[str] | None = None,
    ):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise ImportError(
                "ultralytics is required for YOLODetector. Check if env has it."
            ) from exc

        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.vehicle_classes = {c.lower() for c in (vehicle_classes or VEHICLE_CLASSES)}

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        for res in results:
            boxes = res.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = res.names.get(cls_id, str(cls_id))
                norm_name = cls_name.lower()
                if norm_name not in self.vehicle_classes:
                    # If YOLO gives an unexpected vehicle-ish label, fall back to car.
                    if "car" in norm_name or "vehicle" in norm_name:
                        norm_name = "car"
                    else:
                        continue
                cls_label = CANONICAL_LABEL.get(norm_name, norm_name)
                detections.append(
                    Detection(
                        box=(int(x1), int(y1), int(x2), int(y2)),
                        score=score,
                        cls=cls_label,
                    )
                )
        return detections

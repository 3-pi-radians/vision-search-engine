import logging

from PIL import Image
from ultralytics import YOLO

import config
from detectors.base_detector import BaseDetector, DetectionResult, MAX_DETECTIONS

logger = logging.getLogger(__name__)


class YOLOv8Detector(BaseDetector):
    def __init__(self, model_name: str = config.DETECTOR) -> None:
        self._model = YOLO(f"{model_name}.pt")
        logger.info("YOLOv8Detector loaded: %s", model_name)

    def detect(self, image: Image.Image) -> DetectionResult:
        w, h = image.size

        results = self._model(image, verbose=False)

        # collect all boxes above threshold, sort by confidence descending
        boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf >= config.YOLO_CONF_THRESHOLD:
                    boxes.append((conf, box))

        boxes.sort(key=lambda x: x[0], reverse=True)

        # cap at MAX_DETECTIONS — beyond this it's likely a lookbook over-detection
        if len(boxes) > MAX_DETECTIONS:
            logger.debug("Capping %d detections to top-%d", len(boxes), MAX_DETECTIONS)
            boxes = boxes[:1]  # fall back to top-1 when over the cap

        if not boxes:
            logger.debug("YOLO fallback: no detections above threshold %.3f", config.YOLO_CONF_THRESHOLD)
            return DetectionResult(
                crops=[image],
                bboxes=[(0, 0, w, h)],
                confidences=[0.0],
                used_fallback=True,
            )

        crops, bboxes, confidences = [], [], []
        for conf, box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crops.append(image.crop((x1, y1, x2, y2)))
            bboxes.append((x1, y1, x2, y2))
            confidences.append(conf)

        return DetectionResult(
            crops=crops,
            bboxes=bboxes,
            confidences=confidences,
            used_fallback=False,
        )


if __name__ == "__main__":
    import numpy as np

    print("=== yolov8_detector.py smoke test ===")
    detector = YOLOv8Detector()
    dummy = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
    result = detector.detect(dummy)
    print(f"used_fallback  : {result.used_fallback}")
    print(f"num crops      : {len(result.crops)}")
    print(f"confidences    : {result.confidences}")
    print(f"bboxes         : {result.bboxes}")
    print("OK")

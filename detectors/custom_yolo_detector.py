# Custom YOLO detector — Project 1 fine-tuned weights
# Weights: yolov8s_prj1.pt
# Classes: short_sleeve_top, trousers, shorts, long_sleeve_top, skirt
# Use this detector for images similar to Project 1 training data

import logging

from PIL import Image
from ultralytics import YOLO

import config
from detectors.base_detector import BaseDetector, DetectionResult, MAX_DETECTIONS

logger = logging.getLogger(__name__)


class CustomYOLODetector(BaseDetector):
    def __init__(self) -> None:
        self._model = YOLO(str(config.CUSTOM_YOLO_WEIGHTS_PATH))
        logger.info("CustomYOLODetector loaded: %s", config.CUSTOM_YOLO_WEIGHTS_PATH)

    def detect(self, image: Image.Image) -> DetectionResult:
        w, h = image.size

        results = self._model(image, verbose=False, conf=config.YOLO_CONF_THRESHOLD)

        boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                boxes.append((float(box.conf[0]), box))

        boxes.sort(key=lambda x: x[0], reverse=True)

        if len(boxes) > MAX_DETECTIONS:
            logger.debug("Capping %d detections to top-%d", len(boxes), MAX_DETECTIONS)
            boxes = boxes[:MAX_DETECTIONS]

        if not boxes:
            logger.debug("CustomYOLO fallback: no detections above threshold %.3f", config.YOLO_CONF_THRESHOLD)
            return DetectionResult(
                crops=[image],
                bboxes=[(0, 0, w, h)],
                confidences=[0.0],
                used_fallback=True,
            )

        img_area = w * h
        crops, bboxes, confidences = [], [], []
        for conf, box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) * (y2 - y1) < img_area * 0.03:
                continue
            crops.append(image.crop((x1, y1, x2, y2)))
            bboxes.append((x1, y1, x2, y2))
            confidences.append(conf)

        if not crops:
            return DetectionResult(
                crops=[image],
                bboxes=[(0, 0, w, h)],
                confidences=[0.0],
                used_fallback=True,
            )

        return DetectionResult(
            crops=crops,
            bboxes=bboxes,
            confidences=confidences,
            used_fallback=False,
        )


if __name__ == "__main__":
    import numpy as np

    print("=== custom_yolo_detector.py smoke test ===")
    detector = CustomYOLODetector()
    dummy = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
    result = detector.detect(dummy)
    print(f"used_fallback  : {result.used_fallback}")
    print(f"num crops      : {len(result.crops)}")
    print(f"confidences    : {result.confidences}")
    print(f"bboxes         : {result.bboxes}")
    print("OK")

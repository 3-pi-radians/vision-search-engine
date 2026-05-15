# Fashion-specific YOLO detector
# Model: NovaAstro/YOLOv8m_fashion
# Detects individual clothing items: top, bottom, dress,
# outerwear, skirt, shorts
# Replaces temporary upper/lower body split (Option 1)

import logging

from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

import config
from detectors.base_detector import BaseDetector, DetectionResult, MAX_DETECTIONS

logger = logging.getLogger(__name__)

_MODEL_REPO     = "NovaAstro/YOLOv8m_fashion"
_MODEL_FILENAME = "model_with_updated_labels.pt"
# Main garment classes only — excludes sub-part detections (sleeve, pocket, zipper, etc.)
_GARMENT_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 22, 23]


class FashionYOLODetector(BaseDetector):
    def __init__(self) -> None:
        weights = hf_hub_download(_MODEL_REPO, filename=_MODEL_FILENAME)
        self._model = YOLO(weights)
        logger.info("FashionYOLODetector loaded: %s", _MODEL_REPO)

    def detect(self, image: Image.Image) -> DetectionResult:
        w, h = image.size

        results = self._model(
            image, verbose=False,
            conf=config.YOLO_CONF_THRESHOLD,
            classes=_GARMENT_CLASSES,
        )

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
            logger.debug("FashionYOLO fallback: no detections above threshold %.3f", config.YOLO_CONF_THRESHOLD)
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

    print("=== fashion_yolo_detector.py smoke test ===")
    detector = FashionYOLODetector()
    dummy = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
    result = detector.detect(dummy)
    print(f"used_fallback  : {result.used_fallback}")
    print(f"num crops      : {len(result.crops)}")
    print(f"confidences    : {result.confidences}")
    print(f"bboxes         : {result.bboxes}")
    print("OK")

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class DetectionResult:
    crops: list[Image.Image]                    # one per detected item; [full_image] on fallback
    bboxes: list[tuple[int, int, int, int]]     # (x1, y1, x2, y2) per crop
    confidences: list[float]                    # per crop; 0.0 for fallback
    used_fallback: bool                         # True when no detection met threshold


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: Image.Image) -> DetectionResult:
        """
        Detect clothing items in image.
        Returns up to MAX_DETECTIONS crops sorted by confidence (best first).
        Must never raise — returns full image as single-item fallback.
        """
        ...


MAX_DETECTIONS = 5  # cap; beyond this YOLO is likely over-detecting on a lookbook shot


if __name__ == "__main__":
    print("=== base_detector.py smoke test ===")

    class _DummyDetector(BaseDetector):
        def detect(self, image: Image.Image) -> DetectionResult:
            w, h = image.size
            return DetectionResult(
                crops=[image],
                bboxes=[(0, 0, w, h)],
                confidences=[0.0],
                used_fallback=True,
            )

    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    result = _DummyDetector().detect(img)
    assert result.used_fallback
    assert len(result.crops) == 1
    assert result.confidences[0] == 0.0
    print(f"crops={len(result.crops)}, bbox={result.bboxes[0]}, confidence={result.confidences[0]}, fallback={result.used_fallback}")
    print("OK")

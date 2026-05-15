import logging

import config
from detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, callable] = {}


def _register():
    from detectors.yolov8_detector import YOLOv8Detector
    from detectors.fashion_yolo_detector import FashionYOLODetector
    _REGISTRY["yolov8n"] = lambda: YOLOv8Detector("yolov8n")
    _REGISTRY["yolov8s"] = lambda: YOLOv8Detector("yolov8s")
    _REGISTRY["yolov8m"] = lambda: YOLOv8Detector("yolov8m")
    _REGISTRY["yolov8l"] = lambda: YOLOv8Detector("yolov8l")
    _REGISTRY["yolov8x"] = lambda: YOLOv8Detector("yolov8x")
    _REGISTRY["fashion"] = lambda: FashionYOLODetector()


class DetectorFactory:
    _instance: BaseDetector | None = None

    @classmethod
    def get(cls, name: str = config.DETECTOR) -> BaseDetector:
        if not _REGISTRY:
            _register()

        key = name.lower()
        if key not in _REGISTRY:
            raise ValueError(
                f"Unknown detector '{name}'. Available: {list(_REGISTRY.keys())}"
            )

        if cls._instance is None:
            logger.info("Instantiating detector: %s", name)
            cls._instance = _REGISTRY[key]()

        return cls._instance


if __name__ == "__main__":
    print("=== detector_factory.py smoke test ===")
    detector = DetectorFactory.get(config.DETECTOR)
    print(f"Detector type : {type(detector).__name__}")
    detector2 = DetectorFactory.get(config.DETECTOR)
    assert detector is detector2, "Factory must return the same instance (singleton)"
    print("Singleton check: OK")
    print("OK")

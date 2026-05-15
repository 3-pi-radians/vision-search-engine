import logging

import config
from detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, callable] = {}


def _register():
    from detectors.yolov8_detector import YOLOv8Detector
    from detectors.fashion_yolo_detector import FashionYOLODetector
    from detectors.custom_yolo_detector import CustomYOLODetector
    _REGISTRY["yolov8n"] = lambda: YOLOv8Detector("yolov8n")
    _REGISTRY["yolov8s"] = lambda: YOLOv8Detector("yolov8s")
    _REGISTRY["yolov8m"] = lambda: YOLOv8Detector("yolov8m")
    _REGISTRY["yolov8l"] = lambda: YOLOv8Detector("yolov8l")
    _REGISTRY["yolov8x"] = lambda: YOLOv8Detector("yolov8x")
    _REGISTRY["fashion"] = lambda: FashionYOLODetector()
    _REGISTRY["custom"]  = lambda: CustomYOLODetector()


class DetectorFactory:
    _instances: dict[str, BaseDetector] = {}

    @classmethod
    def get(cls, name: str = None) -> BaseDetector:
        if not _REGISTRY:
            _register()

        name = name or config.DETECTOR
        key = name.lower()
        if key not in _REGISTRY:
            raise ValueError(
                f"Unknown detector '{name}'. Available: {list(_REGISTRY.keys())}"
            )

        if key not in cls._instances:
            logger.info("Instantiating detector: %s", name)
            cls._instances[key] = _REGISTRY[key]()

        return cls._instances[key]


if __name__ == "__main__":
    print("=== detector_factory.py smoke test ===")
    detector = DetectorFactory.get(config.DETECTOR)
    print(f"Detector type : {type(detector).__name__}")
    detector2 = DetectorFactory.get(config.DETECTOR)
    assert detector is detector2, "Factory must return the same instance (singleton)"
    print("Singleton check: OK")
    print("OK")

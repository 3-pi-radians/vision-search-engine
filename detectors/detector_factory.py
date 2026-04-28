import logging

import config
from detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseDetector]] = {}


def _register():
    from detectors.yolov8_detector import YOLOv8Detector
    _REGISTRY["yolov8n"] = YOLOv8Detector
    _REGISTRY["yolov8s"] = YOLOv8Detector
    _REGISTRY["yolov8m"] = YOLOv8Detector
    _REGISTRY["yolov8l"] = YOLOv8Detector
    _REGISTRY["yolov8x"] = YOLOv8Detector


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
            cls._instance = _REGISTRY[key](model_name=name)

        return cls._instance


if __name__ == "__main__":
    print("=== detector_factory.py smoke test ===")
    detector = DetectorFactory.get(config.DETECTOR)
    print(f"Detector type : {type(detector).__name__}")
    detector2 = DetectorFactory.get(config.DETECTOR)
    assert detector is detector2, "Factory must return the same instance (singleton)"
    print("Singleton check: OK")
    print("OK")

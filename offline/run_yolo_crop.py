"""
Offline step 1 — YOLO cropping (plug-and-play detector)
Reads DeepFashion list_eval_partition.txt to get Gallery and Train
images, runs the selected YOLO detector on each, saves crops to disk.
Writes image_paths.json: {index: {path, item_id}} for Gallery split.
Resumable: skips images whose crop already exists.

Usage:
    python offline/run_yolo_crop.py                     # default: yolov8m
    python offline/run_yolo_crop.py --detector fashion  # FashionYOLO
    python offline/run_yolo_crop.py --detector custom   # Project 1 weights

Output paths per detector:
    yolov8m → CROPS_DIR         + IMAGE_PATHS_PATH
    fashion → CROPS_DIR_FASHION + IMAGE_PATHS_PATH_FASHION
    custom  → CROPS_DIR_CUSTOM  + IMAGE_PATHS_PATH_CUSTOM
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_eval_partition(partition_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Returns {"train": [(img_path, item_id), ...], "query": [...], "gallery": [...]}."""
    splits: dict[str, list[tuple[str, str]]] = {"train": [], "query": [], "gallery": []}
    with open(partition_path) as f:
        next(f)  # skip count line
        next(f)  # skip header: image_name item_id evaluation_status
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            img_path, item_id, split = parts[0], parts[1], parts[2].lower()
            if split in splits:
                splits[split].append((img_path, item_id))
    return splits


def _crop_and_save(detector, img_path: Path, base_out: Path) -> tuple[list[Path], bool]:
    """
    Detect clothing items using any BaseDetector implementation.
    Saves each detected crop as {stem}_crop{i}{suffix} under base_out's parent.
    Returns (list of saved paths, used_fallback).
    Works with single-crop (YOLOv8m) and multi-crop (FashionYOLO) detectors.
    """
    image = Image.open(img_path).convert("RGB")
    result = detector.detect(image)
    base_out.parent.mkdir(parents=True, exist_ok=True)
    stem, suffix = base_out.stem, base_out.suffix
    saved = []
    for i, crop in enumerate(result.crops):
        out = base_out.parent / f"{stem}_crop{i}{suffix}"
        crop.save(out)
        saved.append(out)
    return saved, result.used_fallback


def run(splits_to_process: list[str] = ("gallery", "train"), detector_name: str = "yolov8m") -> None:
    from detectors.detector_factory import DetectorFactory

    if detector_name == "fashion":
        crops_dir       = config.CROPS_DIR_FASHION
        image_paths_out = config.IMAGE_PATHS_PATH_FASHION
    elif detector_name == "custom":
        crops_dir       = config.CROPS_DIR_CUSTOM
        image_paths_out = config.IMAGE_PATHS_PATH_CUSTOM
    else:
        crops_dir       = config.CROPS_DIR
        image_paths_out = config.IMAGE_PATHS_PATH

    crops_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Detector: %s", detector_name)
    logger.info("Crops dir: %s", crops_dir)

    detector = DetectorFactory.get(detector_name)

    logger.info("Parsing eval partition: %s", config.LIST_EVAL_PARTITION)
    splits = parse_eval_partition(config.LIST_EVAL_PARTITION)

    # gallery index → {path, item_id, source_image}
    image_paths: dict[int, dict] = {}
    gallery_idx = 0
    fallback_count = 0

    for split in splits_to_process:
        images = splits[split]
        logger.info("Processing %s split: %d images", split, len(images))

        for rel_path, item_id in tqdm(images, desc=split):
            # annotation paths start with "img/" but DATASET_IMAGES_DIR
            # already points to img_highres/
            clean_rel = rel_path[4:] if rel_path.startswith("img/") else rel_path
            img_path = config.DATASET_IMAGES_DIR / clean_rel
            base_out = crops_dir / split / clean_rel
            # _crop0 existence is the resume proxy for "this source image was processed"
            proxy = base_out.parent / f"{base_out.stem}_crop0{base_out.suffix}"

            if not img_path.exists():
                logger.warning("Image not found, skipping: %s", img_path)
                continue

            if proxy.exists():
                # resumable: re-collect all _crop{i} files already saved
                if split == "gallery":
                    i = 0
                    while True:
                        p = base_out.parent / f"{base_out.stem}_crop{i}{base_out.suffix}"
                        if not p.exists():
                            break
                        image_paths[gallery_idx] = {
                            "path":         str(p),
                            "item_id":      item_id,
                            "source_image": str(base_out),
                        }
                        gallery_idx += 1
                        i += 1
                continue

            try:
                saved_paths, used_fallback = _crop_and_save(detector, img_path, base_out)
            except Exception as e:
                logger.error("Failed to process %s: %s", img_path, e)
                continue

            if used_fallback:
                fallback_count += 1

            if split == "gallery":
                for p in saved_paths:
                    image_paths[gallery_idx] = {
                        "path":         str(p),
                        "item_id":      item_id,
                        "source_image": str(base_out),
                    }
                    gallery_idx += 1

        logger.info(
            "Done %s split. Gallery index count so far: %d",
            split, gallery_idx
        )

    total_images = sum(len(splits[s]) for s in splits_to_process)
    logger.info(
        "Fallback count: %d / %d images used full image (no detection above threshold)",
        fallback_count,
        total_images,
    )

    # save gallery index → {path, item_id} mapping
    with open(image_paths_out, "w") as f:
        json.dump(image_paths, f)
    logger.info(
        "Saved image_paths with %d gallery entries → %s",
        len(image_paths), image_paths_out
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector",
        default="yolov8m",
        choices=config.AVAILABLE_DETECTORS,
        help="Detector to use for cropping. Options: " + ", ".join(config.AVAILABLE_DETECTORS),
    )
    args = parser.parse_args()
    run(detector_name=args.detector)
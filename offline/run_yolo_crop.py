"""
Offline step 1 — YOLO cropping
Reads DeepFashion list_eval_partition.txt to get Gallery and Train images,
runs YOLOv8 detection on each, saves crops to CROPS_DIR.
Writes image_paths.json: {index: crop_path} for Gallery split (used for indexing).
Resumable: skips images whose crop already exists.
"""

import json
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# allow imports from project root when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_eval_partition(partition_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Returns {"train": [(img_path, item_id), ...], "query": [...], "gallery": [...]}."""
    splits: dict[str, list[tuple[str, str]]] = {"train": [], "query": [], "gallery": []}
    with open(partition_path) as f:
        next(f)  # skip count line
        next(f)  # skipping header: image_name item_id evaluation_status
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            img_path, item_id, split = parts[0], parts[1], parts[2].lower()
            if split in splits:
                splits[split].append((img_path, item_id))
    return splits


def crop_and_save(
    model: YOLO,
    img_path: Path,
    out_path: Path,
) -> Path:
    """Detect clothing item, save crop. Returns out_path."""
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    results = model(image, verbose=False)
    best_box = None
    best_conf = 0.0

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box

    if best_box is None or best_conf < config.YOLO_CONF_THRESHOLD:
        crop = image
    else:
        x1, y1, x2, y2 = (int(v) for v in best_box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image.crop((x1, y1, x2, y2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return out_path


def run(splits_to_process: list[str] = ("gallery", "train")) -> None:
    config.CROPS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading YOLO model: %s", config.DETECTOR)
    model = YOLO(f"{config.DETECTOR}.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info("Using device: %s", device)

    logger.info("Parsing eval partition: %s", config.LIST_EVAL_PARTITION)
    splits = parse_eval_partition(config.LIST_EVAL_PARTITION)

    # gallery index → {path, item_id} — item_id is ground truth key for eval
    image_paths: dict[int, dict] = {}
    gallery_idx = 0

    for split in splits_to_process:
        images = splits[split]
        logger.info("Processing %s split: %d images", split, len(images))

        for rel_path, item_id in tqdm(images, desc=split):
            img_path = config.DATASET_IMAGES_DIR / rel_path
            out_path = config.CROPS_DIR / split / rel_path

            if not img_path.exists():
                logger.warning("Image not found, skipping: %s", img_path)
                continue

            if out_path.exists():
                # resumable: already cropped
                if split == "gallery":
                    image_paths[gallery_idx] = {"path": str(out_path), "item_id": item_id}
                    gallery_idx += 1
                continue

            try:
                crop_and_save(model, img_path, out_path)
            except Exception as e:
                logger.error("Failed to process %s: %s", img_path, e)
                continue

            if split == "gallery":
                image_paths[gallery_idx] = {"path": str(out_path), "item_id": item_id}
                gallery_idx += 1

        logger.info("Done %s split. Gallery index count so far: %d", split, gallery_idx)

    # save gallery index → {path, item_id} mapping
    with open(config.IMAGE_PATHS_PATH, "w") as f:
        json.dump(image_paths, f)
    logger.info("Saved image_paths.json with %d gallery entries → %s", len(image_paths), config.IMAGE_PATHS_PATH)


if __name__ == "__main__":
    run()
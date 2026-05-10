"""
Offline step 1 — YOLO cropping
Reads DeepFashion list_eval_partition.txt to get Gallery and Train images,
runs YOLOv8 detection on each, saves crops to CROPS_DIR.
Writes image_paths.json: {index: crop_path} for Gallery split (used for indexing).
Resumable: skips images whose crop already exists.

Changes from v1:
- YOLO_CONF_THRESHOLD raised to 0.5 (was 0.3)
- Person class filter (class 0 only) — avoids partial fabric detections
- IoU-based deduplication — prevents overlapping crops of same item
"""

import json
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0  # YOLOv8 COCO class 0 = person


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


def compute_iou(box_a, box_b) -> float:
    """Compute Intersection over Union between two YOLO boxes."""
    ax1, ay1, ax2, ay2 = box_a.xyxy[0].tolist()
    bx1, by1, bx2, by2 = box_b.xyxy[0].tolist()
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def filter_overlapping(boxes, iou_threshold: float = 0.5) -> list:
    """
    Remove overlapping detections using greedy NMS.
    Keeps the highest-confidence box and removes any box
    with IoU > iou_threshold against an already-kept box.
    """
    if not boxes:
        return boxes
    # sort by confidence descending
    boxes = sorted(boxes, key=lambda b: float(b.conf[0]), reverse=True)
    kept = [boxes[0]]
    for box in boxes[1:]:
        if all(compute_iou(box, k) < iou_threshold for k in kept):
            kept.append(box)
    return kept


def crop_and_save(
    model: YOLO,
    img_path: Path,
    out_path: Path,
) -> Path:
    """
    Detect clothing item (via person class), save crop.
    Falls back to full image if no valid detection found.
    Returns out_path.
    """
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    results = model(image, verbose=False)

    # Collect person-class boxes only
    person_boxes = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == PERSON_CLASS_ID and conf >= config.YOLO_CONF_THRESHOLD:
                person_boxes.append(box)

    # Remove overlapping detections
    person_boxes = filter_overlapping(person_boxes, iou_threshold=0.5)

    # Pick highest confidence box
    best_box = person_boxes[0] if person_boxes else None

    if best_box is None:
        # fallback — use full image
        crop = image
        logger.debug("No valid detection for %s — using full image", img_path.name)
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
    logger.info("Confidence threshold: %s", config.YOLO_CONF_THRESHOLD)
    logger.info("Person class filter: ON (class %d only)", PERSON_CLASS_ID)

    logger.info("Parsing eval partition: %s", config.LIST_EVAL_PARTITION)
    splits = parse_eval_partition(config.LIST_EVAL_PARTITION)

    # gallery index → {path, item_id} — item_id is ground truth key for eval
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
            out_path = config.CROPS_DIR / split / clean_rel

            if not img_path.exists():
                logger.warning("Image not found, skipping: %s", img_path)
                continue

            if out_path.exists():
                # resumable: already cropped
                if split == "gallery":
                    image_paths[gallery_idx] = {
                        "path": str(out_path),
                        "item_id": item_id
                    }
                    gallery_idx += 1
                continue

            try:
                crop_and_save(model, img_path, out_path)
            except Exception as e:
                logger.error("Failed to process %s: %s", img_path, e)
                continue

            if split == "gallery":
                image_paths[gallery_idx] = {
                    "path": str(out_path),
                    "item_id": item_id
                }
                gallery_idx += 1

        logger.info(
            "Done %s split. Gallery index count so far: %d",
            split, gallery_idx
        )

    # save gallery index → {path, item_id} mapping
    with open(config.IMAGE_PATHS_PATH, "w") as f:
        json.dump(image_paths, f)
    logger.info(
        "Saved image_paths.json with %d gallery entries → %s",
        len(image_paths), config.IMAGE_PATHS_PATH
    )


if __name__ == "__main__":
    run()
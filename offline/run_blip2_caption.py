"""
Offline step 2 — BLIP-2 captioning
Reads image_paths.json (gallery only), generates one caption per crop,
writes captions.json: {crop_stem: caption_string}.
crop_stem = Path(crop_path).stem, e.g. "01_1_front_crop0".
Resumable: skips crop_stems already present in captions.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def remap_crop_path(old_path: str, fashion: bool = False) -> Path:
    """
    Remaps stale Kaggle session paths to the current crops directory.
    Handles both /crops/ (person mode) and /crops_fashion/ (fashion mode) markers.
    """
    if fashion:
        marker, crops_dir = "/crops_fashion/", config.CROPS_DIR_FASHION
    else:
        marker, crops_dir = "/crops/", config.CROPS_DIR_INPUT
    idx = old_path.find(marker)
    if idx != -1:
        relative = old_path[idx + len(marker):]
        return crops_dir / relative
    return Path(old_path)


def load_existing_captions(captions_path: Path) -> dict[str, str]:
    """Returns {item_id: caption} from existing captions file."""
    if captions_path.exists():
        with open(captions_path) as f:
            raw = json.load(f)
        logger.info("Loaded %d existing captions (resuming)", len(raw))
        return raw
    return {}


def save_captions(captions: dict[str, str], captions_path: Path) -> None:
    with open(captions_path, "w") as f:
        json.dump(captions, f)


def run(fashion: bool = False) -> None:
    image_paths_input = config.IMAGE_PATHS_PATH_FASHION if fashion else config.IMAGE_PATHS_INPUT
    captions_path     = config.CAPTIONS_PATH_FASHION    if fashion else config.CAPTIONS_PATH

    if not image_paths_input.exists():
        raise FileNotFoundError(
            f"image_paths not found at {image_paths_input}. "
            "Run run_yolo_crop.py first."
        )
    with open(image_paths_input) as f:
        image_paths: dict[str, dict] = json.load(f)

    captions = load_existing_captions(captions_path)

    # one caption per crop stem — crop stems are already unique, no deduplication needed
    seen_crops: dict[str, str] = {}   # crop_stem → crop path
    for entry in image_paths.values():
        crop_stem = Path(entry["path"]).stem
        seen_crops[crop_stem] = entry["path"]

    remaining = {stem: path for stem, path in seen_crops.items() if stem not in captions}
    logger.info("Total crops: %d | Already captioned: %d | Remaining: %d",
                len(seen_crops), len(captions), len(remaining))

    if not remaining:
        logger.info("All captions already generated.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading BLIP-2 model: %s on %s", config.BLIP2_MODEL_NAME, device)

    processor = Blip2Processor.from_pretrained(config.BLIP2_MODEL_NAME)
    model = Blip2ForConditionalGeneration.from_pretrained(
        config.BLIP2_MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    logger.info("BLIP-2 loaded.")

    crop_keys = list(remaining.keys())
    batch_size = config.CAPTION_BATCH_SIZE
    save_every = 500

    for batch_start in tqdm(range(0, len(crop_keys), batch_size), desc="captioning"):
        batch_crop_keys = crop_keys[batch_start: batch_start + batch_size]
        images, valid_keys = [], []

        for crop_key in batch_crop_keys:
            crop_path = remap_crop_path(remaining[crop_key], fashion=fashion)
            if not crop_path.exists():
                logger.warning("Crop not found, skipping %s: %s", crop_key, crop_path)
                continue
            try:
                images.append(Image.open(crop_path).convert("RGB"))
                valid_keys.append(crop_key)
            except Exception as e:
                logger.error("Failed to open crop %s: %s", crop_path, e)

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(
            device, torch.float16 if device == "cuda" else torch.float32
        )
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)

        batch_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for crop_key, caption in zip(valid_keys, batch_captions):
            captions[crop_key] = caption.strip()

        if (batch_start // batch_size) % (save_every // batch_size) == 0:
            save_captions(captions, captions_path)

    save_captions(captions, captions_path)
    logger.info("Done. Saved %d captions → %s", len(captions), captions_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fashion", action="store_true",
        help="Use fashion-YOLO image_paths and write captions_fashion.json"
    )
    args = parser.parse_args()
    run(fashion=args.fashion)

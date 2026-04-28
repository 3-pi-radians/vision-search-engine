"""
Offline step 2 — BLIP-2 captioning
Reads image_paths.json (gallery only), generates one caption per crop,
writes captions.json: {item_id: caption_string}.
Resumable: skips item_ids already present in captions.json.
"""

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


def load_existing_captions() -> dict[str, str]:
    """Returns {item_id: caption} from existing captions.json."""
    if config.CAPTIONS_PATH.exists():
        with open(config.CAPTIONS_PATH) as f:
            raw = json.load(f)
        logger.info("Loaded %d existing captions (resuming)", len(raw))
        return raw  # keys are already item_id strings
    return {}


def save_captions(captions: dict[str, str]) -> None:
    with open(config.CAPTIONS_PATH, "w") as f:
        json.dump(captions, f)


def run() -> None:
    if not config.IMAGE_PATHS_PATH.exists():
        raise FileNotFoundError(
            f"image_paths.json not found at {config.IMAGE_PATHS_PATH}. "
            "Run run_yolo_crop.py first."
        )
    with open(config.IMAGE_PATHS_PATH) as f:
        image_paths: dict[str, dict] = json.load(f)

    captions = load_existing_captions()

    # deduplicate: one caption per item_id, pick first crop encountered
    seen_items: dict[str, str] = {}   # item_id → crop path
    for entry in image_paths.values():
        item_id = entry["item_id"]
        if item_id not in seen_items:
            seen_items[item_id] = entry["path"]

    remaining = {iid: path for iid, path in seen_items.items() if iid not in captions}
    logger.info("Unique items: %d | Already captioned: %d | Remaining: %d",
                len(seen_items), len(captions), len(remaining))

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

    item_ids = list(remaining.keys())
    batch_size = config.CAPTION_BATCH_SIZE
    save_every = 500

    for batch_start in tqdm(range(0, len(item_ids), batch_size), desc="captioning"):
        batch_item_ids = item_ids[batch_start: batch_start + batch_size]
        images, valid_ids = [], []

        for item_id in batch_item_ids:
            crop_path = Path(remaining[item_id])
            if not crop_path.exists():
                logger.warning("Crop not found, skipping item_id %s: %s", item_id, crop_path)
                continue
            try:
                images.append(Image.open(crop_path).convert("RGB"))
                valid_ids.append(item_id)
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

        for item_id, caption in zip(valid_ids, batch_captions):
            captions[item_id] = caption.strip()

        if (batch_start // batch_size) % (save_every // batch_size) == 0:
            save_captions(captions)

    save_captions(captions)
    logger.info("Done. Saved %d captions → %s", len(captions), config.CAPTIONS_PATH)


if __name__ == "__main__":
    run()

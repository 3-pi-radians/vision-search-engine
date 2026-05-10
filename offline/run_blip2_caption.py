"""
Offline step 2 — BLIP-2 captioning (blip2-flan-t5-xl)
Reads image_paths.json (gallery only), generates one structured caption per unique
item_id using category-aware prompts, writes captions.json: {item_id: caption_string}.

Key behaviors:
- Category extracted from crop path folder structure → category-specific prompt
- Front-view image (01_1_front) preferred per item_id; fallback to first available
- All generation params driven by config constants
- Post-processing: strips person-reference prefixes, flags short captions
Resumable: skips item_ids already present in captions.json, checkpoints every 500.
"""

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from blip2_prompts import CATEGORY_PROMPTS, DEFAULT_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Output path: writable working dir on Kaggle, same as CAPTIONS_PATH locally
CAPTIONS_OUTPUT = config.WORK_DIR / "captions.json" if config.KAGGLE else config.CAPTIONS_PATH

# Strips leading person-reference phrases ("a woman wearing", "a man in", etc.)
_PERSON_RE = re.compile(
    r"^(a\s+(?:woman|man|person)\s+(?:wearing|in)|"
    r"the\s+(?:woman|man|person)\s+is\s+wearing|"
    r"(?:woman|man|person)\s+wearing)"
    r"[,\s]*",
    re.IGNORECASE,
)

_MIN_CAPTION_WORDS = 6


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def remap_crop_path(old_path: str) -> Path:
    """Remaps stale /kaggle/working/crops/... paths to config.CROPS_DIR_INPUT."""
    marker = "/crops/"
    idx = old_path.find(marker)
    if idx != -1:
        relative = old_path[idx + len(marker):]
        return config.CROPS_DIR_INPUT / relative
    return Path(old_path)


def extract_category(path: str) -> str:
    """
    Extracts DeepFashion category from crop path.
    e.g. .../crops/gallery/WOMEN/Blouses_Shirts/id_00000001/... → 'Blouses_Shirts'
    """
    parts = Path(path).parts
    try:
        gallery_idx = next(i for i, p in enumerate(parts) if p == "gallery")
        return parts[gallery_idx + 2]  # gallery / WOMEN|MEN / <category>
    except (StopIteration, IndexError):
        return ""


# ---------------------------------------------------------------------------
# Caption helpers
# ---------------------------------------------------------------------------

def get_prompt(path: str) -> str:
    category = extract_category(path)
    return CATEGORY_PROMPTS.get(category, DEFAULT_PROMPT)


def postprocess(caption: str, item_id: str, path: str) -> str:
    caption = _PERSON_RE.sub("", caption.strip()).lstrip(",").strip()
    if len(caption.split()) < _MIN_CAPTION_WORDS:
        logger.warning(
            "Short caption (%d words) item_id=%s path=%s: %r",
            len(caption.split()), item_id, path, caption,
        )
    return caption


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_existing_captions() -> dict[str, str]:
    if config.CAPTIONS_PATH.exists():
        with open(config.CAPTIONS_PATH) as f:
            raw = json.load(f)
        logger.info("Loaded %d existing captions (resuming)", len(raw))
        return raw
    return {}


def save_captions(captions: dict[str, str]) -> None:
    CAPTIONS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CAPTIONS_OUTPUT, "w") as f:
        json.dump(captions, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    src = config.IMAGE_PATHS_INPUT
    if not src.exists():
        raise FileNotFoundError(
            f"image_paths.json not found at {src}. Run run_yolo_crop.py first."
        )
    with open(src) as f:
        image_paths: dict[str, dict] = json.load(f)

    captions = load_existing_captions()

    # Group all gallery crops by item_id; prefer front-view image per item
    by_item: dict[str, list[str]] = defaultdict(list)
    for entry in image_paths.values():
        by_item[entry["item_id"]].append(entry["path"])

    seen_items: dict[str, str] = {}
    for item_id, paths in by_item.items():
        front = next((p for p in paths if "01_1_front" in p), None)
        seen_items[item_id] = front if front else paths[0]

    remaining = {iid: path for iid, path in seen_items.items() if iid not in captions}
    logger.info(
        "Unique items: %d | Already captioned: %d | Remaining: %d",
        len(seen_items), len(captions), len(remaining),
    )

    if not remaining:
        logger.info("All captions already generated.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading BLIP-2: %s on %s", config.BLIP2_MODEL_NAME, device)

    processor = Blip2Processor.from_pretrained(config.BLIP2_MODEL_NAME)
    model = Blip2ForConditionalGeneration.from_pretrained(
        config.BLIP2_MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    logger.info("BLIP-2 loaded.")

    item_ids   = list(remaining.keys())
    batch_size = config.CAPTION_BATCH_SIZE
    save_every = 500  # checkpoint interval in items

    for batch_start in tqdm(range(0, len(item_ids), batch_size), desc="captioning"):
        batch_ids = item_ids[batch_start: batch_start + batch_size]
        images, prompts, valid_ids, valid_paths = [], [], [], []

        for item_id in batch_ids:
            crop_path = remap_crop_path(remaining[item_id])
            if not crop_path.exists():
                logger.warning("Crop not found, skipping item_id %s: %s", item_id, crop_path)
                continue
            try:
                images.append(Image.open(crop_path).convert("RGB"))
                prompts.append(get_prompt(str(crop_path)))
                valid_ids.append(item_id)
                valid_paths.append(str(crop_path))
            except Exception as e:
                logger.error("Failed to open crop %s: %s", crop_path, e)

        if not images:
            continue

        dtype = torch.float16 if device == "cuda" else torch.float32
        inputs = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
        ).to(device, dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=config.BLIP2_MAX_NEW_TOKENS,
                num_beams=config.BLIP2_NUM_BEAMS,
                min_length=config.BLIP2_MIN_LENGTH,
                repetition_penalty=config.BLIP2_REPETITION_PENALTY,
            )

        batch_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for item_id, caption, path in zip(valid_ids, batch_captions, valid_paths):
            captions[item_id] = postprocess(caption, item_id, path)

        items_done = batch_start + len(valid_ids)
        if items_done % save_every < batch_size:
            save_captions(captions)
            logger.info("Checkpoint: %d captions saved", len(captions))

    save_captions(captions)
    logger.info("Done. Saved %d captions → %s", len(captions), CAPTIONS_OUTPUT)


if __name__ == "__main__":
    run()

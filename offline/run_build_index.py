"""
Offline step 4 — Build HNSW index
Generates fused CLIP embeddings for all gallery crops across configs A, B, C
and builds one HNSW index per config. Saves index files to WORK_DIR.
"""

import json
import logging
import sys
from pathlib import Path

import hnswlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_clip(finetuned: bool, device: str) -> CLIPModel:
    model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(device)
    if finetuned:
        ckpt_path = config.CLIP_WEIGHTS_DIR / "clip_finetuned.pt"
        if not ckpt_path.exists():
            logger.warning("Fine-tuned weights not found at %s — using pretrained for config C", ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt["model_state"] if "model_state" in ckpt else ckpt
            model.load_state_dict(state)
            logger.info("Loaded fine-tuned weights from %s", ckpt_path)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_embeddings(
    image_paths: dict[str, dict],
    captions: dict[str, str],
    model: CLIPModel,
    processor: CLIPProcessor,
    cfg: dict,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    indices = sorted(image_paths.keys(), key=int)
    all_embeddings = []
    alpha = cfg["alpha"]

    for batch_start in tqdm(range(0, len(indices), batch_size), desc="embedding"):
        batch_idx = indices[batch_start: batch_start + batch_size]
        images, texts = [], []

        for idx in batch_idx:
            entry = image_paths[idx]
            img = Image.open(entry["path"]).convert("RGB")
            images.append(img)
            if cfg["use_captions"]:
                item_id = entry["item_id"]
                texts.append(captions.get(item_id, "a clothing item"))

        img_inputs = processor(images=images, return_tensors="pt").to(device)
        img_emb = model.get_image_features(**img_inputs)
        img_emb = F.normalize(img_emb, dim=-1)

        if cfg["use_captions"]:
            txt_inputs = processor(
                text=texts, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            txt_emb = model.get_text_features(**txt_inputs)
            txt_emb = F.normalize(txt_emb, dim=-1)
            fused = alpha * img_emb + (1.0 - alpha) * txt_emb
            fused = F.normalize(fused, dim=-1)
        else:
            fused = img_emb

        all_embeddings.append(fused.cpu().float().numpy())

    return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_hnsw(embeddings: np.ndarray, index_path: Path) -> None:
    n, dim = embeddings.shape
    index = hnswlib.Index(space=config.HNSW_SPACE, dim=dim)
    index.init_index(
        max_elements=n,
        ef_construction=config.HNSW_EF_CONSTRUCTION,
        M=config.HNSW_M,
    )
    index.add_items(embeddings, list(range(n)))
    index.set_ef(config.HNSW_EF_SEARCH)
    index.save_index(str(index_path))
    logger.info("Saved HNSW index (%d items) → %s", n, index_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    if not config.IMAGE_PATHS_PATH.exists():
        raise FileNotFoundError(f"image_paths.json not found. Run run_yolo_crop.py first.")

    with open(config.IMAGE_PATHS_PATH) as f:
        image_paths: dict[str, dict] = json.load(f)
    logger.info("Gallery size: %d", len(image_paths))

    captions: dict[str, str] = {}
    if config.CAPTIONS_PATH.exists():
        with open(config.CAPTIONS_PATH) as f:
            captions = json.load(f)
        logger.info("Loaded %d captions", len(captions))
    else:
        logger.warning("captions.json not found — configs B and C will use fallback caption text")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)

    for cfg_name, cfg in config.CONFIGS.items():
        index_path = config.HNSW_INDEX_PATHS[cfg_name]

        if index_path.exists():
            logger.info("Index for config %s already exists, skipping.", cfg_name)
            continue

        logger.info("=== Building index for config %s ===", cfg_name)
        model = load_clip(finetuned=cfg["clip_finetuned"], device=device)

        embeddings = generate_embeddings(
            image_paths, captions, model, processor, cfg, device
        )
        logger.info("Config %s: embeddings shape %s", cfg_name, embeddings.shape)

        build_hnsw(embeddings, index_path)

        # free GPU memory before next config
        del model
        torch.cuda.empty_cache()

    logger.info("All indexes built.")


if __name__ == "__main__":
    run()

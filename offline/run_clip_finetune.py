"""
Offline step 3 — CLIP fine-tuning (Config C)
Freezes all CLIP layers except the last FINETUNE_UNFREEZE_BLOCKS vision encoder blocks.
Builds positive pairs from Train split (same item_id) and trains with image-only
InfoNCE contrastive loss. Saves best checkpoint to clip_weights/clip_finetuned.pt.
Resumable: continues from last saved epoch if checkpoint exists.

Changes from v1:
- Early stopping with patience (FINETUNE_PATIENCE)
- Saves best checkpoint (lowest loss) not just last epoch
- Cosine LR scheduler
- remap_crop_path() for stale Kaggle paths
- LR logged per epoch
"""

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path remapping (handles stale /kaggle/working/crops/... paths)
# ---------------------------------------------------------------------------

def remap_crop_path(old_path: str) -> str:
    """
    Remaps stale /kaggle/working/crops/... paths to current config.CROPS_DIR.
    Returns path unchanged if no /crops/ marker found.
    """
    marker = "/crops/"
    idx = old_path.find(marker)
    if idx != -1:
        relative = old_path[idx + len(marker):]
        return str(config.CROPS_DIR / relative)
    return old_path


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """Each sample is a (anchor, positive) pair from the same item_id."""

    def __init__(self, pairs: list[tuple[str, str]], processor: CLIPProcessor) -> None:
        self.pairs = pairs
        self.processor = processor

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        a_path, p_path = self.pairs[idx]
        anchor   = Image.open(a_path).convert("RGB")
        positive = Image.open(p_path).convert("RGB")
        a_inputs = self.processor(images=anchor,   return_tensors="pt")
        p_inputs = self.processor(images=positive, return_tensors="pt")
        return {
            "anchor":   a_inputs["pixel_values"].squeeze(0),
            "positive": p_inputs["pixel_values"].squeeze(0),
        }


_CROP_IDX_RE = re.compile(r"_crop(\d+)$")


def build_pairs(image_paths_train: dict[str, dict]) -> list[tuple[str, str]]:
    """
    Group train crops by (item_id, crop_index) and build positive pairs.
    crop_index is extracted from the filename suffix _crop{i}, acting as a
    garment-type proxy so only same-garment crops are paired.
    Falls back to grouping by item_id alone when no _crop{i} suffix is present
    (backward-compatible with person-mode crops).
    """
    by_garment: dict[str, list[str]] = defaultdict(list)
    for entry in image_paths_train.values():
        item_id = entry["item_id"]
        stem = Path(entry["path"]).stem
        m = _CROP_IDX_RE.search(stem)
        group_key = f"{item_id}_crop{m.group(1)}" if m else item_id
        by_garment[group_key].append(entry["path"])

    pairs = []
    for group_key, paths in by_garment.items():
        existing = [p for p in paths if Path(p).exists()]
        if len(existing) < 2:
            continue
        random.shuffle(existing)
        for i in range(0, len(existing) - 1, 2):
            pairs.append((existing[i], existing[i + 1]))

    logger.info("Built %d positive pairs from %d garment groups", len(pairs), len(by_garment))
    return pairs


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def freeze_clip(model: CLIPModel, unfreeze_blocks: int) -> None:
    """Freeze everything except the last `unfreeze_blocks` vision encoder layers."""
    for param in model.parameters():
        param.requires_grad = False

    encoder_layers = model.vision_model.encoder.layers
    for layer in encoder_layers[-unfreeze_blocks:]:
        for param in layer.parameters():
            param.requires_grad = True

    # always keep the final layer norm trainable
    for param in model.vision_model.post_layernorm.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %d / %d (%.1f%%)", trainable, total, 100 * trainable / total)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def infonce_loss(a_emb: torch.Tensor, p_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE loss over a batch of (anchor, positive) pairs."""
    a_emb = F.normalize(a_emb, dim=-1)
    p_emb = F.normalize(p_emb, dim=-1)
    logits = (a_emb @ p_emb.T) / temperature
    labels = torch.arange(len(logits), device=logits.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_p = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_p) / 2


# ---------------------------------------------------------------------------
# Train image paths loader
# ---------------------------------------------------------------------------

def load_train_image_paths(fashion: bool = False) -> dict[str, dict]:
    """Scan train crops directory and build path → item_id lookup."""
    base = config.CROPS_DIR_FASHION if fashion else config.CROPS_DIR_INPUT
    train_crops_dir = base / "train"
    if not train_crops_dir.exists():
        raise FileNotFoundError(
            f"Train crops not found at {train_crops_dir}. Run run_yolo_crop.py first."
        )

    entries: dict[str, dict] = {}
    for i, p in enumerate(train_crops_dir.rglob("*.jpg")):
        parts = p.parts
        item_id = next((part for part in parts if part.startswith("id_")), "unknown")
        entries[str(i)] = {
            "path":    str(p),
            "item_id": item_id
        }

    logger.info("Found %d train crops", len(entries))
    return entries


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run(fashion: bool = False) -> None:
    ckpt_path = config.CLIP_WEIGHTS_FASHION if fashion else config.CLIP_WEIGHTS_DIR / "clip_finetuned.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info("Config: epochs=%d, unfreeze_blocks=%d, lr=%s, temperature=%s",
                config.FINETUNE_EPOCHS, config.FINETUNE_UNFREEZE_BLOCKS,
                config.FINETUNE_LR, config.FINETUNE_TEMPERATURE)

    logger.info("Loading CLIP: %s", config.CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
    model     = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(device)

    freeze_clip(model, config.FINETUNE_UNFREEZE_BLOCKS)

    # Resume from checkpoint if exists
    start_epoch = 0
    best_loss   = float("inf")
    if ckpt_path.exists():
        logger.info("Resuming from checkpoint: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss   = ckpt.get("best_loss", float("inf"))
        logger.info("Resuming from epoch %d, best loss so far: %.4f", start_epoch, best_loss)

    train_entries = load_train_image_paths(fashion=fashion)
    pairs         = build_pairs(train_entries)

    dataset    = PairDataset(pairs, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=config.FINETUNE_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.FINETUNE_LR,
    )

    # Cosine annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.FINETUNE_EPOCHS,
        eta_min=1e-7,
    )

    patience_counter = 0

    for epoch in range(start_epoch, config.FINETUNE_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"epoch {epoch+1}/{config.FINETUNE_EPOCHS}"):
            anchor   = batch["anchor"].to(device)
            positive = batch["positive"].to(device)

            a_emb = model.visual_projection(
                model.vision_model(pixel_values=anchor).pooler_output
            )
            p_emb = model.visual_projection(
                model.vision_model(pixel_values=positive).pooler_output
            )

            loss = infonce_loss(a_emb, p_emb, config.FINETUNE_TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            "Epoch %d/%d — avg loss: %.4f | lr: %.2e",
            epoch + 1, config.FINETUNE_EPOCHS, avg_loss, current_lr
        )

        scheduler.step()

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch":       epoch,
                "best_loss":   best_loss,
            }, ckpt_path)
            logger.info("New best loss %.4f — checkpoint saved → %s", best_loss, ckpt_path)
        else:
            patience_counter += 1
            logger.info(
                "No improvement — patience %d/%d (best: %.4f)",
                patience_counter, config.FINETUNE_PATIENCE, best_loss
            )
            if patience_counter >= config.FINETUNE_PATIENCE:
                logger.info(
                    "Early stopping at epoch %d — best loss %.4f",
                    epoch + 1, best_loss
                )
                break

    logger.info(
        "Fine-tuning complete. Best loss: %.4f. Weights saved at %s",
        best_loss, ckpt_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fashion", action="store_true",
        help="Use fashion-YOLO train crops and save clip_finetuned_fashion.pt"
    )
    args = parser.parse_args()
    run(fashion=args.fashion)
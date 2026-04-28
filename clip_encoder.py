import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import config

logger = logging.getLogger(__name__)


class CLIPEncoder:
    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("CLIPEncoder device: %s", self._device)

        self._processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)

        # pretrained model — used for configs A and B, and as base for C
        self._pretrained = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(self._device)
        self._pretrained.eval()

        # fine-tuned model for config C (loaded lazily)
        self._finetuned: CLIPModel | None = None
        if config.CLIP_WEIGHTS_DIR.exists():
            self._finetuned = self._load_finetuned()

    def _load_finetuned(self) -> CLIPModel:
        ckpt_path = config.CLIP_WEIGHTS_DIR / "clip_finetuned.pt"
        if not ckpt_path.exists():
            logger.warning("Fine-tuned weights dir exists but no checkpoint found at %s; falling back to pretrained for config C", ckpt_path)
            return self._pretrained

        logger.info("Loading fine-tuned CLIP weights from %s", ckpt_path)
        model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
        state = torch.load(ckpt_path, map_location=self._device)
        model.load_state_dict(state)
        model.to(self._device)
        model.eval()
        return model

    def _model_for(self, config_name: str) -> CLIPModel:
        cfg = config.CONFIGS[config_name]
        if cfg["clip_finetuned"]:
            if self._finetuned is None:
                logger.warning("Config C requested but no fine-tuned weights loaded; using pretrained")
                return self._pretrained
            return self._finetuned
        return self._pretrained

    @torch.no_grad()
    def encode(
        self,
        crop: Image.Image,
        config_name: str,
        caption: str | None = None,
    ) -> np.ndarray:
        cfg = config.CONFIGS[config_name]
        alpha: float = cfg["alpha"]
        model = self._model_for(config_name)

        # image embedding
        image_inputs = self._processor(images=crop, return_tensors="pt").to(self._device)
        image_emb = model.visual_projection(model.vision_model(pixel_values=image_inputs["pixel_values"]).pooler_output)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

        if not cfg["use_captions"] or not caption:
            return image_emb.squeeze(0).cpu().numpy().astype(np.float32)

        # text embedding
        text_inputs = self._processor(text=[caption], return_tensors="pt", padding=True, truncation=True).to(self._device)
        text_emb = model.text_projection(model.text_model(**text_inputs).pooler_output)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # fused embedding
        fused = alpha * image_emb + (1.0 - alpha) * text_emb
        fused = fused / fused.norm(dim=-1, keepdim=True)
        return fused.squeeze(0).cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    import numpy as np

    print("=== clip_encoder.py smoke test ===")
    encoder = CLIPEncoder()
    dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    for cfg_name in ["A", "B", "C"]:
        emb = encoder.encode(dummy_img, config_name=cfg_name, caption="a red dress")
        assert emb.shape == (config.HNSW_DIM,), f"Expected dim {config.HNSW_DIM}, got {emb.shape}"
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5, "Embedding must be unit-normalised"
        print(f"Config {cfg_name}: shape={emb.shape}, norm={np.linalg.norm(emb):.6f} OK")
    print("All configs OK")

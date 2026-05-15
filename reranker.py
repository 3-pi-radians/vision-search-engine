import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval

import config

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = None
        self._model = None

        if not config.ENABLE_RERANKER:
            logger.info("Reranker disabled — skipping BLIP-ITM load")
            return

        logger.info("Loading BLIP-ITM: %s on %s", config.BLIP2_RERANK_MODEL_NAME, self._device)
        self._processor = BlipProcessor.from_pretrained(config.BLIP2_RERANK_MODEL_NAME)
        self._model = BlipForImageTextRetrieval.from_pretrained(
            config.BLIP2_RERANK_MODEL_NAME,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        ).to(self._device)
        self._model.eval()
        logger.info("BLIP-ITM loaded on %s", self._device)

    @torch.no_grad()
    def _itm_scores(
        self,
        query_crop: Image.Image,
        candidate_images: list[Image.Image],
        captions: list[str],
    ) -> list[float]:
        scores = []
        batch_size = config.RERANK_BATCH_SIZE

        for i in range(0, len(candidate_images), batch_size):
            batch_images   = candidate_images[i : i + batch_size]
            batch_captions = captions[i : i + batch_size]

            for img, caption in zip(batch_images, batch_captions):
                inputs = self._processor(
                    img,
                    caption,
                    return_tensors="pt",
                ).to(self._device)
                itm_output = self._model(**inputs)[0]  # shape [batch, 2]
                itm_score = itm_output[0][1].item()    # index 1 = positive match score
                scores.append(itm_score)

        return scores

    def rerank(
        self,
        query_crop: Image.Image,
        candidate_paths: list[str],
        config_name: str,
    ) -> list[int]:
        """
        Returns indices into candidate_paths sorted best-first.
        For config A (no captions), returns original order unchanged.
        """
        if not config.ENABLE_RERANKER or self._model is None:
            return list(range(len(candidate_paths)))

        cfg = config.CONFIGS[config_name]
        if not cfg["use_captions"]:
            return list(range(len(candidate_paths)))

        candidate_images = []
        captions         = []
        valid_indices    = []

        for i, path in enumerate(candidate_paths):
            p = Path(path)
            if not p.exists():
                logger.warning("Candidate crop not found, skipping: %s", path)
                continue
            try:
                candidate_images.append(Image.open(p).convert("RGB"))
                # Use path stem as a lightweight caption proxy if no caption store
                captions.append(p.stem.replace("_", " "))
                valid_indices.append(i)
            except Exception as e:
                logger.error("Failed to open candidate %s: %s", path, e)

        if not candidate_images:
            return list(range(len(candidate_paths)))

        scores = self._itm_scores(query_crop, candidate_images, captions)

        # sort valid indices by score descending
        ranked = sorted(zip(valid_indices, scores), key=lambda x: x[1], reverse=True)
        ranked_indices = [idx for idx, _ in ranked]

        # append any skipped indices at the end
        skipped = [i for i in range(len(candidate_paths)) if i not in valid_indices]
        return ranked_indices + skipped


if __name__ == "__main__":
    import numpy as np

    print("=== reranker.py smoke test ===")
    reranker = Reranker()

    dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    # config A — should return unchanged order without loading BLIP-ITM scoring
    result_a = reranker.rerank(dummy, ["a.jpg", "b.jpg", "c.jpg"], "A")
    assert result_a == [0, 1, 2], f"Config A should preserve order, got {result_a}"
    print(f"Config A (no rerank): {result_a} OK")
    print("Smoke test passed (BLIP-ITM scoring requires real images + GPU to fully verify)")


import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import config

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = None
        self._model = None

        if not config.ENABLE_RERANKER:
            logger.info("Reranker disabled — skipping BLIP-2 load")
            return

        logger.info("Loading BLIP-2: %s on %s", config.BLIP2_MODEL_NAME, self._device)
        self._processor = Blip2Processor.from_pretrained(config.BLIP2_MODEL_NAME)
        if self._device == "cuda":
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                config.BLIP2_MODEL_NAME,
                load_in_8bit=True,
                device_map="auto",
            )
        else:
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                config.BLIP2_MODEL_NAME,
                torch_dtype=torch.float32,
                device_map="auto",
            )
        self._model.eval()
        logger.info("BLIP-2 loaded on %s", self._device)

    @torch.no_grad()
    def _itm_scores(
        self,
        query_crop: Image.Image,
        candidate_images: list[Image.Image],
    ) -> list[float]:
        scores = []
        batch_size = config.RERANK_BATCH_SIZE

        for i in range(0, len(candidate_images), batch_size):
            batch = candidate_images[i: i + batch_size]
            # 8-bit model manages its own dtype — only move to device, no dtype cast
            inputs = self._processor(
                images=batch,
                return_tensors="pt",
            ).to(self._device)

            generated = self._model.generate(**inputs, max_new_tokens=20)
            captions = self._processor.batch_decode(generated, skip_special_tokens=True)

            query_inputs = self._processor(
                images=[query_crop] * len(batch),
                text=captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self._device)

            # use language model logits as a proxy ITM score
            outputs = self._model(**query_inputs, labels=query_inputs["input_ids"])
            # negative loss = higher is better
            batch_scores = [-outputs.loss.item()] * len(batch)
            scores.extend(batch_scores)

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
        valid_indices = []

        for i, path in enumerate(candidate_paths):
            p = Path(path)
            if not p.exists():
                logger.warning("Candidate crop not found, skipping: %s", path)
                continue
            try:
                candidate_images.append(Image.open(p).convert("RGB"))
                valid_indices.append(i)
            except Exception as e:
                logger.error("Failed to open candidate %s: %s", path, e)

        if not candidate_images:
            return list(range(len(candidate_paths)))

        scores = self._itm_scores(query_crop, candidate_images)

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

    # config A — should return unchanged order without loading BLIP-2 scoring
    result_a = reranker.rerank(dummy, ["a.jpg", "b.jpg", "c.jpg"], "A")
    assert result_a == [0, 1, 2], f"Config A should preserve order, got {result_a}"
    print(f"Config A (no rerank): {result_a} OK")
    print("Smoke test passed (BLIP-2 ITM scoring requires real images + GPU to fully verify)")

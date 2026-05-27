import json
import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def remap_crop_path(old_path: str) -> str:
    marker = "/crops/"
    idx = old_path.find(marker)
    if idx != -1:
        relative = old_path[idx + len(marker):]
        return str(config.CROPS_DIR / relative)
    return old_path

class ImageFetcher:
    def __init__(self) -> None:
        with open(config.IMAGE_PATHS_PATH) as f:
            raw = json.load(f)
        # normalise keys to int and remap paths eagerly
        self._image_paths: dict[int, dict] = {
            int(k): {
                "item_id": v["item_id"],
                "path": remap_crop_path(v["path"])
            }
            for k, v in raw.items()
        }

        self._captions: dict[str, str] = {}
        if config.CAPTIONS_PATH.exists():
            with open(config.CAPTIONS_PATH) as f:
                self._captions = json.load(f)
        else:
            logger.warning("captions.json not found — captions will be empty strings")

        logger.info(
            "ImageFetcher loaded: %d gallery entries, %d captions",
            len(self._image_paths),
            len(self._captions),
        )

    def fetch(self, labels: list[int]) -> list[dict]:
        """
        Resolve HNSW integer labels to image metadata.
        Returns list[dict] with keys: label, path, item_id, caption.
        Missing labels are skipped with a warning.
        """
        results = []
        for label in labels:
            entry = self._image_paths.get(label)
            if entry is None:
                logger.warning("Label %d not found in image_paths.json", label)
                continue
            item_id  = entry["item_id"]
            crop_key = f"{item_id}_{Path(entry['path']).stem}"
            results.append({
                "label":   label,
                "path":    entry["path"],
                "item_id": item_id,
                "caption": self._captions.get(crop_key, self._captions.get(item_id, "")),
            })
        return results


if __name__ == "__main__":
    print("=== image_fetcher.py smoke test ===")

    if not config.IMAGE_PATHS_PATH.exists():
        print(f"Skipping: image_paths.json not found at {config.IMAGE_PATHS_PATH}")
    else:
        fetcher = ImageFetcher()
        results = fetcher.fetch([0, 1, 2])
        for r in results:
            print(f"  label={r['label']} item_id={r['item_id']} caption={r['caption'][:60]!r}")
            print(f"    path={r['path']}")
        print("OK")

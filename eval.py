"""
Batch evaluation script — Recall@K, NDCG@K, mAP@K for K=5,10,15.

For each query image in the DeepFashion query split:
  1. Run YOLO crop via POST /crop
  2. Send top crop to POST /retrieve
  3. Compare returned item_ids against ground-truth item_ids for that item

Ground truth: all gallery images sharing the same item_id as the query are positives.
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import requests
from PIL import Image

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8504"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def crop_image(image_bytes: bytes, timeout: int = 30) -> list[dict]:
    resp = requests.post(
        f"{API_BASE}/crop",
        files={"file": ("query.jpg", image_bytes, "image/jpeg")},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["crops"]


def retrieve(crop_bytes: bytes, config_name: str, timeout: int = 60) -> list[dict]:
    resp = requests.post(
        f"{API_BASE}/retrieve",
        files={"file": ("crop.jpg", crop_bytes, "image/jpeg")},
        data={"config_name": config_name},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["results"]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def recall_at_k(ranked_item_ids: list[str], query_item_id: str, relevant_count: int, k: int) -> float:
    if not relevant_count:
        return 0.0
    hits = sum(1 for iid in ranked_item_ids[:k] if iid == query_item_id)
    return min(hits / relevant_count, 1.0)


def ndcg_at_k(ranked_item_ids: list[str], query_item_id: str, relevant_count: int, k: int) -> float:
    if not relevant_count:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, iid in enumerate(ranked_item_ids[:k])
        if iid == query_item_id
    )
    ideal_hits = min(relevant_count, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def ap_at_k(ranked_item_ids: list[str], query_item_id: str, relevant_count: int, k: int) -> float:
    if not relevant_count:
        return 0.0
    hits, precision_sum = 0, 0.0
    for i, iid in enumerate(ranked_item_ids[:k]):
        if iid == query_item_id:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / min(relevant_count, k)


# ---------------------------------------------------------------------------
# Build ground truth from image_paths.json
# ---------------------------------------------------------------------------

def build_ground_truth(image_paths: dict[int, dict]) -> dict[str, int]:
    """item_id → number of gallery images with that item_id."""
    from collections import defaultdict
    counts: dict[str, int] = defaultdict(int)
    for v in image_paths.values():
        counts[v["item_id"]] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Load query list from partition file
# ---------------------------------------------------------------------------

def load_query_entries(image_paths: dict[int, dict]) -> list[dict]:
    """
    Returns query-split entries from list_eval_partition.txt merged with image_paths.
    We use image_paths (gallery + train) and only include entries marked 'query'.
    """
    partition_path = config.LIST_EVAL_PARTITION
    if not partition_path.exists():
        raise FileNotFoundError(f"Partition file not found: {partition_path}")

    # build path → item_id map from image_paths for quick lookup
    path_to_meta: dict[str, dict] = {}
    for entry in image_paths.values():
        path_to_meta[entry["path"]] = entry

    queries = []
    with open(partition_path) as f:
        lines = f.readlines()[2:]  # skip header lines

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        rel_path, item_id, split = parts[0], parts[1], parts[2].lower()
        if split != "query":
            continue

        # resolve actual image path
        clean_rel = rel_path[4:] if rel_path.startswith("img/") else rel_path
        img_path = config.DATASET_IMAGES_DIR / clean_rel
        queries.append({"path": str(img_path), "item_id": item_id})

    return queries


# ---------------------------------------------------------------------------
# Remap crop path (same pattern as other scripts)
# ---------------------------------------------------------------------------

def remap_crop_path(stored: str) -> Path:
    marker = "/crops/"
    idx = stored.find(marker)
    if idx != -1:
        return config.CROPS_DIR / stored[idx + len(marker):]
    return Path(stored)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    cfg_name: str,
    k_values: list[int],
    max_queries: int = 0,
    skip_crop: bool = False,
) -> dict:
    logger.info("Loading image_paths.json …")
    with open(config.IMAGE_PATHS_PATH) as f:
        raw = json.load(f)
    image_paths: dict[int, dict] = {int(k): v for k, v in raw.items()}

    gt = build_ground_truth(image_paths)
    queries = load_query_entries(image_paths)

    if max_queries > 0:
        queries = queries[:max_queries]

    logger.info(
        "Evaluating config %s on %d queries (K=%s)",
        cfg_name, len(queries), k_values,
    )

    max_k = max(k_values)

    accumulators: dict[str, list[float]] = {
        f"recall@{k}": [] for k in k_values
    }
    accumulators.update({f"ndcg@{k}": [] for k in k_values})
    accumulators.update({f"map@{k}": [] for k in k_values})

    skipped = 0

    for q_idx, query in enumerate(queries):
        q_path = Path(query["path"])
        q_item_id = query["item_id"]
        relevant_count = gt.get(q_item_id, 1)

        if not q_path.exists():
            logger.warning("Query image not found, skipping: %s", q_path)
            skipped += 1
            continue

        try:
            image_bytes = q_path.read_bytes()

            if skip_crop:
                # use full image directly (faster for quick sanity checks)
                crop_bytes = image_bytes
            else:
                crops = crop_image(image_bytes)
                if not crops:
                    crop_bytes = image_bytes
                else:
                    import base64, io as _io
                    top_crop_b64 = crops[0]["image_b64"]
                    crop_img = Image.open(_io.BytesIO(base64.b64decode(top_crop_b64))).convert("RGB")
                    buf = _io.BytesIO()
                    crop_img.save(buf, format="JPEG", quality=85)
                    crop_bytes = buf.getvalue()

            results = retrieve(crop_bytes, cfg_name)

        except Exception as e:
            logger.error("Query %d failed: %s", q_idx, e)
            skipped += 1
            continue

        ranked_ids = [r["item_id"] for r in results[:max_k]]

        for k in k_values:
            accumulators[f"recall@{k}"].append(recall_at_k(ranked_ids, q_item_id, relevant_count, k))
            accumulators[f"ndcg@{k}"].append(ndcg_at_k(ranked_ids, q_item_id, relevant_count, k))
            accumulators[f"map@{k}"].append(ap_at_k(ranked_ids, q_item_id, relevant_count, k))

        if (q_idx + 1) % 50 == 0:
            logger.info("  %d / %d queries done (%d skipped)", q_idx + 1, len(queries), skipped)

    n = len(queries) - skipped
    if n == 0:
        logger.warning("No queries evaluated.")
        return {}

    metrics: dict[str, float] = {}
    for key, vals in accumulators.items():
        metrics[key] = round(sum(vals) / n, 4) if vals else 0.0

    metrics["queries_evaluated"] = n
    metrics["queries_skipped"]   = skipped
    metrics["config"]            = cfg_name

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval pipeline")
    parser.add_argument("--config",      default="A",    choices=["A", "B", "C"])
    parser.add_argument("--k",           action="append", type=int, default=None,
                        help="K value (repeatable, e.g. --k 5 --k 10 --k 15)")
    parser.add_argument("--max_queries", type=int, default=0,
                        help="Limit number of queries (0 = all)")
    parser.add_argument("--skip_crop",   action="store_true",
                        help="Skip YOLO crop step; send full query image directly")
    args = parser.parse_args()

    k_values = sorted(set(args.k)) if args.k else [5, 10, 15]

    t0 = time.time()
    metrics = evaluate(args.config, k_values, args.max_queries, args.skip_crop)
    elapsed = time.time() - t0

    if metrics:
        print("\n=== Evaluation Results ===")
        for k in k_values:
            print(
                f"  K={k:2d}  "
                f"Recall={metrics.get(f'recall@{k}', 0):.4f}  "
                f"NDCG={metrics.get(f'ndcg@{k}', 0):.4f}  "
                f"mAP={metrics.get(f'map@{k}', 0):.4f}"
            )
        print(f"\n  Queries evaluated: {metrics['queries_evaluated']}")
        print(f"  Queries skipped:   {metrics['queries_skipped']}")
        print(f"  Elapsed:           {elapsed:.1f}s")
        print(f"\n{metrics}")   # JSON-parseable block for Streamlit


if __name__ == "__main__":
    main()

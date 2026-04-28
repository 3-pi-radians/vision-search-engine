import logging
from pathlib import Path

import hnswlib
import numpy as np

import config

logger = logging.getLogger(__name__)


class HNSWSearch:
    def __init__(self, index: hnswlib.Index) -> None:
        self._index = index
        self._index.set_ef(config.HNSW_EF_SEARCH)

    @classmethod
    def load(cls, index_path: Path) -> "HNSWSearch":
        if not index_path.exists():
            raise FileNotFoundError(f"HNSW index not found at {index_path}")

        index = hnswlib.Index(space=config.HNSW_SPACE, dim=config.HNSW_DIM)
        index.load_index(str(index_path))
        logger.info("Loaded HNSW index from %s (%d items)", index_path, index.get_current_count())
        return cls(index)

    def search(self, query_emb: np.ndarray, k: int = config.TOP_K_RETRIEVAL) -> list[int]:
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        labels, _ = self._index.knn_query(query_emb, k=min(k, self._index.get_current_count()))
        return labels[0].tolist()

    @property
    def size(self) -> int:
        return self._index.get_current_count()


if __name__ == "__main__":
    print("=== hnsw_search.py smoke test ===")

    # build a tiny in-memory index to test search without a saved file
    dim = config.HNSW_DIM
    n = 200

    index = hnswlib.Index(space=config.HNSW_SPACE, dim=dim)
    index.init_index(max_elements=n, ef_construction=config.HNSW_EF_CONSTRUCTION, M=config.HNSW_M)

    vecs = np.random.rand(n, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index.add_items(vecs, list(range(n)))

    searcher = HNSWSearch(index)
    query = vecs[0]
    results = searcher.search(query, k=10)

    assert len(results) == 10, f"Expected 10 results, got {len(results)}"
    assert results[0] == 0, "Top result should be the query itself"
    print(f"Top-10 indices: {results}")
    print(f"Index size: {searcher.size}")
    print("OK")

import base64
import io
import logging
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

import config
from clip_encoder import CLIPEncoder
from detectors import DetectorFactory
from hnsw_search import HNSWSearch
from image_fetcher import ImageFetcher
from reranker import Reranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared app state
# ---------------------------------------------------------------------------

class AppState:
    detector:      object = None
    encoder:       CLIPEncoder = None
    indexes:       dict[str, HNSWSearch] = {}
    reranker:      Reranker = None
    fetcher:       ImageFetcher = None

state = AppState()


# ---------------------------------------------------------------------------
# Lifespan — load everything once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Loading models and artifacts ===")

    state.detector = DetectorFactory.get(config.DETECTOR)
    logger.info("Detector ready: %s", config.DETECTOR)

    state.encoder = CLIPEncoder()
    logger.info("CLIPEncoder ready")

    for cfg_name, index_path in config.HNSW_INDEX_PATHS.items():
        if index_path.exists():
            state.indexes[cfg_name] = HNSWSearch.load(index_path)
            logger.info("HNSW index loaded: config %s", cfg_name)
        else:
            logger.warning("HNSW index not found for config %s: %s", cfg_name, index_path)

    state.reranker = Reranker()
    logger.info("Reranker ready")

    if config.IMAGE_PATHS_PATH.exists():
        state.fetcher = ImageFetcher()
        logger.info("ImageFetcher ready")
    else:
        logger.warning("image_paths.json not found at %s — fetcher disabled", config.IMAGE_PATHS_PATH)

    logger.info("=== Startup complete ===")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Vision Search Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CropItem(BaseModel):
    index:      int
    bbox:       list[int]        # [x1, y1, x2, y2]
    confidence: float
    image_b64:  str              # base64-encoded JPEG


class CropResponse(BaseModel):
    crops:        list[CropItem]
    used_fallback: bool


class ResultItem(BaseModel):
    rank:     int
    label:    int
    item_id:  str
    caption:  str
    path:     str
    score:    float


class RetrieveResponse(BaseModel):
    config_name: str
    results:     list[ResultItem]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/crop", response_model=CropResponse)
async def crop_endpoint(file: Annotated[UploadFile, File()]):
    """
    Accept an uploaded image, run YOLO detection, return up to 5 crops.
    """
    image_data = await file.read()
    try:
        image = bytes_to_pil(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    result = state.detector.detect(image)

    crops = []
    for i, (crop, bbox, conf) in enumerate(
        zip(result.crops, result.bboxes, result.confidences)
    ):
        crops.append(CropItem(
            index=i,
            bbox=list(bbox),
            confidence=round(conf, 4),
            image_b64=pil_to_b64(crop),
        ))

    return CropResponse(crops=crops, used_fallback=result.used_fallback)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(
    file: Annotated[UploadFile, File()],
    config_name: Annotated[str, Form()] = "A",
):
    """
    Accept a cropped image + config name, return top-K ranked results.
    """
    if config_name not in config.CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown config '{config_name}'. Choose A, B or C.")

    if config_name not in state.indexes:
        raise HTTPException(status_code=503, detail=f"HNSW index for config {config_name} not loaded.")

    if state.fetcher is None:
        raise HTTPException(status_code=503, detail="image_paths.json not loaded — artifacts unavailable locally.")

    image_data = await file.read()
    try:
        crop = bytes_to_pil(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # 1. encode
    embedding = state.encoder.encode(crop, config_name=config_name)

    # 2. HNSW search — fetch more candidates than needed for re-ranking
    labels, distances = state.indexes[config_name].search(embedding, k=config.TOP_K_RETRIEVAL)

    # 3. fetch metadata for candidates
    candidates = state.fetcher.fetch(labels)
    candidate_paths = [c["path"] for c in candidates]

    # 4. re-rank
    ranked_indices = state.reranker.rerank(crop, candidate_paths, config_name)

    # 5. build final response (top K_RERANK)
    # hnswlib cosine space returns 1 - cosine_similarity as distance
    results = []
    for rank, idx in enumerate(ranked_indices[: config.TOP_K_RERANK]):
        c = candidates[idx]
        similarity = round(1.0 - distances[idx], 4)
        results.append(ResultItem(
            rank=rank + 1,
            label=c["label"],
            item_id=c["item_id"],
            caption=c["caption"],
            path=c["path"],
            score=similarity,
        ))

    return RetrieveResponse(config_name=config_name, results=results)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "indexes_loaded": list(state.indexes.keys()),
        "detector": config.DETECTOR,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host=config.API_HOST, port=config.API_PORT, reload=False)

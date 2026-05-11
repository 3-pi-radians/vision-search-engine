import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
KAGGLE = os.path.exists("/kaggle")

# ---------------------------------------------------------------------------
# Paths — switches automatically between Kaggle and local
# ---------------------------------------------------------------------------
if KAGGLE:
    DATASET_IMAGES_DIR  = Path("/kaggle/input/datasets/hserdaraltan/deepfashion-inshop-clothes-retrieval/img_highres")
    DATASET_ANNO_DIR    = Path("/kaggle/input/datasets/pankajdeopa/deepfashion-inshop-annotations")
    CAPTIONS_PATH       = Path("/kaggle/input/datasets/pankajdeopa/deepfashion-inshop-captions/captions.json")
    WORK_DIR            = Path("/kaggle/working")

    # READ paths — existing published datasets
    CROPS_DIR_INPUT     = Path("/kaggle/input/datasets/pankajdeopa/deepfashion-inshop-crops/crops")
    IMAGE_PATHS_INPUT   = Path("/kaggle/input/datasets/pankajdeopa/deepfashion-inshop-crops/image_paths.json")
    CLIP_WEIGHTS_INPUT  = Path("/kaggle/input/datasets/pankajdeopa/deepfashion-clip-weights")

    # WRITE paths — new artifacts go here
    CROPS_DIR           = WORK_DIR / "crops"
    IMAGE_PATHS_PATH    = WORK_DIR / "image_paths.json"
    CLIP_WEIGHTS_DIR    = WORK_DIR / "clip_weights"
else:
    DATASET_IMAGES_DIR  = Path("data/img_highres")
    DATASET_ANNO_DIR    = Path("data/annotations")
    CAPTIONS_PATH       = Path("data/captions.json")
    WORK_DIR            = Path("data/")
    CROPS_DIR_INPUT     = Path("data/crops")
    IMAGE_PATHS_INPUT   = Path("data/image_paths.json")
    CLIP_WEIGHTS_INPUT  = Path("data/clip_weights")
    CROPS_DIR           = Path("data/crops")
    IMAGE_PATHS_PATH    = Path("data/image_paths.json")
    CLIP_WEIGHTS_DIR    = Path("data/clip_weights")

LIST_EVAL_PARTITION = DATASET_ANNO_DIR / "list_eval_partition.txt"
LIST_BBOX_INSHOP    = DATASET_ANNO_DIR / "list_bbox_inshop.txt"
LIST_ITEM_INSHOP    = DATASET_ANNO_DIR / "list_item_inshop.txt"

HNSW_INDEX_PATHS = {
    "A": WORK_DIR / "hnsw_index_A.bin",
    "B": WORK_DIR / "hnsw_index_B.bin",
    "C": WORK_DIR / "hnsw_index_C.bin",
}

# ---------------------------------------------------------------------------
# YOLO detector
# ---------------------------------------------------------------------------
DETECTOR            = "yolov8m"
YOLO_CONF_THRESHOLD = 0.5
YOLO_MAX_DETECTIONS = 5

# ---------------------------------------------------------------------------
# BLIP-2 captioning (offline — run_blip2_caption.py)
# ---------------------------------------------------------------------------
BLIP2_MODEL_NAME         = "Salesforce/blip2-flan-t5-xl"  # structured caption generation
CAPTION_BATCH_SIZE       = 4                               # reduced for flan-t5-xl VRAM budget
BLIP2_NUM_BEAMS          = 3                               # beam search width
BLIP2_MAX_NEW_TOKENS     = 50                              # max caption length in tokens
BLIP2_MIN_LENGTH         = 10                              # prevents degenerate 1–2 word outputs
BLIP2_REPETITION_PENALTY = 1.2                            # discourages repeated attribute phrases

# BLIP-ITM reranker (online — reranker.py) — separate smaller model, purpose-built for scoring
BLIP2_RERANK_MODEL_NAME  = "Salesforce/blip-itm-base-coco"
RERANK_BATCH_SIZE        = 4
ENABLE_RERANKER          = KAGGLE

# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME          = "openai/clip-vit-base-patch16"
FINETUNE_UNFREEZE_BLOCKS = 6
FINETUNE_EPOCHS          = 10
FINETUNE_BATCH_SIZE      = 64
FINETUNE_LR              = 1e-5
FINETUNE_TEMPERATURE     = 0.05
FINETUNE_PATIENCE        = 3

# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------
CONFIGS = {
    "A": {"clip_finetuned": False, "use_captions": False, "alpha": 1.0},
    "B": {"clip_finetuned": False, "use_captions": True,  "alpha": 0.7},
    "C": {"clip_finetuned": True,  "use_captions": True,  "alpha": 0.7},
}

# ---------------------------------------------------------------------------
# HNSW index
# ---------------------------------------------------------------------------
HNSW_SPACE           = "cosine"
HNSW_DIM             = 512
HNSW_EF_CONSTRUCTION = 400
HNSW_M               = 32
HNSW_EF_SEARCH       = 100

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K           = 15
TOP_K_RETRIEVAL = 50
TOP_K_RERANK    = 15
ALPHA           = 0.7

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8504

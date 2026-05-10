import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
KAGGLE = os.path.exists("/kaggle")

# ---------------------------------------------------------------------------
# Paths — switches automatically between Kaggle and local
if KAGGLE:
    DATASET_IMAGES_DIR = Path("/kaggle/input/datasets/hserdaraltan/deepfashion-inshop-clothes-retrieval/img_highres")
    DATASET_ANNO_DIR   = Path("/kaggle/input/datasets/pankajdeopa/deepfashion-inshop-annotations")
    CAPTIONS_PATH      = Path("/kaggle/input/datasets/pankajdeopa/deepfashion-inshop-captions/captions.json")
    CLIP_WEIGHTS_DIR   = Path("/kaggle/input/datasets/pankajdeopa/deepfashion-clip-weights")
    WORK_DIR           = Path("/kaggle/working")
    CROPS_DIR          = WORK_DIR / "crops"                  # ← was input path, now writable
    IMAGE_PATHS_PATH   = WORK_DIR / "image_paths.json"       # ← was input path, now writable
else:
    DATASET_IMAGES_DIR = Path("data/img_highres")
    DATASET_ANNO_DIR   = Path("data/annotations")
    CROPS_DIR          = Path("data/crops")
    IMAGE_PATHS_PATH   = Path("data/image_paths.json")
    CAPTIONS_PATH      = Path("data/captions.json")
    CLIP_WEIGHTS_DIR   = Path("data/clip_weights")
    WORK_DIR           = Path("data/")

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
# BLIP-2
# ---------------------------------------------------------------------------
BLIP2_MODEL_NAME   = "Salesforce/blip-itm-base-coco"
CAPTION_BATCH_SIZE = 8
RERANK_BATCH_SIZE  = 4
# False on local (MacBook can't fit 16 GB model); auto-True on Kaggle GPU for eval
ENABLE_RERANKER    = KAGGLE

# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME          = "openai/clip-vit-base-patch32"
FINETUNE_UNFREEZE_BLOCKS = 4
FINETUNE_EPOCHS          = 5
FINETUNE_BATCH_SIZE      = 64
FINETUNE_LR              = 1e-5
FINETUNE_TEMPERATURE     = 0.07

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
HNSW_EF_CONSTRUCTION = 200
HNSW_M               = 16
HNSW_EF_SEARCH       = 50

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

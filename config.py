import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
KAGGLE = os.path.exists("/kaggle")

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
if KAGGLE:
    DATASET_IMAGES_DIR = Path("/kaggle/input/datasets/hserdaraltan/deepfashion-inshop-clothes-retrieval/img_highres")
    DATASET_ANNO_DIR   = Path("/kaggle/input/datasets/pankajdeopa/deepfashion-inshop-annotations")
else:
    DATASET_IMAGES_DIR = Path("data/img_highres")
    DATASET_ANNO_DIR   = Path("data/annotations")

LIST_EVAL_PARTITION = DATASET_ANNO_DIR / "list_eval_partition.txt"
LIST_BBOX_INSHOP    = DATASET_ANNO_DIR / "list_bbox_inshop.txt"
LIST_ITEM_INSHOP    = DATASET_ANNO_DIR / "list_item_inshop.txt"

# ---------------------------------------------------------------------------
# Output / artifact paths (writable)
# ---------------------------------------------------------------------------
WORK_DIR         = Path("/kaggle/working") if KAGGLE else Path("data/working")
CROPS_DIR        = Path("/kaggle/input/deepfashion-inshop-crops/crops") if KAGGLE else WORK_DIR / "crops"
IMAGE_PATHS_PATH = Path("/kaggle/input/deepfashion-inshop-crops/image_paths.json") if KAGGLE else WORK_DIR / "image_paths.json"

# ---------------------------------------------------------------------------
# YOLO detector
# ---------------------------------------------------------------------------
DETECTOR            = "yolov8s"
YOLO_CONF_THRESHOLD = 0.3
YOLO_MAX_DETECTIONS = 5    # cap for multi-item images; beyond this fall back to top-1

# ---------------------------------------------------------------------------
# BLIP-2 captioning
# ---------------------------------------------------------------------------
BLIP2_MODEL_NAME   = "Salesforce/blip2-opt-2.7b"
CAPTIONS_PATH      = WORK_DIR / "captions.json"
CAPTION_BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# CLIP fine-tuning (Config C)
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME          = "openai/clip-vit-base-patch32"
CLIP_WEIGHTS_DIR         = WORK_DIR / "clip_weights"
FINETUNE_UNFREEZE_BLOCKS = 4
FINETUNE_EPOCHS          = 5
FINETUNE_BATCH_SIZE      = 64
FINETUNE_LR              = 1e-5
FINETUNE_TEMPERATURE     = 0.07

# ---------------------------------------------------------------------------
# CLIP / embeddings
# ---------------------------------------------------------------------------
TOP_K  = 15    # max K for retrieval (eval uses 5, 10, 15)
ALPHA  = 0.7   # default alpha for configs B and C

# Ablation configs
CONFIGS = {
    "A": {"clip_finetuned": False, "use_captions": False, "alpha": 1.0},
    "B": {"clip_finetuned": False, "use_captions": True,  "alpha": 0.7},
    "C": {"clip_finetuned": True,  "use_captions": True,  "alpha": 0.7},
}

# ---------------------------------------------------------------------------
# HNSW index
# ---------------------------------------------------------------------------
HNSW_INDEX_PATHS = {
    "A": WORK_DIR / "hnsw_index_A.bin",
    "B": WORK_DIR / "hnsw_index_B.bin",
    "C": WORK_DIR / "hnsw_index_C.bin",
}
HNSW_SPACE            = "cosine"
HNSW_DIM              = 512   # CLIP ViT-B/32 embedding dimension
HNSW_EF_CONSTRUCTION  = 200
HNSW_M                = 16
HNSW_EF_SEARCH        = 50

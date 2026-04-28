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
CROPS_DIR        = WORK_DIR / "crops"
IMAGE_PATHS_PATH = WORK_DIR / "image_paths.json"   # index → crop path (gallery)

# ---------------------------------------------------------------------------
# YOLO detector
# ---------------------------------------------------------------------------
DETECTOR            = "yolov8s"
YOLO_CONF_THRESHOLD = 0.3
YOLO_MAX_DETECTIONS = 5    # cap for multi-item images; beyond this fall back to top-1

# ---------------------------------------------------------------------------
# CLIP / embeddings  (more settings added when CLIP script is written)
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
TOP_K           = 15       # max K for retrieval (eval uses 5, 10, 15)
ALPHA           = 0.7      # default alpha for configs B and C

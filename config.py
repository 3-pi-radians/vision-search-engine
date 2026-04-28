from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset paths (Kaggle input — read-only)
# ---------------------------------------------------------------------------
DATASET_IMAGES_DIR  = Path("/kaggle/input/datasets/deepfashion-inshop-clothes-retrieval/img")
DATASET_ANNO_DIR    = Path("/kaggle/input/datasets/deepfashion-inshop-annotations")

LIST_EVAL_PARTITION = DATASET_ANNO_DIR / "list_eval_partition.txt"
LIST_BBOX_INSHOP    = DATASET_ANNO_DIR / "list_bbox_inshop.txt"

# ---------------------------------------------------------------------------
# Output / artifact paths (Kaggle working dir — writable)
# ---------------------------------------------------------------------------
WORK_DIR         = Path("/kaggle/working")
CROPS_DIR        = WORK_DIR / "crops"
IMAGE_PATHS_PATH = WORK_DIR / "image_paths.json"

# ---------------------------------------------------------------------------
# YOLO detector
# ---------------------------------------------------------------------------
DETECTOR            = "yolov8s"
YOLO_CONF_THRESHOLD = 0.3

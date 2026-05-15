# Visual Product Search Engine

> Visual Recognition Course Project — IIIT Bangalore  
> Team: Chaitanya Nemade · Yuvraj Deshmukh · Pankaj Deopa · Sandiri Rohith

---

## Overview

A fashion image retrieval system built on the DeepFashion In-Shop Clothes Retrieval dataset. Users upload a clothing photo, confirm a YOLO-detected crop, and receive the top-K most visually similar items from a gallery of ~12,600 products — ranked by fused CLIP embeddings and re-ranked with BLIP-2 ITM scoring.

The system supports three ablation configurations (A, B, C) controlled by a single alpha parameter, enabling direct comparison of pretrained vs. fine-tuned CLIP and image-only vs. image+text retrieval.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     OFFLINE PIPELINE                        │
│  DeepFashion → YOLO Crop → BLIP-2 Caption → CLIP Encode   │
│              → HNSW Index Build (configs A / B / C)        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     ONLINE RUNTIME                          │
│  Streamlit UI ←→ FastAPI Backend                           │
│                   ├── POST /crop   (YOLO detection)        │
│                   └── POST /retrieve (CLIP + HNSW + BLIP2) │
└─────────────────────────────────────────────────────────────┘
```

### Ablation Configurations

| Config | CLIP                       | BLIP-2                | Alpha |
| ------ | -------------------------- | --------------------- | ----- |
| A      | Pretrained, vision only    | Not used              | 1.0   |
| B      | Frozen pretrained          | Frozen, captions used | 0.7   |
| C      | Fine-tuned (last 6 blocks) | Frozen, captions used | 0.7   |

### Fused Embedding

```
v = alpha × CLIP_image(crop) + (1 - alpha) × CLIP_text(caption)
```

---

## Repository Structure

```
vision-search-engine/
├── config.py                      # All paths, hyperparams, flags
├── offline/
│   ├── run_yolo_crop.py           # Step 1 — YOLO cropping
│   ├── run_blip2_caption.py       # Step 2 — BLIP-2 captioning
│   ├── run_clip_finetune.py       # Step 3 — CLIP fine-tuning
│   └── run_build_index.py         # Step 4 — HNSW index build
├── detectors/
│   ├── base_detector.py           # Abstract detector interface
│   ├── yolov8_detector.py         # YOLOv8s implementation
│   └── detector_factory.py        # Plug-and-play factory
├── clip_encoder.py                # Fused embedding (image + text)
├── hnsw_search.py                 # HNSW nearest-neighbor search
├── reranker.py                    # BLIP-2 ITM re-ranking
├── image_fetcher.py               # item_id → image path lookup
├── main.py                        # FastAPI app + endpoints
├── streamlit_app.py               # Streamlit UI
└── eval.py                        # Standalone batch evaluation
```

---

## Setup

### Prerequisites

```bash
pip install ultralytics hnswlib transformers accelerate fastapi uvicorn streamlit torch pillow
```

### Clone

```bash
git clone https://github.com/3-pi-radians/vision-search-engine.git
cd vision-search-engine
```

### Download Pre-built Artifacts (Local Setup)

To run the project locally without re-running the offline pipeline, download the pre-built artifacts from Kaggle:

```bash
mkdir -p data/clip_weights data/crops

# Indexes + image_paths.json (85 MB)
kaggle datasets download pankajdeopa/deepfashion-hnsw-indexes \
    -p data/ --unzip

# Captions (253 KB — instant)
kaggle datasets download pankajdeopa/deepfashion-inshop-captions \
    -p data/ --unzip

# CLIP weights (605 MB)
kaggle datasets download pankajdeopa/deepfashion-clip-weights \
    -p data/clip_weights/ --unzip

# Crops (2.4 GB)
kaggle datasets download pankajdeopa/deepfashion-inshop-crops \
    -p data/crops/ --unzip
```

### Local paths (edit config.py)

```python
KAGGLE = False
DATASET_IMAGES_DIR = Path("data/img_highres")
DATASET_ANNO_DIR   = Path("data/annotations")
CROPS_DIR          = Path("data/working/crops")
```

---

## Running the Offline Pipeline

> ⚠️ All offline steps were run on Kaggle GPU T4. Artifacts are saved as permanent Kaggle datasets. You do not need to re-run these unless reproducing from scratch.

Run in order — each step depends on the previous:

```bash
# Step 1 — YOLO cropping (~27 min on GPU T4)
python offline/run_yolo_crop.py

# Step 2 — BLIP-2 captioning (~45 min on GPU T4 x2)
python offline/run_blip2_caption.py

# Step 3 — CLIP fine-tuning (~16 min on GPU T4, 5 epochs)
python offline/run_clip_finetune.py

# Step 4 — Build HNSW indexes for configs A, B, C (~20 min on CPU)
python offline/run_build_index.py
```

All scripts are **resumable** — they skip already-processed items on re-run.

### Offline Artifacts Produced

| Artifact                         | Description                                                |
| -------------------------------- | ---------------------------------------------------------- |
| `crops/`                         | 38,494 YOLO-cropped images (12,612 gallery + 25,882 train) |
| `image_paths.json`               | `{index: {path, item_id}}` — gallery index mapping         |
| `captions.json`                  | `{item_id: caption}` — 3,985 BLIP-2 captions               |
| `clip_weights/clip_finetuned.pt` | Fine-tuned CLIP vision encoder (605 MB)                    |
| `hnsw_index_A.bin`               | HNSW index — pretrained CLIP, image only                   |
| `hnsw_index_B.bin`               | HNSW index — pretrained CLIP + captions                    |
| `hnsw_index_C.bin`               | HNSW index — fine-tuned CLIP + captions                    |

---

## Running the API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8504 --reload
```

### Endpoints

**POST /crop**

```
Input:  multipart image upload
Output: {bbox, confidence, crop_b64, detections[]}
```

**POST /retrieve**

```
Input:  {crop_b64, config: "A"|"B"|"C", alpha: float}
Output: [{item_id, image_path, score}, ...]
```

---

## Running the Streamlit UI

```bash
streamlit run streamlit_app.py --server.port 8502
```

The UI runs at `http://localhost:8502`. Make sure the FastAPI server is running first.

### UI Flow

1. Select ablation config (A / B / C) and alpha slider
2. Upload a clothing image
3. Confirm or reject the YOLO crop (if multiple items detected, select one)
4. View top-K results with item IDs and structured attribute tags
5. Optionally run batch evaluation mode

---

## Running Batch Evaluation

`eval.py` is standalone — no FastAPI or Streamlit required.

```bash
python eval.py --config A
python eval.py --config B
python eval.py --config C
```

---

## Results

Evaluated on the DeepFashion In-Shop Clothes Retrieval benchmark.  
Gallery: 12,612 images · 3,985 unique items · Query set: 14,218 images

| Config                     | R@5    | R@10   | R@15   | NDCG@5 | NDCG@10 | NDCG@15 | mAP@5  | mAP@10 | mAP@15 |
| -------------------------- | ------ | ------ | ------ | ------ | ------- | ------- | ------ | ------ | ------ |
| A (pretrained, image only) | 0.2392 | 0.2925 | 0.3259 | 0.2896 | 0.2950  | 0.3039  | 0.2189 | 0.2112 | 0.2124 |
| B (pretrained + captions)  | 0.2359 | 0.2907 | 0.3250 | 0.2841 | 0.2903  | 0.2996  | 0.2147 | 0.2073 | 0.2088 |
| C (fine-tuned + captions)  | 0.5909 | 0.6820 | 0.7273 | 0.6829 | 0.6916  | 0.7035  | 0.6095 | 0.6011 | 0.6051 |

_Config C evaluated on 14,218 queries (reranker disabled for eval speed)_

---

## Design Decisions

| Decision                     | Rationale                                                               |
| ---------------------------- | ----------------------------------------------------------------------- |
| Single FastAPI process       | Avoids inter-service overhead — modularization achieves same separation |
| No database                  | All storage fits in memory as Python dicts + binary index               |
| Plug-and-play detector       | Strategy pattern via BaseDetector — swap any YOLO variant in config.py  |
| Full image fallback          | Pipeline never errors on failed YOLO detection                          |
| Pre-stored captions          | BLIP-2 runs offline — no per-query caption generation at runtime        |
| Read-only storage at runtime | Eliminates concurrency issues for 2–3 simultaneous users                |

---

## Dataset

DeepFashion In-Shop Clothes Retrieval  
Ziwei Liu et al., CVPR 2016  
http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html

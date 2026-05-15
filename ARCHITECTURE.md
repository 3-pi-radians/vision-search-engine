# ARCHITECTURE.md
# Project file registry — purpose, design choices, tradeoffs

---

## config.py
**Purpose:** Single source of truth for all paths, model names, hyperparameters, and flags. Every module imports from here — no magic numbers elsewhere.

**Design choices:**
- Plain Python constants (no dataclass/pydantic) — dead simple to read and edit
- `KAGGLE = os.path.exists("/kaggle")` flag splits all paths into Kaggle vs local variants — both branches defined in one if/else block
- `BLIP2_MODEL_NAME` = captioning model (`blip2-opt-2.7b`); `BLIP2_RERANK_MODEL_NAME` = ITM reranker model (`blip-itm-base-coco`) — kept separate so each can be upgraded independently
- `ENABLE_RERANKER = KAGGLE` — auto-disables BLIP-ITM on local machines (~895 MB but still avoids loading on CPU); auto-enables on Kaggle GPU
- `RERANK_BATCH_SIZE = 4` — controls BLIP-ITM batching to avoid OOM during reranking
- Ablation configs A/B/C defined as a dict — adding a new config is one line
- `DETECTOR = "fashion"` — default detector; overridable at runtime via API form field or Streamlit sidebar
- `AVAILABLE_DETECTORS = ["fashion", "yolov8m", "custom"]` — allowlist surfaced in the UI radio
- `CUSTOM_YOLO_WEIGHTS_PATH = Path("yolov8s_prj1.pt")` — path to Project 1 fine-tuned weights loaded by `CustomYOLODetector`

**Tradeoffs:**
- No validation of paths at import time (avoids side effects); scripts check existence themselves

---

## detectors/base_detector.py
**Purpose:** Abstract base class defining the contract every detector must satisfy. Enables plug-and-play swapping via `DetectorFactory`.

**Design choices:**
- `DetectionResult` dataclass returns *lists* of crops/bboxes/confidences — supports multi-item detection
- `MAX_DETECTIONS = 5` cap defined here — beyond this, YOLO is likely over-detecting on a lookbook shot

**Tradeoffs:**
- List return type adds a tiny bit of complexity vs single-crop, but necessary for the multi-item UX flow

---

## detectors/yolov8_detector.py
**Purpose:** Person-detection detector using YOLOv8m. Used for studio shots where a single person bbox is the expected output. Falls back to the full image if no person is found above threshold.

**Design choices:**
- `classes=[0]` — restricts YOLO to the person class; avoids spurious object detections on accessories
- `iou=0.4` — relaxed NMS threshold to avoid collapsing overlapping person crops in crowded lookbook shots
- Confidence threshold: `config.YOLO_CONF_THRESHOLD = 0.4`
- Detections sorted by confidence descending, capped at `MAX_DETECTIONS`
- Min-area filter: crops smaller than 5% of image area are discarded (likely background noise)
- Bbox coords clamped to image bounds to prevent invalid crops
- Model loaded once in `__init__`, never per-call

**Tradeoffs:**
- Person bbox is a loose crop for fashion retrieval — `FashionYOLODetector` returns tighter garment-level crops; `yolov8m` is best for clean studio images

---

## detectors/fashion_yolo_detector.py
**Purpose:** Garment-level detector using `NovaAstro/YOLOv8m_fashion` (46 classes). Returns tight crops of individual clothing items — tops, bottoms, dresses, outerwear, skirts, shorts. Best detector for multi-item or lifestyle shots.

**Design choices:**
- Weights downloaded from HuggingFace Hub via `hf_hub_download` at first init; cached locally by HF thereafter
- `_GARMENT_CLASSES = [0..11, 22, 23]` — filters to 14 relevant garment classes, excludes shoes/bags/accessories
- Confidence threshold: `config.YOLO_CONF_THRESHOLD = 0.4`; min-area filter 3% of image area
- Falls back to full image if no garment detected above threshold

**Tradeoffs:**
- `hf_hub_download` adds a network round-trip on first load; unavoidable for HuggingFace-hosted weights

---

## detectors/custom_yolo_detector.py
**Purpose:** Fine-tuned YOLOv8s detector trained on Project 1 data. Detects 5 classes: short_sleeve_top, long_sleeve_top, trousers, shorts, skirt. Local weights from `yolov8s_prj1.pt`.

**Design choices:**
- Loads weights from `config.CUSTOM_YOLO_WEIGHTS_PATH` — path is configurable without touching this file
- No class filtering (`classes=None`) — model was trained on only 5 relevant garment classes
- Same confidence threshold and min-area filter as other detectors for consistency

**Tradeoffs:**
- Narrower class set than `FashionYOLODetector`; stronger recall on the specific garment types it was trained on

---

## detectors/detector_factory.py
**Purpose:** Single entry point to instantiate any detector by name. Decouples the codebase from concrete detector classes. Supports runtime detector switching per request.

**Design choices:**
- Lambda registry dict — adding a new detector is one line; lambdas defer imports until first use
- Per-name instance cache (`_instances[key]`) — each named detector is loaded once and reused; multiple detectors can coexist in memory simultaneously (needed for runtime switching)
- `_register()` called lazily on first `get()` call — avoids circular import at module load time

**Tradeoffs:**
- All instantiated detectors stay in memory for the process lifetime; with three detectors this is acceptable (~300–700 MB total across all three)

---

## detectors/__init__.py
**Purpose:** Makes `detectors/` a proper package and exposes a clean public API.

**Design choices:**
- Re-exports `BaseDetector`, `DetectionResult`, `DetectorFactory` so callers use `from detectors import ...`

---

## clip_encoder.py
**Purpose:** Encodes a cropped image (and optional caption) into a fused CLIP embedding using `v = alpha * image_emb + (1-alpha) * text_emb`. Supports all three ablation configs at runtime.

**Design choices:**
- Config is passed per `encode()` call (runtime-switchable), not locked at init
- Fine-tuned weights loaded lazily from `CLIP_WEIGHTS_DIR`; gracefully falls back to pretrained if missing
- Uses `model.visual_projection(model.vision_model(...).pooler_output)` explicitly — avoids `get_image_features()` which returns a model output object in some transformer versions

**Tradeoffs:**
- Loading two CLIP models (pretrained + finetuned) doubles GPU memory for config C; acceptable since only one is active per request

---

## hnsw_search.py
**Purpose:** Wraps the hnswlib index for nearest-neighbour search. Loads a pre-built index from disk at startup and serves top-K candidate integer labels for a query embedding.

**Design choices:**
- Uses hnswlib (not faiss) — lightweight, no CUDA-specific build requirements
- One `HNSWSearch` instance per config index file — search state isolated per config
- `load()` classmethod to instantiate from disk path

**Tradeoffs:**
- hnswlib is approximate — trades perfect recall for speed; acceptable for top-50 candidate retrieval

---

## reranker.py
**Purpose:** Re-ranks HNSW candidates using `Salesforce/blip-itm-base-coco` — a purpose-built Image-Text Matching (ITM) model. Only active for configs B and C on Kaggle GPU.

**Design choices:**
- `ENABLE_RERANKER = False` (local) → `__init__` returns immediately, zero RAM usage, `rerank()` is a passthrough
- `ENABLE_RERANKER = True` (Kaggle) → BLIP-ITM loaded in `float16` on CUDA / `float32` on CPU (~895 MB total)
- Config A returns candidates unchanged even when enabled — no BLIP-ITM call, zero overhead
- Uses `BlipForImageTextRetrieval` — returns a direct ITM probability score per (image, text) pair; no text generation or LM-loss proxy needed
- Caption proxy: path stem used as text input when no external caption store is available (e.g. local runs)
- Batched inference via `config.RERANK_BATCH_SIZE` to avoid OOM
- Skipped/missing crops appended at end rather than raising

**Tradeoffs:**
- Swapped from BLIP-2 OPT-2.7B (~15 GB) to BLIP-ITM-base-coco (~895 MB) — dramatically lower memory, purpose-built ITM head vs LM-loss proxy
- Path-stem caption proxy is a weak text signal; on Kaggle the real `captions.json` captions can be wired in for higher reranking quality

---

## offline/run_yolo_crop.py
**Purpose:** Offline step 1. Runs YOLO detection on every DeepFashion gallery + train image, saves crops to `CROPS_DIR`, writes `image_paths.json` mapping gallery index → `{path, item_id}`.

**Design choices:**
- Resumable — skips images whose crop already exists (Kaggle session timeout safe)
- Parses `list_eval_partition.txt` columns as `parts[0]=path, parts[1]=item_id, parts[2]=split`
- Strips leading `img/` from annotation paths before joining with `DATASET_IMAGES_DIR`
- `image_paths.json` stores `item_id` alongside path — ground truth key for eval

**Tradeoffs:**
- Crops both gallery and train splits in one run — train crops needed for CLIP fine-tuning positive pairs

---

## offline/run_blip2_caption.py
**Purpose:** Offline step 2. Generates one BLIP-2 caption per unique `item_id` from gallery crops using `blip2-opt-2.7b`, writes `captions.json: {item_id: caption_string}`.

**Design choices:**
- Model: `blip2-opt-2.7b` — OPT-based language head; runs unconditional image captioning (no text prompt)
- Reads source gallery metadata from `IMAGE_PATHS_INPUT` (Kaggle read path) — not the write-path `IMAGE_PATHS_PATH`
- **Deduplication:** one caption per `item_id`; picks the first crop encountered in `image_paths.json`
- `max_new_tokens=50` — sufficient for a short clothing description
- Checkpoints every 500 items to survive Kaggle session timeouts; resumable by loading existing `CAPTIONS_PATH`
- Keyed by `item_id` — reranker and eval look up by item_id

**Tradeoffs:**
- No category-specific prompting — model generates whatever it sees; captions may include person descriptions ("a woman wearing...")
- One caption per item (not per image) keeps `captions.json` compact and avoids redundancy at query time

---

## offline/run_clip_finetune.py
**Purpose:** Offline step 3. Fine-tunes the last 4 CLIP vision encoder blocks using image-only InfoNCE contrastive loss. Positive pairs = two crops of the same `item_id` from the train split.

**Design choices:**
- Freezes all layers except last 4 vision blocks + post_layernorm — 18.7% trainable params
- Symmetric InfoNCE loss — both anchor→positive and positive→anchor directions
- Checkpoint saved after every epoch — resumable if session times out
- Uses `model.visual_projection(model.vision_model(...).pooler_output)` to get tensors directly

**Tradeoffs:**
- Image-only contrastive (not image-text) — simpler, faster, sufficient for Config C ablation
- 5 epochs chosen; loss converged 0.9280 → 0.1496

---

## image_fetcher.py
**Purpose:** Resolves HNSW integer labels → image metadata (`path`, `item_id`, `caption`) needed by the API response and Streamlit results grid.

**Design choices:**
- Loads `image_paths.json` and `captions.json` once at init — read-only at runtime
- `remap_crop_path()` translates stale `/kaggle/working/crops/...` paths to current `CROPS_DIR`
- Missing labels skipped with a warning rather than raising

**Tradeoffs:**
- Entire `image_paths.json` held in memory (~12K entries, ~1.5 MB) — fast lookup, negligible footprint

---

## main.py
**Purpose:** FastAPI backend. Exposes `POST /crop`, `POST /retrieve`, and `GET /health`. All models and artifacts loaded once at startup via lifespan context.

**Design choices:**
- FastAPI lifespan for clean startup/shutdown — models never loaded inside request handlers
- `POST /crop` returns base64-encoded JPEG crops — no temp files, stateless
  - `detector_name` form field (default: `config.DETECTOR`) — allows runtime detector selection per request
  - Returns 400 for unknown detector names, 503 if the detector fails to load
- `POST /retrieve` pipeline: encode → HNSW search (top-50) → fetch metadata → rerank → return top-15
- `config_name` passed as form field alongside the crop image
- CORS middleware enabled for Streamlit ↔ FastAPI communication
- Graceful fallback if `image_paths.json` is missing at startup — logs warning, fetcher set to None; `/retrieve` returns 503 instead of crashing

**Tradeoffs:**
- Base64 encoding adds ~33% payload overhead vs multipart binary; acceptable for crop preview images
- Single `AppState` singleton — simple, but means server is stateful (one config loaded per process)

---

## streamlit_app.py
**Purpose:** Streamlit frontend. Two tabs: (1) Search — upload image → detector crop picker → retrieve results grid; (2) Compare Configs — same flow run against configs A, B, C simultaneously for side-by-side comparison.

**Design choices:**
- All API calls go through `api_crop()` / `api_retrieve()` helpers — single place to change API_BASE
- `api_crop(image_bytes, detector)` passes `detector_name` to `/crop` — detector is selected via a sidebar radio and forwarded per call
- Crop picker shows all ≤5 crops side-by-side; user clicks "Use crop #N" to set `session_state["selected_crop_idx"]`
- Results grid: 5 columns per row, reads image directly from `ResultItem.path`; gracefully shows placeholder div if path is missing
- All result images displayed using PIL contain canvas (white letterbox) — `thumbnail` + `Image.new("RGB", (W,H), (255,255,255))` + `paste` — prevents distortion at fixed grid dimensions
- **Caption display:** `render_caption_tags()` parses structured `"attr: value, attr: value"` captions into inline badge pills (values only, max 6); falls back to plain italic text for old-format captions without `:`; uses `html.escape()` to prevent injection
- **Stale state clearing:** session state keys `crop_data`, `selected_crop_idx`, `results_data`, `search_elapsed` are wiped when a new filename is detected or the uploader is cleared
- Sidebar detector radio exposes `AVAILABLE_DETECTORS` with one-line captions describing each model's intended use

**Tradeoffs:**
- Images shown from local `path` — works on Kaggle notebooks; on a remote deployment the images won't be accessible (would need a `/image` endpoint)
- Compare Configs tab runs three sequential API calls — no parallelism; latency triples but keeps code simple

---

## eval.py
**Purpose:** Batch evaluation script. For each DeepFashion query image: calls `/crop` → `/retrieve` → compares returned `item_id`s against ground-truth. Computes Recall@K, NDCG@K, mAP@K for configurable K values (default 5, 10, 15).

**Design choices:**
- Ground truth: all gallery images sharing the same `item_id` as the query are positives (standard DeepFashion retrieval protocol)
- Queries loaded from `list_eval_partition.txt` (split=`query`), same path-stripping as `run_yolo_crop.py`
- `--skip_crop` flag bypasses YOLO for quick sanity runs (sends full query image)
- Prints final metrics dict as a raw Python repr — Streamlit eval tab parses the last `{...}` block as JSON
- `--max_queries N` for quick smoke tests without running all ~14K query images

**Tradeoffs:**
- Calls live API (not offline) — requires `main.py` to be running; simplifies code by reusing the same pipeline
- No multiprocessing — sequential query loop; acceptable since bottleneck is GPU inference in the API server

---

## offline/run_build_index.py
**Purpose:** Offline step 4. Generates fused CLIP embeddings for all gallery crops and builds one HNSW index per ablation config (A, B, C). Saves index `.bin` files to `WORK_DIR`.

**Design choices:**
- Skips any config whose index file already exists — safe to re-run
- Frees GPU memory between configs (`del model; torch.cuda.empty_cache()`) to avoid OOM
- `remap_crop_path()` applied when opening crops
- Falls back to `"a clothing item"` caption if `item_id` not in `captions.json`

**Tradeoffs:**
- Builds all 3 indexes in one run — simpler orchestration vs separate scripts per config

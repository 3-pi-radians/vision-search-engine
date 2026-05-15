# ARCHITECTURE.md
# Project file registry — purpose, design choices, tradeoffs

---

## config.py
**Purpose:** Single source of truth for all paths, model names, hyperparameters, and flags. Every module imports from here — no magic numbers elsewhere.

**Design choices:**
- Plain Python constants (no dataclass/pydantic) — dead simple to read and edit
- `KAGGLE = os.path.exists("/kaggle")` flag splits all paths into Kaggle vs local variants — both branches defined in one if/else block
- `BLIP2_MODEL_NAME` = captioning model (`blip2-flan-t5-xl`); `BLIP2_RERANK_MODEL_NAME` = ITM reranker model (`blip-itm-base-coco`) — kept separate so upgrading one doesn't break the other
- `ENABLE_RERANKER = KAGGLE` — auto-disables BLIP-ITM on local machines (~895 MB but still avoids loading on CPU); auto-enables on Kaggle GPU
- `RERANK_BATCH_SIZE = 4` — controls BLIP-ITM batching to avoid OOM during reranking
- Ablation configs A/B/C defined as a dict — adding a new config is one line

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
**Purpose:** Concrete YOLOv8 implementation of `BaseDetector`. Runs inference, returns top-N crops sorted by confidence, falls back to full image if nothing meets the threshold.

**Design choices:**
- Model loaded once in `__init__`, never per-call
- Capped at `MAX_DETECTIONS`; if >5 boxes detected, collapses to top-1 (likely over-detection)
- Bbox coords clamped to image bounds to prevent invalid crops

**Tradeoffs:**
- YOLOv8s chosen (small) — fast on CPU, sufficient for fashion detection

---

## detectors/detector_factory.py
**Purpose:** Single entry point to instantiate any detector by name. Decouples the codebase from concrete detector classes.

**Design choices:**
- Registry dict over if/elif chain — adding a new detector is one line
- Singleton pattern — `DetectorFactory.get()` always returns the same instance, preventing double model loads

**Tradeoffs:**
- Singleton means detector config is fixed at first call; acceptable since detector is set via `config.DETECTOR`

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

## offline/blip2_prompts.py
**Purpose:** Stores all 17 category-specific BLIP-2 prompts and a default fallback. Kept separate from `config.py` to avoid bloating the config with long strings.

**Design choices:**
- Keys match DeepFashion folder names exactly (`Blouses_Shirts`, `Denim`, `Shirts_Polos`, etc.) — looked up by `extract_category()` at runtime
- Covers all unique category folder names in the dataset (17 unique names across WOMEN + MEN folders)
- **v4 sentence format:** each prompt asks the model to *describe the garment in one concise sentence* listing visible attributes (color, fit, neckline, etc.) and to omit unclear ones — no structured templates, no placeholders
- `DEFAULT_PROMPT` catches any unknown/future category gracefully
- Exports `get_prompt(category: str) -> str` — callers use this instead of importing `CATEGORY_PROMPTS` directly

**Prompt evolution (why v4):**
- v1: `attribute: value, attribute: value` format — model copied the template text literally instead of filling values
- v2: `[value]` placeholder format — same problem, model output `[color]` verbatim
- v4: free-text sentence — model generates natural descriptions; `render_caption_tags()` badge pills in Streamlit are dormant for v4 captions (no `:` separator) and fall back to plain italic display

**Tradeoffs:**
- Free-text sentences are less parse-friendly than structured key:value, but produce far more accurate captions from the model
- Prompt strings are long — separating from `config.py` keeps config readable and prompts independently editable

---

## offline/run_blip2_caption.py
**Purpose:** Offline step 2. Generates one structured BLIP-2 caption per unique `item_id` from gallery crops using `blip2-flan-t5-xl`, writes `captions.json: {item_id: caption_string}`.

**Design choices:**
- Model: `blip2-flan-t5-xl` — encoder-decoder with flan-t5 language head; produces richer, more controllable structured output than OPT-based variants
- Reads source gallery metadata from `IMAGE_PATHS_INPUT` (Kaggle read path) — not the write-path `IMAGE_PATHS_PATH`; output written to `WORK_DIR/captions.json`
- **Front-view preference:** groups all crops per `item_id`, selects the one with `01_1_front` in filename; falls back to first available — ensures consistent viewpoint across all items
- **Category extraction:** parses `gallery/WOMEN|MEN/<Category>/` from crop path at runtime; calls `get_prompt(path)` from `blip2_prompts.py`; falls back to `DEFAULT_PROMPT` for unknown categories
- **Generation params** all driven by config: `num_beams=3`, `max_new_tokens=50`, `min_length=10`, `repetition_penalty=1.2`
- **Post-processing:** `_PERSON_RE` regex strips leading person-reference prefixes ("a woman wearing", "a man in", etc.) before storing; captions under 6 words are logged as warnings but stored as-is (partial caption > missing caption)
- Output format: free-text sentence (e.g. "A red floral midi dress with a v-neckline and short sleeves.") — Streamlit falls back to plain italic display since v4 captions contain no `:` separator
- Keyed by `item_id` — reranker and eval look up by item_id
- Checkpoints every 500 items to survive Kaggle session timeouts; resumable by loading existing `CAPTIONS_PATH`

**Tradeoffs:**
- Per-item prompts differ across a batch — processor handles padding to longest prompt, slight throughput reduction vs uniform prompts; `CAPTION_BATCH_SIZE=4` mitigates VRAM pressure
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
- `POST /retrieve` pipeline: encode → HNSW search (top-50) → fetch metadata → rerank → return top-15
- `config_name` passed as form field alongside the crop image
- CORS middleware enabled for Streamlit ↔ FastAPI communication
- Graceful fallback if `image_paths.json` is missing at startup — logs warning, fetcher set to None; `/retrieve` returns 503 instead of crashing

**Tradeoffs:**
- Base64 encoding adds ~33% payload overhead vs multipart binary; acceptable for crop preview images
- Single `AppState` singleton — simple, but means server is stateful (one config loaded per process)

---

## streamlit_app.py
**Purpose:** Streamlit frontend. Two tabs: (1) Search — upload image → YOLO crop picker → retrieve results grid; (2) Batch Eval — launches `eval.py` subprocess and displays metrics.

**Design choices:**
- All API calls go through `api_crop()` / `api_retrieve()` helpers — single place to change API_BASE
- Crop picker shows all ≤5 crops side-by-side; user clicks "Use crop #N" to set `session_state["selected_crop_idx"]`
- Results grid: 5 columns per row, reads image directly from `ResultItem.path`; gracefully shows placeholder div if path is missing
- **Caption display:** `render_caption_tags()` parses structured `"attr: value, attr: value"` captions into inline badge pills (values only, max 6); falls back to plain italic text for old-format captions without `:`; uses `html.escape()` to prevent injection
- **Stale state clearing:** session state keys `crop_data`, `selected_crop_idx`, `results_data`, `search_elapsed` are wiped when a new filename is detected or the uploader is cleared
- Batch eval launches `eval.py` via subprocess so the Streamlit process never blocks on long eval loops
- Attempts to parse a JSON block from eval stdout for structured metric display

**Tradeoffs:**
- subprocess approach for eval means no live progress bar; acceptable since eval is a one-shot operation
- Images shown from local `path` — works on Kaggle notebooks; on a remote deployment the images won't be accessible (would need a `/image` endpoint)
- Score bar uses rank-based proxy (rank 1 ≈ 100%, rank 15 ≈ 9%) since the API response does not return raw similarity scores

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

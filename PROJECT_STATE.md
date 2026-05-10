# PROJECT_STATE.md

## Last Updated
2026-05-11

## Completed Files
- [x] PROJECT_STATE.md — living progress tracker
- [x] config.py — canonical Kaggle + local paths, all hyperparams, ablation configs A/B/C; BLIP2_MODEL_NAME (captioning) + BLIP2_RERANK_MODEL_NAME (ITM) split
- [x] detectors/base_detector.py — ABC with detect() contract; DetectionResult returns list of crops
- [x] detectors/yolov8_detector.py — YOLOv8 with multi-crop (≤5) + confidence fallback
- [x] detectors/detector_factory.py — singleton factory; swap detector via config.DETECTOR
- [x] detectors/__init__.py — package init
- [x] clip_encoder.py — CLIPEncoder, runtime-switchable config A/B/C, fused embedding
- [x] hnsw_search.py — HNSWSearch wrapping hnswlib; load from disk, search → top-K label IDs
- [x] offline/run_yolo_crop.py — YOLO cropping Gallery+Train, resumable, writes image_paths.json
- [x] offline/run_blip2_caption.py — blip2-flan-t5-xl, category-aware prompts, front-view preference, post-processing
- [x] offline/blip2_prompts.py — 17 category-specific structured prompts + default fallback
- [x] offline/run_clip_finetune.py — CLIP fine-tune last 4 blocks, InfoNCE, resumable per epoch
- [x] offline/run_build_index.py — fused embeddings + HNSW index for A/B/C, remap_crop_path applied

## In Progress
_(none)_

## Pending Kaggle Re-runs (required after BLIP-2 upgrade)
1. **run_blip2_caption.py** — re-run on Kaggle (GPU T4 x2, ~60 min)
   - Attach: deepfashion-inshop-crops (read)
   - Publish output `/kaggle/working/captions.json` as new version of: `deepfashion-inshop-captions`
2. **run_build_index.py** — re-run after new captions published
   - Configs B and C use captions in fused embeddings — all 3 indexes must be rebuilt
   - Publish new `hnsw_index_A/B/C.bin` as new version of: `deepfashion-hnsw-indexes`

## Remaining Files
_(none — all files complete)_

## Archive: Completed Remaining Files
- [x] reranker.py — BLIP-2 ITM re-ranker; no-op for config A, batched scoring for B/C
- [x] image_fetcher.py — resolves HNSW labels → {path, item_id, caption}; remap_crop_path applied
- [x] main.py — FastAPI app; POST /crop, POST /retrieve, GET /health; all models loaded via lifespan
- [x] streamlit_app.py — Streamlit UI; tab 1 search (crop picker + results grid), tab 2 batch eval; structured caption tag badges
- [x] eval.py — batch eval; Recall@K, NDCG@K, mAP@K; calls /crop + /retrieve API per query

## Jira Status
- SCRUM-15 Offline Indexing Pipeline epic: ✅ DONE
- SCRUM-19 Dataset setup: ✅ DONE
- SCRUM-20 YOLO cropping: ✅ DONE
- SCRUM-21 BLIP-2 captioning: ✅ DONE
- SCRUM-22 CLIP fine-tuning: ✅ DONE (subtasks 35, 36, 37 ✅)
- SCRUM-23 HNSW index build: ✅ DONE
- SCRUM-24 to 27 Backend modules (Yuvraj): ✅ DONE
- SCRUM-28 to 31 FastAPI server (Chaitanya): ✅ DONE
- SCRUM-32 to 34 Streamlit UI + Eval (Rohith): ✅ DONE

## Dataset Status
- DeepFashion downloaded: YES
- YOLO crops generated: YES — 12,612 gallery + 25,882 train images
- BLIP-2 captions generated: YES — 3,985 unique item_ids
- CLIP fine-tuned: YES — 5 epochs, loss 0.9280 → 0.1496
- HNSW index built: YES — A/B/C indexes (~27.7 MB each)

## Kaggle Datasets (all permanent, owned by pankajdeopa)
| Slug | Contents |
|------|---------|
| deepfashion-inshop-crops | crops/ + image_paths.json (2.4 GB) |
| deepfashion-inshop-captions | captions.json (3,985 item_ids) |
| deepfashion-clip-weights | clip_finetuned.pt (605 MB) |
| deepfashion-inshop-annotations | list_eval_partition.txt + 2 files |
| deepfashion-hnsw-indexes | hnsw_index_A/B/C.bin + image_paths.json |

Third-party images (hserdaraltan): img_highres/

## Key Facts
- Gallery: 12,612 images, 3,985 unique item_ids (avg 3.2 images/item)
- Train: 25,882 images, 12,163 positive pairs, 3,997 items
- list_eval_partition.txt: parts[0]=path, parts[1]=item_id, parts[2]=split
- image_paths.json schema: {"0": {"path": "...", "item_id": "id_XXXXX"}, ...}
- captions.json schema: {"id_XXXXX": "caption text", ...}
- CROP PATH REMAPPING: image_paths.json stores stale /kaggle/working/crops/... paths.
  All scripts use remap_crop_path() to resolve to config.CROPS_DIR at runtime.

## Key Decisions Made
- Multi-item detection: YOLO returns ≤5 crops; Streamlit shows side-by-side picker
- YOLO fallback threshold: 0.3
- Alpha for configs B and C: 0.7
- HNSW: cosine distance, dim=512 (CLIP ViT-B/32)
- Captions keyed by item_id (not index) for reranker/eval compatibility
- CLIP embedding: explicit vision_model().pooler_output → visual_projection

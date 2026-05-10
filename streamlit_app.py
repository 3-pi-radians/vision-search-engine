"""
Streamlit frontend for the Vision Search Engine.

Tab 1 — Search: upload image → YOLO crop picker → retrieve results grid
Tab 2 — Batch Eval: run recall/NDCG/mAP evaluation against the gallery
"""

import base64
import io
import time
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8504"
CONFIGS  = ["A", "B", "C"]
CONFIG_LABELS = {
    "A": "A — Pretrained CLIP, image-only",
    "B": "B — Pretrained CLIP + captions (α=0.7)",
    "C": "C — Fine-tuned CLIP + captions (α=0.7)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=85)
    return buf.getvalue()


def api_crop(image_bytes: bytes) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/crop",
            files={"file": ("upload.jpg", image_bytes, "image/jpeg")},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Crop request failed: {e}")
        return None


def api_retrieve(crop_bytes: bytes, config_name: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/retrieve",
            files={"file": ("crop.jpg", crop_bytes, "image/jpeg")},
            data={"config_name": config_name},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Retrieve request failed: {e}")
        return None


def api_health() -> dict | None:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Vision Search Engine",
    page_icon="👗",
    layout="wide",
)

st.title("👗 Vision Search Engine")

# Sidebar: config + health
with st.sidebar:
    st.header("Settings")
    selected_config = st.radio(
        "Ablation config",
        options=CONFIGS,
        format_func=lambda c: CONFIG_LABELS[c],
        index=0,
    )
    st.divider()
    if st.button("Check API health"):
        health = api_health()
        if health:
            st.success(f"API up — indexes: {health.get('indexes_loaded', [])}")
        else:
            st.error("API unreachable")

tab_search, tab_eval = st.tabs(["🔍 Search", "📊 Batch Eval"])


# ===========================================================================
# Tab 1 — Search
# ===========================================================================

with tab_search:
    st.subheader("Upload a fashion image")
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "webp"], key="search_upload"
    )

    if uploaded is not None:
        image_bytes = uploaded.read()
        st.image(image_bytes, caption="Uploaded image", width=300)

        # --- Step 1: crop ---
        if st.button("Detect clothing items", key="btn_crop"):
            with st.spinner("Running YOLO detection…"):
                crop_data = api_crop(image_bytes)

            if crop_data:
                st.session_state["crop_data"] = crop_data
                st.session_state["selected_crop_idx"] = 0

        # --- Step 2: pick crop ---
        if "crop_data" in st.session_state and st.session_state["crop_data"]:
            crop_data = st.session_state["crop_data"]
            crops = crop_data["crops"]
            used_fallback = crop_data.get("used_fallback", False)

            if used_fallback:
                st.info("No clothing detected — using full image as crop.")

            st.markdown(f"**{len(crops)} crop(s) detected.** Select one to search:")

            cols = st.columns(min(len(crops), 5))
            for i, crop_item in enumerate(crops):
                crop_img = b64_to_pil(crop_item["image_b64"])
                label = f"#{i+1}  conf={crop_item['confidence']:.2f}"
                with cols[i]:
                    st.image(crop_img, caption=label, use_container_width=True)
                    if st.button(f"Use crop #{i+1}", key=f"pick_crop_{i}"):
                        st.session_state["selected_crop_idx"] = i

            sel_idx = st.session_state.get("selected_crop_idx", 0)
            selected_crop = b64_to_pil(crops[sel_idx]["image_b64"])
            selected_bytes = pil_to_bytes(selected_crop)

            st.markdown(f"**Selected: crop #{sel_idx+1}**")
            st.image(selected_crop, width=200)

            # --- Step 3: retrieve ---
            if st.button("Search similar items", key="btn_retrieve"):
                with st.spinner(f"Searching with config {selected_config}…"):
                    t0 = time.time()
                    results_data = api_retrieve(selected_bytes, selected_config)
                    elapsed = time.time() - t0

                if results_data:
                    st.session_state["results_data"] = results_data
                    st.session_state["search_elapsed"] = elapsed

        # --- Step 4: show results ---
        if "results_data" in st.session_state:
            results_data = st.session_state["results_data"]
            elapsed     = st.session_state.get("search_elapsed", 0)
            results     = results_data.get("results", [])
            cfg_used    = results_data.get("config_name", "?")

            st.divider()
            st.markdown(
                f"**Top {len(results)} results** — config **{cfg_used}**"
                f"  ·  {elapsed:.2f}s"
            )

            COLS_PER_ROW = 5
            for row_start in range(0, len(results), COLS_PER_ROW):
                row_items = results[row_start: row_start + COLS_PER_ROW]
                cols = st.columns(len(row_items))
                for col, item in zip(cols, row_items):
                    with col:
                        img_path = Path(item["path"])
                        if img_path.exists():
                            st.image(str(img_path), use_container_width=True)
                        else:
                            st.markdown("_(image not found)_")
                        st.caption(
                            f"**#{item['rank']}** `{item['item_id']}`\n\n"
                            f"{item['caption'][:80] if item['caption'] else '—'}"
                        )


# ===========================================================================
# Tab 2 — Batch Eval
# ===========================================================================

with tab_eval:
    st.subheader("Batch Evaluation")
    st.markdown(
        "Runs the offline `eval.py` script (Recall@K, NDCG@K, mAP@K for K=5,10,15) "
        "against the full query split. This requires the API to be running."
    )

    eval_config = st.selectbox(
        "Config to evaluate",
        options=CONFIGS,
        format_func=lambda c: CONFIG_LABELS[c],
        key="eval_config",
    )
    k_values = st.multiselect(
        "K values", options=[5, 10, 15], default=[5, 10, 15], key="eval_k"
    )
    max_queries = st.number_input(
        "Max queries (0 = all)", min_value=0, value=0, step=10, key="eval_max_q"
    )

    if st.button("Run evaluation", key="btn_eval"):
        import subprocess, sys, json as _json

        args = [sys.executable, "eval.py", "--config", eval_config]
        for k in sorted(k_values):
            args += ["--k", str(k)]
        if max_queries > 0:
            args += ["--max_queries", str(max_queries)]

        with st.spinner(f"Evaluating config {eval_config}… (may take several minutes)"):
            try:
                proc = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )
                stdout = proc.stdout.strip()
                stderr = proc.stderr.strip()

                if proc.returncode == 0:
                    st.success("Evaluation complete.")
                    # try to parse JSON result block
                    try:
                        json_start = stdout.rfind("{")
                        if json_start != -1:
                            metrics = _json.loads(stdout[json_start:])
                            st.json(metrics)
                        else:
                            st.text(stdout)
                    except Exception:
                        st.text(stdout)
                else:
                    st.error("Evaluation failed.")
                    st.text(stderr or stdout)

            except subprocess.TimeoutExpired:
                st.error("Evaluation timed out (30 min limit).")
            except Exception as e:
                st.error(f"Failed to launch eval: {e}")

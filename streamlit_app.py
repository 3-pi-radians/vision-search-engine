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
CONFIG_ALPHA  = {"A": 1.0, "B": 0.7, "C": 0.7}
CONFIG_COLORS = {"A": "#4A90D9", "B": "#27AE60", "C": "#E67E22"}
GALLERY_SIZE  = 12612

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
# Page config + CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Visual Fashion Search",
    page_icon="👗",
    layout="wide",
)

st.markdown("""
<style>
/* Fixed-size result images */
[data-testid="stImage"] img {
    width: 100% !important;
    height: 220px !important;
    object-fit: cover !important;
    border-radius: 8px !important;
}

/* Hide fullscreen/zoom button on images */
button[title="View fullscreen"] {
    display: none !important;
}

/* Result card */
.result-card {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 12px;
    border: 1px solid #2d2d4e;
    margin-top: 6px;
    transition: border-color 0.2s;
}
.result-card:hover {
    border-color: #5a5aae;
}
.rank-badge {
    display: inline-block;
    font-size: 1.05em;
    font-weight: 700;
    color: #a78bfa;
    margin: 4px 0 2px;
}
.item-id {
    font-family: monospace;
    font-size: 0.76em;
    color: #9ca3af;
    margin-bottom: 4px;
    word-break: break-all;
}
.score-bar-wrap {
    background: #2d2d4e;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.score-bar-fill {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
}
.caption-text {
    font-style: italic;
    color: #6b7280;
    font-size: 0.76em;
    line-height: 1.4;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

/* Upload hint */
.upload-hint {
    border: 2px dashed #4a5568;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    color: #9ca3af;
    margin-bottom: 10px;
    font-size: 0.9em;
}

/* Config badge */
.config-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.85em;
    margin-left: 6px;
}

/* Results header */
.results-header {
    font-size: 1.2em;
    font-weight: 700;
    color: #e2e8f0;
    margin: 12px 0 8px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

_, col_title, _ = st.columns([1, 3, 1])
with col_title:
    st.markdown("""
    <div style="text-align:center; padding:16px 0 8px;">
        <h1 style="margin:0; font-size:2.2em;">👗 Visual Fashion Search</h1>
        <p style="color:#9ca3af; margin:4px 0 0; font-size:1em;">
            Powered by CLIP &nbsp;·&nbsp; BLIP-2 &nbsp;·&nbsp; HNSW
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    selected_config = st.radio(
        "Ablation config",
        options=CONFIGS,
        format_func=lambda c: CONFIG_LABELS[c],
        index=0,
    )

    color = CONFIG_COLORS[selected_config]
    st.markdown(
        f'<span class="config-badge" style="background:{color}22; color:{color}; '
        f'border:1px solid {color}55;">Config {selected_config} active</span>',
        unsafe_allow_html=True,
    )

    if selected_config in ("B", "C"):
        st.slider(
            "Alpha (image weight)",
            min_value=0.0,
            max_value=1.0,
            value=CONFIG_ALPHA[selected_config],
            step=0.05,
            disabled=True,
            help="α controls image vs. caption weight in the fused embedding. Fixed by config.",
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

    # Upload section
    st.markdown(
        '<div class="upload-hint">Upload a fashion photo — JPG, PNG, or WebP</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="search_upload",
        label_visibility="collapsed",
    )

    # Clear all results when the file uploader is cleared (X button clicked)
    if uploaded is None and st.session_state.get("last_uploaded_filename"):
        for key in ("crop_data", "selected_crop_idx", "results_data", "search_elapsed", "last_uploaded_filename"):
            st.session_state.pop(key, None)

    # BUG 1 & 2: clear all stale state when a new file is uploaded
    if uploaded is not None:
        if uploaded.name != st.session_state.get("last_uploaded_filename"):
            for key in ("crop_data", "selected_crop_idx", "results_data", "search_elapsed"):
                st.session_state.pop(key, None)
            st.session_state["last_uploaded_filename"] = uploaded.name

        image_bytes = uploaded.read()
        file_kb = len(image_bytes) / 1024
        size_str = f"{file_kb:.1f} KB" if file_kb < 1024 else f"{file_kb / 1024:.1f} MB"
        st.markdown(f"**{uploaded.name}** &nbsp;·&nbsp; {size_str}")

        col_prev, _ = st.columns([1, 3])
        with col_prev:
            st.image(image_bytes, caption="Uploaded image", use_container_width=True)

        # --- Step 1: detect crop ---
        if st.button("Detect clothing items", key="btn_crop"):
            with st.spinner("Detecting clothing item..."):
                crop_data = api_crop(image_bytes)
            if crop_data:
                st.session_state["crop_data"] = crop_data
                st.session_state["selected_crop_idx"] = 0

        # --- Step 2: pick crop ---
        if "crop_data" in st.session_state and st.session_state["crop_data"]:
            crop_data = st.session_state["crop_data"]
            crops        = crop_data["crops"]
            used_fallback = crop_data.get("used_fallback", False)

            if used_fallback:
                st.info("No clothing detected — using full image as crop.")

            st.markdown(f"**{len(crops)} crop(s) detected.** Select one to search:")

            crop_cols = st.columns(min(len(crops), 5))
            for i, crop_item in enumerate(crops):
                crop_img = b64_to_pil(crop_item["image_b64"])
                conf = crop_item["confidence"]
                with crop_cols[i]:
                    st.image(crop_img, use_container_width=True)
                    st.markdown(
                        f'<div style="text-align:center; font-size:0.8em;">'
                        f'conf <strong>{conf:.2f}</strong></div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(f"Use crop #{i+1}", key=f"pick_crop_{i}"):
                        st.session_state["selected_crop_idx"] = i

            sel_idx      = st.session_state.get("selected_crop_idx", 0)
            selected_crop = b64_to_pil(crops[sel_idx]["image_b64"])
            sel_conf      = crops[sel_idx]["confidence"]
            selected_bytes = pil_to_bytes(selected_crop)

            # Original vs selected crop side by side
            st.markdown("---")
            col_orig, col_crop = st.columns(2)
            with col_orig:
                st.markdown("**Original**")
                st.image(image_bytes, use_container_width=True)
            with col_crop:
                st.markdown(
                    f'**Detected Crop** &nbsp;'
                    f'<span style="background:#6366f122; color:#a78bfa; '
                    f'border:1px solid #6366f155; padding:2px 8px; '
                    f'border-radius:8px; font-size:0.8em;">conf {sel_conf:.2f}</span>',
                    unsafe_allow_html=True,
                )
                st.image(selected_crop, use_container_width=True)

            # --- Step 3: retrieve ---
            if st.button("Search similar items", key="btn_retrieve"):
                with st.spinner(f"Searching {GALLERY_SIZE:,} gallery items..."):
                    t0 = time.time()
                    results_data = api_retrieve(selected_bytes, selected_config)
                    elapsed = time.time() - t0
                if results_data:
                    st.session_state["results_data"] = results_data
                    st.session_state["search_elapsed"] = elapsed

    # --- Step 4: show results ---
    if "results_data" in st.session_state:
        results_data = st.session_state["results_data"]
        elapsed   = st.session_state.get("search_elapsed", 0)
        results   = results_data.get("results", [])
        cfg_used  = results_data.get("config_name", "?")
        badge_col = CONFIG_COLORS.get(cfg_used, "#6366f1")

        st.divider()
        st.markdown(
            f'<div class="results-header">'
            f'Top {len(results)} results &nbsp;'
            f'<span class="config-badge" style="background:{badge_col}22; color:{badge_col}; '
            f'border:1px solid {badge_col}55;">config {cfg_used}</span>'
            f'&nbsp;<span style="color:#9ca3af; font-size:0.8em; font-weight:400;">'
            f'{elapsed:.2f}s</span>'
            f'</div>',
            unsafe_allow_html=True,
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
                        st.markdown(
                            '<div style="height:220px; background:#1a1a2e; border-radius:8px; '
                            'display:flex; align-items:center; justify-content:center; '
                            'color:#6b7280; font-size:0.8em;">image not found</div>',
                            unsafe_allow_html=True,
                        )

                    score     = item.get("score", 0.0)
                    score_pct = round(score * 100, 1)
                    caption   = item.get("caption", "") or ""
                    caption_display = (caption[:100] + "…") if len(caption) > 100 else caption

                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="rank-badge">#{item["rank"]}</div>'
                        f'<div class="item-id">{item["item_id"]}</div>'
                        f'<div style="display:flex; justify-content:space-between; '
                        f'align-items:center; margin: 4px 0 2px;">'
                        f'<div class="score-bar-wrap" style="flex:1; margin:0;">'
                        f'<div class="score-bar-fill" style="width:{score_pct}%;"></div>'
                        f'</div>'
                        f'<span style="font-size:0.75em; color:#a78bfa; margin-left:8px; '
                        f'white-space:nowrap;">{score_pct}%</span>'
                        f'</div>'
                        f'<div class="caption-text">{caption_display if caption_display else "—"}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
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

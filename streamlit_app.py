"""
Streamlit frontend for the Vision Search Engine.

Tab 1 — Search: upload image → YOLO crop picker → retrieve results grid
Tab 2 — Batch Eval: run recall/NDCG/mAP evaluation against the gallery
"""

import base64
import html as _html
import io
import time
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components
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


def render_caption_tags(caption: str) -> str:
    """
    Structured caption "attr: value, attr: value" → badge pills (value only, max 6).
    Falls back to plain italic text if caption has no ':' (old-format captions).
    Returns an HTML string for embedding inside a card div.
    """
    caption = caption.strip()
    if not caption:
        return '<span style="color:#4b5563; font-style:italic; font-size:0.76em;">—</span>'

    if ":" not in caption:
        return (
            f'<span style="font-style:italic; color:#6b7280; font-size:0.76em;">'
            f'{_html.escape(caption[:120])}</span>'
        )

    tags = []
    for pair in caption.split(","):
        pair = pair.strip()
        if not pair:
            continue
        value = pair.split(":", 1)[1].strip() if ":" in pair else pair
        if value:
            tags.append(_html.escape(value))

    badges = "".join(
        f'<span class="caption-tag">{tag}</span>' for tag in tags[:6]
    )
    return f'<div style="margin-top:4px; line-height:1.9;">{badges}</div>'


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
/* ── App background ─────────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #fff0f8 0%, #f3e8ff 40%, #e8f4ff 100%);
    min-height: 100vh;
}
.main .block-container {
    background: transparent;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fdf4ff 0%, #ede8ff 100%) !important;
    border-right: 1px solid rgba(167, 139, 250, 0.35);
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: #4c1d95 !important;
}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 2px solid rgba(167, 139, 250, 0.3);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #7c6fa0;
    font-weight: 600;
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    background: transparent;
    transition: color 0.2s, background 0.2s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #7c3aed;
    background: rgba(167, 139, 250, 0.08);
}
.stTabs [aria-selected="true"] {
    color: #7c3aed !important;
    background: rgba(167, 139, 250, 0.12) !important;
    border-bottom: 2px solid #7c3aed !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.02em;
    box-shadow: 0 3px 12px rgba(124, 58, 237, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.5);
    border: none;
    color: #ffffff;
}
.stButton > button:active {
    transform: translateY(0px);
    box-shadow: 0 2px 8px rgba(124, 58, 237, 0.3);
}

/* ── File uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.5);
    border-radius: 12px;
    padding: 4px;
}
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed rgba(124, 58, 237, 0.45) !important;
    border-radius: 10px !important;
    background: rgba(255, 255, 255, 0.6) !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(124, 58, 237, 0.8) !important;
    background: rgba(243, 232, 255, 0.6) !important;
}
[data-testid="stFileUploaderDropzone"] button {
    transform: none !important;
}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploaderDropzone"] button:focus {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    color: white !important;
    transform: none !important;
    box-shadow: none !important;
    border: none !important;
}

/* ── Images ─────────────────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    width: 100% !important;
    height: 280px !important;
    object-fit: cover !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.18);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stImage"] img:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 30px rgba(124, 58, 237, 0.3);
}

/* Hide fullscreen/zoom button on images */
button[title="View fullscreen"] {
    display: none !important;
}

/* ── Result card ─────────────────────────────────────────────────────────── */
.result-card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 14px;
    padding: 12px;
    border: 1px solid rgba(167, 139, 250, 0.3);
    margin-top: 6px;
    box-shadow: 0 4px 18px rgba(124, 58, 237, 0.1);
    transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}
.result-card:hover {
    border-color: rgba(124, 58, 237, 0.55);
    box-shadow: 0 10px 32px rgba(124, 58, 237, 0.22);
    transform: translateY(-3px);
}

/* ── Item ID ─────────────────────────────────────────────────────────────── */
.item-id {
    font-family: monospace;
    font-size: 0.76em;
    color: #7c3aed;
    margin-bottom: 4px;
    word-break: break-all;
}

/* ── Caption tags ────────────────────────────────────────────────────────── */
.caption-tag {
    display: inline-block;
    background: rgba(167, 139, 250, 0.15);
    border: 1px solid rgba(124, 58, 237, 0.25);
    color: #6d28d9;
    border-radius: 5px;
    padding: 1px 7px;
    font-size: 0.72em;
    margin: 2px 2px 0 0;
    white-space: nowrap;
    transition: background 0.15s;
}
.caption-tag:hover {
    background: rgba(167, 139, 250, 0.3);
}

/* ── Upload hint ─────────────────────────────────────────────────────────── */
.upload-hint {
    border: 2px dashed rgba(124, 58, 237, 0.4);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    color: #7c3aed;
    margin-bottom: 10px;
    font-size: 0.9em;
    background: rgba(255, 255, 255, 0.5);
}

/* ── Config badge ────────────────────────────────────────────────────────── */
.config-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.85em;
    margin-left: 6px;
}

/* ── Results header ──────────────────────────────────────────────────────── */
.results-header {
    font-size: 1.2em;
    font-weight: 700;
    background: linear-gradient(90deg, #1e1b4b, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 12px 0 8px;
}

/* ── Spinner ─────────────────────────────────────────────────────────────── */
.stSpinner p,
[data-testid="stSpinner"] p,
.stSpinner > div,
div[class*="StatusWidget"] p {
    color: #1a1a2e !important;
}
.stSpinner svg circle,
[data-testid="stSpinner"] svg circle {
    stroke: #7c3aed !important;
}

/* ── Headings & body text ────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    color: #1a1a2e !important;
    font-weight: 700 !important;
}
.stMarkdown p, .stMarkdown li, .stMarkdown span {
    color: #1a1a2e;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span {
    color: #4c1d95 !important;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr {
    border-color: rgba(167, 139, 250, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# Button colour overrides by text label (JS MutationObserver approach)
components.html("""
<script>
function applyButtonStyles() {
    const doc = window.parent.document;
    doc.querySelectorAll('button').forEach(btn => {
        const text = btn.innerText.trim();
        if (text === 'Reset') {
            btn.style.background = 'linear-gradient(135deg,#dc2626,#ef4444)';
            btn.style.boxShadow  = '0 3px 12px rgba(220,38,38,0.4)';
        } else if (text === 'Re-crop' || text === 'Detect clothing items') {
            btn.style.background = 'linear-gradient(135deg,#0891b2,#06b6d4)';
            btn.style.boxShadow  = '0 3px 12px rgba(8,145,178,0.4)';
        } else if (text === 'Check API health') {
            btn.style.background = 'linear-gradient(135deg,#059669,#10b981)';
            btn.style.boxShadow  = '0 3px 12px rgba(5,150,105,0.4)';
        }
    });
}
applyButtonStyles();
new MutationObserver(applyButtonStyles)
    .observe(window.parent.document.body, {childList:true, subtree:true});
</script>
""", height=0)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

_, col_title, _ = st.columns([1, 3, 1])
with col_title:
    st.markdown("""
    <div style="text-align:center; padding:16px 0 8px;">
        <h1 style="margin:0; font-size:2.2em;">👗 Visual Fashion Search</h1>
        <p style="color:#6d28d9; margin:4px 0 0; font-size:1em; font-weight:600;">
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

    selected_config = st.selectbox(
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

_SEARCH_STATE_KEYS = (
    "uploaded_bytes", "crops", "results", "last_uploaded_filename",
)

with tab_search:

    # ── Top-right action bar: Reset only ────────────────────────────────────
    _, col_reset = st.columns([9, 1])
    with col_reset:
        if st.button("Reset", key="btn_reset"):
            for _k in _SEARCH_STATE_KEYS:
                st.session_state.pop(_k, None)
            st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
            st.rerun()

    # ── File uploader ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="upload-hint">Upload a fashion photo — JPG, PNG, or WebP</div>',
        unsafe_allow_html=True,
    )
    uploader_key = st.session_state.get("uploader_key", 0)
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"search_upload_{uploader_key}",
        label_visibility="collapsed",
    )

    # Clear all state when uploader X is clicked
    if uploaded is None and st.session_state.get("last_uploaded_filename"):
        for _k in _SEARCH_STATE_KEYS:
            st.session_state.pop(_k, None)

    if uploaded is not None:
        if uploaded.name != st.session_state.get("last_uploaded_filename"):
            for _k in _SEARCH_STATE_KEYS:
                st.session_state.pop(_k, None)
            st.session_state["last_uploaded_filename"] = uploaded.name
        st.session_state["uploaded_bytes"] = uploaded.read()

    # ── Always show uploaded image once available ───────────────────────────
    if "uploaded_bytes" in st.session_state:
        image_bytes = st.session_state["uploaded_bytes"]
        col_img, col_act = st.columns([1, 3])
        with col_img:
            st.image(image_bytes, caption="Uploaded image", use_container_width=True)
        with col_act:
            btn_label = "Re-crop" if "crops" in st.session_state else "Detect clothing items"
            if st.button(btn_label, key="btn_detect"):
                with st.spinner("Detecting clothing items..."):
                    crop_data = api_crop(image_bytes)
                if crop_data:
                    st.session_state["crops"] = crop_data.get("crops", [])
                    st.session_state.pop("results", None)
                    st.rerun()

    # ── Step 2: crop picker (hidden once results are fetched) ───────────────
    if "crops" in st.session_state and "results" not in st.session_state:
        crops = st.session_state["crops"]
        image_bytes = st.session_state["uploaded_bytes"]

        st.markdown(f"**{len(crops)} crop(s) detected.** Select one to search:")

        all_cols = st.columns(1 + min(len(crops), 5))

        with all_cols[0]:
            st.image(image_bytes, caption="Original", use_container_width=True)
            if st.button("Use original", key="btn_use_original"):
                with st.spinner(f"Searching {GALLERY_SIZE:,} gallery items..."):
                    resp = api_retrieve(image_bytes, selected_config)
                if resp:
                    st.session_state["results"] = resp.get("results", [])
                    st.rerun()

        for i, crop_item in enumerate(crops[:5]):
            crop_img = b64_to_pil(crop_item["image_b64"])
            conf = crop_item["confidence"]
            with all_cols[i + 1]:
                st.image(crop_img, caption=f"conf {conf:.2f}", use_container_width=True)
                if st.button(f"Use crop #{i + 1}", key=f"btn_crop_{i}"):
                    with st.spinner(f"Searching {GALLERY_SIZE:,} gallery items..."):
                        resp = api_retrieve(pil_to_bytes(crop_img), selected_config)
                    if resp:
                        st.session_state["results"] = resp.get("results", [])
                        st.rerun()

    # ── Step 4: results grid ────────────────────────────────────────────────
    if "results" in st.session_state:
        results = st.session_state["results"]

        st.divider()
        st.markdown(
            f'<div class="results-header">Top {len(results)} results</div>',
            unsafe_allow_html=True,
        )

        COLS_PER_ROW = 4
        for row_start in range(0, len(results), COLS_PER_ROW):
            if row_start > 0:
                st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
            row_items = results[row_start: row_start + COLS_PER_ROW]
            cols = st.columns(len(row_items))
            for col, item in zip(cols, row_items):
                with col:
                    try:
                        img_path = Path(item["path"])
                        if not img_path.exists():
                            raise FileNotFoundError
                        st.image(str(img_path), use_container_width=True)
                    except Exception:
                        st.markdown(
                            '<div style="height:260px; background:#f3e8ff; border-radius:10px; '
                            'display:flex; align-items:center; justify-content:center; '
                            'color:#6b7280; font-size:0.8em;">image not found</div>',
                            unsafe_allow_html=True,
                        )

                    score = item.get("score", 0.0)
                    score_pct = round(score * 100, 1)
                    caption = item.get("caption", "") or ""

                    st.markdown(
                        f'<div class="result-card" style="position:relative;">'
                        f'<span style="position:absolute; top:10px; right:10px; '
                        f'background:#7c3aed; color:white; font-size:11px; '
                        f'font-weight:600; padding:3px 8px; border-radius:999px; '
                        f'white-space:nowrap;">{score_pct}%</span>'
                        f'<div class="item-id" style="padding-right:52px;">'
                        f'{_html.escape(item["item_id"])}</div>'
                        f'<div style="display:-webkit-box; -webkit-line-clamp:2; '
                        f'-webkit-box-orient:vertical; overflow:hidden; '
                        f'line-height:1.4em; max-height:2.8em;">'
                        f'{render_caption_tags(caption)}</div>'
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

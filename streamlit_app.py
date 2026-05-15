"""
Streamlit frontend — Visual Fashion Search
Search: upload → detect clothing → pick crop → retrieve similar items
Compare: run same query across all three retrieval configs side-by-side
"""

import base64
import html as _html
import io
import time
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

API_BASE      = "http://localhost:8504"
CONFIGS       = ["A", "B", "C"]
CONFIG_LABELS = {
    "A": "Config A — Pretrained CLIP",
    "B": "Config B — CLIP + Captions",
    "C": "Config C — Fine-tuned CLIP",
}
CONFIG_DESC = {
    "A": "Image-only · pretrained CLIP · cosine similarity",
    "B": "Pretrained CLIP + BLIP-2 captions · alpha = 0.7",
    "C": "Fine-tuned CLIP + BLIP-2 captions · alpha = 0.7",
}
CONFIG_ALPHA  = {"A": 1.0, "B": 0.7, "C": 0.7}
CONFIG_COLORS = {"A": "#4A90D9", "B": "#10B981", "C": "#F59E0B"}
BBOX_COLORS   = ["#7C3AED", "#06B6D4", "#10B981", "#F59E0B", "#EF4444"]
GALLERY_SIZE  = 12_612

_SEARCH_STATE_KEYS = (
    "uploaded_bytes", "crops", "annotated_bytes",
    "results", "last_uploaded_filename", "timing",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def b64_to_pil(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def pil_to_bytes(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=85)
    return buf.getvalue()


def draw_detections(image_bytes, crops_data):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i, crop in enumerate(crops_data[:5]):
        x1, y1, x2, y2 = crop["bbox"]
        r, g, b = tuple(int(BBOX_COLORS[i % len(BBOX_COLORS)].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 35), outline=(r, g, b, 220), width=3)
        label = f" #{i+1} {crop['confidence']:.0%} "
        tw = len(label) * 7
        draw.rectangle([x1, max(0, y1 - 22), x1 + tw, y1], fill=(r, g, b, 210))
        draw.text((x1 + 4, max(0, y1 - 20)), label.strip(), fill=(255, 255, 255, 255))
    composite = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    composite.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def render_caption_tags(caption):
    caption = caption.strip()
    if not caption:
        return '<span style="color:#9CA3AF;font-style:italic;font-size:0.75em;">-</span>'
    if ":" not in caption:
        return f'<span style="font-style:italic;color:#6B7280;font-size:0.75em;">{_html.escape(caption[:120])}</span>'
    tags = []
    for pair in caption.split(","):
        pair = pair.strip()
        if not pair:
            continue
        value = pair.split(":", 1)[1].strip() if ":" in pair else pair
        if value:
            tags.append(_html.escape(value))
    badges = "".join(f'<span class="caption-tag">{t}</span>' for t in tags[:5])
    return f'<div style="margin-top:5px;line-height:2;">{badges}</div>'


def api_crop(image_bytes):
    try:
        resp = requests.post(
            f"{API_BASE}/crop",
            files={"file": ("upload.jpg", image_bytes, "image/jpeg")},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Detection failed: {e}")
        return None


def api_retrieve(crop_bytes, config_name):
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
        st.error(f"Retrieval failed: {e}")
        return None


def api_health():
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _stat(label, value, sub):
    return (
        f'<div class="stat-card">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-sub">{sub}</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

def inject_css(dark):
    bg         = "#0F172A" if dark else "#F8F9FC"
    bg2        = "#1E293B" if dark else "#F1F5F9"
    card       = "#1E293B" if dark else "#FFFFFF"
    sb         = "#111827" if dark else "#FFFFFF"
    tpri       = "#F8FAFC" if dark else "#111827"
    tsec       = "#94A3B8" if dark else "#6B7280"
    tmut       = "#64748B" if dark else "#9CA3AF"
    brd        = "#334155" if dark else "#E5E7EB"
    acl        = "rgba(139,92,246,0.14)" if dark else "rgba(124,58,237,0.08)"
    shd        = "0 4px 24px rgba(0,0,0,0.32)" if dark else "0 4px 24px rgba(0,0,0,0.07)"
    shdh       = "0 12px 40px rgba(0,0,0,0.48)" if dark else "0 12px 40px rgba(124,58,237,0.16)"
    tabbg      = "#1E293B" if dark else "#F1F5F9"
    tabact     = "#0F172A" if dark else "#FFFFFF"

    css = f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

*,*::before,*::after{{font-family:'Manrope',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;box-sizing:border-box;}}
.material-symbols-rounded,.material-symbols-outlined,.material-symbols-sharp,.material-icons{{font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Symbols Sharp','Material Icons'!important;font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 24;font-feature-settings:normal;text-rendering:optimizeLegibility;-webkit-font-smoothing:antialiased;}}
[data-testid="collapsedControl"] *{{font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Symbols Sharp','Material Icons'!important;}}
:root{{--bg:{bg};--bg2:{bg2};--card:{card};--sb:{sb};--tpri:{tpri};--tsec:{tsec};--tmut:{tmut};--brd:{brd};--acl:{acl};--shd:{shd};--shdh:{shdh};--acc:#7C3AED;}}

.stApp{{background:{bg}!important;}}
.main .block-container{{background:transparent!important;padding-top:0.75rem!important;padding-left:2rem!important;padding-right:2rem!important;max-width:1440px!important;}}

[data-testid="stSidebar"]{{background:{sb}!important;border-right:1px solid {brd}!important;}}
[data-testid="stSidebar"]>div:first-child{{padding:1.25rem 1rem!important;}}
[data-testid="stSidebar"] .stMarkdown p,[data-testid="stSidebar"] p{{color:{tsec}!important;font-size:0.82em!important;}}

h1,h2,h3,h4,h5,h6{{color:{tpri}!important;font-weight:800!important;letter-spacing:-0.025em!important;}}
p,li{{color:{tpri};}}
.stMarkdown p{{color:{tpri}!important;}}

.stButton>button{{background:linear-gradient(135deg,#7C3AED 0%,#9333EA 100%)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:700!important;font-size:0.85rem!important;padding:0.45rem 1.1rem!important;box-shadow:0 2px 10px rgba(124,58,237,0.28)!important;transition:transform 0.18s ease,box-shadow 0.18s ease!important;white-space:nowrap!important;}}
.stButton>button:hover{{transform:translateY(-2px)!important;box-shadow:0 6px 22px rgba(124,58,237,0.42)!important;border:none!important;color:#fff!important;}}
.stButton>button:active{{transform:translateY(0)!important;}}

[data-testid="stFileUploaderDropzone"]{{background:{card}!important;border:2px dashed {brd}!important;border-radius:16px!important;padding:1rem!important;transition:border-color 0.25s ease,background 0.25s ease!important;}}
[data-testid="stFileUploaderDropzone"]:hover{{border-color:#7C3AED!important;background:{acl}!important;}}
[data-testid="stFileUploaderDropzone"] p{{color:{tsec}!important;font-size:0.85rem!important;}}
[data-testid="stFileUploaderDropzone"] button{{background:linear-gradient(135deg,#7C3AED,#9333EA)!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:700!important;transform:none!important;white-space:nowrap!important;width:auto!important;padding:0.4rem 1.2rem!important;}}
[data-testid="stFileUploaderDropzone"] button:hover{{background:linear-gradient(135deg,#6D28D9,#7C3AED)!important;transform:none!important;box-shadow:none!important;}}
[data-testid="stFileUploaderDropzone"] button *{{color:#fff!important;font-size:0.875rem!important;}}

[data-testid="stImage"] img{{border-radius:12px!important;width:100%!important;object-fit:cover!important;transition:transform 0.22s ease,box-shadow 0.22s ease!important;display:block!important;}}
[data-testid="stImage"] img:hover{{transform:scale(1.015)!important;box-shadow:0 8px 32px rgba(0,0,0,0.18)!important;}}
button[title="View fullscreen"]{{display:none!important;}}

.vfs-card{{background:{card};border:1px solid {brd};border-radius:14px;padding:1rem 1.1rem;box-shadow:{shd};transition:box-shadow 0.2s,border-color 0.2s,transform 0.2s;}}
.vfs-card:hover{{box-shadow:{shdh};border-color:rgba(124,58,237,0.38);transform:translateY(-2px);}}

.result-card{{background:{card};border:1px solid {brd};border-top:none;border-radius:0 0 14px 14px;padding:9px 11px 11px;box-shadow:{shd};transition:box-shadow 0.2s,border-color 0.2s;position:relative;}}
.result-card:hover{{border-color:rgba(124,58,237,0.4);box-shadow:{shdh};}}

.score-badge{{display:inline-flex;align-items:center;background:linear-gradient(135deg,#7C3AED,#9333EA);color:#fff;font-size:10.5px;font-weight:800;padding:2px 9px;border-radius:999px;letter-spacing:0.03em;}}
.item-id{{font-size:0.68em;color:{tmut};margin-bottom:4px;word-break:break-all;font-weight:500;}}

.caption-tag{{display:inline-block;background:{acl};border:1px solid rgba(124,58,237,0.22);color:#7C3AED;border-radius:6px;padding:2px 7px;font-size:0.71em;font-weight:600;margin:2px 2px 0 0;white-space:nowrap;}}

.stat-card{{background:{card};border:1px solid {brd};border-radius:12px;padding:0.85rem 1rem;box-shadow:{shd};}}
.stat-label{{font-size:0.68em;font-weight:800;text-transform:uppercase;letter-spacing:0.08em;color:{tmut};margin-bottom:3px;}}
.stat-value{{font-size:1.45em;font-weight:800;color:{tpri};letter-spacing:-0.025em;line-height:1.15;}}
.stat-sub{{font-size:0.7em;color:{tsec};margin-top:2px;font-weight:500;}}

.section-heading{{font-size:0.7em;font-weight:800;text-transform:uppercase;letter-spacing:0.09em;color:{tmut};margin:1.5rem 0 0.75rem;padding-bottom:0.5rem;border-bottom:1px solid {brd};}}
.sidebar-label{{font-size:0.65em;font-weight:800;text-transform:uppercase;letter-spacing:0.1em;color:{tmut};margin:1.1rem 0 0.4rem;display:block;}}
.model-chip{{display:inline-flex;align-items:center;background:{acl};border:1px solid rgba(124,58,237,0.22);color:#7C3AED;border-radius:6px;padding:2px 7px;font-size:0.68em;font-weight:700;margin-right:4px;}}

.status-dot{{width:7px;height:7px;border-radius:50%;display:inline-block;margin-right:5px;vertical-align:middle;}}
.status-dot.online{{background:#10B981;box-shadow:0 0 0 3px rgba(16,185,129,0.22);}}
.status-dot.offline{{background:#EF4444;}}

.stTabs [data-baseweb="tab-list"]{{background:{tabbg}!important;border-radius:12px!important;padding:4px!important;gap:2px!important;border:none!important;}}
.stTabs [data-baseweb="tab"]{{border-radius:9px!important;font-weight:700!important;font-size:0.875rem!important;color:{tsec}!important;padding:8px 20px!important;border:none!important;background:transparent!important;transition:all 0.18s ease!important;}}
.stTabs [aria-selected="true"]{{background:{tabact}!important;color:#7C3AED!important;box-shadow:0 2px 10px rgba(0,0,0,0.1)!important;}}

hr{{border-color:{brd}!important;margin:1.25rem 0!important;}}
.stSpinner svg circle,[data-testid="stSpinner"] svg circle{{stroke:#7C3AED!important;}}
[data-testid="stSpinner"] p{{color:{tsec}!important;}}
.stSelectbox label,.stSlider label,.stNumberInput label{{color:{tsec}!important;font-size:0.8em!important;font-weight:700!important;}}
[data-testid="stAlert"]{{border-radius:12px!important;border:none!important;}}
#MainMenu{{visibility:hidden;}}footer{{visibility:hidden;}}header{{visibility:hidden;}}
.cmp-header{{border-radius:10px;padding:10px 13px;margin-bottom:10px;font-size:0.82em;}}
</style>"""
    st.markdown(css, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Visual Fashion Search",
    page_icon="&#128",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode
inject_css(dark)

# ─────────────────────────────────────────────────────────────────────────────
# JS button colour overrides
# ─────────────────────────────────────────────────────────────────────────────

components.html("""
<script>
function applyBtnStyles() {
    window.parent.document.querySelectorAll('button').forEach(function(btn) {
        var t = btn.innerText.trim();
        if (t === 'Reset') {
            btn.style.background = 'linear-gradient(135deg,#DC2626,#EF4444)';
            btn.style.boxShadow  = '0 2px 10px rgba(220,38,38,0.32)';
        } else if (t === 'Re-crop' || t === 'Detect clothing items') {
            btn.style.background = 'linear-gradient(135deg,#0891B2,#06B6D4)';
            btn.style.boxShadow  = '0 2px 10px rgba(8,145,178,0.32)';
        } else if (t.indexOf('API') >= 0 || t.indexOf('health') >= 0) {
            btn.style.background = 'linear-gradient(135deg,#059669,#10B981)';
            btn.style.boxShadow  = '0 2px 10px rgba(5,150,105,0.32)';
        } else if (t.indexOf('Compare') >= 0 || t.indexOf('Detect &') >= 0) {
            btn.style.background = 'linear-gradient(135deg,#D97706,#F59E0B)';
            btn.style.boxShadow  = '0 2px 10px rgba(217,119,6,0.32)';
        }
    });
}
applyBtnStyles();
new MutationObserver(applyBtnStyles)
    .observe(window.parent.document.body, {childList:true, subtree:true});
</script>
""", height=0)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    col_brand, col_dm = st.columns([3, 1])
    with col_brand:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:7px;margin-bottom:2px;">'
            '<span style="font-size:1.35em;">&#128;</span>'
            '<span style="font-weight:800;font-size:1.05em;letter-spacing:-0.025em;">VisionSearch</span>'
            '</div>'
            '<div style="font-size:0.7em;color:var(--tmut);font-weight:600;letter-spacing:0.04em;">AI FASHION RETRIEVAL</div>',
            unsafe_allow_html=True,
        )
    with col_dm:
        dm_icon = "&#9728;" if dark else "&#9790;"
        if st.button(dm_icon, key="btn_dark", help="Toggle dark / light mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.divider()
    st.markdown('<span class="sidebar-label">Search Settings</span>', unsafe_allow_html=True)

    selected_config = st.selectbox(
        "Retrieval config",
        options=CONFIGS,
        format_func=lambda c: CONFIG_LABELS[c],
        index=2,
        key="selected_config",
    )

    cfg_color = CONFIG_COLORS[selected_config]
    st.markdown(
        f'<div style="background:rgba({_rgb(cfg_color)},0.1);border:1px solid rgba({_rgb(cfg_color)},0.3);'
        f'border-radius:9px;padding:7px 10px;margin:4px 0 8px;font-size:0.76em;'
        f'color:{cfg_color};font-weight:600;">{CONFIG_DESC[selected_config]}</div>',
        unsafe_allow_html=True,
    )

    if selected_config in ("B", "C"):
        st.slider(
            "Image weight (alpha)",
            0.0, 1.0, CONFIG_ALPHA[selected_config], 0.05,
            disabled=True,
            help="Fixed per config. alpha=0.7 means 70% image + 30% caption embedding.",
        )

    top_k = st.select_slider("Results to show", options=[5, 10, 15, 20], value=10, key="top_k")

    st.divider()
    st.markdown('<span class="sidebar-label">Models</span>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.8em;line-height:2.3;color:var(--tsec);">'
        '<div><span class="model-chip">CLIP</span> clip-vit-base-patch16</div>'
        '<div><span class="model-chip">BLIP-2</span> blip2-opt-2.7b</div>'
        '<div><span class="model-chip">Reranker</span> blip-itm-base-coco</div>'
        '<div><span class="model-chip">Detector</span> YOLOv8m</div>'
        f'<div><span class="model-chip">Index</span> HNSW {GALLERY_SIZE:,} items</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown('<span class="sidebar-label">System Status</span>', unsafe_allow_html=True)

    if st.button("Check API health", key="btn_health"):
        st.session_state["api_status"] = api_health()
        st.rerun()

    api_status = st.session_state.get("api_status", None)
    if api_status is not None:
        if api_status:
            loaded = api_status.get("indexes_loaded", [])
            st.markdown(
                f'<div style="font-size:0.8em;margin-top:6px;color:var(--tsec);">'
                f'<span class="status-dot online"></span>'
                f'<strong style="color:#10B981;">API Online</strong><br>'
                f'<span style="font-size:0.9em;">Indexes: {", ".join(loaded) or "none"}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-size:0.8em;margin-top:6px;">'
                '<span class="status-dot offline"></span>'
                '<strong style="color:#EF4444;">API Offline</strong>'
                '</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div style="font-size:0.7em;color:var(--tmut);margin-top:10px;font-weight:500;">'
        f'Gallery {GALLERY_SIZE:,} indexed items</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

_, col_hdr, _ = st.columns([1, 5, 1])
with col_hdr:
    st.markdown(
        '<div style="text-align:center;padding:1.1rem 0 0.4rem;">'
        '<h1 style="font-size:2.6em;margin:0;font-weight:800;letter-spacing:-0.04em;">Visual Fashion Search</h1>'
        '<p style="font-size:0.9em;color:var(--tsec);margin:6px 0 0;font-weight:500;">'
        'AI-powered apparel retrieval &nbsp;&middot;&nbsp; CLIP &nbsp;&middot;&nbsp; BLIP-2 &nbsp;&middot;&nbsp; HNSW vector search'
        '</p></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="height:1px;background:var(--brd);margin:0.6rem 0 1.25rem;"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_search, tab_compare = st.tabs(["  Search", "  Compare Configs"])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Search
# ─────────────────────────────────────────────────────────────────────────────

with tab_search:

    _, col_rst = st.columns([11, 1])
    with col_rst:
        if st.button("Reset", key="btn_reset"):
            for _k in _SEARCH_STATE_KEYS:
                st.session_state.pop(_k, None)
            st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
            st.rerun()

    uploader_key = st.session_state.get("uploader_key", 0)
    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"search_upload_{uploader_key}",
        label_visibility="collapsed",
    )

    if uploaded is None and st.session_state.get("last_uploaded_filename"):
        for _k in _SEARCH_STATE_KEYS:
            st.session_state.pop(_k, None)

    if uploaded is not None:
        if uploaded.name != st.session_state.get("last_uploaded_filename"):
            for _k in _SEARCH_STATE_KEYS:
                st.session_state.pop(_k, None)
            st.session_state["last_uploaded_filename"] = uploaded.name
        st.session_state["uploaded_bytes"] = uploaded.read()

    if "uploaded_bytes" in st.session_state:
        image_bytes = st.session_state["uploaded_bytes"]
        col_img, col_meta = st.columns([1, 2])

        with col_img:
            display_bytes = st.session_state.get("annotated_bytes", image_bytes)
            _preview = Image.open(io.BytesIO(display_bytes)).convert("RGB")
            _preview.thumbnail((400, 400), Image.LANCZOS)
            _pcanvas = Image.new("RGB", (400, 400), (255, 255, 255))
            _pcanvas.paste(_preview, ((400 - _preview.width) // 2, (400 - _preview.height) // 2))
            st.image(_pcanvas, use_container_width=True)

        with col_meta:
            filename = st.session_state.get("last_uploaded_filename", "image")
            size_kb = len(image_bytes) / 1024
            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
            try:
                pil_img = Image.open(io.BytesIO(image_bytes))
                dims = f"{pil_img.width} x {pil_img.height} px"
            except Exception:
                dims = "-"

            crops_detected = len(st.session_state.get("crops", []))
            results_count  = len(st.session_state.get("results", []))

            extra = ""
            if crops_detected:
                extra += f'<div><div class="stat-label">Crops</div><div style="font-size:0.88em;font-weight:700;color:var(--tpri);">{crops_detected}</div></div>'
            if results_count:
                extra += f'<div><div class="stat-label">Results</div><div style="font-size:0.88em;font-weight:700;color:var(--tpri);">{results_count}</div></div>'

            st.markdown(
                f'<div class="vfs-card" style="margin-bottom:0.85rem;">'
                f'<div style="font-weight:700;font-size:0.95em;margin-bottom:10px;color:var(--tpri);word-break:break-all;">{_html.escape(filename)}</div>'
                f'<div style="display:flex;gap:20px;flex-wrap:wrap;">'
                f'<div><div class="stat-label">Size</div><div style="font-size:0.88em;font-weight:700;color:var(--tpri);">{size_str}</div></div>'
                f'<div><div class="stat-label">Dimensions</div><div style="font-size:0.88em;font-weight:700;color:var(--tpri);">{dims}</div></div>'
                f'<div><div class="stat-label">Config</div><div style="font-size:0.88em;font-weight:700;color:{cfg_color};">{selected_config}</div></div>'
                f'{extra}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            btn_label = "Re-crop" if "crops" in st.session_state else "Detect clothing items"
            if st.button(btn_label, key="btn_detect"):
                t0 = time.time()
                with st.spinner("Detecting clothing items..."):
                    crop_data = api_crop(image_bytes)
                if crop_data:
                    crops = crop_data.get("crops", [])
                    st.session_state["crops"] = crops
                    st.session_state.pop("results", None)
                    if crops:
                        st.session_state["annotated_bytes"] = draw_detections(image_bytes, crops)
                    timing = st.session_state.get("timing", {})
                    timing["detect_time"] = round(time.time() - t0, 2)
                    st.session_state["timing"] = timing
                    st.rerun()

    if "crops" in st.session_state and "results" not in st.session_state:
        crops       = st.session_state["crops"]
        image_bytes = st.session_state["uploaded_bytes"]

        st.markdown('<div class="section-heading">Select clothing item to search</div>', unsafe_allow_html=True)
        if not crops:
            st.info("No clothing detected - you can still search using the full image.")

        num_cols = 1 + min(len(crops), 5)
        all_cols = st.columns(num_cols)

        with all_cols[0]:
            _full = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            _full.thumbnail((300, 200), Image.LANCZOS)
            _canvas = Image.new("RGB", (300, 200), (255, 255, 255))
            _canvas.paste(_full, ((300 - _full.width) // 2, (200 - _full.height) // 2))
            st.image(_canvas, use_container_width=True)
            st.markdown(
                '<div style="text-align:center;margin:4px 0 6px;">'
                '<span style="font-size:0.72em;font-weight:700;color:var(--tmut);">FULL IMAGE</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            if st.button("Use full image", key="btn_use_original"):
                t0 = time.time()
                with st.spinner(f"Searching {GALLERY_SIZE:,} items..."):
                    resp = api_retrieve(image_bytes, selected_config)
                if resp:
                    timing = st.session_state.get("timing", {})
                    timing["retrieve_time"] = round(time.time() - t0, 2)
                    st.session_state["timing"] = timing
                    st.session_state["results"] = resp.get("results", [])
                    st.rerun()

        for i, crop_item in enumerate(crops[:5]):
            crop_img = b64_to_pil(crop_item["image_b64"])
            conf     = crop_item["confidence"]
            color    = BBOX_COLORS[i % len(BBOX_COLORS)]
            with all_cols[i + 1]:
                _thumb = crop_img.convert("RGB").copy()
                _thumb.thumbnail((300, 200), Image.LANCZOS)
                _canvas = Image.new("RGB", (300, 200), (255, 255, 255))
                _canvas.paste(_thumb, ((300 - _thumb.width) // 2, (200 - _thumb.height) // 2))
                st.image(_canvas, use_container_width=True)
                st.markdown(
                    f'<div style="text-align:center;margin:4px 0 6px;">'
                    f'<span style="font-size:0.72em;font-weight:800;color:{color};">CROP #{i+1}</span>'
                    f'<span style="font-size:0.7em;color:var(--tmut);margin-left:5px;">{conf:.0%}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"Use crop #{i+1}", key=f"btn_crop_{i}"):
                    t0 = time.time()
                    with st.spinner(f"Searching {GALLERY_SIZE:,} items..."):
                        resp = api_retrieve(pil_to_bytes(crop_img), selected_config)
                    if resp:
                        timing = st.session_state.get("timing", {})
                        timing["retrieve_time"] = round(time.time() - t0, 2)
                        st.session_state["timing"] = timing
                        st.session_state["results"] = resp.get("results", [])
                        st.rerun()

    if "results" in st.session_state:
        results = st.session_state["results"]
        timing  = st.session_state.get("timing", {})

        st.markdown('<div style="height:1px;background:var(--brd);margin:1.5rem 0 1rem;"></div>', unsafe_allow_html=True)

        hcol1, hcol2 = st.columns([3, 1])
        with hcol1:
            retrieve_t = timing.get("retrieve_time", None)
            timing_str = f" {retrieve_t:.2f}s" if retrieve_t else ""
            st.markdown(
                f'<div style="font-weight:800;font-size:1.25em;letter-spacing:-0.02em;color:var(--tpri);">Top {min(len(results), top_k)} results</div>'
                f'<div style="font-size:0.78em;color:var(--tmut);margin-top:2px;font-weight:500;">Config {selected_config} {GALLERY_SIZE:,} items{timing_str}</div>',
                unsafe_allow_html=True,
            )
        with hcol2:
            if results:
                top_score = results[0].get("score", 0)
                st.markdown(
                    f'<div style="text-align:right;">'
                    f'<div class="stat-label">Top match</div>'
                    f'<div style="font-size:1.5em;font-weight:800;color:#7C3AED;letter-spacing:-0.03em;">{top_score:.1%}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)

        COLS_PER_ROW    = 4
        display_results = results[:top_k]

        for row_start in range(0, len(display_results), COLS_PER_ROW):
            row_items = display_results[row_start: row_start + COLS_PER_ROW]
            cols = st.columns(COLS_PER_ROW)
            for col_i, (col, item) in enumerate(zip(cols, row_items)):
                with col:
                    try:
                        img_path = Path(item["path"])
                        if not img_path.exists():
                            raise FileNotFoundError
                        pil_img = Image.open(str(img_path)).convert("RGB")
                        pil_img.thumbnail((400, 280), Image.LANCZOS)
                        canvas = Image.new("RGB", (400, 280), (255, 255, 255))
                        canvas.paste(pil_img, ((400 - pil_img.width) // 2, (280 - pil_img.height) // 2))
                        st.image(canvas, use_container_width=True)
                    except Exception:
                        st.markdown(
                            '<div style="height:280px;background:var(--bg2);border-radius:14px 14px 0 0;'
                            'display:flex;align-items:center;justify-content:center;'
                            'color:var(--tmut);font-size:0.85em;border:1px solid var(--brd);">image not found</div>',
                            unsafe_allow_html=True,
                        )
                    score     = item.get("score", 0.0)
                    score_pct = round(score * 100, 1)
                    caption   = item.get("caption", "") or ""
                    rank      = item.get("rank", row_start + col_i + 1)
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">'
                        f'<span class="score-badge">{score_pct}%</span>'
                        f'<span style="font-size:0.7em;color:var(--tmut);font-weight:700;">#{rank}</span>'
                        f'</div>'
                        f'<div class="item-id">{_html.escape(item["item_id"])}</div>'
                        f'{render_caption_tags(caption)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            if row_start + COLS_PER_ROW < len(display_results):
                st.markdown('<div style="height:5px;"></div>', unsafe_allow_html=True)

        st.markdown('<div style="height:1px;background:var(--brd);margin:2rem 0 1rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Session Analytics</div>', unsafe_allow_html=True)

        detect_t   = timing.get("detect_time", None)
        retrieve_t = timing.get("retrieve_time", None)
        a1, a2, a3, a4, a5 = st.columns(5)
        with a1:
            st.markdown(_stat("Detection", f"{detect_t:.2f}s" if detect_t else "-", "YOLOv8m" if detect_t else "not run"), unsafe_allow_html=True)
        with a2:
            st.markdown(_stat("Retrieval", f"{retrieve_t:.2f}s" if retrieve_t else "-", "CLIP+HNSW+BLIP-ITM" if retrieve_t else "not run"), unsafe_allow_html=True)
        with a3:
            st.markdown(_stat("Config", selected_config, CONFIG_DESC[selected_config].split("·")[0].strip()), unsafe_allow_html=True)
        with a4:
            top_s = results[0].get("score", 0) if results else 0
            st.markdown(_stat("Top Score", f"{top_s:.1%}", "cosine similarity"), unsafe_allow_html=True)
        with a5:
            st.markdown(_stat("Shown", str(min(len(results), top_k)), f"of {GALLERY_SIZE:,} indexed"), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Compare Configs
# ─────────────────────────────────────────────────────────────────────────────

with tab_compare:
    st.markdown(
        '<div style="margin-bottom:1.25rem;">'
        '<div style="font-weight:800;font-size:1.3em;letter-spacing:-0.025em;color:var(--tpri);">Config Comparison</div>'
        '<div style="font-size:0.82em;color:var(--tmut);margin-top:3px;font-weight:500;">'
        'Run the same query across all three retrieval configs to compare quality.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    cmp_up_key = st.session_state.get("cmp_uploader_key", 0)
    cmp_uploaded = st.file_uploader(
        "Upload image for comparison",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"cmp_upload_{cmp_up_key}",
        label_visibility="collapsed",
    )

    if cmp_uploaded is not None:
        cmp_bytes = cmp_uploaded.read()
        if cmp_bytes != st.session_state.get("cmp_bytes_raw"):
            st.session_state["cmp_bytes_raw"] = cmp_bytes
            st.session_state["cmp_bytes"]     = cmp_bytes
            st.session_state["cmp_name"]      = cmp_uploaded.name
            st.session_state.pop("cmp_crops", None)
            st.session_state.pop("cmp_results", None)

    if "cmp_bytes" in st.session_state:
        cmp_bytes = st.session_state["cmp_bytes"]
        c_img, c_info = st.columns([1, 2])
        with c_img:
            st.image(st.session_state.get("cmp_annotated", cmp_bytes), use_container_width=True)
        with c_info:
            cmp_name = st.session_state.get("cmp_name", "image")
            st.markdown(
                f'<div style="font-weight:700;font-size:0.92em;color:var(--tpri);margin-bottom:10px;word-break:break-all;">{_html.escape(cmp_name)}</div>',
                unsafe_allow_html=True,
            )
            if "cmp_crops" not in st.session_state:
                if st.button("Detect & Compare all configs", key="btn_cmp_detect"):
                    with st.spinner("Detecting clothing..."):
                        crop_data = api_crop(cmp_bytes)
                    if crop_data:
                        crops = crop_data.get("crops", [])
                        st.session_state["cmp_crops"] = crops
                        if crops:
                            st.session_state["cmp_annotated"] = draw_detections(cmp_bytes, crops)
                        st.rerun()
            else:
                cmp_crops = st.session_state["cmp_crops"]
                if cmp_crops:
                    st.markdown(f'<div style="font-size:0.82em;color:var(--tmut);margin-bottom:8px;">{len(cmp_crops)} crop(s) detected. Pick one:</div>', unsafe_allow_html=True)
                    crop_cols = st.columns(min(len(cmp_crops), 5))
                    for i, cr in enumerate(cmp_crops[:5]):
                        ci_img = b64_to_pil(cr["image_b64"])
                        color  = BBOX_COLORS[i % len(BBOX_COLORS)]
                        with crop_cols[i]:
                            st.image(ci_img, use_container_width=True)
                            st.markdown(
                                f'<div style="text-align:center;font-size:0.72em;font-weight:800;color:{color};margin:3px 0 5px;">#{i+1} {cr["confidence"]:.0%}</div>',
                                unsafe_allow_html=True,
                            )
                            if st.button(f"Compare #{i+1}", key=f"cmp_crop_{i}"):
                                cb = pil_to_bytes(ci_img)
                                cmp_out = {}
                                with st.spinner("Running all 3 configs..."):
                                    for cfg in CONFIGS:
                                        r = api_retrieve(cb, cfg)
                                        cmp_out[cfg] = r.get("results", []) if r else []
                                st.session_state["cmp_results"] = cmp_out
                                st.rerun()
                else:
                    if st.button("Compare using full image", key="btn_cmp_full"):
                        cmp_out = {}
                        with st.spinner("Running all 3 configs..."):
                            for cfg in CONFIGS:
                                r = api_retrieve(cmp_bytes, cfg)
                                cmp_out[cfg] = r.get("results", []) if r else []
                        st.session_state["cmp_results"] = cmp_out
                        st.rerun()

    if "cmp_results" in st.session_state:
        cmp_results = st.session_state["cmp_results"]
        st.markdown('<div style="height:1px;background:var(--brd);margin:1.5rem 0 1rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-weight:800;font-size:1.15em;margin-bottom:1rem;color:var(--tpri);">Retrieval Comparison</div>', unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        for col, cfg in zip([col_a, col_b, col_c], CONFIGS):
            with col:
                color     = CONFIG_COLORS[cfg]
                c_results = cmp_results.get(cfg, [])
                top_s     = c_results[0].get("score", 0) if c_results else 0
                st.markdown(
                    f'<div class="cmp-header" style="background:rgba({_rgb(color)},0.08);border:1px solid rgba({_rgb(color)},0.28);">'
                    f'<div style="font-weight:800;color:{color};font-size:0.88em;">{CONFIG_LABELS[cfg]}</div>'
                    f'<div style="font-size:0.7em;color:var(--tmut);margin-top:1px;">{CONFIG_DESC[cfg]}</div>'
                    f'<div style="font-size:0.78em;font-weight:700;color:{color};margin-top:4px;">Top match: {top_s:.1%}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                for item in c_results[:5]:
                    try:
                        img_path = Path(item["path"])
                        if not img_path.exists():
                            raise FileNotFoundError
                        pil_img = Image.open(str(img_path)).convert("RGB")
                        scale = 400 / pil_img.width
                        new_h = int(pil_img.height * scale)
                        pil_img = pil_img.resize((400, new_h), Image.LANCZOS)
                        if new_h > 280:
                            top = (new_h - 280) // 2
                            pil_img = pil_img.crop((0, top, 400, top + 280))
                        st.image(pil_img, use_container_width=True)
                    except Exception:
                        st.markdown(
                            '<div style="height:280px;background:var(--bg2);border-radius:10px 10px 0 0;'
                            'display:flex;align-items:center;justify-content:center;'
                            'color:var(--tmut);font-size:0.8em;">image not found</div>',
                            unsafe_allow_html=True,
                        )
                    score   = item.get("score", 0.0)
                    caption = item.get("caption", "") or ""
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">'
                        f'<span class="score-badge">{score:.1%}</span>'
                        f'<span style="font-size:0.68em;color:var(--tmut);font-weight:700;">#{item.get("rank","")}</span>'
                        f'</div>'
                        f'<div class="item-id">{_html.escape(item["item_id"])}</div>'
                        f'<div style="font-size:0.72em;color:var(--tmut);margin-top:4px;'
                        f'display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;'
                        f'overflow:hidden;max-width:100%;">'
                        f'{_html.escape(caption) if caption else "-"}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

                for _ in range(5 - len(c_results[:5])):
                    st.markdown(
                        '<div style="height:280px;background:var(--bg2);border-radius:10px 10px 0 0;'
                        'display:flex;align-items:center;justify-content:center;'
                        'color:var(--tmut);font-size:0.8em;">no result</div>'
                        '<div class="result-card" style="min-height:52px;"></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

        st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
        if st.button("Clear comparison", key="btn_cmp_reset"):
            for _k in ["cmp_bytes", "cmp_bytes_raw", "cmp_crops", "cmp_results", "cmp_name", "cmp_annotated"]:
                st.session_state.pop(_k, None)
            st.session_state["cmp_uploader_key"] = st.session_state.get("cmp_uploader_key", 0) + 1
            st.rerun()

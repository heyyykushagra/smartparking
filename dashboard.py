"""
Streamlit dashboard for the Car Parking Detection project.

    streamlit run dashboard.py

Tabs: Live (single image + parking guidance), Hourly trends (log analytics),
Video (sampled analysis), Layout (view / build the CarParkPos layout).
"""

from __future__ import annotations

import html
import os
import shutil
import tempfile
from datetime import datetime, time

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import config
import occupancy_analytics as oa
import parking_core as pc

# Palette (validated for colour-blind separation; see .streamlit/config.toml)
BLUE = "#0057B8"  # parking-sign blue / primary
NAVY = "#0B1F3A"
FREE_COLOR = "#1BA39C"  # free spaces
OCC_COLOR = "#D1342F"  # occupied spaces
AMBER = "#E8A317"  # entrance / accent (always paired with a text label)
INK = "#14213D"
MUTED = "#5B6B82"
GRID = "#E6EBF2"

st.set_page_config(page_title="Smart Parking Dashboard", page_icon="🅿️", layout="wide")

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* Hero banner */
.hero {{
  background: linear-gradient(120deg, {NAVY} 0%, #0F3B7A 55%, {BLUE} 100%);
  border-radius: 16px; padding: 22px 28px; color: #fff; margin-bottom: 18px;
  display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
  box-shadow: 0 8px 24px rgba(11,31,58,.18);
}}
.hero .sign {{
  width: 58px; height: 58px; border-radius: 12px; background: #fff; color: {BLUE};
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 34px; border: 3px solid {BLUE}; outline: 3px solid #fff;
  flex: 0 0 auto;
}}
.hero h1 {{ margin: 0; font-size: 1.7rem; font-weight: 800; color: #fff; letter-spacing: -.01em; padding: 0; }}
.hero p {{ margin: 2px 0 0 0; color: #C9D8EE; font-size: .95rem; }}
.hero .chips {{ margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; }}
.chip {{
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22);
  color: #fff; border-radius: 999px; padding: 5px 12px; font-size: .8rem; font-weight: 500;
  white-space: nowrap;
}}
.chip.warn {{ background: rgba(232,163,23,.22); border-color: rgba(232,163,23,.6); }}

/* KPI cards */
.kpi-row {{ display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; margin: 4px 0 16px; }}
@media (max-width: 900px) {{ .kpi-row {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
.kpi {{
  background: #fff; border: 1px solid {GRID}; border-radius: 14px; padding: 16px 18px;
  border-top: 4px solid var(--accent); box-shadow: 0 1px 3px rgba(20,33,61,.06);
}}
.kpi .label {{ color: {MUTED}; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; display:flex; align-items:center; gap:8px; }}
.kpi .dot {{ width: 10px; height: 10px; border-radius: 3px; background: var(--accent); display:inline-block; }}
.kpi .value {{ color: {INK}; font-size: 2rem; font-weight: 800; line-height: 1.15; margin-top: 6px; }}
.kpi .sub {{ color: {MUTED}; font-size: .82rem; margin-top: 2px; }}

/* Capacity bar */
.capacity {{ background:#fff; border:1px solid {GRID}; border-radius: 14px; padding: 14px 18px; margin-bottom: 16px; }}
.capacity .bar {{ display:flex; height: 14px; border-radius: 7px; overflow:hidden; gap: 2px; background: #fff; }}
.capacity .bar span {{ display:block; height:100%; }}
.capacity .legend {{ display:flex; gap:18px; margin-top:8px; font-size:.82rem; color:{MUTED}; flex-wrap: wrap; }}
.capacity .legend b {{ color: {INK}; }}
.sw {{ width:10px; height:10px; border-radius:3px; display:inline-block; margin-right:6px; vertical-align:middle; }}

/* Recommendation card */
.rec {{
  display:flex; gap:18px; align-items:center; background:#fff; border:1px solid {GRID};
  border-left: 6px solid {BLUE}; border-radius: 14px; padding: 16px 20px; margin-bottom: 14px;
  box-shadow: 0 1px 3px rgba(20,33,61,.06); flex-wrap: wrap;
}}
.rec .badge {{
  background:{BLUE}; color:#fff; border-radius: 12px; min-width: 84px; padding: 10px 12px;
  text-align:center; flex: 0 0 auto;
}}
.rec .badge small {{ display:block; font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; opacity:.85; }}
.rec .badge b {{ font-size: 1.9rem; font-weight: 800; line-height: 1.1; }}
.rec h3 {{ margin:0; font-size:1.15rem; color:{INK}; font-weight:700; padding:0; }}
.rec p {{ margin:2px 0 0; color:{MUTED}; font-size:.88rem; }}
.rec .alts {{ margin-left:auto; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
.rec .alt {{ border:1px solid {GRID}; background:#F5F7FB; border-radius:999px; padding:5px 12px; font-size:.82rem; color:{INK}; }}
.rec.full {{ border-left-color: {OCC_COLOR}; }}
.rec.full .badge {{ background: {OCC_COLOR}; }}

/* Section titles */
.section {{ display:flex; align-items:baseline; gap:10px; margin: 10px 0 8px; }}
.section h3 {{ margin:0; font-size:1.1rem; font-weight:700; color:{INK}; padding:0; }}
.section span {{ color:{MUTED}; font-size:.86rem; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {GRID}; }}
.stTabs [data-baseweb="tab"] {{
  height: 44px; padding: 0 18px; border-radius: 10px 10px 0 0; font-weight: 600; color: {MUTED};
}}
.stTabs [aria-selected="true"] {{ color: {BLUE} !important; background: #fff; }}

/* Images & containers */
[data-testid="stImage"] img {{ border-radius: 12px; box-shadow: 0 2px 10px rgba(20,33,61,.10); }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; }}

/* Sidebar */
section[data-testid="stSidebar"] .sb-brand {{
  display:flex; align-items:center; gap:10px; margin: 4px 0 14px;
}}
section[data-testid="stSidebar"] .sb-brand .p {{
  width:34px; height:34px; border-radius:8px; background:#fff; color:{BLUE};
  font-weight:800; font-size:20px; display:flex; align-items:center; justify-content:center;
}}
section[data-testid="stSidebar"] .sb-brand b {{ font-size:1.05rem; color:#fff; }}
section[data-testid="stSidebar"] .sb-head {{
  font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
  color:#8FB3E8; margin: 18px 0 4px;
}}
</style>
""",
    unsafe_allow_html=True,
)


def esc(text) -> str:
    return html.escape(str(text))


def section(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="section"><h3>{esc(title)}</h3><span>{esc(subtitle)}</span></div>',
        unsafe_allow_html=True,
    )


def kpi_cards(cards) -> None:
    """cards: list of (label, value, sub, accent_hex)."""
    items = "".join(
        f'<div class="kpi" style="--accent:{a}"><div class="label"><span class="dot"></span>'
        f'{esc(lbl)}</div><div class="value">{esc(v)}</div><div class="sub">{esc(sub)}</div></div>'
        for lbl, v, sub, a in cards
    )
    st.markdown(f'<div class="kpi-row">{items}</div>', unsafe_allow_html=True)


def capacity_bar(free: int, occupied: int) -> None:
    total = max(free + occupied, 1)
    fp, op = 100 * free / total, 100 * occupied / total
    st.markdown(
        f"""<div class="capacity">
  <div class="bar" role="img" aria-label="{occupied} occupied, {free} free">
    <span style="width:{op:.2f}%;background:{OCC_COLOR}"></span>
    <span style="width:{fp:.2f}%;background:{FREE_COLOR}"></span>
  </div>
  <div class="legend">
    <span><i class="sw" style="background:{OCC_COLOR}"></i>Occupied <b>{occupied}</b> ({op:.1f}%)</span>
    <span><i class="sw" style="background:{FREE_COLOR}"></i>Free <b>{free}</b> ({fp:.1f}%)</span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def style_chart(chart: alt.Chart) -> alt.Chart:
    """Recessive axes and grid, Inter font, no outer border."""
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelFont="Inter", titleFont="Inter", labelColor=MUTED, titleColor=MUTED,
            gridColor=GRID, domainColor=GRID, tickColor=GRID, labelFontSize=11,
            titleFontSize=12, titleFontWeight=600,
        )
        .configure_legend(labelFont="Inter", titleFont="Inter", labelColor=INK,
                          orient="top", title=None, symbolType="square")
    )


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
def _mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_data(show_spinner=False)
def load_image_from_path(path: str, _mtime_key: float):
    return cv2.imread(path)


@st.cache_data(show_spinner=False)
def decode_image(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


@st.cache_data(show_spinner=False)
def cached_positions(path: str, _mtime_key: float):
    return pc.load_positions(path)


@st.cache_data(show_spinner=False)
def cached_log(path: str, _mtime_key: float) -> pd.DataFrame:
    return oa.load_log(path)


@st.cache_data(show_spinner=False)
def cached_analysis(frame: np.ndarray, positions: tuple, threshold: int, ref_size):
    return pc.analyze_frame(frame, list(positions), threshold=threshold, ref_size=ref_size)


# Cache keys include the file's mtime, so edits on disk are picked up.
positions = cached_positions(config.POSITION_FILE, _mtime(config.POSITION_FILE))
ref_img = load_image_from_path(config.DEFAULT_IMAGE_PATH, _mtime(config.DEFAULT_IMAGE_PATH))
ref_size = (ref_img.shape[1], ref_img.shape[0]) if ref_img is not None else None

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
sb = st.sidebar
sb.markdown('<div class="sb-brand"><div class="p">P</div><b>Smart Parking</b></div>',
            unsafe_allow_html=True)


def sb_head(text: str) -> None:
    sb.markdown(f'<div class="sb-head">{esc(text)}</div>', unsafe_allow_html=True)


sb_head("Image source")
source = sb.radio("Image", ["Sample image", "Upload"], label_visibility="collapsed",
                  horizontal=True)
frame = None
if source == "Sample image":
    frame = ref_img
    if frame is None:
        sb.error(f"{config.DEFAULT_IMAGE_PATH} not found.")
else:
    up = sb.file_uploader("Parking-lot image", type=["jpg", "jpeg", "png", "bmp"])
    if up is not None:
        frame = decode_image(up.getvalue())
        if frame is None:
            sb.error("Could not decode that image.")

sb_head("Detection")
threshold = sb.slider(
    "Occupancy threshold (pixels)",
    min_value=100,
    max_value=3000,
    value=int(config.OCCUPANCY_THRESHOLD),
    step=25,
    help="A space counts as occupied when its edge-pixel count is at or above this value "
    "(scaled automatically for larger video frames).",
)

sb_head("Parking guidance")
_guide_size = ref_size or ((frame.shape[1], frame.shape[0]) if frame is not None else (1280, 720))
_default_entrance = config.ENTRANCE_POINT or pc.default_entrance(_guide_size)
e1, e2 = sb.columns(2)
entrance = (
    int(e1.number_input("Entrance x", 0, _guide_size[0] - 1, int(_default_entrance[0]))),
    int(e2.number_input("Entrance y", 0, _guide_size[1] - 1, int(_default_entrance[1]))),
)
top_n = sb.slider("Suggestions to show", 1, 10, 3)
sb.caption("Entrance is in the marked image's pixels. Set a permanent default "
           "with `ENTRANCE_POINT` in config.py.")

sb_head("Occupancy log")
real_log = oa.default_log_path()
log_choice = sb.radio(
    "Log file",
    ["Real log", "Sample data (synthetic)"],
    help=f"Real: {real_log}  ·  Sample: {oa.SAMPLE_LOG}",
    label_visibility="collapsed",
)
is_sample = log_choice != "Real log"
log_path = oa.SAMPLE_LOG if is_sample else real_log
sb.caption(f"`{log_path}`")
if is_sample:
    sb.warning("Showing SYNTHETIC sample data, not real detections.")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
chips = [f"{len(positions)} spaces marked", f"Threshold {threshold}px"]
chips_html = "".join(f'<span class="chip">{esc(c)}</span>' for c in chips)
if is_sample:
    chips_html += '<span class="chip warn">Sample data</span>'
st.markdown(
    f"""<div class="hero">
  <div class="sign">P</div>
  <div><h1>Smart Parking Dashboard</h1>
  <p>Live space detection · nearest-space guidance · hourly occupancy trends</p></div>
  <div class="chips">{chips_html}</div>
</div>""",
    unsafe_allow_html=True,
)

tab_live, tab_hourly, tab_video, tab_layout = st.tabs(
    ["📷  Live", "📈  Hourly trends", "🎬  Video", "🧩  Layout"]
)

# --------------------------------------------------------------------------- #
# Live
# --------------------------------------------------------------------------- #
with tab_live:
    if frame is None:
        st.info("Choose the sample image or upload one in the sidebar.")
    elif not positions:
        st.warning("No parking spaces are marked yet. Use the **Layout** tab or `python run.py`.")
        st.image(frame, channels="BGR", width="stretch")
    else:
        result = cached_analysis(frame, tuple(positions), threshold, ref_size)
        if ref_size and result["scale"] == 1.0 and (frame.shape[1], frame.shape[0]) != ref_size:
            st.warning(
                "This image's size/aspect ratio differs from the image the spaces were "
                "marked on, so positions may not line up. Re-mark spaces for this image."
            )

        # Percentages from raw counts so the cards and the capacity bar agree
        occ_pct = 100 * result["occupied"] / max(result["total"], 1)
        if occ_pct >= 85:
            busy_label = "busy"
        elif occ_pct >= 50:
            busy_label = "moderate"
        else:
            busy_label = "plenty of room"
        kpi_cards([
            ("Total spaces", result["total"], "marked in the layout", BLUE),
            ("Occupied", result["occupied"], f"{occ_pct:.1f}% of the lot", OCC_COLOR),
            ("Free", result["free"], f"{100 - occ_pct:.1f}% available", FREE_COLOR),
            ("Occupancy", f"{occ_pct:.1f}%", busy_label, NAVY),
        ])
        capacity_bar(result["free"], result["occupied"])

        # ---- Where should a new car park? ----
        recs = pc.recommend_spaces(result, entrance, top_n=top_n)
        s = result["scale"]
        entrance_px = (int(entrance[0] * s), int(entrance[1] * s))
        if recs:
            best = recs[0]
            alts = "".join(
                f'<span class="alt">#{r["id"]} · {r["distance_px"]:.0f}px</span>' for r in recs[1:]
            )
            alts_html = f'<div class="alts"><span style="color:{MUTED};font-size:.8rem">Alternatives</span>{alts}</div>' if alts else ""
            st.markdown(
                f"""<div class="rec">
  <div class="badge"><small>Space</small><b>#{best['id']}</b></div>
  <div><h3>🚗 Park here: space #{best['id']}</h3>
  <p>The nearest free space to the entrance, about {best['distance_px']:.0f} px away in a straight line.</p></div>
  {alts_html}
</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="rec full"><div class="badge"><small>Status</small><b>FULL</b></div>
  <div><h3>Lot full</h3><p>There are no free spaces right now.</p></div></div>""",
                unsafe_allow_html=True,
            )

        annotated = pc.annotate(frame, result)
        annotated = pc.annotate_recommendation(annotated, recs, entrance_px)
        st.image(annotated, channels="BGR", width="stretch")
        st.markdown(
            f'<div class="capacity legend" style="border:none;background:transparent;padding:4px 2px;margin:0">'
            f'<span><i class="sw" style="background:{FREE_COLOR}"></i>Free</span> &nbsp; '
            f'<span><i class="sw" style="background:{OCC_COLOR}"></i>Occupied</span> &nbsp; '
            f'<span><i class="sw" style="background:{BLUE}"></i>Suggested (#1 = nearest)</span> &nbsp; '
            f'<span><i class="sw" style="background:{AMBER}"></i>Entrance</span></div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([1, 3])
        if c1.button("💾  Save this reading to log", type="primary", width="stretch"):
            path = oa.append_reading(result["total"], result["occupied"], result["free"],
                                     path=oa.DEFAULT_LOG)
            st.toast(f"Saved to {path} at {datetime.now():%H:%M:%S}", icon="✅")

        with st.expander("Per-space details"):
            st.dataframe(
                pd.DataFrame(result["spaces"]),
                hide_index=True,
                column_config={
                    "occupied": st.column_config.CheckboxColumn("Occupied"),
                    "in_frame": st.column_config.CheckboxColumn("In frame"),
                    "pixel_count": st.column_config.ProgressColumn(
                        "Edge pixels", min_value=0,
                        max_value=int(max(result["threshold"] * 2, 1)), format="%d"),
                },
            )

# --------------------------------------------------------------------------- #
# Hourly trends
# --------------------------------------------------------------------------- #
with tab_hourly:
    if is_sample:
        st.info("📊 **Sample data**: generated by `generate_sample_data.py`, not real detections.")
    try:
        log = cached_log(log_path, _mtime(log_path))
    except ValueError as e:
        st.error(str(e))
        log = oa._empty_log()

    if log.empty:
        st.info(
            f"No readings in `{log_path}` yet.\n\n"
            "- Run detection: **Live** tab → *Save this reading to log*, the **Video** tab, "
            "or `python run.py` (press **D**).\n"
            "- Or create demo data: `python generate_sample_data.py`, then pick "
            "*Sample data* in the sidebar."
        )
    else:
        days = sorted(log["Timestamp"].dt.date.unique(), reverse=True)
        fcol, _ = st.columns([1, 3])
        day = fcol.selectbox(
            "Day", ["All days"] + days,
            format_func=lambda d: d if isinstance(d, str) else d.strftime("%a %d %b %Y"),
        )
        sel = oa.filter_day(log, None if day == "All days" else day)
        summary = oa.hourly_summary(sel)
        bq = oa.busiest_and_quietest(summary)

        avg_occ = sel["Occupancy %"].mean()
        cards = [("Readings", f"{len(sel):,}", "in the selected period", BLUE)]
        if bq["busiest"]:
            cards += [
                ("Busiest hour", oa.hour_label(bq["busiest"]["hour"]),
                 f"{bq['busiest']['avg_occupancy_pct']:.1f}% average occupancy", OCC_COLOR),
                ("Quietest hour", oa.hour_label(bq["quietest"]["hour"]),
                 f"{bq['quietest']['avg_occupancy_pct']:.1f}% average occupancy", FREE_COLOR),
            ]
        cards.append(("Average occupancy", f"{avg_occ:.1f}%", "across all readings", NAVY))
        kpi_cards(cards)

        section("Average occupancy by hour", "Peak hour highlighted · hover a bar for details")
        chart_df = summary.copy()
        peak_hour = bq["busiest"]["hour"] if bq["busiest"] else -1
        chart_df["is_peak"] = chart_df["hour"] == peak_hour
        chart_df["label"] = chart_df["hour"].map(lambda h: f"{h:02d}:00")
        tooltip = [
            alt.Tooltip("label:N", title="Hour"),
            alt.Tooltip("readings:Q", title="Readings"),
            alt.Tooltip("avg_occupancy_pct:Q", title="Avg occupancy %", format=".1f"),
            alt.Tooltip("peak_occupancy_pct:Q", title="Peak occupancy %", format=".1f"),
            alt.Tooltip("avg_available:Q", title="Avg free", format=".1f"),
            alt.Tooltip("min_available:Q", title="Min free"),
        ]
        base = alt.Chart(chart_df).encode(
            x=alt.X("hour:O", title="Hour of day", axis=alt.Axis(labelAngle=0)),
        )
        bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=22).encode(
            y=alt.Y("avg_occupancy_pct:Q", title="Average occupancy (%)",
                    scale=alt.Scale(domain=[0, 100])),
            color=alt.condition("datum.is_peak", alt.value(OCC_COLOR), alt.value(BLUE)),
            tooltip=tooltip,
        )
        peak_text = f"Peak {bq['busiest']['avg_occupancy_pct']:.0f}%" if bq["busiest"] else ""
        peak_lbl = base.transform_filter("datum.is_peak").mark_text(
            dy=-10, color=INK, fontWeight="bold", font="Inter", fontSize=12
        ).encode(y="avg_occupancy_pct:Q", text=alt.value(peak_text))
        st.altair_chart(style_chart((bars + peak_lbl).properties(height=340)), width="stretch")

        # Day-of-week x hour heatmap (only meaningful with several days of data)
        if day == "All days" and sel["Timestamp"].dt.date.nunique() >= 3:
            section("Weekly pattern", "Average occupancy by weekday and hour")
            hm = sel.assign(
                weekday=sel["Timestamp"].dt.day_name().str[:3],
                hour=sel["Timestamp"].dt.hour,
            ).groupby(["weekday", "hour"], as_index=False)["Occupancy %"].mean()
            order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            heat = alt.Chart(hm).mark_rect(cornerRadius=3, stroke="#fff", strokeWidth=2).encode(
                x=alt.X("hour:O", title="Hour of day", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("weekday:N", sort=order, title=None),
                color=alt.Color("Occupancy %:Q", title="Avg occupancy %",
                                scale=alt.Scale(scheme="blues", domain=[0, 100])),
                tooltip=[alt.Tooltip("weekday:N", title="Day"),
                         alt.Tooltip("hour:O", title="Hour"),
                         alt.Tooltip("Occupancy %:Q", title="Avg occupancy %", format=".1f")],
            ).properties(height=250)
            st.altair_chart(style_chart(heat).configure_legend(orient="right"), width="stretch")

        section("Hourly summary", "Hours with at least one reading")
        st.dataframe(
            summary[summary["readings"] > 0],
            hide_index=True,
            width="stretch",
            column_config={
                "hour": st.column_config.NumberColumn("Hour", format="%02d:00"),
                "readings": st.column_config.NumberColumn("Readings"),
                "avg_occupancy_pct": st.column_config.ProgressColumn(
                    "Avg occupancy", min_value=0, max_value=100, format="%.1f%%"),
                "peak_occupancy_pct": st.column_config.NumberColumn("Peak %", format="%.1f%%"),
                "avg_available": st.column_config.NumberColumn("Avg free", format="%.1f"),
                "min_available": st.column_config.NumberColumn("Min free"),
            },
        )
        st.download_button(
            "⬇  Download hourly summary (CSV)",
            summary.to_csv(index=False).encode(),
            file_name=f"hourly_summary_{'all' if day == 'All days' else day}.csv",
            mime="text/csv",
        )

# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #
with tab_video:
    with st.container(border=True):
        section("Analyse a video", "Samples a frame every N seconds and counts free spaces")
        vsrc = st.radio("Video", [f"Sample ({config.DEFAULT_VIDEO_PATH})", "Upload"],
                        horizontal=True, label_visibility="collapsed")
        video_path = None
        if vsrc == "Upload":
            vup = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
            if vup is not None:
                suffix = os.path.splitext(vup.name)[1] or ".mp4"
                tmp = os.path.join(tempfile.gettempdir(),
                                   f"parking_upload_{abs(hash(vup.name))}{suffix}")
                with open(tmp, "wb") as f:
                    f.write(vup.getbuffer())
                video_path = tmp
        elif os.path.exists(config.DEFAULT_VIDEO_PATH):
            video_path = config.DEFAULT_VIDEO_PATH
        else:
            st.warning(f"{config.DEFAULT_VIDEO_PATH} not found.")

        c1, c2 = st.columns(2)
        every = c1.number_input("Sample every N seconds", min_value=0.5, max_value=600.0,
                                value=5.0, step=0.5)
        use_start = c2.checkbox("Recording started at…",
                                help="Adds real timestamps (start + video offset) to each sample.")
        start_dt = None
        if use_start:
            d1, d2 = st.columns(2)
            sd = d1.date_input("Date", value=datetime.now().date())
            stime = d2.time_input("Time", value=time(9, 0), step=60)
            start_dt = datetime.combine(sd, stime)

        run = st.button("▶  Analyse video", type="primary",
                        disabled=video_path is None or not positions)

    if run:
        bar = st.progress(0.0, text="Analysing…")
        try:
            samples = pc.analyze_video(
                video_path, float(every), start_dt, positions=positions,
                threshold=threshold, ref_size=ref_size,
                progress=lambda p: bar.progress(min(p, 1.0), text=f"Analysing… {p:.0%}"),
            )
            st.session_state["video_samples"] = samples
            st.session_state["video_start"] = start_dt
        except Exception as e:  # noqa: BLE001
            st.error(f"Video analysis failed: {e}")
        finally:
            bar.empty()

    samples = st.session_state.get("video_samples")
    if samples:
        vdf = pd.DataFrame(samples)
        has_ts = "timestamp" in vdf.columns
        kpi_cards([
            ("Samples", len(vdf), f"every {every:g}s", BLUE),
            ("Avg occupied", f"{vdf['occupied'].mean():.0f}", f"max {vdf['occupied'].max()}", OCC_COLOR),
            ("Avg free", f"{vdf['free'].mean():.0f}", f"min {vdf['free'].min()}", FREE_COLOR),
            ("Avg occupancy", f"{vdf['occupancy_pct'].mean():.1f}%",
             f"of {vdf['total'].iloc[0]} spaces", NAVY),
        ])
        section("Spaces over time", "Positions are scaled to the video resolution when it differs")
        x = "timestamp" if has_ts else "offset_s"
        long = vdf.melt(id_vars=[x], value_vars=["occupied", "free"],
                        var_name="status", value_name="spaces")
        long["status"] = long["status"].str.capitalize()
        line = alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40)).encode(
            x=alt.X(f"{x}:{'T' if has_ts else 'Q'}",
                    title="Time" if has_ts else "Video offset (s)"),
            y=alt.Y("spaces:Q", title="Spaces"),
            color=alt.Color("status:N", scale=alt.Scale(domain=["Occupied", "Free"],
                                                        range=[OCC_COLOR, FREE_COLOR])),
            tooltip=[alt.Tooltip(f"{x}:{'T' if has_ts else 'Q'}",
                                 title="Time" if has_ts else "Offset (s)"),
                     alt.Tooltip("status:N", title="Status"),
                     alt.Tooltip("spaces:Q", title="Spaces")],
        ).properties(height=320)
        st.altair_chart(style_chart(line), width="stretch")
        with st.expander("Per-sample results"):
            st.dataframe(vdf, hide_index=True, width="stretch")
        if has_ts:
            if st.button("📝  Log these samples (real timestamps)"):
                for smp in samples:
                    oa.append_reading(smp["total"], smp["occupied"], smp["free"],
                                      timestamp=smp["timestamp"], path=oa.DEFAULT_LOG)
                st.success(f"Logged {len(samples)} readings to `{oa.DEFAULT_LOG}`.")
        else:
            st.caption("Tick *Recording started at…* and re-analyse to log samples with real timestamps.")
    elif not positions:
        st.info("Mark parking spaces first (Layout tab).")

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
with tab_layout:
    base_img = frame if frame is not None else ref_img
    left, right = st.columns(2, gap="large")
    with left:
        section("Current layout", f"{len(positions)} spaces in {config.POSITION_FILE}")
        if base_img is not None:
            st.image(pc.draw_positions(base_img, positions, color=pc.COLOR_GUIDE),
                     channels="BGR", width="stretch")
        st.caption("The shipped `CarParkPos` is a starter layout. Re-mark spaces for your own image.")

    with right:
        with st.container(border=True):
            section("Grid builder", "Add a block of spaces with a live preview")
            first = positions[0] if positions else (100, 60)
            g1, g2 = st.columns(2)
            sx = g1.number_input("Start x", 0, 10000, int(first[0]))
            sy = g2.number_input("Start y", 0, 10000, int(first[1]))
            cols = g1.number_input("Columns", 1, 200, 1)
            rows = g2.number_input("Rows", 1, 200, 10)
            cstep = g1.number_input("Column step (px)", 1, 2000, config.PARKING_WIDTH)
            rstep = g2.number_input("Row step (px)", 1, 2000, config.PARKING_HEIGHT)
            mode = st.radio("On save", ["Add to current layout", "Replace current layout"],
                            horizontal=True)
            grid = pc.grid_positions(sx, sy, cols, rows, cstep, rstep)
            if mode.startswith("Add"):
                new_layout = list(positions) + [p for p in grid if p not in set(positions)]
            else:
                new_layout = grid
            if base_img is not None:
                preview = pc.draw_positions(base_img, positions if mode.startswith("Add") else [],
                                            color=(160, 160, 160))
                preview = pc.draw_positions(preview, grid, color=pc.COLOR_ENTRANCE)
                st.image(preview, channels="BGR", width="stretch",
                         caption=f"Preview: {len(grid)} new spaces (amber) → "
                         f"{len(new_layout)} total")
            if st.button("💾  Save layout", type="primary"):
                if os.path.exists(config.POSITION_FILE):
                    shutil.copyfile(config.POSITION_FILE, config.POSITION_FILE + ".bak")
                pc.save_positions(new_layout, config.POSITION_FILE)
                st.toast(f"Saved {len(new_layout)} spaces to {config.POSITION_FILE} "
                         f"(backup: {config.POSITION_FILE}.bak).")
                st.cache_data.clear()
                st.rerun()

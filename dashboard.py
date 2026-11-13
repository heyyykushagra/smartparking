"""
Streamlit dashboard for the Car Parking Detection project.

    streamlit run dashboard.py

Tabs: Live (single image), Hourly trends (log analytics), Video (sampled
analysis), Layout (view / build the CarParkPos layout).
"""

from __future__ import annotations

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

FREE_COLOR = "#1E9E4A"
OCC_COLOR = "#D62728"
BLUE = "#0057B8"

st.set_page_config(page_title="Parking Occupancy Dashboard", page_icon="🅿️", layout="wide")


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
st.sidebar.title("🅿️ Parking dashboard")

st.sidebar.subheader("Image source")
source = st.sidebar.radio("Image", ["Sample image", "Upload"], label_visibility="collapsed")
frame = None
if source == "Sample image":
    frame = ref_img
    if frame is None:
        st.sidebar.error(f"{config.DEFAULT_IMAGE_PATH} not found.")
else:
    up = st.sidebar.file_uploader("Parking-lot image", type=["jpg", "jpeg", "png", "bmp"])
    if up is not None:
        frame = decode_image(up.getvalue())
        if frame is None:
            st.sidebar.error("Could not decode that image.")

st.sidebar.subheader("Detection")
threshold = st.sidebar.slider(
    "Occupancy threshold (pixels)",
    min_value=100,
    max_value=3000,
    value=int(config.OCCUPANCY_THRESHOLD),
    step=25,
    help="A space counts as occupied when its edge-pixel count is at or above this value "
    "(scaled automatically for larger video frames).",
)

st.sidebar.subheader("Occupancy log")
real_log = oa.default_log_path()
log_choice = st.sidebar.radio(
    "Log file",
    ["Real log", "Sample data (synthetic)"],
    help=f"Real: {real_log}  ·  Sample: {oa.SAMPLE_LOG}",
)
log_path = real_log if log_choice == "Real log" else oa.SAMPLE_LOG
st.sidebar.caption(f"`{log_path}`")
if log_choice != "Real log":
    st.sidebar.warning("Showing SYNTHETIC sample data, not real detections.")
st.sidebar.caption(f"Layout: **{len(positions)}** spaces from `{config.POSITION_FILE}`")

st.title("Parking occupancy")
tab_live, tab_hourly, tab_video, tab_layout = st.tabs(
    ["Live", "Hourly trends", "Video", "Layout"]
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
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total spaces", result["total"])
        c2.metric("Occupied", result["occupied"])
        c3.metric("Free", result["free"])
        c4.metric("Occupancy", f"{result['occupancy_pct']:.1f}%")

        st.image(
            pc.annotate(frame, result),
            channels="BGR",
            width="stretch",
            caption="Green = free · Red = occupied",
        )

        if st.button("Save this reading to log", type="primary"):
            path = oa.append_reading(result["total"], result["occupied"], result["free"],
                                     path=oa.DEFAULT_LOG)
            st.success(f"Saved to `{path}` at {datetime.now():%H:%M:%S}.")

        with st.expander("Per-space details"):
            st.dataframe(pd.DataFrame(result["spaces"]), hide_index=True)

# --------------------------------------------------------------------------- #
# Hourly trends
# --------------------------------------------------------------------------- #
with tab_hourly:
    if log_choice != "Real log":
        st.info("📊 **Sample data** - generated by `generate_sample_data.py`, not real detections.")
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
        day = st.selectbox(
            "Day", ["All days"] + days,
            format_func=lambda d: d if isinstance(d, str) else d.strftime("%a %d %b %Y"),
        )
        sel = oa.filter_day(log, None if day == "All days" else day)
        summary = oa.hourly_summary(sel)
        bq = oa.busiest_and_quietest(summary)

        c1, c2, c3 = st.columns(3)
        c1.metric("Readings", len(sel))
        if bq["busiest"]:
            c2.metric("Busiest hour", oa.hour_label(bq["busiest"]["hour"]),
                      f"{bq['busiest']['avg_occupancy_pct']:.1f}% avg", delta_color="off")
            c3.metric("Quietest hour", oa.hour_label(bq["quietest"]["hour"]),
                      f"{bq['quietest']['avg_occupancy_pct']:.1f}% avg", delta_color="off")

        chart_df = summary.copy()
        peak_hour = bq["busiest"]["hour"] if bq["busiest"] else -1
        chart_df["is_peak"] = chart_df["hour"] == peak_hour
        base = alt.Chart(chart_df).encode(
            x=alt.X("hour:O", title="Hour of day"),
        )
        bars = base.mark_bar().encode(
            y=alt.Y("avg_occupancy_pct:Q", title="Avg occupancy (%)",
                    scale=alt.Scale(domain=[0, 100])),
            color=alt.condition("datum.is_peak", alt.value(OCC_COLOR), alt.value(BLUE)),
            tooltip=["hour", "readings", "avg_occupancy_pct", "peak_occupancy_pct",
                     "avg_available", "min_available"],
        )
        peak_lbl = base.transform_filter("datum.is_peak").mark_text(
            dy=-8, color=OCC_COLOR, fontWeight="bold"
        ).encode(
            y="avg_occupancy_pct:Q",
            text=alt.value("▼ peak"),
        )
        st.altair_chart(bars + peak_lbl, width="stretch")

        st.subheader("Hourly summary")
        st.dataframe(summary[summary["readings"] > 0], hide_index=True)
        st.download_button(
            "Download hourly summary (CSV)",
            summary.to_csv(index=False).encode(),
            file_name=f"hourly_summary_{'all' if day == 'All days' else day}.csv",
            mime="text/csv",
        )

# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #
with tab_video:
    vsrc = st.radio("Video", [f"Sample ({config.DEFAULT_VIDEO_PATH})", "Upload"],
                    horizontal=True)
    video_path = None
    if vsrc == "Upload":
        vup = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
        if vup is not None:
            suffix = os.path.splitext(vup.name)[1] or ".mp4"
            tmp = os.path.join(tempfile.gettempdir(), f"parking_upload_{abs(hash(vup.name))}{suffix}")
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

    if st.button("Analyse video", type="primary", disabled=video_path is None or not positions):
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
        st.caption(f"{len(vdf)} samples · {vdf['total'].iloc[0]} spaces · positions scaled "
                   "to the video resolution when it differs from the marked image")
        x = "timestamp" if has_ts else "offset_s"
        st.line_chart(vdf, x=x, y=["occupied", "free"], color=[OCC_COLOR, FREE_COLOR],
                      x_label="Time" if has_ts else "Video offset (s)", y_label="Spaces")
        st.dataframe(vdf, hide_index=True)
        if has_ts:
            if st.button("Log these samples (real timestamps)"):
                for s in samples:
                    oa.append_reading(s["total"], s["occupied"], s["free"],
                                      timestamp=s["timestamp"], path=oa.DEFAULT_LOG)
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
    left, right = st.columns(2)
    with left:
        st.subheader(f"Current layout ({len(positions)} spaces)")
        if base_img is not None:
            st.image(pc.draw_positions(base_img, positions), channels="BGR", width="stretch")
        st.caption("The shipped `CarParkPos` is a starter layout - re-mark spaces for your own image.")

    with right:
        st.subheader("Grid builder")
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
            preview = pc.draw_positions(preview, grid, color=(184, 87, 0))
            st.image(preview, channels="BGR", width="stretch",
                     caption=f"Preview: {len(grid)} new (blue) → {len(new_layout)} total")
        if st.button("Save layout", type="primary"):
            if os.path.exists(config.POSITION_FILE):
                shutil.copyfile(config.POSITION_FILE, config.POSITION_FILE + ".bak")
            pc.save_positions(new_layout, config.POSITION_FILE)
            st.toast(f"Saved {len(new_layout)} spaces to {config.POSITION_FILE} "
                     f"(backup: {config.POSITION_FILE}.bak).")
            st.cache_data.clear()
            st.rerun()

"""
Headless parking-occupancy core.

Same CV pipeline as EnhancedParkingDetector.process_frame
(gray -> GaussianBlur -> adaptiveThreshold -> medianBlur -> dilate ->
countNonZero per space vs config.OCCUPANCY_THRESHOLD), but with no GUI
windows and no YOLO import, so it can run inside Streamlit and in tests.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import config

Position = Tuple[int, int]

# Dashboard palette in BGR (pair validated for colour-blind separation).
# The desktop tool (run.py) keeps its own colours from config.py.
COLOR_EMPTY = (156, 163, 27)  # #1BA39C teal-green = free
COLOR_OCCUPIED = (47, 52, 209)  # #D1342F red = occupied


# --------------------------------------------------------------------------- #
# Parking positions
# --------------------------------------------------------------------------- #
def _normalise(items) -> List[Position]:
    out: List[Position] = []
    for item in items:
        x, y = item[0], item[1]
        out.append((int(x), int(y)))
    return out


def load_positions(path: str = config.POSITION_FILE) -> List[Position]:
    """Load parking positions from a pickle (what the GUI saves) or a
    plain-text file with one "x,y" pair per line. Missing file -> []."""
    if not path or not os.path.exists(path):
        return []

    with open(path, "rb") as f:
        raw = f.read()
    if not raw.strip():
        return []

    # 1) Pickle
    try:
        data = pickle.loads(raw)
        if isinstance(data, (list, tuple)):
            return _normalise(data)
    except Exception:
        pass

    # 2) Plain text "x,y" lines (blank lines and # comments ignored)
    positions: List[Position] = []
    text = raw.decode("utf-8", errors="strict")
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(";", ",").replace(" ", ",").split(",")
        parts = [p for p in parts if p]
        if len(parts) < 2:
            raise ValueError(f"{path}:{lineno}: expected 'x,y', got {line!r}")
        positions.append((int(float(parts[0])), int(float(parts[1]))))
    return positions


def save_positions(positions: Sequence[Position], path: str = config.POSITION_FILE) -> None:
    """Save positions as a pickled list of (x, y) tuples (GUI-compatible)."""
    with open(path, "wb") as f:
        pickle.dump(_normalise(positions), f)


def grid_positions(
    start_x: int, start_y: int, cols: int, rows: int, col_step: int, row_step: int
) -> List[Position]:
    """Build a simple grid layout, column by column (same order as run.py)."""
    return [
        (int(start_x + c * col_step), int(start_y + r * row_step))
        for c in range(max(0, cols))
        for r in range(max(0, rows))
    ]


# --------------------------------------------------------------------------- #
# Resolution scaling
# --------------------------------------------------------------------------- #
def scale_factor(
    frame_size: Tuple[int, int], ref_size: Optional[Tuple[int, int]], tol: float = 0.01
) -> float:
    """Scale from reference image to frame (sizes are (w, h)).

    Returns 1.0 when there is no reference or the aspect ratios differ."""
    if not ref_size:
        return 1.0
    fw, fh = frame_size
    rw, rh = ref_size
    if min(fw, fh, rw, rh) <= 0:
        return 1.0
    sx, sy = fw / rw, fh / rh
    if abs(sx - sy) / max(sx, sy) > tol:
        return 1.0
    return (sx + sy) / 2.0


def scale_layout(
    positions: Sequence[Position],
    width: int,
    height: int,
    threshold: float,
    scale: float,
) -> Tuple[List[Position], int, int, float]:
    """Scale positions and box size by `scale`, threshold by `scale**2`."""
    if scale == 1.0:
        return list(positions), width, height, threshold
    pos = [(int(round(x * scale)), int(round(y * scale))) for x, y in positions]
    return pos, int(round(width * scale)), int(round(height * scale)), threshold * scale * scale


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def preprocess(frame: np.ndarray) -> np.ndarray:
    """Identical to EnhancedParkingDetector.process_frame."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
    )
    median = cv2.medianBlur(thresh, 5)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(median, kernel, iterations=1)


def analyze_frame(
    frame: np.ndarray,
    positions: Sequence[Position],
    width: int = config.PARKING_WIDTH,
    height: int = config.PARKING_HEIGHT,
    threshold: float = config.OCCUPANCY_THRESHOLD,
    ref_size: Optional[Tuple[int, int]] = None,
) -> Dict:
    """Analyse one BGR frame.

    If `ref_size` (w, h of the image the positions were marked on) is given and
    the frame has the same aspect ratio but a different size, positions, box
    size and threshold are scaled to the frame.
    """
    h_img, w_img = frame.shape[:2]
    scale = scale_factor((w_img, h_img), ref_size)
    positions, width, height, threshold = scale_layout(positions, width, height, threshold, scale)

    processed = preprocess(frame)
    spaces = []
    for idx, (x, y) in enumerate(positions, start=1):
        crop = processed[max(0, y) : y + height, max(0, x) : x + width]
        count = int(cv2.countNonZero(crop)) if crop.size else 0
        spaces.append(
            {
                "id": idx,
                "x": int(x),
                "y": int(y),
                "w": int(width),
                "h": int(height),
                "pixel_count": count,
                "occupied": count >= threshold,
                # Box fully inside the frame (spaces outside it can't be checked)
                "in_frame": x >= 0 and y >= 0 and x + width <= w_img and y + height <= h_img,
            }
        )

    total = len(spaces)
    occupied = sum(1 for s in spaces if s["occupied"])
    free = total - occupied
    return {
        "spaces": spaces,
        "total": total,
        "occupied": occupied,
        "free": free,
        "occupancy_pct": round(100.0 * occupied / total, 2) if total else 0.0,
        "scale": scale,
        "threshold": threshold,
    }


def annotate(
    frame: np.ndarray, result: Dict, show_ids: bool = True, thickness: int = 2
) -> np.ndarray:
    """Return a copy of `frame` with spaces drawn (green = empty, red = occupied)."""
    img = frame.copy()
    for s in result["spaces"]:
        color = COLOR_OCCUPIED if s["occupied"] else COLOR_EMPTY
        p1 = (s["x"], s["y"])
        p2 = (s["x"] + s["w"], s["y"] + s["h"])
        cv2.rectangle(img, p1, p2, color, thickness + (0 if s["occupied"] else 1))
        if show_ids:
            cv2.putText(
                img,
                str(s["id"]),
                (s["x"] + 4, s["y"] + s["h"] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * max(result.get("scale", 1.0), 1.0),
                color,
                1,
                cv2.LINE_AA,
            )
    return img


# --------------------------------------------------------------------------- #
# Parking guidance: nearest free space
# --------------------------------------------------------------------------- #
COLOR_GUIDE = (184, 87, 0)  # BGR parking-sign blue
COLOR_ENTRANCE = (0, 200, 255)  # BGR amber


def default_entrance(frame_size: Tuple[int, int]) -> Position:
    """Bottom-centre of a frame of size (w, h)."""
    w, h = frame_size
    return (w // 2, h - 1)


def recommend_spaces(
    result: Dict,
    entrance: Optional[Position] = None,
    frame_size: Optional[Tuple[int, int]] = None,
    top_n: int = 3,
) -> List[Dict]:
    """Rank free spaces by straight-line distance from the entrance.

    `entrance` is in the reference image's coordinates (where the spaces were
    marked); it is scaled with `result["scale"]` like the spaces. When it is
    None, config.ENTRANCE_POINT is used, else the bottom-centre of the frame
    (`frame_size` = (w, h) of the analysed frame).

    Returns up to `top_n` dicts (nearest first) with id, x, y, w, h, the space
    centre (cx, cy), distance_px (in frame pixels) and rank. Spaces whose box
    lies outside the frame are skipped. An empty list means the lot is full.
    """
    scale = result.get("scale", 1.0)
    if entrance is None:
        entrance = config.ENTRANCE_POINT
    if entrance is not None:
        ex, ey = entrance[0] * scale, entrance[1] * scale
    elif frame_size is not None:
        ex, ey = default_entrance(frame_size)
    else:
        raise ValueError("Give an entrance point or the frame size")

    ranked = []
    for s in result["spaces"]:
        if s["occupied"] or not s.get("in_frame", True):
            continue
        cx, cy = s["x"] + s["w"] / 2, s["y"] + s["h"] / 2
        dist = float(np.hypot(cx - ex, cy - ey))
        ranked.append({**s, "cx": int(cx), "cy": int(cy), "distance_px": round(dist, 1)})

    ranked.sort(key=lambda r: (r["distance_px"], r["id"]))
    for i, r in enumerate(ranked[: max(0, top_n)], start=1):
        r["rank"] = i
    return ranked[: max(0, top_n)]


def annotate_recommendation(
    img: np.ndarray,
    recommendations: Sequence[Dict],
    entrance: Position,
) -> np.ndarray:
    """Draw the entrance, an arrow to the best space and highlight alternatives.

    `entrance` must be in the same (frame) coordinates as `img`.
    Returns a copy; `img` is not modified.
    """
    out = img.copy()
    h_img, w_img = out.shape[:2]
    lw = max(2, int(round(min(w_img, h_img) / 400)))
    ex, ey = int(entrance[0]), int(entrance[1])

    for r in reversed(list(recommendations)):  # best drawn last (on top)
        best = r["rank"] == 1
        p1, p2 = (r["x"], r["y"]), (r["x"] + r["w"], r["y"] + r["h"])
        cv2.rectangle(out, p1, p2, COLOR_GUIDE, lw * (3 if best else 1) + 1)
        cv2.putText(out, f"#{r['rank']}", (r["x"] + 4, r["y"] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GUIDE, 2, cv2.LINE_AA)

    if recommendations:
        best = recommendations[0]
        cv2.arrowedLine(out, (ex, ey), (best["cx"], best["cy"]), COLOR_GUIDE, lw + 1,
                        cv2.LINE_AA, tipLength=0.04)

    cv2.circle(out, (ex, ey), lw * 6, COLOR_ENTRANCE, -1)
    cv2.circle(out, (ex, ey), lw * 6, (0, 0, 0), 2)
    label_y = ey - lw * 8 if ey - lw * 8 > 20 else ey + lw * 12
    cv2.putText(out, "ENTRANCE", (max(0, ex - 50), label_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, COLOR_ENTRANCE, 2, cv2.LINE_AA)
    return out


def draw_positions(
    frame: np.ndarray,
    positions: Sequence[Position],
    width: int = config.PARKING_WIDTH,
    height: int = config.PARKING_HEIGHT,
    color: Tuple[int, int, int] = config.COLOR_REGULAR,
    thickness: Optional[int] = None,
) -> np.ndarray:
    """Draw a layout (no analysis) onto a copy of the frame.

    thickness=None picks one that stays visible when a large image is shown small."""
    img = frame.copy()
    if thickness is None:
        thickness = max(2, int(round(max(img.shape[:2]) / 900)))
    for x, y in positions:
        cv2.rectangle(img, (x, y), (x + width, y + height), color, thickness)
    return img


def reference_size(path: str = config.DEFAULT_IMAGE_PATH) -> Optional[Tuple[int, int]]:
    """(w, h) of the image the positions were marked on, or None."""
    if path and os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            return (img.shape[1], img.shape[0])
    return None


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #
def analyze_video(
    path: str,
    every_seconds: float = 5.0,
    start_time: Optional[datetime] = None,
    positions: Optional[Sequence[Position]] = None,
    width: int = config.PARKING_WIDTH,
    height: int = config.PARKING_HEIGHT,
    threshold: float = config.OCCUPANCY_THRESHOLD,
    ref_size: Optional[Tuple[int, int]] = None,
    progress: Optional[Callable[[float], None]] = None,
) -> List[Dict]:
    """Sample a video every `every_seconds` and analyse each sampled frame.

    Each sample has offset_s, total, occupied, free, occupancy_pct, and
    `timestamp` (start_time + offset) when start_time is given.
    If ref_size is None, the size of config.DEFAULT_IMAGE_PATH is used.
    """
    if positions is None:
        positions = load_positions()
    if ref_size is None:
        ref_size = reference_size()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(every_seconds * fps)))
        targets = list(range(0, n_frames, step)) if n_frames > 0 else []

        samples: List[Dict] = []
        for i, frame_no in enumerate(targets):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                break
            res = analyze_frame(frame, positions, width, height, threshold, ref_size)
            offset = frame_no / fps
            sample = {
                "frame": frame_no,
                "offset_s": round(offset, 2),
                "total": res["total"],
                "occupied": res["occupied"],
                "free": res["free"],
                "occupancy_pct": res["occupancy_pct"],
            }
            if start_time is not None:
                sample["timestamp"] = start_time + timedelta(seconds=offset)
            samples.append(sample)
            if progress:
                progress((i + 1) / len(targets))
        return samples
    finally:
        cap.release()

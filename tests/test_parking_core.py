"""Tests for parking_core (headless; synthetic images only, no GUI/YOLO)."""

import os
import pickle
import sys
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import parking_core as pc

W, H = config.PARKING_WIDTH, config.PARKING_HEIGHT


def make_lot(size=(400, 300), occupied=((20, 20),), empty=((200, 20),)):
    """Plain grey lot; 'occupied' spaces get a busy checkerboard (lots of edges)."""
    w, h = size
    img = np.full((h, w, 3), 128, np.uint8)
    for x, y in occupied:
        patch = np.indices((H, W)).sum(axis=0) // 4 % 2 * 255
        img[y : y + H, x : x + W] = patch[..., None].astype(np.uint8)
    return img, list(occupied) + list(empty)


# ----------------------------------------------------------------- positions
def test_load_positions_pickle(tmp_path):
    p = tmp_path / "pos"
    with open(p, "wb") as f:
        pickle.dump([(1, 2), (3, 4)], f)
    assert pc.load_positions(str(p)) == [(1, 2), (3, 4)]


def test_load_positions_pickle_lists_normalised(tmp_path):
    p = tmp_path / "pos"
    with open(p, "wb") as f:
        pickle.dump([[1, 2], [3.0, 4.0]], f)
    assert pc.load_positions(str(p)) == [(1, 2), (3, 4)]


def test_load_positions_text(tmp_path):
    p = tmp_path / "pos.txt"
    p.write_text("# header\n10,20\n\n 30 , 40 \n")
    assert pc.load_positions(str(p)) == [(10, 20), (30, 40)]


def test_load_positions_missing_and_empty(tmp_path):
    assert pc.load_positions(str(tmp_path / "nope")) == []
    e = tmp_path / "empty"
    e.write_bytes(b"")
    assert pc.load_positions(str(e)) == []


def test_load_positions_bad_text(tmp_path):
    p = tmp_path / "bad"
    p.write_text("hello\n")
    with pytest.raises(ValueError):
        pc.load_positions(str(p))


def test_save_positions_roundtrip_is_pickle(tmp_path):
    p = tmp_path / "pos"
    pc.save_positions([(5, 6), [7, 8]], str(p))
    with open(p, "rb") as f:
        assert pickle.load(f) == [(5, 6), (7, 8)]  # GUI-compatible


def test_grid_positions():
    g = pc.grid_positions(10, 20, cols=2, rows=3, col_step=100, row_step=50)
    assert len(g) == 6
    assert g[0] == (10, 20) and g[2] == (10, 120) and g[3] == (110, 20)


# ----------------------------------------------------------------- pipeline
def test_preprocess_shape_and_binary():
    img, _ = make_lot()
    out = pc.preprocess(img)
    assert out.shape == img.shape[:2]
    assert out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 255}


def test_preprocess_matches_detector_pipeline():
    img, _ = make_lot()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 16)
    expected = cv2.dilate(cv2.medianBlur(th, 5), np.ones((3, 3), np.uint8), iterations=1)
    assert np.array_equal(pc.preprocess(img), expected)


def test_analyze_frame_occupied_vs_empty():
    img, pos = make_lot()
    r = pc.analyze_frame(img, pos)
    assert r["total"] == 2
    assert r["occupied"] == 1 and r["free"] == 1
    assert r["occupancy_pct"] == 50.0
    assert r["spaces"][0]["occupied"] is True
    assert r["spaces"][1]["occupied"] is False
    assert r["spaces"][0]["id"] == 1


def test_analyze_frame_no_positions():
    img, _ = make_lot()
    r = pc.analyze_frame(img, [])
    assert r["total"] == 0 and r["occupancy_pct"] == 0.0


def test_threshold_changes_result():
    img, pos = make_lot()
    r = pc.analyze_frame(img, pos, threshold=10**9)
    assert r["occupied"] == 0


def test_annotate_colors_and_copy():
    img, pos = make_lot()
    r = pc.analyze_frame(img, pos)
    out = pc.annotate(img, r, show_ids=False)
    assert out is not img and out.shape == img.shape
    # top edge of occupied box is red, of empty box is green (BGR)
    assert tuple(out[20, 20 + W // 2]) == config.COLOR_OCCUPIED
    assert tuple(out[20, 200 + W // 2]) == config.COLOR_EMPTY
    assert np.array_equal(img, make_lot()[0])  # original untouched


# ----------------------------------------------------------------- scaling
def test_scale_factor():
    assert pc.scale_factor((3700, 2082), (2560, 1440)) == pytest.approx(1.445, abs=0.002)
    assert pc.scale_factor((800, 800), (2560, 1440)) == 1.0  # aspect mismatch
    assert pc.scale_factor((800, 600), None) == 1.0


def test_scale_layout():
    pos, w, h, t = pc.scale_layout([(10, 20)], 100, 50, 900, 2.0)
    assert pos == [(20, 40)] and (w, h) == (200, 100) and t == 3600


def test_analyze_frame_scales_to_larger_frame():
    small, pos = make_lot()
    big = cv2.resize(small, (800, 600), interpolation=cv2.INTER_NEAREST)
    r = pc.analyze_frame(big, pos, ref_size=(400, 300))
    assert r["scale"] == pytest.approx(2.0)
    assert r["spaces"][1]["x"] == 400 and r["spaces"][1]["w"] == 2 * W
    assert r["threshold"] == pytest.approx(config.OCCUPANCY_THRESHOLD * 4)
    assert r["occupied"] == 1 and r["free"] == 1


# ----------------------------------------------------------------- video
@pytest.fixture
def tiny_video(tmp_path):
    path = str(tmp_path / "v.avi")
    img, pos = make_lot()
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (400, 300))
    if not vw.isOpened():
        pytest.skip("No video writer available")
    for _ in range(30):  # 3 s at 10 fps
        vw.write(img)
    vw.release()
    return path, pos


def test_analyze_video_samples(tiny_video):
    path, pos = tiny_video
    samples = pc.analyze_video(path, every_seconds=1, positions=pos, ref_size=(400, 300))
    assert len(samples) == 3
    assert [s["offset_s"] for s in samples] == [0.0, 1.0, 2.0]
    assert all(s["total"] == 2 for s in samples)
    assert "timestamp" not in samples[0]


def test_analyze_video_with_start_time(tiny_video):
    path, pos = tiny_video
    start = datetime(2026, 1, 5, 9, 0, 0)
    calls = []
    samples = pc.analyze_video(path, 1, start, positions=pos, ref_size=(400, 300),
                               progress=calls.append)
    assert samples[2]["timestamp"] == start + timedelta(seconds=2)
    assert calls[-1] == pytest.approx(1.0)


def test_analyze_video_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        pc.analyze_video(str(tmp_path / "missing.mp4"), positions=[(0, 0)])


def test_no_gui_or_yolo_imports():
    src = open(pc.__file__, encoding="utf-8").read()
    assert "imshow" not in src
    assert "ultralytics" not in src and "car_detector" not in src

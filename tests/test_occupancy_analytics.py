"""Tests for occupancy_analytics and generate_sample_data (temp CSVs only)."""

import os
import sys
from datetime import datetime

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_sample_data as gsd
import occupancy_analytics as oa

HEADER = "Total Slots,Occupied Slots,Available Slots,Timestamp\n"


@pytest.fixture
def log_file(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text(
        HEADER
        + "100,10,90,2026-01-05 08:15:00\n"
        + "100,30,70,2026-01-05 08:45:00\n"
        + "100,90,10,2026-01-05 13:00:00\n"
        + "100,50,50,2026-01-06 13:30:00\n"
        + "100,5,95,2026-01-06 22:00:00\n"
    )
    return str(p)


def test_load_log(log_file):
    df = oa.load_log(log_file)
    assert len(df) == 5
    assert pd.api.types.is_datetime64_any_dtype(df["Timestamp"])
    assert list(df["Occupancy %"]) == [10.0, 30.0, 90.0, 50.0, 5.0]


def test_load_log_old_format_file():
    """The legacy ./parking_status.csv in the repo must still parse."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    legacy = os.path.join(root, "parking_status.csv")
    if not os.path.exists(legacy):
        pytest.skip("legacy log not present")
    assert len(oa.load_log(legacy)) >= 1


def test_load_log_missing_or_empty(tmp_path):
    assert oa.load_log(str(tmp_path / "missing.csv")).empty
    e = tmp_path / "empty.csv"
    e.write_text("")
    assert oa.load_log(str(e)).empty
    h = tmp_path / "header.csv"
    h.write_text(HEADER)
    assert oa.load_log(str(h)).empty


def test_load_log_bad_rows_dropped(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text(HEADER + "100,10,90,not-a-date\n100,20,80,2026-01-05 09:00:00\n")
    assert len(oa.load_log(str(p))) == 1


def test_load_log_missing_columns(tmp_path):
    p = tmp_path / "wrong.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        oa.load_log(str(p))


def test_append_reading_creates_and_appends(tmp_path):
    p = str(tmp_path / "sub" / "log.csv")
    oa.append_reading(10, 4, timestamp=datetime(2026, 1, 1, 7, 0, 0), path=p)
    oa.append_reading(10, 6, 4, timestamp=datetime(2026, 1, 1, 8, 0, 0), path=p)
    lines = open(p).read().splitlines()
    assert lines[0] == HEADER.strip()  # exact schema
    assert lines[1] == "10,4,6,2026-01-01 07:00:00"
    assert lines[2] == "10,6,4,2026-01-01 08:00:00"
    assert len(oa.load_log(p)) == 2


def test_hourly_summary(log_file):
    s = oa.hourly_summary(oa.load_log(log_file))
    assert list(s["hour"]) == list(range(24))
    assert list(s.columns) == oa.SUMMARY_COLUMNS
    h8 = s[s["hour"] == 8].iloc[0]
    assert h8["readings"] == 2
    assert h8["avg_occupancy_pct"] == 20.0
    assert h8["peak_occupancy_pct"] == 30.0
    assert h8["avg_available"] == 80.0
    assert h8["min_available"] == 70
    h13 = s[s["hour"] == 13].iloc[0]
    assert h13["avg_occupancy_pct"] == 70.0
    assert s[s["hour"] == 3].iloc[0]["readings"] == 0


def test_hourly_summary_empty():
    s = oa.hourly_summary(oa._empty_log())
    assert len(s) == 24 and s["readings"].sum() == 0


def test_busiest_and_quietest(log_file):
    bq = oa.busiest_and_quietest(oa.hourly_summary(oa.load_log(log_file)))
    assert bq["busiest"]["hour"] == 13
    assert bq["quietest"]["hour"] == 22


def test_busiest_and_quietest_empty():
    assert oa.busiest_and_quietest(oa.hourly_summary(oa._empty_log())) == {
        "busiest": None,
        "quietest": None,
    }


def test_filter_day(log_file):
    df = oa.load_log(log_file)
    assert len(oa.filter_day(df, "2026-01-05")) == 3
    assert len(oa.filter_day(df, None)) == 5
    assert oa.filter_day(df, "2025-01-01").empty


def test_plot_hourly(log_file, tmp_path):
    out = oa.plot_hourly(oa.hourly_summary(oa.load_log(log_file)), str(tmp_path / "c.png"))
    assert os.path.getsize(out) > 0


def test_cli(log_file, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # chart goes to tmp_path/reports
    assert oa.main(["--csv", log_file, "--date", "2026-01-05"]) == 0
    out = capsys.readouterr().out
    assert "Busiest hour : 13:00-13:59" in out
    assert "Quietest hour: 08:00-08:59" in out
    assert any(f.endswith(".png") for f in os.listdir(tmp_path / "reports"))


def test_cli_empty(tmp_path, capsys):
    assert oa.main(["--csv", str(tmp_path / "none.csv")]) == 1
    assert "generate_sample_data" in capsys.readouterr().out


# ----------------------------------------------------------------- sample data
def test_generate_sample_data_pattern():
    df = gsd.generate(days=14, slots=50, end=datetime(2026, 1, 12))
    assert list(df.columns) == oa.COLUMNS
    assert (df["Occupied Slots"] + df["Available Slots"] == 50).all()
    ts = pd.to_datetime(df["Timestamp"])
    weekday = df[ts.dt.weekday < 5]
    wts = pd.to_datetime(weekday["Timestamp"])
    midday = weekday[wts.dt.hour.between(10, 11)]["Occupied Slots"].mean()
    night = weekday[wts.dt.hour.between(1, 4)]["Occupied Slots"].mean()
    assert midday > night * 3  # office pattern


def test_generate_sample_refuses_real_log(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        gsd.main(["--out", oa.DEFAULT_LOG])


def test_generate_sample_writes_separate_file(tmp_path):
    out = tmp_path / "sample.csv"
    assert gsd.main(["--out", str(out), "--days", "1"]) == 0
    assert len(oa.load_log(str(out))) == 96

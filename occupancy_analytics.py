"""
Hourly occupancy analytics for the parking log.

Log schema (unchanged, so old logs keep working):
    Total Slots, Occupied Slots, Available Slots, Timestamp (%Y-%m-%d %H:%M:%S)

CLI:
    python occupancy_analytics.py [--csv PATH] [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from typing import Dict, Optional, Union

import pandas as pd

import config

COLUMNS = ["Total Slots", "Occupied Slots", "Available Slots", "Timestamp"]
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_LOG = os.path.join(config.DATA_DIR, config.CSV_FILE)
LEGACY_LOG = config.CSV_FILE  # ./parking_status.csv
SAMPLE_LOG = os.path.join(config.DATA_DIR, "sample_parking_status.csv")

SUMMARY_COLUMNS = [
    "hour",
    "readings",
    "avg_occupancy_pct",
    "peak_occupancy_pct",
    "avg_available",
    "min_available",
]


def default_log_path() -> str:
    """data/parking_status.csv, falling back to ./parking_status.csv."""
    if os.path.exists(DEFAULT_LOG):
        return DEFAULT_LOG
    if os.path.exists(LEGACY_LOG):
        return LEGACY_LOG
    return DEFAULT_LOG


def _empty_log() -> pd.DataFrame:
    df = pd.DataFrame(columns=COLUMNS + ["Occupancy %"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def load_log(path: Optional[str] = None) -> pd.DataFrame:
    """Load the occupancy log. Adds an 'Occupancy %' column.

    Missing/empty file -> empty DataFrame. Bad rows are dropped."""
    path = path or default_log_path()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return _empty_log()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return _empty_log()

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df = df[COLUMNS].copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format=TS_FORMAT, errors="coerce")
    for c in COLUMNS[:3]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    for c in COLUMNS[:3]:
        df[c] = df[c].astype(int)

    total = df["Total Slots"].where(df["Total Slots"] > 0)
    df["Occupancy %"] = (100.0 * df["Occupied Slots"] / total).fillna(0.0).round(2)
    return df.sort_values("Timestamp").reset_index(drop=True)


def append_reading(
    total: int,
    occupied: int,
    available: Optional[int] = None,
    timestamp: Optional[datetime] = None,
    path: Optional[str] = None,
) -> str:
    """Append one reading in the existing schema. Returns the file path."""
    path = path or DEFAULT_LOG
    if available is None:
        available = int(total) - int(occupied)
    ts = (timestamp or datetime.now()).strftime(TS_FORMAT)
    row = pd.DataFrame(
        [[int(total), int(occupied), int(available), ts]], columns=COLUMNS
    )
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    row.to_csv(path, mode="a", header=write_header, index=False)
    return path


def filter_day(df: pd.DataFrame, day: Union[None, str, date]) -> pd.DataFrame:
    """Keep only rows from `day` (YYYY-MM-DD string or date). None -> all."""
    if day is None or df.empty:
        return df
    if isinstance(day, str):
        day = datetime.strptime(day, "%Y-%m-%d").date()
    return df[df["Timestamp"].dt.date == day].reset_index(drop=True)


def hourly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per hour 0-23 (hours with no readings have readings=0, NaN stats)."""
    hours = pd.DataFrame({"hour": range(24)})
    if df.empty:
        out = hours.copy()
        out["readings"] = 0
        for c in SUMMARY_COLUMNS[2:]:
            out[c] = float("nan")
        return out[SUMMARY_COLUMNS]

    g = df.assign(hour=df["Timestamp"].dt.hour).groupby("hour")
    stats = pd.DataFrame(
        {
            "readings": g.size(),
            "avg_occupancy_pct": g["Occupancy %"].mean().round(2),
            "peak_occupancy_pct": g["Occupancy %"].max().round(2),
            "avg_available": g["Available Slots"].mean().round(1),
            "min_available": g["Available Slots"].min(),
        }
    ).reset_index()
    out = hours.merge(stats, on="hour", how="left")
    out["readings"] = out["readings"].fillna(0).astype(int)
    return out[SUMMARY_COLUMNS]


def busiest_and_quietest(summary: pd.DataFrame) -> Dict[str, Optional[Dict]]:
    """Busiest / quietest hour by average occupancy (hours with data only)."""
    data = summary[summary["readings"] > 0]
    if data.empty:
        return {"busiest": None, "quietest": None}
    b = data.loc[data["avg_occupancy_pct"].idxmax()]
    q = data.loc[data["avg_occupancy_pct"].idxmin()]
    as_dict = lambda r: {"hour": int(r["hour"]), "avg_occupancy_pct": float(r["avg_occupancy_pct"])}
    return {"busiest": as_dict(b), "quietest": as_dict(q)}


def hour_label(h: int) -> str:
    return f"{h:02d}:00-{h:02d}:59"


def plot_hourly(
    summary: pd.DataFrame,
    out_path: Optional[str] = None,
    title: str = "Average occupancy by hour",
) -> str:
    """Save a bar chart of average occupancy per hour to reports/. Returns path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_path is None:
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(config.REPORTS_DIR, f"hourly_occupancy_{stamp}.png")
    else:
        d = os.path.dirname(out_path)
        if d:
            os.makedirs(d, exist_ok=True)

    vals = summary["avg_occupancy_pct"].fillna(0)
    peak = busiest_and_quietest(summary)["busiest"]
    colors = ["#1f5fbf"] * 24
    if peak:
        colors[peak["hour"]] = "#d62728"

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(summary["hour"], vals, color=colors)
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average occupancy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    if peak:
        ax.annotate(
            f"Peak {peak['avg_occupancy_pct']:.1f}%",
            (peak["hour"], peak["avg_occupancy_pct"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#d62728",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Hourly parking occupancy summary")
    p.add_argument("--csv", default=None, help="Occupancy log (default: data/parking_status.csv)")
    p.add_argument("--date", default=None, help="Only this day, YYYY-MM-DD")
    args = p.parse_args(argv)

    path = args.csv or default_log_path()
    try:
        df = filter_day(load_log(path), args.date)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    label = f"{path}" + (f" ({args.date})" if args.date else "")
    if df.empty:
        print(f"No readings in {label}.")
        print("Run detection (python run.py / dashboard) or: python generate_sample_data.py")
        return 1

    summary = hourly_summary(df)
    print(f"Hourly occupancy - {label} - {len(df)} readings\n")
    with pd.option_context("display.max_rows", 30, "display.width", 120):
        print(summary[summary["readings"] > 0].to_string(index=False))

    bq = busiest_and_quietest(summary)
    print()
    print(f"Busiest hour : {hour_label(bq['busiest']['hour'])} "
          f"({bq['busiest']['avg_occupancy_pct']:.1f}% avg)")
    print(f"Quietest hour: {hour_label(bq['quietest']['hour'])} "
          f"({bq['quietest']['avg_occupancy_pct']:.1f}% avg)")

    title = "Average occupancy by hour" + (f" - {args.date}" if args.date else "")
    if os.path.abspath(path) == os.path.abspath(SAMPLE_LOG):
        title += " (SAMPLE DATA)"
    out = plot_hourly(summary, title=title)
    print(f"\nChart saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

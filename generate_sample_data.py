"""
Generate SYNTHETIC demo occupancy data (SAMPLE DATA - NOT REAL READINGS).

Writes an office-lot pattern (quiet nights, morning rush, lunch dip,
evening drain-out, quieter weekends) to data/sample_parking_status.csv,
using the same schema as the real log. It never touches the real log
(data/parking_status.csv).

    python generate_sample_data.py [--days 7] [--slots 69] [--seed 42]
"""

from __future__ import annotations

import argparse
import math
import os
import random
from datetime import datetime, timedelta

import pandas as pd

import config
from occupancy_analytics import COLUMNS, DEFAULT_LOG, SAMPLE_LOG, TS_FORMAT


def office_profile(hour: float, weekend: bool) -> float:
    """Expected occupancy fraction (0-1) for a fractional hour of day."""
    def bump(center, width, height):
        return height * math.exp(-((hour - center) ** 2) / (2 * width**2))

    if weekend:
        return 0.05 + bump(12.5, 2.5, 0.20)
    base = 0.05
    workday = 0.80 / (1 + math.exp(-(hour - 8.5) * 2.2)) - 0.80 / (1 + math.exp(-(hour - 17.5) * 1.6))
    lunch_dip = -bump(12.75, 0.6, 0.12)
    return max(0.0, min(1.0, base + workday + lunch_dip))


def generate(
    days: int = 7,
    slots: int = 69,
    every_minutes: int = 15,
    seed: int = 42,
    end: datetime | None = None,
) -> pd.DataFrame:
    rng = random.Random(seed)
    end = (end or datetime.now()).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    rows = []
    t = start
    while t < end:
        h = t.hour + t.minute / 60
        frac = office_profile(h, t.weekday() >= 5) + rng.gauss(0, 0.04)
        occ = max(0, min(slots, round(frac * slots)))
        rows.append([slots, occ, slots - occ, t.strftime(TS_FORMAT)])
        t += timedelta(minutes=every_minutes)
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate SYNTHETIC sample occupancy data")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--slots", type=int, default=69)
    p.add_argument("--every", type=int, default=15, help="minutes between readings")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=SAMPLE_LOG)
    args = p.parse_args(argv)

    if os.path.abspath(args.out) in {
        os.path.abspath(DEFAULT_LOG),
        os.path.abspath(config.CSV_FILE),
    }:
        p.error("Refusing to write sample data to the real occupancy log.")

    df = generate(args.days, args.slots, args.every, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} SYNTHETIC sample readings to {args.out}")
    print("This is demo data, not real detections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

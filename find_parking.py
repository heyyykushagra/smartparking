"""
Tell an arriving car where to park: the nearest free space to the entrance.

    python find_parking.py                                  # sample image
    python find_parking.py --image lot.jpg --entrance 1280,1439 --top 5
    python find_parking.py --video carPark.mp4 --at 30      # frame at 30 s
    python find_parking.py --save                           # also save a guide image

"Nearest" is the straight-line distance from the entrance to the centre of each
free space, not the driving distance along the lanes.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import cv2

import config
import parking_core as pc


def parse_point(text: str):
    try:
        x, y = (int(float(v)) for v in text.split(","))
        return (x, y)
    except ValueError:
        raise argparse.ArgumentTypeError("expected x,y (e.g. 1280,1439)")


def read_video_frame(path: str, seconds: float):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"No frame at {seconds}s in {path}")
        return frame
    finally:
        cap.release()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Suggest the nearest free parking space")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--image", default=None, help=f"default: {config.DEFAULT_IMAGE_PATH}")
    src.add_argument("--video", default=None)
    p.add_argument("--at", type=float, default=0.0, help="seconds into the video")
    p.add_argument("--entrance", type=parse_point, default=None,
                   help="x,y in the marked image's pixels (default: config.ENTRANCE_POINT "
                        "or bottom-centre)")
    p.add_argument("--top", type=int, default=3, help="how many suggestions")
    p.add_argument("--positions", default=config.POSITION_FILE)
    p.add_argument("--threshold", type=float, default=config.OCCUPANCY_THRESHOLD)
    p.add_argument("--save", action="store_true", help="save a guide image to reports/")
    args = p.parse_args(argv)

    positions = pc.load_positions(args.positions)
    if not positions:
        print(f"No parking spaces in {args.positions}. Mark them with run.py or the dashboard.")
        return 2

    ref_size = pc.reference_size()
    try:
        if args.video:
            frame = read_video_frame(args.video, args.at)
        else:
            path = args.image or config.DEFAULT_IMAGE_PATH
            frame = cv2.imread(path)
            if frame is None:
                raise FileNotFoundError(f"Cannot read image: {path}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    result = pc.analyze_frame(frame, positions, threshold=args.threshold, ref_size=ref_size)
    entrance = args.entrance or config.ENTRANCE_POINT or pc.default_entrance(
        ref_size or (frame.shape[1], frame.shape[0]))
    recs = pc.recommend_spaces(result, entrance, top_n=args.top)

    print(f"Free {result['free']} / {result['total']} spaces "
          f"({result['occupancy_pct']:.1f}% occupied). Entrance at {entrance}.")
    if not recs:
        print("Lot full - no free spaces right now.")
    else:
        best = recs[0]
        print(f"\n>>> Park in space #{best['id']}  "
              f"(~{best['distance_px']:.0f} px from the entrance)\n")
        if len(recs) > 1:
            print("Alternatives:")
            for r in recs[1:]:
                print(f"  #{r['rank']}: space {r['id']:>4}  ~{r['distance_px']:.0f} px")

    if args.save:
        s = result["scale"]
        img = pc.annotate_recommendation(
            pc.annotate(frame, result), recs, (int(entrance[0] * s), int(entrance[1] * s)))
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        out = os.path.join(config.REPORTS_DIR,
                           f"parking_guide_{datetime.now():%Y%m%d_%H%M%S}.png")
        cv2.imwrite(out, img)
        print(f"\nGuide image saved to {out}")
    return 0 if recs else 1


if __name__ == "__main__":
    sys.exit(main())

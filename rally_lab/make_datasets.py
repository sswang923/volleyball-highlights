#!/usr/bin/env python3
"""
Pseudo-labelled training sets from games whose ball tracks we trust most.

  livedead/  (YOLO classify layout)  {train,val}/{live,dead}/*.jpg
      live = 1 fps samples inside confident rally tracks
      dead = 1 fps samples > 4 s away from any ball-activity track
  ball/      (YOLO detect layout)    images/{train,val}, labels/{train,val}
      every 6th frame inside confident rally tracks, box = ball position +-12 px;
      plus a few dead-time frames with empty labels as background

    python3 rally_lab/make_datasets.py --games game01 game03 --val game04 \
        --work /mnt/d/volleyball/work --out /mnt/d/volleyball/work/datasets
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build_golden_v1 as bg  # noqa: E402

BALL_HALF = 12          # px at 1280x720
BALL_EVERY = 6          # frames
BG_SHARE = 0.1          # background images per positive image


def activity(tracks_dir):
    """(confident rallies, any ball activity) as merged [start, end] lists, plus per-frame ball xy."""
    strict, anyt, ball_xy = [], [], {}
    for p in sorted(Path(tracks_dir).glob("track_*.json")):
        d = json.load(open(p))
        rf, rc = d.get("rally_features", {}), d.get("rally_classification", {})
        fps = float(d.get("fps") or 30)
        pos = d["positions"]
        if len(pos) < 3:
            continue
        s, e = pos[0][1] / fps, pos[-1][1] / fps
        if rf.get("path_len_px", 0) >= 300:
            anyt.append([s, e])
        if rc.get("is_rally") and rc.get("rally_confidence", 0) >= 0.7:
            strict.append([s, e])
            for (x, y), f in pos:
                ball_xy[int(f)] = (x, y)
    return strict, anyt, ball_xy


def inside(t, spans, pad=0.0):
    return any(a - pad <= t <= b + pad for a, b in spans)


def build(game, split, args, rng):
    video = Path(args.work) / f"proxy_20261004_{game}.mp4"
    tracks = Path(args.work) / "out" / f"proxy_20261004_{game}" / "tracks"
    strict, anyt, ball_xy = activity(tracks)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    want = {}
    for f in range(0, n, int(fps)):                       # 1 fps live/dead samples
        t = f / fps
        if inside(t, [[a + 1, b - 0.5] for a, b in strict]):
            want.setdefault(f, set()).add("live")
        elif not inside(t, anyt, pad=4):
            want.setdefault(f, set()).add("dead")
    pos_frames = [f for f in sorted(ball_xy) if f % BALL_EVERY == 0]
    for f in pos_frames:
        want.setdefault(f, set()).add("ball")
    dead_frames = [f for f, k in want.items() if "dead" in k]
    for f in rng.sample(dead_frames, min(len(dead_frames), int(len(pos_frames) * BG_SHARE))):
        want[f].add("bg")

    out = Path(args.out)
    counts = {"live": 0, "dead": 0, "ball": 0, "bg": 0}
    f = 0
    while f < n:
        if not cap.grab():
            break
        if f in want:
            ok, img = cap.retrieve()
            if ok:
                name = f"{game}_{f:06d}.jpg"
                for k in want[f]:
                    if k in ("live", "dead"):
                        d = out / "livedead" / split / k
                        d.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(d / name), cv2.resize(img, (320, 180)))
                    else:
                        di = out / "ball" / "images" / split
                        dl = out / "ball" / "labels" / split
                        di.mkdir(parents=True, exist_ok=True)
                        dl.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(di / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        h, w = img.shape[:2]
                        with open(dl / name.replace(".jpg", ".txt"), "w") as fh:
                            if k == "ball":
                                x, y = ball_xy[f]
                                fh.write(f"0 {x / w:.6f} {y / h:.6f} {2 * BALL_HALF / w:.6f} {2 * BALL_HALF / h:.6f}\n")
                    counts[k] += 1
        f += 1
    print(game, split, counts, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(0)
    for g in args.games:
        build(g, "train", args, rng)
    for g in args.val:
        build(g, "val", args, rng)
    (Path(args.out) / "ball" / "data.yaml").write_text(
        f"path: {Path(args.out) / 'ball'}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")


if __name__ == "__main__":
    main()

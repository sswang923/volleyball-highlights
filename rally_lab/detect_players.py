#!/usr/bin/env python3
"""
Person detections at a fixed sample rate (YOLOv8n, GPU) -> compressed .npz.

    MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=-1 \
    python3 rally_lab/detect_players.py --video proxy.mp4 --out players.npz --fps 5

Saved arrays: t (sample times, s), boxes (N x 6: sample_idx, x1, y1, x2, y2, conf).
"""

import argparse

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / args.fps))

    times, rows, batch, batch_t = [], [], [], []

    def flush():
        res = model.predict(batch, imgsz=args.imgsz, classes=[0], conf=0.25, half=False,
                            device=0, verbose=False)
        for t, r in zip(batch_t, res):
            idx = len(times)
            times.append(t)
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                rows.append((idx, x1, y1, x2, y2, c))
        batch.clear()
        batch_t.clear()

    f = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if f % step == 0:
            ok, img = cap.retrieve()
            if ok:
                batch.append(img)
                batch_t.append(f / src_fps)
                if len(batch) == args.batch:
                    flush()
        f += 1
        if f % (step * 3000) == 0:
            print(f"{f}/{n_frames}", flush=True)
    if batch:
        flush()
    np.savez_compressed(args.out, t=np.array(times, np.float32),
                        boxes=np.array(rows, np.float32).reshape(-1, 6))
    print(f"done: {len(times)} samples, {len(rows)} boxes -> {args.out}")


if __name__ == "__main__":
    main()

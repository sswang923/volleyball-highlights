#!/usr/bin/env python3
"""
Run a trained YOLO model over a proxy video.

  livedead:  p(live) at --fps samples          -> npz {t, p_live}
  ball:      ball detections on every --step-th frame -> ball.csv
             (same columns as the ONNX tracker: Frame,Visibility,X,Y,Radius;
              Frame is the sample index, so pass fps/step to track_calculator)

    python3 rally_lab/infer.py livedead --model best.pt --video proxy.mp4 --out ld.npz --fps 2
    python3 rally_lab/infer.py ball --model best.pt --video proxy.mp4 --out ball.csv --step 2
"""

import argparse

import cv2
import numpy as np
from ultralytics import YOLO


def frames(video, step):
    cap = cv2.VideoCapture(video)
    f = 0
    while cap.grab():
        if f % step == 0:
            ok, img = cap.retrieve()
            if ok:
                yield f, img
        f += 1


def batched(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["livedead", "ball"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()

    model = YOLO(args.model)
    src_fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS) or 30.0

    if args.mode == "livedead":
        step = max(1, round(src_fps / args.fps))
        names = model.names
        live_idx = [k for k, v in names.items() if v == "live"][0]
        ts, ps = [], []
        for chunk in batched(frames(args.video, step), 64):
            imgs = [cv2.resize(im, (320, 180)) for _, im in chunk]
            res = model.predict(imgs, imgsz=args.imgsz or 256, half=False, device=0, verbose=False)
            for (f, _), r in zip(chunk, res):
                ts.append(f / src_fps)
                ps.append(float(r.probs.data[live_idx]))
        np.savez_compressed(args.out, t=np.array(ts, np.float32), p_live=np.array(ps, np.float32))
        print(f"done: {len(ts)} samples, mean p_live {np.mean(ps):.2f}")
    else:
        rows = []
        for chunk in batched(frames(args.video, args.step), 8):
            res = model.predict([im for _, im in chunk], imgsz=args.imgsz or 1280, conf=args.conf,
                                half=False, device=0, verbose=False, max_det=1)
            for (f, _), r in zip(chunk, res):
                i = f // args.step
                if len(r.boxes):
                    x1, y1, x2, y2 = r.boxes.xyxy[0].tolist()
                    rows.append((i, 1, (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2))
                else:
                    rows.append((i, 0, -1, -1, 0))
        with open(args.out, "w") as fh:
            fh.write("Frame,Visibility,X,Y,Radius\n")
            for r in rows:
                fh.write(f"{r[0]},{r[1]},{r[2]:.1f},{r[3]:.1f},{r[4]:.1f}\n")
        print(f"done: {len(rows)} frames, ball seen in {sum(r[1] for r in rows)}")


if __name__ == "__main__":
    main()

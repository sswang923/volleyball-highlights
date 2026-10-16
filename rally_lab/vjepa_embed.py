#!/usr/bin/env python3
"""
V-JEPA 2 clip embeddings for a game: one 1024-d vector every --stride seconds (default 2 s).

Each clip = 8 frames covering ~2 s (every 8th frame of the 30 fps working copy), squashed to
224x224 so the whole court stays in view, centred on the timestamp. A small classifier trained
on these vectors (rally_lab/train_live.py) decides "in play" vs "not in play".

Runs in the separate ~/vb-tools/vlm environment (Python 3.12, recent torch/transformers):
    cd ~/vb-tools/vlm && uv run python /mnt/c/Users/stanl/volleyball-highlights/rally_lab/vjepa_embed.py \
        --video /mnt/d/volleyball/work/20261004_game03/proxy.mp4 --out /mnt/d/volleyball/work/20261004_game03/vjepa.npz
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch
from transformers import AutoModel

MODEL = "facebook/vjepa2-vitl-fpc64-256"
FRAMES, STEP, SIZE = 8, 8, 224           # defaults: 8 frames, every 8th source frame (~2.1 s at 30 fps), 224 px
# (16 frames at 256 px is ~3x slower on a GTX 1650 and not worth it for in-play / not-in-play)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=float, default=2.0, help="seconds between clips")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--step", type=int, default=STEP, help="use every Nth source frame")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--limit", type=float, default=0, help="stop after this many seconds (testing)")
    a = ap.parse_args()

    frames_n, step, size = a.frames, a.step, a.size
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    span = frames_n * step                                 # source frames per clip
    stride = max(1, round(a.stride * fps))

    buf = deque(maxlen=frames_n)                           # last sampled frames (every step-th)
    times, embs, batch, batch_t = [], [], [], []
    t_start = time.time()

    def flush():
        x = torch.from_numpy(np.stack(batch)).to("cuda", torch.float16)   # B, T, C, H, W
        with torch.no_grad():
            out = model.get_vision_features(pixel_values_videos=x) if hasattr(model, "get_vision_features") \
                else model(pixel_values_videos=x).last_hidden_state
        embs.append(out.float().mean(dim=1).cpu().numpy().astype(np.float16))
        times.extend(batch_t)
        batch.clear(); batch_t.clear()

    f = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if f % step == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
            buf.append(((img - MEAN) / STD).transpose(2, 0, 1))
        # emit a clip every `stride` frames once the buffer is full; clip centre = f - span/2
        if len(buf) == frames_n and f % stride == 0:
            batch.append(np.stack(buf)); batch_t.append((f - span / 2) / fps)
            if len(batch) == a.batch:
                flush()
        f += 1
        if a.limit and f / fps >= a.limit:
            break
        if f % (fps * 300) < 1:
            el = time.time() - t_start
            print(f"{f / fps / 60:.0f} min of video in {el / 60:.1f} min", flush=True)
    if batch:
        flush()
    emb = np.concatenate(embs)
    np.savez_compressed(a.out, t=np.array(times, np.float32), emb=emb)
    print(f"done: {len(times)} clips, dim {emb.shape[1]}, {(time.time() - t_start) / 60:.1f} min, "
          f"gpu {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()

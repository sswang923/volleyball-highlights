"""
Per-time-bin signals for rally detection and the shared signal -> rallies step.

All signals live on one grid of BIN_S-second bins starting at t=0.
"""

import csv
import json
from pathlib import Path

import numpy as np

BIN_S = 0.5
FRAME_W, FRAME_H = 1280, 720

# Playing area in proxy pixels (camera behind the near end line): excludes the
# sideline crowd on the left and the scorer's table / benches on the right.
COURT_POLY = np.array([[0, 600], [175, 360], [200, 285], [960, 285], [1010, 330], [1280, 520], [1280, 720], [0, 720]], np.float32)


def grid(duration):
    return np.arange(0, duration, BIN_S, dtype=np.float32)


# ── ball (ball.csv from the tracker, or from our YOLO ball detector) ─────────

def load_ball_csv(path, fps):
    t, vis, x, y = [], [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            t.append(int(row["Frame"]) / fps)
            vis.append(int(row["Visibility"]))
            x.append(float(row["X"]))
            y.append(float(row["Y"]))
    return {"t": np.array(t), "vis": np.array(vis, bool), "x": np.array(x), "y": np.array(y)}


def ball_activity(ball, bins):
    """Share of frames in each bin where the ball is visible and moving."""
    t, vis, x, y = ball["t"], ball["vis"], ball["x"], ball["y"]
    sp = np.zeros_like(t)
    both = vis[1:] & vis[:-1]
    d = np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 1e-3)
    sp[1:][both] = d[both]
    moving = vis & (sp > 60) & (sp < 4000)       # px/s; rejects static balls and jumps
    idx = np.minimum((t / BIN_S).astype(int), len(bins) - 1)
    act = np.bincount(idx, weights=moving, minlength=len(bins)) / np.maximum(np.bincount(idx, minlength=len(bins)), 1)
    seen = np.bincount(idx, weights=vis, minlength=len(bins)) / np.maximum(np.bincount(idx, minlength=len(bins)), 1)
    return act.astype(np.float32), seen.astype(np.float32)


# ── players (detect_players.py output) ───────────────────────────────────────

def _in_court(px, py):
    import cv2
    return np.array([cv2.pointPolygonTest(COURT_POLY, (float(a), float(b)), False) >= 0 for a, b in zip(px, py)], bool)


def player_features(npz_path, bins):
    """Players on court (count) and their mean foot speed (px/s) per bin."""
    d = np.load(npz_path)
    ts, boxes = d["t"], d["boxes"]
    feet_x = (boxes[:, 1] + boxes[:, 3]) / 2
    feet_y = boxes[:, 4]
    on = _in_court(feet_x, feet_y) if len(boxes) else np.zeros(0, bool)
    per = [[] for _ in ts]
    for (i, *_), fx, fy, o in zip(boxes, feet_x, feet_y, on):
        if o:
            per[int(i)].append((fx, fy))
    count = np.array([len(p) for p in per], np.float32)
    speed = np.zeros(len(ts), np.float32)
    for k in range(1, len(ts)):
        a, b = np.array(per[k - 1]), np.array(per[k])
        if len(a) and len(b):
            dist = np.hypot(b[:, None, 0] - a[None, :, 0], b[:, None, 1] - a[None, :, 1]).min(axis=1)
            dist = dist[dist < 120]                    # unmatched -> ignore
            if len(dist):
                speed[k] = dist.mean() / (ts[k] - ts[k - 1])
    return resample(ts, count, bins), resample(ts, speed, bins)


# ── live/dead classifier (classify_livedead.py output) ───────────────────────

def livedead(npz_path, bins):
    d = np.load(npz_path)
    return resample(d["t"], d["p_live"], bins)


def whistle_density(whistle_json, bins, width=1.0):
    w = np.array(json.load(open(whistle_json))["whistles"])
    out = np.zeros(len(bins), np.float32)
    for t in w:
        out[np.abs(bins - t) <= width] = 1.0
    return out


def resample(ts, vals, bins):
    return np.interp(bins, ts, vals).astype(np.float32) if len(ts) else np.zeros(len(bins), np.float32)


def smooth(x, k=3):
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same").astype(np.float32)


# ── signal -> rallies ────────────────────────────────────────────────────────

def segment(score, bins, hi=0.5, lo=None, min_len=2.0, merge_gap=2.0):
    lo = hi * 0.6 if lo is None else lo
    segs, on, start = [], False, 0.0
    for t, s in zip(bins, score):
        if not on and s >= hi:
            on, start = True, t
        elif on and s < lo:
            on = False
            segs.append([start, t])
    if on:
        segs.append([start, bins[-1]])
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [s for s in merged if s[1] - s[0] >= min_len]


def segment_to_count(score, bins, target, lo_t=0.05, hi_t=0.95, return_threshold=False, **kw):
    """Threshold that yields `target` segments (count constraint from the set score)."""
    best, best_hi = None, None
    for hi in np.linspace(lo_t, hi_t, 91):
        segs = segment(score, bins, hi=hi, **kw)
        if best is None or abs(len(segs) - target) < abs(len(best) - target):
            best, best_hi = segs, float(hi)
    return (best, best_hi) if return_threshold else best


def serve_side_raw(ball, start):
    """Near/far serve from the first ball sighting around the rally start (see build_golden_v1.serve_side).

    Detected rally starts can land a little before or after the serve, so look
    from 0.5 s before (earlier catches balls tossed back to the server) to 2 s
    after, and judge from the first second of sightings.
    """
    m = np.where(ball["vis"] & (ball["t"] >= start - 0.5) & (ball["t"] <= start + 2.0))[0]
    if len(m) < 4:
        return None
    first = m[ball["t"][m] <= ball["t"][m[0]] + 1.0]
    xs = ball["x"][first]
    if len(xs) < 4:
        return None
    if np.median(xs[:3]) / FRAME_W > 0.8 and xs[-1] < xs[0]:
        return "near"
    return "far"


def score_rallies(segs, set_spans, ball):
    """Side-out scoring; home = near side in set 1, sides swap every set."""
    rallies = []
    for si, (a, b) in enumerate(set_spans, 1):
        near_team = "home" if si % 2 == 1 else "away"
        far_team = "away" if near_team == "home" else "home"
        rs = [s for s in segs if a - 5 <= s[0] <= b + 5]
        sides = [serve_side_raw(ball, s[0]) for s in rs]
        h = w = 0
        for k, s in enumerate(rs):
            nxt = sides[k + 1] if k + 1 < len(rs) else None
            if nxt:
                by = near_team if nxt == "near" else far_team
            else:
                by = "home" if h >= w else "away"
            h += by == "home"
            w += by == "away"
            rallies.append({"start": s[0], "end": s[1], "set": si, "score_by": by})
    return rallies


def set_spans_from_rallies(rallies, gap=150.0):
    spans = []
    for r in sorted(rallies, key=lambda r: r[0]):
        if spans and r[0] - spans[-1][1] <= gap:
            spans[-1][1] = max(spans[-1][1], r[1])
        else:
            spans.append([r[0], r[1]])
    return [s for s in spans if s[1] - s[0] > 120]

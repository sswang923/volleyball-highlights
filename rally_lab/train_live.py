#!/usr/bin/env python3
"""
Train / evaluate "ball in play" classifiers on a fully reviewed game.

Truth comes from the review app (labels/<game>/review_<game>.json): every 0.5 s bin between game
start and game end is "in play" if it lies inside a checked rally, otherwise "not in play"
(including set breaks and the possible rallies you dismissed).

Honest split: train on set 1, test on set 2 (and the reverse), so a model is never scored on
time it was fitted to.

    python3 rally_lab/train_live.py --key 20261004_game03                 # signal-based models
    python3 rally_lab/train_live.py --key 20261004_game03 --vjepa          # + V-JEPA 2 clip vectors

Prints per-bin accuracy / F1 and rally-level recall / precision for each model.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import signals as sg  # noqa: E402

WORK = Path("/mnt/d/volleyball/work")
LABELS = HERE.parent / "labels"
FEATS = ["ball_act", "ball_seen", "p_speed", "p_live"]


def truth(key):
    d = json.load(open(LABELS / key / f"review_{key}.json"))
    rallies = [r for r in d["rallies"] if r["status"] == "ok"]
    sets = d.get("sets") or []
    return d, rallies, sets


def bin_labels(bins, d, rallies):
    gs = d.get("game_start") or 0
    ge = d.get("game_end") or bins[-1]
    inside = (bins >= gs) & (bins <= ge)
    live = np.zeros(len(bins), bool)
    for r in rallies:
        live[(bins >= r["start"]) & (bins <= r["end"])] = True
    return inside, live


def context(X, widths=(3, 9)):
    """Add smoothed copies of each feature (±0.75 s, ±2.25 s) so the model sees a little time context."""
    cols = [X]
    for w in widths:
        k = np.ones(w) / w
        cols.append(np.stack([np.convolve(X[:, j], k, mode="same") for j in range(X.shape[1])], 1))
    return np.concatenate(cols, 1)


def fit_logistic(X, y, iters=2000, lr=0.3, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    pos = y.mean()
    sw = np.where(y, 0.5 / pos, 0.5 / (1 - pos))            # balance classes
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = sw * (p - y)
        w -= lr * (Z.T @ g / len(y) + l2 * w)
        b -= lr * float(g.mean())
    return lambda Xn: 1 / (1 + np.exp(-(((Xn - mu) / sd) @ w + b)))


def rally_eval(prob, bins, mask, gold, thr=0.5):
    """Cut the probability into rallies and match them to the reviewed rallies."""
    p = np.where(mask, prob, 0)
    segs = sg.segment(p.astype(np.float32), bins, hi=thr, lo=thr * 0.6, min_len=2.0, merge_gap=1.5)
    used, hit = set(), 0
    for a, b in segs:
        for j, r in enumerate(gold):
            if j in used:
                continue
            ov = min(b, r["end"]) - max(a, r["start"])
            if ov > 0.3 * min(b - a, r["end"] - r["start"]):
                used.add(j); hit += 1
                break
    return {"found": hit, "of": len(gold), "recall": round(hit / max(1, len(gold)), 3),
            "detections": len(segs), "precision": round(hit / max(1, len(segs)), 3)}


def bin_scores(prob, y, thr=0.5):
    pred = prob >= thr
    tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
    acc = (pred == y).mean()
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return {"acc": round(float(acc), 3), "f1": round(float(f1), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="20261004_game03")
    ap.add_argument("--vjepa", action="store_true")
    a = ap.parse_args()
    d, gold, sets = truth(a.key)
    s = np.load(WORK / a.key / "signals.npz")
    bins = s["bins"]
    inside, y = bin_labels(bins, d, gold)
    if len(sets) < 2:
        raise SystemExit("needs two marked sets for the set-1 / set-2 split")
    split = sets[1]["start"]
    parts = {"set1": inside & (bins < split), "set2": inside & (bins >= split)}
    gold_by = {"set1": [r for r in gold if r["start"] < split], "set2": [r for r in gold if r["start"] >= split]}
    print(f"{a.key}: {len(gold)} reviewed rallies, {inside.sum() * 0.5 / 60:.0f} min of game, "
          f"{y[inside].mean():.0%} of it in play")

    X = np.stack([s[k] for k in FEATS if k in s.files], 1).astype(np.float64)
    models = {
        "current detector (no training on your labels)": lambda tr: (lambda m: s["fused"]),
        "signals refit on your labels": lambda tr: fit_logistic(context(X)[tr], y[tr]),
    }
    feats = {"current detector (no training on your labels)": None, "signals refit on your labels": context(X)}

    if a.vjepa:
        v = np.load(WORK / a.key / "vjepa.npz")
        E = np.stack([np.interp(bins, v["t"], v["emb"][:, j].astype(np.float32)) for j in range(v["emb"].shape[1])], 1)
        feats["V-JEPA 2 clips"] = E
        models["V-JEPA 2 clips"] = lambda tr: fit_logistic(E[tr], y[tr], iters=600, lr=0.05, l2=1e-2)
        EX = np.concatenate([E, context(X)], 1)
        feats["V-JEPA 2 + signals"] = EX
        models["V-JEPA 2 + signals"] = lambda tr: fit_logistic(EX[tr], y[tr], iters=600, lr=0.05, l2=1e-2)

    for name, make in models.items():
        print(f"\n{name}")
        for train, test in (("set1", "set2"), ("set2", "set1")):
            f = make(parts[train])
            prob = f(feats[name]) if feats[name] is not None else f(None)
            prob = sg.smooth(np.asarray(prob, np.float32), 3)
            m = parts[test]
            bs = bin_scores(prob[m], y[m])
            re = rally_eval(prob, bins, m, gold_by[test])
            print(f"  train {train} -> test {test}: per-0.5s acc {bs['acc']:.2f} F1 {bs['f1']:.2f} | "
                  f"rallies found {re['found']}/{re['of']} (recall {re['recall']:.2f}), "
                  f"{re['detections']} detections (precision {re['precision']:.2f})")


if __name__ == "__main__":
    main()

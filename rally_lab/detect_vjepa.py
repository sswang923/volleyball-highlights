#!/usr/bin/env python3
"""
Rally detection with the in-play classifier trained on reviewed games (V-JEPA 2 clips + signals).

Trains on every in-game half-second of the reviewed game(s), scores the target game, cuts rallies
(>= 3 s, gaps < 3 s joined), and writes them as the target's auto detections. The previous auto
detections and signals are kept as *_old files; a reviewed game's own files are never touched.

    python3 rally_lab/detect_vjepa.py --target 20261004_game04 [--train 20261004_game03]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import apply_live as al     # noqa: E402
import pipeline as pl       # noqa: E402
import signals as sg        # noqa: E402
import train_live as tl     # noqa: E402

SEG = dict(hi=0.5, lo=0.3, min_len=3.0, merge_gap=3.0)


def norm(F):
    """Each input relative to its own game's average and spread. Without this the classifier's
    confidence drifts between games (lighting, time of day): trained on game 3 it called 76% of
    game 4 'in play'. With it: ~39% on games 2 and 4, game 2 rally count off by 1.6 vs 3.4 before."""
    return (F - F.mean(0)) / (F.std(0) + 1e-6)


MAX_SET_RALLIES, MIN_BREAK_S = 55, 45.0


def split_long_sets(spans, segs):
    """A 'set' with more rallies than a real set can hold is two sets whose break was too short for
    the 150 s rule (game 4: sets 2 and 3, 76 s apart). Split it at its longest quiet gap."""
    out = []
    for a, b in spans:
        rs = sorted(s for s in segs if a - 5 <= s[0] <= b + 5)
        while len(rs) > MAX_SET_RALLIES:
            gaps = [(rs[i + 1][0] - rs[i][1], i) for i in range(10, len(rs) - 10)]   # keep >= 10 rallies a side
            if not gaps:
                break
            gap, i = max(gaps)
            if gap < MIN_BREAK_S:
                break
            out.append([a, rs[i][1]])
            a, rs = rs[i + 1][0], rs[i + 1:]
        out.append([a, b])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--train", nargs="+", default=["20261004_game03"])
    ap.add_argument("--set-scores", default="")
    a = ap.parse_args()
    if a.target in a.train:
        raise SystemExit("target must not be one of the training games")

    Fs, ys = [], []
    for key in a.train:
        d, gold, _ = tl.truth(key)
        _, bins_tr, F_tr = al.features(key)
        inside, y = tl.bin_labels(bins_tr, d, gold)
        Fs.append(norm(F_tr)[inside]); ys.append(y[inside])
    model = tl.fit_logistic(np.concatenate(Fs), np.concatenate(ys), iters=600, lr=0.05, l2=1e-2)

    g = pl.Game(a.target)
    s, bins, F = al.features(a.target)
    prob = sg.smooth(np.asarray(model(norm(F)), np.float32), 3)
    spans = s["spans"].tolist()
    totals = [h + w for h, w in pl.parse_set_scores(a.set_scores)]

    segs = []
    for i, (lo_t, hi_t) in enumerate(spans):
        m = (bins >= lo_t - 5) & (bins <= hi_t + 5)
        if i < len(totals):
            s_ = sg.segment_to_count(prob[m], bins[m], totals[i], min_len=SEG["min_len"], merge_gap=SEG["merge_gap"])
        else:
            s_ = sg.segment(prob[m], bins[m], **SEG)
        segs += [[x - 0.5, y + 1.0] for x, y in s_]
    spans = split_long_sets(spans, segs)
    ball = sg.load_ball_csv(g.ball_csv, pl.FPS)
    rallies = sg.score_rallies(segs, spans, ball)

    # keep the previous detections / signals, then write the new ones
    auto = g.labels / f"auto_events_{g.key}.json"
    backup = auto.with_name(f"auto_events_{g.key}_old.json")
    if auto.exists() and not backup.exists():           # keep the original detector's output, once
        shutil.copy2(auto, backup)
    sig = g.dir / "signals.npz"
    if sig.exists() and not (g.dir / "signals_old.npz").exists():
        shutil.copy2(sig, g.dir / "signals_old.npz")
    old = dict(np.load(sig))
    old["fused"] = prob
    old["thresholds"] = np.full(len(spans), SEG["hi"], np.float32)
    np.savez_compressed(sig, **old)
    method = "vjepa2+signals trained on " + ",".join(a.train)
    pl.write_events(g, rallies, spans, pl.parse_set_scores(a.set_scores), method)

    print(json.dumps({"game": g.key, "method": method, "rallies": len(rallies),
                      "sets": [sum(1 for r in rallies if r["set"] == i + 1) for i in range(len(spans))],
                      "in_play_share": round(float((prob >= 0.5).mean()), 3),
                      "auto_events": str(auto), "old_kept": str(auto.with_name(f'auto_events_{g.key}_old.json'))}))


if __name__ == "__main__":
    main()

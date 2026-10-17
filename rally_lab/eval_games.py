#!/usr/bin/env python3
"""
Cross-game evaluation of the in-play classifier (V-JEPA 2 clips + detector signals, per-game
normalised) against fully reviewed games, plus the original detector as a baseline.

    python3 rally_lab/eval_games.py

- reviewed game vs reviewed game: train on one, test on the other (rally recall / precision,
  per-half-second F1, start/end error of matched rallies)
- train on 1 vs 2 reviewed games -> test on game 2 against its scoreboard checkpoints
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import apply_live as al        # noqa: E402
import detect_vjepa as dv      # noqa: E402
import eval_rallies as er      # noqa: E402
import signals as sg           # noqa: E402
import train_live as tl        # noqa: E402

WORK = Path("/mnt/d/volleyball/work")
LABELS = HERE.parent / "labels"
REVIEWED = ["20261004_game03", "20261004_game04"]
SEG_NEW = dict(hi=0.5, lo=0.3, min_len=3.0, merge_gap=3.0)
SEG_OLD = dict(hi=0.5, lo=0.3, min_len=2.5, merge_gap=2.5)


def baseline_prob(key):
    """The original detector's score (game 4's signals.npz now holds the V-JEPA score)."""
    p = WORK / key / "signals_old.npz"
    return np.load(p if p.exists() else WORK / key / "signals.npz")["fused"]


def train(keys):
    Fs, ys = [], []
    for k in keys:
        d, gold, _ = tl.truth(k)
        _, b, F = al.features(k)
        inside, y = tl.bin_labels(b, d, gold)
        Fs.append(dv.norm(F)[inside]); ys.append(y[inside])
    return tl.fit_logistic(np.concatenate(Fs), np.concatenate(ys), iters=600, lr=0.05, l2=1e-2)


def score_vs_truth(prob, key, seg):
    d, gold, _ = tl.truth(key)
    s = np.load(WORK / key / "signals.npz")
    b = s["bins"]
    inside, y = tl.bin_labels(b, d, gold)
    p = np.where(inside, prob, 0).astype(np.float32)
    segs = sg.segment(p, b, **seg)
    used, pairs = set(), []
    for a, e in segs:
        best, bj = 0.0, None
        for j, r in enumerate(gold):
            if j in used:
                continue
            ov = min(e, r["end"]) - max(a, r["start"])
            frac = ov / max(1e-3, min(e - a, r["end"] - r["start"]))
            if frac > best:
                best, bj = frac, j
        if bj is not None and best >= 0.3:
            used.add(bj); pairs.append((a, e, gold[bj]))
    starts = [abs(a - g["start"]) for a, e, g in pairs]
    ends = [abs(e - g["end"]) for a, e, g in pairs]
    bs = tl.bin_scores(prob[inside], y[inside])
    return (f"found {len(pairs):3d}/{len(gold)} ({len(pairs) / len(gold):.0%}), {len(segs):3d} detections "
            f"({len(pairs) / max(1, len(segs)):.0%} real), F1 {bs['f1']:.2f}, "
            f"start/end off by {np.median(starts):.1f}/{np.median(ends):.1f} s")


def main():
    print("REVIEWED GAME vs REVIEWED GAME")
    for test in REVIEWED:
        train_on = [k for k in REVIEWED if k != test]
        print(f"\n  test {test[-6:]}:")
        print(f"    original detector          : {score_vs_truth(baseline_prob(test), test, SEG_OLD)}")
        m = train(train_on)
        _, _, F = al.features(test)
        p = sg.smooth(np.asarray(m(dv.norm(F)), np.float32), 3)
        print(f"    V-JEPA trained on {','.join(k[-6:] for k in train_on):8s}: {score_vs_truth(p, test, SEG_NEW)}")

    print("\nGAME 2 vs ITS SCOREBOARD (rally count error at 26 checkpoints, no set scores)")
    cps = json.load(open(LABELS / "20261004_game02" / "scoreboard_checkpoints.json"))["checkpoints"]
    s2, b2, F2 = al.features("20261004_game02")
    spans = s2["spans"].tolist()

    def count_err(prob, seg):
        segs = []
        for a, e in spans:
            m_ = (b2 >= a - 5) & (b2 <= e + 5)
            segs += [[x - 0.5, y + 1.0] for x, y in sg.segment(prob[m_], b2[m_], **seg)]
        r = er.evaluate([{"start": x, "end": y, "score_by": None} for x, y in segs], cps)
        return f"{len(segs):3d} rallies, off by {r['count_mae']:.2f} on average (bias {r['count_bias']:+.2f})"

    print(f"    original detector             : {count_err(s2['fused'], SEG_OLD)}")
    for keys in (["20261004_game03"], ["20261004_game04"], REVIEWED):
        m = train(keys)
        p = sg.smooth(np.asarray(m(dv.norm(F2)), np.float32), 3)
        print(f"    V-JEPA trained on {'+'.join(k[-2:] for k in keys):10s}: {count_err(p, SEG_NEW)}")


if __name__ == "__main__":
    main()

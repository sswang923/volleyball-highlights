#!/usr/bin/env python3
"""
Train the in-play classifier (V-JEPA 2 clips + detector signals) on one reviewed game and test it
on another game, against that game's scoreboard checkpoints and any rallies checked in its review.

    python3 rally_lab/apply_live.py --train 20261004_game03 --test 20261004_game02 --set-totals 44,43
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import eval_rallies as er      # noqa: E402
import signals as sg           # noqa: E402
import train_live as tl        # noqa: E402

WORK = Path("/mnt/d/volleyball/work")
LABELS = HERE.parent / "labels"
SEG = dict(min_len=2.5, merge_gap=2.5)


def features(key):
    s = np.load(WORK / key / "signals.npz")
    bins = s["bins"]
    X = tl.context(np.stack([s[k] for k in tl.FEATS if k in s.files], 1).astype(np.float64))
    v = np.load(WORK / key / "vjepa.npz")
    E = np.stack([np.interp(bins, v["t"], v["emb"][:, j].astype(np.float32)) for j in range(v["emb"].shape[1])], 1)
    return s, bins, np.concatenate([E, X], 1)


def rallies_from(prob, bins, spans, totals):
    segs = []
    for i, (a, b) in enumerate(spans):
        m = (bins >= a - 5) & (bins <= b + 5)
        if i < len(totals):
            s_ = sg.segment_to_count(prob[m], bins[m], totals[i], **SEG)
        else:
            s_ = sg.segment(prob[m], bins[m], hi=0.5, **SEG)
        segs += [[x - 0.5, y + 1.0] for x, y in s_]
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="20261004_game03")
    ap.add_argument("--test", default="20261004_game02")
    ap.add_argument("--set-totals", default="44,43")
    a = ap.parse_args()

    # train on every in-game bin of the reviewed game
    d, gold, _ = tl.truth(a.train)
    s_tr, bins_tr, F_tr = features(a.train)
    inside, y = tl.bin_labels(bins_tr, d, gold)
    model = tl.fit_logistic(F_tr[inside], y[inside], iters=600, lr=0.05, l2=1e-2)

    s, bins, F = features(a.test)
    p_new = sg.smooth(np.asarray(model(F), np.float32), 3)
    p_old = s["fused"]
    spans = s["spans"].tolist()
    cps = json.load(open(LABELS / a.test / "scoreboard_checkpoints.json"))["checkpoints"]
    totals = [int(x) for x in a.set_totals.split(",") if x]

    print(f"trained on {a.train} ({int(inside.sum())} half-seconds, {len(gold)} rallies) -> testing on {a.test}")
    for name, p in (("current detector", p_old), ("V-JEPA 2 + signals", p_new)):
        for label, tot in (("no set scores", []), ("with set scores", totals)):
            segs = rallies_from(p, bins, spans, tot)
            res = er.evaluate([{"start": x, "end": y, "score_by": None} for x, y in segs], cps)
            print(f"  {name:20s} {label:16s}: {len(segs):3d} rallies | vs scoreboard: count off by "
                  f"{res['count_mae']:.2f} on average (bias {res['count_bias']:+.2f})")
    np.savez_compressed(WORK / a.test / "live_vjepa.npz", t=bins, p_live=p_new)

    # any rallies already checked in the test game's review
    rv = LABELS / a.test / f"review_{a.test}.json"
    if rv.exists():
        dg = json.load(open(rv))
        gg = [r for r in dg["rallies"] if r["status"] == "ok"]
        if gg:
            m = np.zeros(len(bins), bool)
            for r in gg:
                m |= (bins >= r["start"] - 20) & (bins <= r["end"] + 20)
            for name, p in (("current detector", p_old), ("V-JEPA 2 + signals", p_new)):
                re = tl.rally_eval(p, bins, m, gg)
                print(f"  vs your {len(gg)} checked rallies in {a.test} ({name}): found {re['found']}/{re['of']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare rally-detection variants on one game against its scoreboard checkpoints.

  A   ball tracker (ONNX VballNet) -> build_golden_v1 rules       (existing event log)
  A0  same, confident tracks only (no short-rally rule)
  B   live/dead frame classifier alone
  C   fusion: ball activity + live/dead + players on court + player speed
  C+  C with the per-set rally count fixed to the set score (AES constraint)
  D   our YOLOv8n ball detector -> tracker rules

    python3 rally_lab/run_variants.py --game game02
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import build_golden_v1 as bg      # noqa: E402
import eval_rallies as ev         # noqa: E402
import signals as sg              # noqa: E402

W = Path("/mnt/d/volleyball/work")


def tracker_rallies(tracks_dir, strict_only=False):
    tracks = bg.load_tracks(Path(tracks_dir))
    if strict_only:
        tracks = [t for t in tracks if t["strict"]]
    return [[r["start"] - bg.PRE_ROLL_S, r["end"] + bg.POST_ROLL_S] for r in bg.merge(tracks)], \
           [[r["start"], r["end"]] for r in bg.merge([t for t in tracks if t["strict"]])]


FUSION_FEATS = ["ball_act", "ball_seen", "p_speed", "p_live"]


def game_features(g, bins, ball):
    feats = {}
    act, seen = sg.ball_activity(ball, bins)
    feats["ball_act"] = sg.smooth(act, 5)
    feats["ball_seen"] = sg.smooth(seen, 5)
    ld = W / f"livedead_{g}.npz"
    if ld.exists():
        feats["p_live"] = sg.smooth(sg.livedead(ld, bins), 3)
    pl = W / f"players_{g}.npz"
    if pl.exists():
        _, spd = sg.player_features(pl, bins)
        feats["p_speed"] = np.clip(sg.smooth(spd, 5) / 100.0, 0, 3)
    return feats


def pseudo_labels(bins, strict):
    live = np.zeros(len(bins), bool)
    near = np.zeros(len(bins), bool)
    for a, b in strict:
        live[(bins >= a + 1) & (bins <= b - 0.5)] = True
        near[(bins >= a - 4) & (bins <= b + 4)] = True
    return live, ~near


def livedead_threshold(train_games, test_game):
    """p_live threshold with the best balanced accuracy on the training games' pseudo-labels."""
    ps, ys = [], []
    for tg in train_games:
        if tg == test_game or not (W / f"livedead_{tg}.npz").exists():
            continue
        tkey = f"20261004_{tg}"
        tball = sg.load_ball_csv(W / "out" / f"proxy_{tkey}" / "ball.csv", 30.0)
        tbins = sg.grid(float(tball["t"].max()))
        p = sg.smooth(sg.livedead(W / f"livedead_{tg}.npz", tbins), 3)
        _, tstrict = tracker_rallies(W / "out" / f"proxy_{tkey}" / "tracks")
        live, dead = pseudo_labels(tbins, tstrict)
        ps.append(p[live | dead])
        ys.append(live[live | dead])
    p, y = np.concatenate(ps), np.concatenate(ys)
    best = max(np.linspace(0.02, 0.9, 89), key=lambda t: ((p >= t) & y).sum() / y.sum() + ((p < t) & ~y).sum() / (~y).sum())
    return float(best)


def fit_logistic(X, y, iters=3000, lr=0.5, l2=1e-3):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
        b -= lr * float(np.mean(p - y))
    return w, b


def report(name, rallies, cps, rows):
    res = ev.evaluate(rallies, cps)
    rows.append((name, res))
    print(f"{name:4s} rallies={res['n_rallies']:4d}  count MAE={res['count_mae']:5.2f}  "
          f"bias={res['count_bias']:+6.2f}  score acc={res['score_acc']:.2f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="game02")
    ap.add_argument("--train-games", default="game01,game03,game04")
    ap.add_argument("--set-totals", default="44,43", help="final points per set (from AES), for C+")
    args = ap.parse_args()
    g = args.game
    key = f"20261004_{g}"
    cps = json.load(open(HERE.parent / "labels" / key / "scoreboard_checkpoints.json"))["checkpoints"]
    tracks = W / "out" / f"proxy_{key}" / "tracks"
    fps = 30.0
    ball = sg.load_ball_csv(W / "out" / f"proxy_{key}" / "ball.csv", fps)
    duration = float(ball["t"].max())
    bins = sg.grid(duration)
    rows = []

    # A / A0
    segs_a, strict = tracker_rallies(tracks)
    spans = sg.set_spans_from_rallies(strict)
    print("set spans:", [(round(a), round(b)) for a, b in spans])
    report("A", sg.score_rallies(segs_a, spans, ball), cps, rows)
    segs_a0, _ = tracker_rallies(tracks, strict_only=True)
    report("A0", sg.score_rallies(segs_a0, spans, ball), cps, rows)

    def in_sets(segs):
        return [s for s in segs if any(a - 5 <= s[0] <= b + 5 for a, b in spans)]

    feats = game_features(g, bins, ball)
    if "p_live" in feats:
        thr = livedead_threshold(args.train_games.split(","), g)
        print(f"   B threshold {thr:.2f} (chosen on training games)")
        segs_b = in_sets(sg.segment(feats["p_live"], bins, hi=thr, min_len=3.0, merge_gap=1.5))
        segs_b = [[a, b + 1.0] for a, b in segs_b]
        report("B", sg.score_rallies(segs_b, spans, ball), cps, rows)

    # C: logistic fusion fitted on the OTHER games' pseudo-labels, applied here
    names = [k for k in FUSION_FEATS if k in feats]
    train_games = [x for x in args.train_games.split(",") if x != g]
    X, y = [], []
    for tg in train_games:
        tkey = f"20261004_{tg}"
        tcsv = W / "out" / f"proxy_{tkey}" / "ball.csv"
        if not tcsv.exists():
            continue
        tball = sg.load_ball_csv(tcsv, fps)
        tbins = sg.grid(float(tball["t"].max()))
        tf = game_features(tg, tbins, tball)
        if not all(k in tf for k in names):
            continue
        _, tstrict = tracker_rallies(W / "out" / f"proxy_{tkey}" / "tracks")
        live, dead = pseudo_labels(tbins, tstrict)
        F = np.stack([tf[k] for k in names], 1)
        X.append(F[live | dead])
        y.append(live[live | dead])
    if X:
        X, y = np.concatenate(X), np.concatenate(y).astype(np.float32)
        w, b = fit_logistic(X, y)
        print(f"   C trained on {len(train_games)} game(s), {len(y)} bins: "
              + ", ".join(f"{k}={v:+.2f}" for k, v in zip(names, w)) + f", bias={b:+.2f}")
        fused = 1 / (1 + np.exp(-(np.stack([feats[k] for k in names], 1) @ w + b)))
        fused = sg.smooth(fused.astype(np.float32), 3)
        np.save(W / f"fused_{g}.npy", np.stack([bins, fused]))
        segs_c = in_sets(sg.segment(fused, bins, hi=0.5, min_len=2.5, merge_gap=2.5))
        segs_c = [[a - 0.5, b + 1.0] for a, b in segs_c]
        report("C", sg.score_rallies(segs_c, spans, ball), cps, rows)

        # C+: per-set count from the final set score
        totals = [int(x) for x in args.set_totals.split(",")]
        segs_cp = []
        for i, (a, b) in enumerate(spans):
            m = (bins >= a - 5) & (bins <= b + 5)
            if i < len(totals):
                s_ = sg.segment_to_count(fused[m], bins[m], totals[i], min_len=2.5, merge_gap=2.5)
            else:
                s_ = sg.segment(fused[m], bins[m], hi=0.5, min_len=2.5, merge_gap=2.5)
            segs_cp += [[x - 0.5, y_ + 1.0] for x, y_ in s_]
        report("C+", sg.score_rallies(segs_cp, spans, ball), cps, rows)

    # D: YOLO ball csv -> tracker rules
    dcsv = W / "out" / f"yolo_{key}" / "ball.csv"
    if dcsv.exists():
        dtr = W / "out" / f"yolo_{key}" / "tracks"
        if not dtr.exists():
            subprocess.run(["bash", "-lc",
                            f"cd ~/vb-tools/fast-volleyball-tracking-inference && PATH=$HOME/.local/bin:$PATH "
                            f"uv run src/track_calculator.py --csv_path {dcsv} --output_dir {W / 'out'} "
                            f"--fps 15 --video_width 1280 --video_height 720 > /dev/null 2>&1"], check=True)
        dball = sg.load_ball_csv(dcsv, 15.0)
        segs_d, _ = tracker_rallies(dtr)
        report("D", sg.score_rallies(segs_d, spans, dball), cps, rows)

    json.dump({n: {k: v for k, v in r.items() if k != "rows"} for n, r in rows},
              open(W / f"variants_{g}.json", "w"), indent=2)


if __name__ == "__main__":
    main()

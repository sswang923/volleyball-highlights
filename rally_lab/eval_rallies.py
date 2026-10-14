#!/usr/bin/env python3
"""
Score a rally event log against scoreboard checkpoints (points played so far).

    python3 rally_lab/eval_rallies.py labels/20261004_game02/event_log_20261004_game02.json \
        labels/20261004_game02/scoreboard_checkpoints.json

Metrics
  count MAE   mean |rallies found so far in the set - board total| over checkpoints
  count bias  mean signed error (+ = over-counting)
  score acc   share of checkpoints where per-team counts are within 1 of the board,
              using the best home/away <-> blue/red mapping per set (the board's
              team order is not known)
"""

import json
import sys
from pathlib import Path


def set_windows(cps):
    """Set windows [start, end) from checkpoints: a set runs until the first checkpoint of the next."""
    sets = sorted({c["set"] for c in cps})
    first = {s: min(c["t"] for c in cps if c["set"] == s) for s in sets}
    last = {s: max(c["t"] for c in cps if c["set"] == s) for s in sets}
    win = {}
    for i, s in enumerate(sets):
        lo = 0 if i == 0 else (last[sets[i - 1]] + first[s]) / 2
        hi = float("inf") if i == len(sets) - 1 else (last[s] + first[sets[i + 1]]) / 2
        win[s] = (lo, hi)
    return win


def evaluate(rallies, cps):
    """rallies: list of dicts with start, end, score_by ('home'/'away'/None)."""
    win = set_windows(cps)
    errs, score_ok, rows = [], [], []
    for s, (lo, hi) in win.items():
        rs = sorted((r for r in rallies if lo <= r["start"] < hi), key=lambda r: r["end"])
        scps = [c for c in cps if c["set"] == s]
        best = None
        for home_is_blue in (True, False):
            ok = 0
            for c in scps:
                done = [r for r in rs if r["end"] <= c["t"]]
                h = sum(r.get("score_by") == "home" for r in done)
                a = sum(r.get("score_by") == "away" for r in done)
                b, rd = (h, a) if home_is_blue else (a, h)
                ok += abs(b - c["blue"]) <= 1 and abs(rd - c["red"]) <= 1
            if best is None or ok > best[0]:
                best = (ok, home_is_blue)
        score_ok.append((best[0], len(scps)))
        for c in scps:
            n = sum(1 for r in rs if r["end"] <= c["t"])
            gt = c["blue"] + c["red"]
            errs.append(n - gt)
            rows.append((c["t"], s, gt, n))
    mae = sum(abs(e) for e in errs) / len(errs)
    bias = sum(errs) / len(errs)
    acc = sum(o for o, _ in score_ok) / sum(n for _, n in score_ok)
    return {"count_mae": round(mae, 2), "count_bias": round(bias, 2), "score_acc": round(acc, 2),
            "n_rallies": len(rallies), "rows": rows}


def rallies_from_log(path):
    ev = json.load(open(path))["events"]
    starts = {(e["set"], e["rally_in_set"]): e for e in ev if e["event"] == "rally_start"}
    out = []
    for e in ev:
        if e["event"] == "rally_end":
            s = starts.get((e["set"], e["rally_in_set"]))
            out.append({"start": s["t_s"] if s else e["t_s"] - 5, "end": e["t_s"], "score_by": e.get("score_by")})
    return out


if __name__ == "__main__":
    log, cp = sys.argv[1], sys.argv[2]
    cps = json.load(open(cp))["checkpoints"]
    res = evaluate(rallies_from_log(log), cps)
    print(f"{Path(log).name}: rallies={res['n_rallies']}  count MAE={res['count_mae']}  "
          f"bias={res['count_bias']:+}  score acc={res['score_acc']}")
    if "-v" in sys.argv:
        for t, s, gt, n in res["rows"]:
            print(f"  t={t:5d} set {s}  board {gt:2d}  found {n:2d}  {n - gt:+d}")

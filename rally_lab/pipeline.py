#!/usr/bin/env python3
"""
Rally detection pipeline — one entry point for the tools (MCP server, CLI).

Per game everything lives in /mnt/d/volleyball/work/<key>/  (key = 20261004_game02):
    proxy.mp4        1280x720 30 fps working copy (GPU transcode of the stitched video)
    proxy/ball.csv   ball per frame (VballNet ONNX tracker)
    proxy/tracks/    ball tracks with rally classification
    players.npz      YOLOv8n person boxes at 5 fps
    livedead.npz     live/dead frame classifier p(live) at 2 fps
    summary.json     result of the last detect run

Rally events are written to labels/<key>/auto_events_<key>.json, and also to
labels/<key>/event_log_<key>.json when that file does not exist yet — a log
the user has started correcting is never overwritten.

    python3 rally_lab/pipeline.py detect --key 20261004_game02 [--set-scores 19-25,25-18,15-10]
    python3 rally_lab/pipeline.py fit-fusion --keys 20261004_game01 20261004_game03
    python3 rally_lab/pipeline.py evaluate --key 20261004_game02
    python3 rally_lab/pipeline.py status --key 20261004_game02

Each step prints one JSON line {"step": ..., "status": ...} so a caller can
follow progress without reading tool logs.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import build_golden_v1 as bg  # noqa: E402
import signals as sg          # noqa: E402

VIDEO_DIR = Path(os.environ.get("VB_VIDEO_DIR", "/mnt/d/volleyball"))
WORK = Path(os.environ.get("VB_WORK_DIR", "/mnt/d/volleyball/work"))
MODELS = WORK / "models"
TRACKER = Path.home() / "vb-tools" / "fast-volleyball-tracking-inference"
FUSION_FEATS = ["ball_act", "ball_seen", "p_speed", "p_live"]
FPS = 30.0
MALLOC_ENV = {"MALLOC_MMAP_THRESHOLD_": "134217728", "MALLOC_TRIM_THRESHOLD_": "-1"}


def emit(step, status, **kw):
    print(json.dumps({"step": step, "status": status, "time": datetime.now().isoformat(timespec="seconds"), **kw}),
          flush=True)


def run(cmd, step, env=None):
    t0 = time.time()
    emit(step, "running")
    e = dict(os.environ, PATH=f"{Path.home()}/.local/bin:" + os.environ.get("PATH", ""), **(env or {}))
    res = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, env=e)
    if res.returncode != 0:
        emit(step, "failed", error=(res.stderr or res.stdout)[-800:])
        raise SystemExit(1)
    emit(step, "done", seconds=round(time.time() - t0))


class Game:
    def __init__(self, key):
        self.key = key
        self.dir = WORK / key
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stitched = VIDEO_DIR / f"stitched_{key}.mp4"
        self.proxy = self.dir / "proxy.mp4"
        self.ball_dir = self.dir / "proxy"
        self.ball_csv = self.ball_dir / "ball.csv"
        self.tracks = self.ball_dir / "tracks"
        self.players = self.dir / "players.npz"
        self.livedead = self.dir / "livedead.npz"
        self.labels = REPO / "labels" / key


# ── steps ────────────────────────────────────────────────────────────────────

def step_proxy(g):
    if g.proxy.exists():
        return emit("proxy", "cached")
    if not g.stitched.exists():
        emit("proxy", "failed", error=f"{g.stitched.name} not found — stitch the game first")
        raise SystemExit(1)
    tmp = g.dir / "proxy.tmp.mp4"
    run(f"ffmpeg-nvenc -v error -y -hwaccel cuda -i '{g.stitched}' -vf scale=1280:720,fps=30 "
        f"-c:v h264_nvenc -preset p4 -cq 23 -c:a aac -b:a 128k '{tmp}' && mv '{tmp}' '{g.proxy}'", "proxy")


def step_ball(g):
    if g.tracks.exists() and any(g.tracks.iterdir()):
        return emit("ball", "cached")
    model = sorted(TRACKER.glob("models/VballNetV4c_seq9_grayscale_*.onnx"))[0]
    run(f"cd '{TRACKER}' && uv run src/inference_onnx_seq_gray_v2.py --video_path '{g.proxy}' "
        f"--model_path '{model}' --output_dir '{g.dir}' --only_csv && "
        f"uv run src/track_calculator.py --csv_path '{g.ball_csv}' --output_dir '{g.dir}' "
        f"--fps 30 --video_width 1280 --video_height 720", "ball")


def step_players(g):
    if g.players.exists():
        return emit("players", "cached")
    run(f"python3 '{HERE / 'detect_players.py'}' --video '{g.proxy}' --out '{g.players}' --fps 5",
        "players", env=MALLOC_ENV)


def step_livedead(g):
    model = MODELS / "livedead.pt"
    if g.livedead.exists():
        return emit("livedead", "cached")
    if not model.exists():
        return emit("livedead", "skipped", reason="no live/dead model trained yet")
    run(f"python3 '{HERE / 'infer.py'}' livedead --model '{model}' --video '{g.proxy}' --out '{g.livedead}' --fps 2",
        "livedead", env=MALLOC_ENV)


# ── features / fusion ────────────────────────────────────────────────────────

def features(g, ball, bins):
    feats = {}
    act, seen = sg.ball_activity(ball, bins)
    feats["ball_act"] = sg.smooth(act, 5)
    feats["ball_seen"] = sg.smooth(seen, 5)
    if g.players.exists():
        _, spd = sg.player_features(g.players, bins)
        feats["p_speed"] = np.clip(sg.smooth(spd, 5) / 100.0, 0, 3)
    if g.livedead.exists():
        feats["p_live"] = sg.smooth(sg.livedead(g.livedead, bins), 3)
    return feats


def tracker_spans(g):
    tracks = bg.load_tracks(g.tracks)
    strict = [[r["start"], r["end"]] for r in bg.merge([t for t in tracks if t["strict"]])]
    return strict


def pseudo_labels(bins, strict):
    live = np.zeros(len(bins), bool)
    near = np.zeros(len(bins), bool)
    for a, b in strict:
        live[(bins >= a + 1) & (bins <= b - 0.5)] = True
        near[(bins >= a - 4) & (bins <= b + 4)] = True
    return live, ~near


def fit_logistic(X, y, iters=3000, lr=0.5, l2=1e-3):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
        b -= lr * float(np.mean(p - y))
    return w, b


def cmd_fit_fusion(args):
    X, y, names = [], [], None
    for key in args.keys:
        g = Game(key)
        ball = sg.load_ball_csv(g.ball_csv, FPS)
        bins = sg.grid(float(ball["t"].max()))
        f = features(g, ball, bins)
        names = names or [k for k in FUSION_FEATS if k in f]
        if not all(k in f for k in names):
            emit("fit", "skipped", game=key, reason="missing features")
            continue
        live, dead = pseudo_labels(bins, tracker_spans(g))
        X.append(np.stack([f[k] for k in names], 1)[live | dead])
        y.append(live[live | dead])
    X, y = np.concatenate(X), np.concatenate(y).astype(np.float32)
    w, b = fit_logistic(X, y)
    MODELS.mkdir(parents=True, exist_ok=True)
    out = {"features": names, "weights": [float(v) for v in w], "bias": float(b),
           "trained_on": args.keys, "bins": int(len(y)), "fitted_at": datetime.now().isoformat(timespec="seconds")}
    json.dump(out, open(MODELS / "fusion.json", "w"), indent=2)
    emit("fit", "done", **out)


def fused_score(g, ball, bins):
    feats = features(g, ball, bins)
    fm = MODELS / "fusion.json"
    if fm.exists():
        m = json.load(open(fm))
        if all(k in feats for k in m["features"]):
            z = np.stack([feats[k] for k in m["features"]], 1) @ np.array(m["weights"]) + m["bias"]
            return sg.smooth((1 / (1 + np.exp(-z))).astype(np.float32), 3), "fusion:" + "+".join(m["features"])
    return sg.smooth(np.clip(feats["ball_act"] * 1.4, 0, 1), 3), "ball_act only"


# ── detect ───────────────────────────────────────────────────────────────────

def parse_set_scores(s):
    if not s:
        return []
    out = []
    for part in s.split(","):
        h, a = part.strip().split("-")
        out.append((int(h), int(a)))
    return out


def cmd_detect(args):
    g = Game(args.key)
    t0 = time.time()
    step_proxy(g)
    step_ball(g)
    step_players(g)
    step_livedead(g)

    emit("rallies", "running")
    ball = sg.load_ball_csv(g.ball_csv, FPS)
    bins = sg.grid(float(ball["t"].max()))
    score, method = fused_score(g, ball, bins)
    spans = sg.set_spans_from_rallies(tracker_spans(g))
    set_scores = parse_set_scores(args.set_scores)

    segs = []
    for i, (a, b) in enumerate(spans):
        m = (bins >= a - 5) & (bins <= b + 5)
        if i < len(set_scores):
            s = sg.segment_to_count(score[m], bins[m], sum(set_scores[i]), min_len=2.5, merge_gap=2.5)
        else:
            s = sg.segment(score[m], bins[m], hi=0.5, min_len=2.5, merge_gap=2.5)
        segs += [[x - 0.5, y + 1.0] for x, y in s]
    rallies = sg.score_rallies(segs, spans, ball)

    events = write_events(g, rallies, spans, set_scores, method)
    summary = {
        "game": g.key, "method": method, "set_score_constraint": bool(set_scores),
        "rallies": len(rallies),
        "sets": [{"set": i + 1, "start": bg.fmt(a), "end": bg.fmt(b),
                  "rallies": sum(1 for r in rallies if r["set"] == i + 1),
                  "target": sum(set_scores[i]) if i < len(set_scores) else None}
                 for i, (a, b) in enumerate(spans)],
        "events_file": str(events["auto"]), "event_log_written": events["event_log_written"],
        "seconds": round(time.time() - t0),
        "note": "point winners use a weak serve-side rule; check them in the labeler",
    }
    json.dump(summary, open(g.dir / "summary.json", "w"), indent=2)
    emit("rallies", "done", summary=summary)


def write_events(g, rallies, spans, set_scores, method):
    vf = g.stitched.name
    ev = []

    def add(kind, t, **kw):
        ev.append({"event": kind, "t_s": round(float(t), 3), "t": bg.fmt(t), "video_file": vf, **kw})

    if rallies:
        add("game_start", max(0.0, rallies[0]["start"] - 2), set=None, rally_in_set=None, score_by=None)
    for i, (a, b) in enumerate(spans, 1):
        rs = [r for r in rallies if r["set"] == i]
        if not rs:
            continue
        add("set_start", rs[0]["start"], set=i, rally_in_set=None, score=None)
        h = w = 0
        for k, r in enumerate(rs, 1):
            h += r["score_by"] == "home"
            w += r["score_by"] == "away"
            common = dict(set=i, rally_in_set=k, auto=True, confirmed=False)
            add("rally_start", r["start"], score_by=None, **common)
            add("rally_end", r["end"], score_by=r["score_by"], score_after=f"{h}-{w}", **common)
        add("set_end", rs[-1]["end"] + 1, set=i, rally_in_set=None, score_by=None)
    for n, e in enumerate(ev):
        e["seq"] = n

    g.labels.mkdir(parents=True, exist_ok=True)
    doc = {"game": g.key, "generated_at": datetime.now().isoformat(timespec="seconds"),
           "generator": f"rally_lab/pipeline.py ({method})", "event_count": len(ev), "events": ev}
    auto = g.labels / f"auto_events_{g.key}.json"
    json.dump(doc, open(auto, "w"), indent=2)
    log = g.labels / f"event_log_{g.key}.json"
    wrote = False
    if not log.exists():
        json.dump(doc, open(log, "w"), indent=2)
        wrote = True
    return {"auto": auto, "event_log_written": wrote}


# ── evaluate / status ────────────────────────────────────────────────────────

def load_rallies(path):
    ev = json.load(open(path))["events"]
    starts = {(e["set"], e["rally_in_set"]): e for e in ev if e["event"] == "rally_start"}
    out = []
    for e in ev:
        if e["event"] == "rally_end":
            s = starts.get((e["set"], e["rally_in_set"]))
            out.append({"start": s["t_s"] if s else e["t_s"] - 5, "end": e["t_s"], "set": e["set"],
                        "score_by": e.get("score_by"), "confirmed": bool(e.get("confirmed", not e.get("auto")))})
    return out


def match(pred, gold, min_overlap=0.3):
    """Greedy one-to-one matching by overlap / shorter-span length."""
    used, pairs = set(), []
    for i, p in enumerate(pred):
        best, bj = 0.0, None
        for j, q in enumerate(gold):
            if j in used:
                continue
            ov = min(p["end"], q["end"]) - max(p["start"], q["start"])
            frac = ov / max(1e-3, min(p["end"] - p["start"], q["end"] - q["start"]))
            if frac > best:
                best, bj = frac, j
        if bj is not None and best >= min_overlap:
            used.add(bj)
            pairs.append((i, bj))
    return pairs


def cmd_evaluate(args):
    g = Game(args.key)
    res = {"game": g.key}
    auto = g.labels / f"auto_events_{g.key}.json"
    log = g.labels / f"event_log_{g.key}.json"
    if auto.exists() and log.exists():
        pred = load_rallies(auto)
        gold = [r for r in load_rallies(log) if r["confirmed"]]
        if gold:
            lo, hi = min(r["start"] for r in gold) - 30, max(r["end"] for r in gold) + 30
            pred_in = [p for p in pred if lo <= p["start"] <= hi]
            pairs = match(pred_in, gold)
            starts = [abs(pred_in[i]["start"] - gold[j]["start"]) for i, j in pairs]
            ends = [abs(pred_in[i]["end"] - gold[j]["end"]) for i, j in pairs]
            winners = [pred_in[i]["score_by"] == gold[j]["score_by"] for i, j in pairs if gold[j]["score_by"]]
            res["vs_confirmed_labels"] = {
                "confirmed_rallies": len(gold), "detected_in_range": len(pred_in), "matched": len(pairs),
                "recall": round(len(pairs) / len(gold), 3),
                "precision": round(len(pairs) / max(1, len(pred_in)), 3),
                "median_start_err_s": round(float(np.median(starts)), 2) if starts else None,
                "median_end_err_s": round(float(np.median(ends)), 2) if ends else None,
                "point_winner_acc": round(sum(winners) / len(winners), 3) if winners else None,
            }
        else:
            res["vs_confirmed_labels"] = "no confirmed rallies in the event log yet"
    cp = g.labels / "scoreboard_checkpoints.json"
    if cp.exists() and auto.exists():
        import eval_rallies as er
        r = er.evaluate(load_rallies(auto), json.load(open(cp))["checkpoints"])
        res["vs_scoreboard"] = {k: v for k, v in r.items() if k != "rows"}
    print(json.dumps(res))


def cmd_status(args):
    g = Game(args.key)
    st = {"game": g.key, "stitched": g.stitched.exists(), "proxy": g.proxy.exists(),
          "ball": g.tracks.exists(), "players": g.players.exists(), "livedead": g.livedead.exists()}
    s = g.dir / "summary.json"
    if s.exists():
        st["last_detect"] = json.load(open(s))
    log = g.labels / f"event_log_{g.key}.json"
    if log.exists():
        rs = load_rallies(log)
        st["event_log"] = {"rallies": len(rs), "confirmed": sum(r["confirmed"] for r in rs)}
    print(json.dumps(st))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--key", required=True)
    d.add_argument("--set-scores", default="", help="final score per set, home-away, e.g. 19-25,25-18,15-10")
    f = sub.add_parser("fit-fusion")
    f.add_argument("--keys", nargs="+", required=True)
    for name in ("evaluate", "status"):
        p = sub.add_parser(name)
        p.add_argument("--key", required=True)
    args = ap.parse_args()
    {"detect": cmd_detect, "fit-fusion": cmd_fit_fusion, "evaluate": cmd_evaluate, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()

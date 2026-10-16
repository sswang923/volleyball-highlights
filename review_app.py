#!/usr/bin/env python3
"""
Rally review web app — watch detected rallies and fix them.

    python3 review_app.py [--port 8770] [--host 127.0.0.1]
    open http://localhost:8770/

Per game it plays the 720p working copy (work/<key>/proxy.mp4, falls back to the
stitched video) and loads, in order of preference:
    labels/<key>/review_<key>.json        your saved review
    labels/<key>/auto_events_<key>.json   detector output
    labels/<key>/event_log_<key>.json     older drafts
Every save writes review_<key>.json and rallies_<key>.txt (editable text), and
exports event_log_<key>.json (the format
label_clips.py, evaluation and YouTube chapters read). The first export backs
up any existing event log.
"""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request

HERE = Path(__file__).resolve().parent
LABELS = HERE / "labels"
VIDEO_DIR = Path(os.environ.get("VB_VIDEO_DIR", "/mnt/d/volleyball"))
WORK = Path(os.environ.get("VB_WORK_DIR", "/mnt/d/volleyball/work"))
KEY_RE = re.compile(r"^\d{8}_game\d{2}$")
RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)")

app = Flask(__name__, template_folder=str(HERE / "templates"))
_durations = {}


def check_key(key):
    if not KEY_RE.match(key):
        abort(404)
    return key


def video_path(key):
    proxy = WORK / key / "proxy.mp4"
    return proxy if proxy.exists() else VIDEO_DIR / f"stitched_{key}.mp4"


def duration(p: Path) -> float:
    if p not in _durations:
        r = subprocess.run(["bash", "-lc", f"ffprobe -v error -show_entries format=duration -of csv=p=0 '{p}'"],
                           capture_output=True, text=True)
        try:
            _durations[p] = float(r.stdout.strip())
        except ValueError:
            _durations[p] = 0.0
    return _durations[p]


def fmt(t):
    t = int(max(0, t))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


# ── load / save ──────────────────────────────────────────────────────────────

def rallies_from_events(ev):
    starts = {(e.get("set"), e.get("rally_in_set")): e for e in ev if e.get("event") == "rally_start"}
    out = []
    for e in ev:
        if e.get("event") == "rally_end":
            s = starts.get((e.get("set"), e.get("rally_in_set")))
            if not s:
                continue
            out.append({"start": s["t_s"], "end": e["t_s"], "set": e.get("set") or 1,
                        "winner": e.get("score_by"),
                        "status": "ok" if e.get("confirmed") else "auto"})
    out.sort(key=lambda r: r["start"])
    gs = next((e["t_s"] for e in ev if e.get("event") == "game_start"), None)
    ge = next((e["t_s"] for e in ev if e.get("event") == "game_end"), None)
    return out, gs, ge


def load_review(key):
    lab = LABELS / key
    rv = lab / f"review_{key}.json"
    if rv.exists():
        d = json.load(open(rv))
        d["source"] = rv.name
        return d
    for name in (f"auto_events_{key}.json", f"event_log_{key}.json"):
        p = lab / name
        if p.exists():
            rallies, gs, ge = rallies_from_events(json.load(open(p)).get("events", []))
            # auto point winners are unreliable — start the review without them
            if name.startswith("auto_"):
                for r in rallies:
                    r["winner"] = None
            return {"game": key, "game_start": gs, "game_end": ge, "rallies": rallies, "source": name}
    return {"game": key, "game_start": None, "game_end": None, "rallies": [], "source": None}


def export_event_log(key, d):
    vf = f"stitched_{key}.mp4"
    ev = []

    def add(kind, t, **kw):
        ev.append({"event": kind, "t_s": round(float(t), 3), "t": fmt(t), "video_file": vf, **kw})

    rallies = sorted((r for r in d["rallies"] if r.get("status") != "deleted"), key=lambda r: r["start"])
    if d.get("game_start") is not None:
        add("game_start", d["game_start"], set=None, rally_in_set=None, score_by=None)
    for s in sorted({r["set"] for r in rallies}):
        rs = [r for r in rallies if r["set"] == s]
        add("set_start", rs[0]["start"], set=s, rally_in_set=None, score=None)
        for k, r in enumerate(rs, 1):
            ok = r.get("status") == "ok"
            common = dict(set=s, rally_in_set=k, auto=not ok, confirmed=ok)
            add("rally_start", r["start"], score_by=None, **common)
            add("rally_end", r["end"], score_by=r.get("winner"), **common)
        add("set_end", rs[-1]["end"] + 1, set=s, rally_in_set=None, score_by=None)
    if d.get("game_end") is not None:
        add("game_end", d["game_end"], set=None, rally_in_set=None, score_by=None)
    ev.sort(key=lambda e: e["t_s"])
    for i, e in enumerate(ev):
        e["seq"] = i
    log = LABELS / key / f"event_log_{key}.json"
    if log.exists() and not (LABELS / key / f"review_{key}.json").exists():
        log.rename(log.with_name(f"{log.name}.bak.{datetime.now():%Y%m%d%H%M%S}"))
    tmp = log.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"game": key, "generated_at": datetime.now().isoformat(timespec="seconds"),
                               "generator": "review_app.py", "event_count": len(ev), "events": ev}, indent=2))
    tmp.replace(log)


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    games = []
    for p in sorted(VIDEO_DIR.glob("stitched_*.mp4")):
        key = p.stem[len("stitched_"):]
        if KEY_RE.match(key):
            d = load_review(key)
            rs = [r for r in d["rallies"] if r.get("status") != "deleted"]
            games.append({"key": key, "rallies": len(rs), "ok": sum(r.get("status") == "ok" for r in rs),
                          "source": d["source"]})
    return render_template("review.html", game=None, games=games)


@app.route("/g/<key>")
def game_page(key):
    check_key(key)
    return render_template("review.html", game=key, games=[])


@app.route("/api/<key>/rallies", methods=["GET", "PUT"])
def api_rallies(key):
    check_key(key)
    if request.method == "GET":
        d = load_review(key)
        d["duration"] = duration(video_path(key))
        return jsonify(d)
    d = request.get_json(force=True) or {}
    clean = []
    for r in d.get("rallies", []):
        s, e = float(r["start"]), float(r["end"])
        if e <= s:
            e = s + 1.0
        clean.append({"start": round(s, 2), "end": round(e, 2), "set": int(r.get("set") or 1),
                      "winner": r.get("winner") if r.get("winner") in ("home", "away") else None,
                      "status": r.get("status") if r.get("status") in ("auto", "ok", "deleted") else "auto"})
    clean.sort(key=lambda r: r["start"])
    out = {"game": key, "saved_at": datetime.now().isoformat(timespec="seconds"),
           "game_start": d.get("game_start"), "game_end": d.get("game_end"), "rallies": clean}
    (LABELS / key).mkdir(parents=True, exist_ok=True)
    export_event_log(key, out)
    rv = LABELS / key / f"review_{key}.json"
    tmp = rv.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(rv)
    if isinstance(d.get("text"), str):          # editable text twin of the review
        (LABELS / key / f"rallies_{key}.txt").write_text(d["text"], encoding="utf-8")
    return jsonify({"ok": True, "saved_at": out["saved_at"],
                    "rallies": sum(r["status"] != "deleted" for r in clean),
                    "reviewed": sum(r["status"] == "ok" for r in clean)})


@app.route("/api/<key>/text")
def api_text(key):
    """The rally list as text (labels/<key>/rallies_<key>.txt), for editing outside the app."""
    check_key(key)
    p = LABELS / key / f"rallies_{key}.txt"
    return jsonify({"text": p.read_text(encoding="utf-8") if p.exists() else None})


@app.route("/api/<key>/signals")
def api_signals(key):
    """Detector signals for a time window (from work/<key>/signals.npz, written by pipeline detect)."""
    check_key(key)
    p = WORK / key / "signals.npz"
    if not p.exists():
        return jsonify({"available": False})
    import numpy as np
    s = np.load(p)
    t0, t1 = float(request.args.get("t0", 0)), float(request.args.get("t1", 60))
    m = (s["bins"] >= t0) & (s["bins"] <= t1)
    r = lambda a, n: [round(float(x), n) for x in a]
    out = {"available": True, "t": r(s["bins"][m], 2)}
    for k in ("fused", "ball_seen", "ball_act", "p_live"):
        if k in s.files:
            out[k] = r(s[k][m], 3)
    mid = (t0 + t1) / 2
    thr = 0.5
    for (a, b), h in zip(s["spans"].tolist(), s["thresholds"].tolist()):
        if a - 5 <= mid <= b + 5:
            thr = h
    out["hi"], out["lo"] = round(thr, 3), round(0.6 * thr, 3)
    return jsonify(out)


@app.route("/video/<key>")
def video(key):
    check_key(key)
    path = video_path(key)
    if not path.exists():
        abort(404)
    size = path.stat().st_size
    m = RANGE_RE.search(request.headers.get("Range", ""))
    start = int(m.group(1)) if m else 0
    end = min(int(m.group(2)) if m and m.group(2) else size - 1, size - 1)
    if start > end:
        abort(416)

    def gen():
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 18, left))
                if not chunk:
                    break
                left -= len(chunk)
                yield chunk

    rv = Response(gen(), status=206 if m else 200, mimetype="video/mp4", direct_passthrough=True)
    rv.headers["Accept-Ranges"] = "bytes"
    rv.headers["Content-Length"] = str(end - start + 1)
    if m:
        rv.headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return rv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    print(f"Rally review: http://localhost:{a.port}/", flush=True)
    app.run(host=a.host, port=a.port, threaded=True)

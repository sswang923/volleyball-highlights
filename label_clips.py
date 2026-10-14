#!/usr/bin/env python3
"""
Golden-data clip labeller for the volleyball highlight detector.

Run:
    python3 label_clips.py \\
        --video /mnt/d/volleyball/stitched_20260501_game01.mp4 \\
        --detector-output /path/to/event_log_or_point_timeline.json   # optional
        --player 12                                                   # optional
        --n-random 30 \\
        --window-pad 1.5 \\
        --port 8765

Then open http://localhost:8765/ in any browser.

Outputs live in:
    labels/<game_key>/manifest.json   (candidate clips, generated once)
    labels/<game_key>/labels.json     (your labels, autosaved on every change)

Hotkeys are documented in templates/index.html.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)

log = logging.getLogger("label_clips")

HERE = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# Global app state (set once at startup from CLI args)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AppState:
    video_path: Path = None
    game_key: str = ""
    player: Optional[int] = None
    duration_s: float = 0.0
    labels_dir: Path = None
    manifest: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)


STATE = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ffprobe_duration(video: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video),
        ],
        stderr=subprocess.STDOUT,
    )
    return float(out.strip())


def derive_game_key(video: Path) -> str:
    """stitched_20260501_game01.mp4 → 20260501_game01"""
    stem = video.stem
    return re.sub(r"^stitched_", "", stem)


def parse_detector_output(path: Path) -> list[tuple[float, float, str]]:
    """
    Return a list of (t_start, t_end, label) tuples extracted from one of:
      - event_log_*.json (events: point_start/point_end pairs)
      - point_timeline_*.json (points list with t_start/t_end or start_s/end_s)
      - highlight reel segments JSON (segments with abs_start/abs_end)
    """
    data = json.loads(path.read_text())
    spans = []

    # Case A: event_log format with events[]
    if isinstance(data, dict) and "events" in data:
        events = data["events"]
        # pair point_start with point_end
        open_pt = None
        for e in events:
            kind = e.get("event")
            t_s = e.get("t_s")
            if t_s is None:
                continue
            if kind == "point_start":
                open_pt = (float(t_s), e.get("point_in_set"))
            elif kind == "point_end" and open_pt is not None:
                t_start, pt = open_pt
                spans.append((t_start, float(t_s), f"point_{pt}" if pt else "point"))
                open_pt = None

    # Case B: point_timeline format with points[]
    if not spans and isinstance(data, dict) and "points" in data:
        for p in data["points"]:
            ts = p.get("t_start") or p.get("start_s") or p.get("t_s_start")
            te = p.get("t_end") or p.get("end_s") or p.get("t_s_end")
            if ts is None or te is None:
                continue
            label = p.get("label") or f"point_{p.get('point_in_set', '')}".rstrip("_")
            spans.append((float(ts), float(te), label or "point"))

    # Case C: segments[] (from highlight detector)
    if not spans and isinstance(data, dict) and "segments" in data:
        for s in data["segments"]:
            ts = s.get("start_time") or s.get("abs_start")
            te = s.get("end_time") or s.get("abs_end")
            if ts is None or te is None:
                continue
            spans.append((float(ts), float(te), "segment"))

    return spans


def build_manifest(
    video: Path,
    duration_s: float,
    detector_spans: list[tuple[float, float, str]],
    n_random: int,
    window_pad: float,
    window_size: float,
    edge_buffer: float = 30.0,
    seed: int = 42,
) -> list[dict]:
    """
    Build the candidate clip list:
      - one clip per detector span (padded)
      - plus N random non-overlapping windows from uncovered regions
    """
    clips: list[dict] = []

    # Detector clips
    occupied: list[tuple[float, float]] = []
    for i, (ts, te, label) in enumerate(detector_spans):
        s = max(0.0, ts - window_pad)
        e = min(duration_s, te + window_pad)
        clips.append({
            "clip_id": f"det_{i:04d}",
            "source": "detector",
            "t_start": round(s, 3),
            "t_end": round(e, 3),
            "label_hint": label,
        })
        occupied.append((s, e))

    # Random sample from un-covered regions
    rng = random.Random(seed)
    occ_sorted = sorted(occupied)
    candidates = []
    tries = 0
    max_tries = n_random * 50
    while len(candidates) < n_random and tries < max_tries:
        tries += 1
        start = rng.uniform(edge_buffer, max(edge_buffer, duration_s - edge_buffer - window_size))
        end = start + window_size
        # check overlap with occupied
        overlap = False
        for (os_, oe_) in occ_sorted:
            if not (end < os_ or start > oe_):
                overlap = True
                break
        if overlap:
            continue
        # check overlap with already-chosen randoms
        for c in candidates:
            if not (end < c[0] or start > c[1]):
                overlap = True
                break
        if overlap:
            continue
        candidates.append((start, end))

    for i, (s, e) in enumerate(sorted(candidates)):
        clips.append({
            "clip_id": f"rnd_{i:04d}",
            "source": "random",
            "t_start": round(s, 3),
            "t_end": round(e, 3),
            "label_hint": None,
        })

    return clips


def load_or_build_manifest(args) -> list[dict]:
    manifest_path = STATE.labels_dir / "manifest.json"
    if manifest_path.exists() and not args.rebuild:
        log.info(f"Loading existing manifest: {manifest_path}")
        return json.loads(manifest_path.read_text())["clips"]

    detector_spans = []
    if args.detector_output:
        detector_spans = parse_detector_output(Path(args.detector_output))
        log.info(f"Parsed {len(detector_spans)} spans from {args.detector_output}")

    clips = build_manifest(
        video=STATE.video_path,
        duration_s=STATE.duration_s,
        detector_spans=detector_spans,
        n_random=args.n_random,
        window_pad=args.window_pad,
        window_size=args.window_size,
    )
    payload = {
        "video": str(STATE.video_path),
        "game_key": STATE.game_key,
        "duration_s": STATE.duration_s,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_detector": sum(1 for c in clips if c["source"] == "detector"),
        "n_random": sum(1 for c in clips if c["source"] == "random"),
        "clips": clips,
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    log.info(f"Wrote manifest ({len(clips)} clips): {manifest_path}")
    return clips


def load_labels() -> dict:
    p = STATE.labels_dir / "labels.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_labels() -> None:
    p = STATE.labels_dir / "labels.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(STATE.labels, indent=2, sort_keys=True))
    tmp.replace(p)


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder=str(HERE / "templates"),
    static_folder=str(HERE / "static"),
)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0   # never cache static assets
app.jinja_env.auto_reload = True


@app.after_request
def _no_cache_static(resp):
    # Belt and suspenders: tell the browser not to cache static or HTML
    if request.path.startswith("/static/") or resp.mimetype == "text/html":
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


def _switch_video_if_requested():
    """If ?game=<key> or ?video=<name> is present and differs from current, swap STATE."""
    if not STATE.video_path:
        return
    name = request.args.get("video")
    game = request.args.get("game")
    if game and not name:
        # Resolve game_key → video file
        videos_dir = STATE.video_path.parent
        candidates = sorted(videos_dir.glob(f"stitched_{game}.mp4"))
        if candidates:
            name = candidates[0].name
    if not name:
        return
    candidate = STATE.video_path.parent / name
    try:
        candidate = candidate.resolve()
    except Exception:
        return
    if not candidate.exists() or candidate == STATE.video_path:
        return
    STATE.video_path = candidate
    STATE.game_key = derive_game_key(candidate)
    STATE.duration_s = ffprobe_duration(candidate)
    STATE.labels_dir = STATE.labels_dir.parent / STATE.game_key
    STATE.labels_dir.mkdir(parents=True, exist_ok=True)
    STATE.manifest = []
    STATE.labels = load_labels()
    log.info(f"Switched video: {STATE.video_path}")


@app.route("/g/<game_key>")
def game_shortlink(game_key):
    """Permalink redirect: /g/<game_key> → /events?game=<game_key>."""
    return redirect(url_for("events_page", game=game_key))


@app.route("/")
def index():
    return redirect(url_for("events_page"))


@app.route("/events")
def events_page():
    _switch_video_if_requested()
    return render_template(
        "events.html",
        game_key=STATE.game_key,
        player=STATE.player,
        video_name=STATE.video_path.name,
        duration_s=STATE.duration_s,
    )


@app.route("/api/videos")
def api_videos():
    videos_dir = STATE.video_path.parent if STATE.video_path else None
    out = []
    if videos_dir and videos_dir.exists():
        for p in sorted(videos_dir.glob("stitched_*.mp4")):
            out.append({
                "name": p.name,
                "game_key": derive_game_key(p),
                "size_mb": round(p.stat().st_size / 1e6),
            })
    return jsonify({
        "current": STATE.video_path.name if STATE.video_path else None,
        "current_game_key": STATE.game_key,
        "videos": out,
    })


# ── Highlight collections (multi) ──────────────────────────────────────────

@app.route("/api/highlights", methods=["GET"])
def api_highlights_list():
    return jsonify(_load_highlights())


@app.route("/api/highlights", methods=["POST"])
def api_highlights_create():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "Untitled").strip()
    data = _load_highlights()
    existing_ids = [c["id"] for c in data["collections"]]
    cid = _unique_slug(name, existing_ids)
    coll = {
        "id": cid,
        "name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rallies": [],
    }
    data["collections"].append(coll)
    _save_highlights(data)
    return jsonify({"ok": True, "collection": coll})


@app.route("/api/highlights/<cid>", methods=["PATCH"])
def api_highlights_update(cid):
    payload = request.get_json(silent=True) or {}
    data = _load_highlights()
    coll = _find_collection(data, cid)
    if not coll:
        abort(404)
    if "name" in payload:
        coll["name"] = payload["name"].strip() or coll["name"]
    _save_highlights(data)
    return jsonify({"ok": True, "collection": coll})


@app.route("/api/highlights/<cid>", methods=["DELETE"])
def api_highlights_delete(cid):
    data = _load_highlights()
    data["collections"] = [c for c in data["collections"] if c.get("id") != cid]
    _save_highlights(data)
    return jsonify({"ok": True})


@app.route("/api/highlights/<cid>/rallies", methods=["POST"])
def api_highlights_add_rally(cid):
    payload = request.get_json(silent=True) or {}
    set_num = payload.get("set")
    rally_num = payload.get("rally_in_set")
    notes = payload.get("notes", "")
    if set_num is None or rally_num is None:
        return jsonify({"error": "set and rally_in_set required"}), 400
    data = _load_highlights()
    coll = _find_collection(data, cid)
    if not coll:
        abort(404)
    # de-dupe
    for r in coll["rallies"]:
        if r.get("set") == set_num and r.get("rally_in_set") == rally_num:
            return jsonify({"ok": True, "already": True, "collection": coll})
    coll["rallies"].append({
        "set": set_num, "rally_in_set": rally_num,
        "notes": notes,
        "added_at": datetime.now().isoformat(timespec="seconds"),
    })
    _save_highlights(data)
    return jsonify({"ok": True, "collection": coll})


@app.route("/api/highlights/<cid>/rallies/<int:set_num>/<int:rally_num>", methods=["DELETE"])
def api_highlights_remove_rally(cid, set_num, rally_num):
    data = _load_highlights()
    coll = _find_collection(data, cid)
    if not coll:
        abort(404)
    before = len(coll["rallies"])
    coll["rallies"] = [
        r for r in coll["rallies"]
        if not (r.get("set") == set_num and r.get("rally_in_set") == rally_num)
    ]
    _save_highlights(data)
    return jsonify({"ok": True, "removed": before - len(coll["rallies"])})


@app.route("/api/highlights/renumber", methods=["POST"])
def api_highlights_renumber():
    """Apply a rename map { set, pairs: [{old, new}, ...] } to all collections."""
    payload = request.get_json(silent=True) or {}
    set_num = payload.get("set")
    pairs = payload.get("pairs", [])
    if set_num is None or not pairs:
        return jsonify({"ok": True, "no_op": True})
    old_to_new = {int(p["old"]): int(p["new"]) for p in pairs}
    data = _load_highlights()
    for coll in data.get("collections", []):
        # Snapshot to avoid in-place re-mapping
        snapshots = [(r.get("set"), r.get("rally_in_set")) for r in coll.get("rallies", [])]
        for i, r in enumerate(coll.get("rallies", [])):
            old_set, old_num = snapshots[i]
            if old_set == set_num and old_num in old_to_new:
                r["rally_in_set"] = old_to_new[old_num]
    _save_highlights(data)
    return jsonify({"ok": True})


@app.route("/api/highlights/<cid>/rallies/<int:set_num>/<int:rally_num>", methods=["PATCH"])
def api_highlights_update_rally(cid, set_num, rally_num):
    payload = request.get_json(silent=True) or {}
    data = _load_highlights()
    coll = _find_collection(data, cid)
    if not coll:
        abort(404)
    for r in coll["rallies"]:
        if r.get("set") == set_num and r.get("rally_in_set") == rally_num:
            if "notes" in payload:
                r["notes"] = payload["notes"]
            break
    _save_highlights(data)
    return jsonify({"ok": True})


# ── Scorebug PNG renderer (for burn-in export) ──────────────────────────────

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


def _find_font(size: int, bold: bool = True) -> "ImageFont.FreeTypeFont":
    candidates = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    ) + [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _scorebug_score_at(events: list[dict], target_t: float) -> tuple[Optional[int], int, int]:
    """Return (set_num, home, away) AT time target_t."""
    sorted_e = sorted(events, key=lambda e: e.get("t_s", 0.0))
    cur_set = None
    for e in sorted_e:
        if e.get("t_s", 0.0) > target_t:
            break
        kind = e.get("event")
        if kind == "set_start": cur_set = e.get("set")
        elif kind == "set_end": cur_set = None
    home, away = 0, 0
    if cur_set is not None:
        for e in sorted_e:
            if e.get("set") != cur_set: continue
            if e.get("t_s", 0.0) > target_t: continue
            if e.get("event") == "rally_end":
                if e.get("score_by") == "home": home += 1
                elif e.get("score_by") == "away": away += 1
    return cur_set, home, away


def _render_scorebug_png(home_name: str, away_name: str,
                         home_score: int, away_score: int,
                         set_num: Optional[int], out_path: Path) -> None:
    """Render the broadcast-style scorebug to a PNG."""
    W, H = 420, 78
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Backdrop
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=6,
                         fill=(8, 9, 12, 235),
                         outline=(255, 255, 255, 36), width=1)

    # Set strip (bottom)
    set_h = 20
    d.rectangle((1, H - set_h, W - 1, H - 1), fill=(20, 18, 4, 200))

    # Fonts
    f_name = _find_font(16)
    f_score = _find_font(34)
    f_set = _find_font(12)
    f_dash = _find_font(28)

    # Helper to measure text width
    def tw(text, font):
        bbox = d.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    body_y_label = 8
    body_y_score = 22

    home_name = (home_name or "HOME").upper()[:14]
    away_name = (away_name or "AWAY").upper()[:14]

    # Left side: HOME label + score
    d.text((22, body_y_label), home_name, fill=(93, 216, 112, 255), font=f_name)
    d.text((22, body_y_score), str(home_score), fill=(93, 216, 112, 255), font=f_score)

    # Center dash
    dash = "—"
    d.text((W // 2 - tw(dash, f_dash) // 2, body_y_score + 4),
           dash, fill=(110, 113, 124, 255), font=f_dash)

    # Right side: AWAY label + score
    aw_label_w = tw(away_name, f_name)
    aw_score_w = tw(str(away_score), f_score)
    d.text((W - 22 - aw_label_w, body_y_label),
           away_name, fill=(255, 79, 74, 255), font=f_name)
    d.text((W - 22 - aw_score_w, body_y_score),
           str(away_score), fill=(255, 79, 74, 255), font=f_score)

    # Set strip text
    set_text = f"SET {set_num}" if set_num is not None else "PRE-SET"
    sw = tw(set_text, f_set)
    d.text(((W - sw) // 2, H - set_h + 3),
           set_text, fill=(255, 176, 0, 230), font=f_set)

    img.save(str(out_path), "PNG")


# ── Highlight reel export ───────────────────────────────────────────────────

_NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


# ── Multi-collection Highlights storage ─────────────────────────────────────
import unicodedata

def _slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s or "untitled"

def _unique_slug(name: str, existing_ids: list[str]) -> str:
    base = _slugify(name)
    if base not in existing_ids:
        return base
    n = 2
    while f"{base}_{n}" in existing_ids:
        n += 1
    return f"{base}_{n}"

def _highlights_path() -> Path:
    return STATE.labels_dir / f"highlights_{STATE.game_key}.json"

def _load_highlights() -> dict:
    p = _highlights_path()
    if p.exists():
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict) and "collections" in data:
                return data
        except Exception:
            log.exception("Failed to read highlights")
    # Migrate any rally_start with starred=true into a default collection
    events = _load_events()
    starred = [
        {"set": e.get("set"), "rally_in_set": e.get("rally_in_set"),
         "notes": e.get("notes", ""),
         "added_at": datetime.now().isoformat(timespec="seconds")}
        for e in events
        if e.get("event") == "rally_start" and e.get("starred")
        and e.get("set") is not None and e.get("rally_in_set") is not None
    ]
    data = {"collections": []}
    if starred:
        data["collections"].append({
            "id": "highlights",
            "name": "Highlights",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "rallies": starred,
        })
    _save_highlights(data)
    return data

def _save_highlights(data: dict) -> None:
    p = _highlights_path()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)

def _find_collection(data: dict, cid: str) -> Optional[dict]:
    for c in data.get("collections", []):
        if c.get("id") == cid:
            return c
    return None

def _reel_path(name: Optional[str] = None) -> Path:
    out_dir = STATE.video_path.parent / "highlight_reels"
    out_dir.mkdir(parents=True, exist_ok=True)
    if name:
        slug = _NAME_SAFE.sub("_", name.strip()).strip("._-")
        if slug:
            return out_dir / f"{slug}.mp4"
    return out_dir / f"reel_{STATE.game_key}.mp4"


@app.route("/api/export-reel", methods=["POST"])
def api_export_reel():
    """Concatenate rallies in a highlight collection into a single MP4 reel.

    Body params:
      name           — output filename slug
      collection_id  — which highlight collection's rallies to export
      with_scorebug  — bool, burn a per-rally scorebug overlay onto each clip
      team_home, team_away — display names for the scorebug
    """
    payload = request.get_json(silent=True) or {}
    reel_name = payload.get("name", "")
    collection_id = payload.get("collection_id", "")
    with_scorebug = bool(payload.get("with_scorebug", False))
    team_home = payload.get("team_home", "HOME")
    team_away = payload.get("team_away", "AWAY")

    events = _load_events()
    # Map (set, rally_in_set) → (start_t, end_t)
    sorted_events = sorted(events, key=lambda e: e.get("t_s", 0.0))
    open_by_key: dict = {}
    rally_times: dict = {}  # (set, rally_in_set) -> (start, end)
    for e in sorted_events:
        kind = e.get("event")
        key = (e.get("set"), e.get("rally_in_set"))
        if kind == "rally_start":
            open_by_key[key] = e
        elif kind == "rally_end" and key in open_by_key:
            start = open_by_key.pop(key)
            rally_times[key] = (float(start["t_s"]), float(e["t_s"]))

    segments: list[dict] = []

    if collection_id:
        # Use rallies from the named highlight collection
        hl = _load_highlights()
        coll = _find_collection(hl, collection_id)
        if not coll:
            return jsonify({"error": f"Collection '{collection_id}' not found"}), 404
        for r in coll["rallies"]:
            key = (r.get("set"), r.get("rally_in_set"))
            if key in rally_times:
                start, end = rally_times[key]
                segments.append({
                    "start": start, "end": end,
                    "set": key[0], "rally": key[1],
                    "notes": r.get("notes", ""),
                })
        # Sort by chronological order within the game
        segments.sort(key=lambda s: s["start"])
        if not reel_name:
            reel_name = coll.get("name") or coll.get("id")
    else:
        # Legacy fallback: rally_start.starred
        for key, (start, end) in rally_times.items():
            # Find the rally_start event
            rs = next((e for e in sorted_events
                       if e.get("event") == "rally_start"
                       and e.get("set") == key[0]
                       and e.get("rally_in_set") == key[1]), None)
            if rs and rs.get("starred"):
                segments.append({
                    "start": start, "end": end,
                    "set": key[0], "rally": key[1],
                    "notes": rs.get("notes", ""),
                })
        segments.sort(key=lambda s: s["start"])

    if not segments:
        return jsonify({"error": "No rallies in this collection."}), 400

    if with_scorebug and not _PIL_OK:
        return jsonify({"error": "Pillow not installed — can't render scorebug. Run: pip install Pillow"}), 500

    out_path = _reel_path(reel_name)
    tmp_dir = Path(tempfile.mkdtemp(prefix="vol_reel_"))
    list_file = tmp_dir / "list.txt"
    clips: list[Path] = []

    try:
        for i, seg in enumerate(segments):
            clip = tmp_dir / f"clip_{i:04d}.mp4"
            if with_scorebug:
                # Score AT start of this rally (frozen for the whole clip)
                set_num, home, away = _scorebug_score_at(events, seg["start"])
                png = tmp_dir / f"sb_{i:04d}.png"
                _render_scorebug_png(team_home, team_away, home, away, set_num, png)
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(seg["start"]),
                    "-i", str(STATE.video_path),
                    "-i", str(png),
                    "-t", str(max(0.05, seg["end"] - seg["start"])),
                    "-filter_complex",
                    "[1:v]scale=420:-1[sb];[0:v][sb]overlay=40:main_h-110[v]",
                    "-map", "[v]", "-map", "0:a?",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                    "-c:a", "aac", "-b:a", "160k",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-avoid_negative_ts", "make_zero",
                    str(clip),
                ]
            else:
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(seg["start"]),
                    "-i", str(STATE.video_path),
                    "-t", str(max(0.05, seg["end"] - seg["start"])),
                    "-c", "copy",
                    "-avoid_negative_ts", "make_zero",
                    str(clip),
                ]
            subprocess.run(cmd, capture_output=True, check=True)
            clips.append(clip)

        list_file.write_text("\n".join(f"file '{c}'" for c in clips))
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
        ]
        if with_scorebug:
            # All re-encoded clips share codec settings; stream copy is safe
            concat_cmd += ["-c", "copy"]
        else:
            concat_cmd += ["-c", "copy"]
        concat_cmd += [str(out_path)]
        subprocess.run(concat_cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": "ffmpeg failed",
            "stderr": (e.stderr or b"").decode("utf-8", "replace")[-800:],
        }), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    total_dur = sum(s["end"] - s["start"] for s in segments)
    return jsonify({
        "ok": True,
        "path": str(out_path),
        "filename": out_path.name,
        "segments": len(segments),
        "total_duration_s": round(total_dur, 2),
        "size_mb": round(out_path.stat().st_size / 1e6, 1),
        "with_scorebug": with_scorebug,
    })


@app.route("/api/reel-download")
def api_reel_download():
    name = request.args.get("name", "")
    path = _reel_path(name)
    if not path.exists():
        abort(404)
    return send_file(str(path), as_attachment=True, download_name=path.name)


# ── Whistle-based rally autodetection ──────────────────────────────────────

_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END_RE   = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


def _detect_whistles(start_t: float, end_t: float,
                     threshold_db: float, min_silence_s: float,
                     hp_hz: int, lp_hz: int) -> list[dict]:
    """Run ffmpeg silencedetect on band-passed audio, return whistle bursts."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-ss", str(start_t),
        "-t", str(max(0.1, end_t - start_t)),
        "-i", str(STATE.video_path),
        "-af", f"highpass=f={hp_hz},lowpass=f={lp_hz},"
               f"silencedetect=n={threshold_db}dB:d={min_silence_s}",
        "-f", "null", "-",
    ]
    log.info(f"whistle detect: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = res.stderr

    silence_starts: list[float] = []
    silence_ends:   list[float] = []
    for m in _SILENCE_START_RE.finditer(out):
        silence_starts.append(float(m.group(1)) + start_t)
    for m in _SILENCE_END_RE.finditer(out):
        silence_ends.append(float(m.group(1)) + start_t)

    # The non-silent windows are: [silence_end_i, silence_start_{i+1}]
    # Audio at the very start before the first silence_start is also non-silent.
    whistles: list[dict] = []
    edges: list[tuple[float, str]] = []
    for s in silence_starts: edges.append((s, "silence_start"))
    for s in silence_ends:   edges.append((s, "silence_end"))
    edges.sort()

    open_t = start_t  # audio starts loud by default
    in_silence = False
    for t, kind in edges:
        if kind == "silence_start" and not in_silence:
            # End of a non-silent window — record it
            dur = t - open_t
            if 0.05 < dur < 3.0:
                whistles.append({"t_start": round(open_t, 3),
                                 "t_end":   round(t, 3),
                                 "duration": round(dur, 3)})
            in_silence = True
        elif kind == "silence_end" and in_silence:
            open_t = t
            in_silence = False
    # Tail (if audio ends loud)
    if not in_silence:
        dur = end_t - open_t
        if 0.05 < dur < 3.0:
            whistles.append({"t_start": round(open_t, 3),
                             "t_end":   round(end_t, 3),
                             "duration": round(dur, 3)})

    # De-dupe whistles within ~0.4s of each other (single continuous blast)
    merged: list[dict] = []
    for w in whistles:
        if merged and w["t_start"] - merged[-1]["t_end"] < 0.4:
            merged[-1]["t_end"] = w["t_end"]
            merged[-1]["duration"] = round(merged[-1]["t_end"] - merged[-1]["t_start"], 3)
        else:
            merged.append(w)
    return merged


def _pair_whistles_into_rallies(whistles: list[dict],
                                 min_rally_s: float,
                                 max_rally_s: float) -> list[dict]:
    """Pair consecutive whistles into (rally_start_t, rally_end_t)."""
    rallies: list[dict] = []
    i = 0
    while i + 1 < len(whistles):
        a, b = whistles[i], whistles[i + 1]
        gap = b["t_start"] - a["t_end"]
        if min_rally_s <= gap <= max_rally_s:
            rallies.append({
                "start_t": a["t_start"],
                "end_t":   b["t_start"],
                "duration": round(b["t_start"] - a["t_start"], 3),
            })
            i += 2
        else:
            # Skip the loner whistle; try pairing the next two
            i += 1
    return rallies


# ── Async job runner for whistle autodetect ────────────────────────────────

_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_EVENTS_LOCK = threading.Lock()   # serialize concurrent writes to event_log_*.json
_DURATION_PROGRESS_RE = re.compile(r"time=(\d+):(\d+):([\d.]+)")


def _job_update(job_id: str, **kwargs):
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(kwargs)


def _highlights_renumber_internal(set_num: int, pairs: list[dict]) -> None:
    """Apply a renumber map { set, [{old,new}, …] } to all highlight collections."""
    if not pairs:
        return
    old_to_new = {int(p["old"]): int(p["new"]) for p in pairs}
    data = _load_highlights()
    for coll in data.get("collections", []):
        snapshots = [(r.get("set"), r.get("rally_in_set")) for r in coll.get("rallies", [])]
        for i, r in enumerate(coll.get("rallies", [])):
            old_set, old_num = snapshots[i]
            if old_set == set_num and old_num in old_to_new:
                r["rally_in_set"] = old_to_new[old_num]
    _save_highlights(data)


def _renumber_rallies_in_set_inplace(events: list[dict], set_num: int) -> list[dict]:
    """Renumber rally_in_set values for a set to 1..N by chronological order
    (in place). Returns the [{old, new}, …] map of changes."""
    starts = sorted(
        [e for e in events if e.get("event") == "rally_start" and e.get("set") == set_num],
        key=lambda e: e.get("t_s", 0.0),
    )
    if not starts:
        return []
    end_by_old = {}
    for e in events:
        if (e.get("event") == "rally_end" and e.get("set") == set_num
                and e.get("rally_in_set") is not None):
            end_by_old[e["rally_in_set"]] = e
    new_nums = {id(rs): idx + 1 for idx, rs in enumerate(starts)}
    pairs = []
    for rs in starts:
        old_num = rs.get("rally_in_set")
        new_num = new_nums[id(rs)]
        if old_num != new_num:
            pairs.append({"old": old_num, "new": new_num})
            end = end_by_old.get(old_num)
            rs["rally_in_set"] = new_num
            if end is not None:
                end["rally_in_set"] = new_num
    return pairs


def _insert_streaming_rally(start_t: float, end_t: float,
                            target_set: Optional[int]) -> Optional[dict]:
    """Append a pending rally pair, renumber the set, and update highlight refs.
    Returns the {set, rally_in_set} the new rally landed at after renumbering,
    or None if it overlapped an existing event."""
    with _EVENTS_LOCK:
        events = _load_events()
        sorted_e = sorted(events, key=lambda e: e.get("t_s", 0.0))
        for e in sorted_e:
            if e.get("event") in ("rally_start", "rally_end"):
                t = float(e.get("t_s", 0.0))
                if start_t - 1.0 <= t <= end_t + 1.0:
                    return None  # overlap
        if target_set is not None:
            set_num = int(target_set)
        else:
            cur_set = None
            for e in sorted_e:
                if e.get("event") == "set_start": cur_set = e.get("set")
                elif e.get("event") == "set_end": cur_set = None
            set_num = cur_set or 1
        # Tentative number = highest in set + 1; will be normalized by renumber
        next_rally = 1
        for e in events:
            if e.get("event") == "rally_start" and e.get("set") == set_num:
                next_rally = max(next_rally, int(e.get("rally_in_set") or 0) + 1)
        new_start = {
            "event": "rally_start", "t_s": round(start_t, 3),
            "set": set_num, "rally_in_set": next_rally,
            "score_by": None, "auto": True, "confirmed": False,
        }
        new_end = {
            "event": "rally_end", "t_s": round(end_t, 3),
            "set": set_num, "rally_in_set": next_rally,
            "score_by": None, "auto": True, "confirmed": False,
        }
        events.append(new_start)
        events.append(new_end)
        # Renumber chronologically, propagate to highlight collections
        pairs = _renumber_rallies_in_set_inplace(events, set_num)
        events.sort(key=lambda e: e.get("t_s", 0.0))
        for i, e in enumerate(events): e["seq"] = i
        _save_events(events)
        if pairs:
            _highlights_renumber_internal(set_num, pairs)
        return {"set": set_num, "rally_in_set": new_start["rally_in_set"]}


def _save_whistle_event(start_t: float, end_t: float) -> None:
    """Save a detected whistle burst into the event log."""
    with _EVENTS_LOCK:
        events = _load_events()
        # Skip if we already have a whistle within 0.3s of this one (re-detect)
        for e in events:
            if e.get("event") == "whistle":
                if abs(float(e.get("t_s", 0)) - start_t) < 0.3:
                    return
        events.append({
            "event": "whistle",
            "t_s": round(start_t, 3),
            "t_end": round(end_t, 3),
            "duration": round(end_t - start_t, 3),
            "auto": True,
        })
        events.sort(key=lambda e: e.get("t_s", 0.0))
        for i, e in enumerate(events): e["seq"] = i
        _save_events(events)


def _run_whistle_job(job_id: str, payload: dict):
    proc = None
    try:
        start_t = float(payload.get("start_t", 0))
        end_t   = float(payload.get("end_t", STATE.duration_s))
        threshold_db = float(payload.get("threshold_db", -28))
        min_silence_s = float(payload.get("min_silence_s", 0.15))
        hp_hz = int(payload.get("hp_hz", 2000))
        lp_hz = int(payload.get("lp_hz", 4500))
        min_rally_s = float(payload.get("min_rally_s", 4))
        max_rally_s = float(payload.get("max_rally_s", 90))
        target_set = payload.get("set_num")
        total_secs = max(0.1, end_t - start_t)
        _job_update(job_id, progress="Starting ffmpeg…", pct=0, rallies_count=0)

        cmd = [
            "ffmpeg", "-hide_banner", "-nostats",
            "-ss", str(start_t),
            "-t", str(total_secs),
            "-i", str(STATE.video_path),
            "-vn",                   # skip video decode — we only need audio
            "-map", "0:a:0",         # take the first audio stream
            "-af", f"highpass=f={hp_hz},lowpass=f={lp_hz},"
                   f"silencedetect=n={threshold_db}dB:d={min_silence_s}",
            "-progress", "pipe:2",
            "-f", "null", "-",
        ]
        log.info(f"whistle job {job_id}: starting ffmpeg")
        proc = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["_proc"] = proc

        # Streaming state machine: each non-silent window is a whistle, pair
        # adjacent whistles within [min_rally_s, max_rally_s] into rallies.
        in_loud = False
        whistle_start: Optional[float] = None
        last_whistle_end: Optional[float] = None
        prev_whistle: Optional[dict] = None      # candidate rally_start whistle
        inserted = 0
        scanned_sec = 0.0

        def process_whistle(w_start: float, w_end: float):
            nonlocal prev_whistle, last_whistle_end, inserted
            duration = w_end - w_start
            if not (0.05 < duration < 3.0):
                return
            # Merge with previous if extremely close (still part of same blast)
            if last_whistle_end is not None and (w_start - last_whistle_end) < 0.4:
                last_whistle_end = w_end
                return
            this_w = {"t_start": w_start, "t_end": w_end}
            last_whistle_end = w_end
            # Save as a whistle event so the user can see what was detected
            _save_whistle_event(w_start, w_end)
            if prev_whistle is None:
                prev_whistle = this_w
                return
            gap = this_w["t_start"] - prev_whistle["t_end"]
            if min_rally_s <= gap <= max_rally_s:
                rally = _insert_streaming_rally(
                    prev_whistle["t_start"], this_w["t_start"], target_set
                )
                if rally is not None:
                    inserted += 1
                    _job_update(
                        job_id,
                        rallies_count=inserted,
                        last_rally_t=this_w["t_start"],
                        last_rally_label=f"Set {rally['set']} · Rally {rally['rally_in_set']}",
                    )
                prev_whistle = None
            elif gap > max_rally_s:
                # Too long, treat this as a new candidate start
                prev_whistle = this_w
            # else: gap < min, swallow this whistle (likely false positive between)

        for line in proc.stderr:
            # Cancel check
            with _JOBS_LOCK:
                if _JOBS.get(job_id, {}).get("cancelled"):
                    break

            m = _SILENCE_START_RE.search(line)
            if m and in_loud and whistle_start is not None:
                w_end = float(m.group(1)) + start_t
                process_whistle(whistle_start, w_end)
                in_loud = False
                whistle_start = None
            m = _SILENCE_END_RE.search(line)
            if m and not in_loud:
                whistle_start = float(m.group(1)) + start_t
                in_loud = True

            m = _DURATION_PROGRESS_RE.search(line)
            if m:
                hh = int(m.group(1)); mm = int(m.group(2)); ss = float(m.group(3))
                scanned_sec = hh * 3600 + mm * 60 + ss
                pct = min(100, int((scanned_sec / total_secs) * 100))
                _job_update(
                    job_id, pct=pct,
                    progress=f"Scanned {int(scanned_sec)}s / {int(total_secs)}s · {inserted} found",
                )

        proc.wait()

        # Was this cancelled?
        with _JOBS_LOCK:
            cancelled = _JOBS.get(job_id, {}).get("cancelled", False)

        if cancelled:
            _job_update(
                job_id, status="cancelled",
                progress=f"Cancelled · {inserted} rallies kept",
                result={"rallies_inserted": inserted},
            )
        else:
            _job_update(
                job_id, status="done", pct=100,
                progress=f"Done · {inserted} rallies",
                result={"rallies_inserted": inserted},
            )
    except Exception as e:
        log.exception("whistle job failed")
        _job_update(job_id, status="error", error=str(e))
    finally:
        # Drop the proc reference
        with _JOBS_LOCK:
            if job_id in _JOBS and "_proc" in _JOBS[job_id]:
                del _JOBS[job_id]["_proc"]


@app.route("/api/auto-rallies/start", methods=["POST"])
def api_auto_rallies_start():
    payload = request.get_json(silent=True) or {}
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "id": job_id,
            "status": "running",
            "pct": 0,
            "progress": "Queued",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "set_num": payload.get("set_num"),
            "start_t": payload.get("start_t"),
            "end_t": payload.get("end_t"),
            "game_key": STATE.game_key,
        }
    threading.Thread(target=_run_whistle_job, args=(job_id, payload), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/auto-rallies/jobs")
def api_auto_rallies_jobs():
    """List all auto-detect jobs that are still running for this game.
    Used by the client to resume polling after a page refresh."""
    with _JOBS_LOCK:
        running = []
        for j in _JOBS.values():
            if j.get("status") != "running": continue
            if j.get("game_key") and j.get("game_key") != STATE.game_key: continue
            running.append({k: v for k, v in j.items() if not k.startswith("_")})
    return jsonify({"jobs": running})


@app.route("/api/auto-rallies/status/<job_id>")
def api_auto_rallies_status(job_id):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            abort(404)
        # Strip non-serializable internals (like the Popen handle)
        public = {k: v for k, v in job.items() if not k.startswith("_")}
    return jsonify(public)


@app.route("/api/auto-rallies/cancel/<job_id>", methods=["POST"])
def api_auto_rallies_cancel(job_id):
    """Mark a job as cancelled and terminate its ffmpeg subprocess."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            abort(404)
        job["cancelled"] = True
        proc = job.get("_proc")
    if proc:
        try:
            proc.terminate()
        except Exception:
            pass
    return jsonify({"ok": True})


@app.route("/api/auto-rallies/apply", methods=["POST"])
def api_auto_rallies_apply():
    """Insert previously-detected rallies into the event log."""
    payload = request.get_json(silent=True) or {}
    rallies = payload.get("rallies", [])
    if not rallies:
        return jsonify({"ok": True, "inserted": 0})

    events = _load_events()
    existing = sorted(events, key=lambda e: e.get("t_s", 0.0))

    def overlaps(a_start, a_end):
        for e in existing:
            if e.get("event") not in ("rally_start", "rally_end"): continue
            t = float(e.get("t_s", 0.0))
            if a_start - 1.0 <= t <= a_end + 1.0:
                return True
        return False

    cur_set = None
    for e in existing:
        if e.get("event") == "set_start": cur_set = e.get("set")
        elif e.get("event") == "set_end": cur_set = None
    set_num = int(payload.get("set_num") or cur_set or 1)

    next_rally = 1
    for e in existing:
        if e.get("event") == "rally_start" and e.get("set") == set_num:
            next_rally = max(next_rally, int(e.get("rally_in_set") or 0) + 1)

    inserted = 0
    for r in rallies:
        if overlaps(r["start_t"], r["end_t"]):
            continue
        events.append({
            "event": "rally_start", "t_s": r["start_t"],
            "set": set_num, "rally_in_set": next_rally,
            "score_by": None, "auto": True, "confirmed": False,
        })
        events.append({
            "event": "rally_end", "t_s": r["end_t"],
            "set": set_num, "rally_in_set": next_rally,
            "score_by": None, "auto": True, "confirmed": False,
        })
        next_rally += 1
        inserted += 1

    events.sort(key=lambda e: e.get("t_s", 0.0))
    for i, e in enumerate(events): e["seq"] = i
    _save_events(events)
    return jsonify({"ok": True, "inserted": inserted, "skipped": len(rallies) - inserted})


def _events_path() -> Path:
    return STATE.labels_dir / f"event_log_{STATE.game_key}.json"


def _load_events() -> list[dict]:
    p = _events_path()
    if p.exists():
        try:
            return json.loads(p.read_text()).get("events", [])
        except Exception:
            log.exception("Failed to read events file; starting empty")
    return []


def _save_events(events: list[dict]) -> None:
    p = _events_path()
    payload = {
        "game": STATE.game_key,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "event_count": len(events),
        "events": events,
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(p)


@app.route("/api/events", methods=["GET", "PUT"])
def api_events():
    if request.method == "GET":
        return jsonify({
            "game": STATE.game_key,
            "video_name": STATE.video_path.name,
            "duration_s": STATE.duration_s,
            "events": _load_events(),
            "path": str(_events_path()),
        })
    payload = request.get_json(force=True) or {}
    events = payload.get("events", [])
    # Sort by t_s ascending and re-number seq
    events = sorted(events, key=lambda e: e.get("t_s", 0.0))
    for i, e in enumerate(events):
        e["seq"] = i
        if "t" not in e or not e["t"]:
            ts = float(e.get("t_s", 0.0))
            e["t"] = f"{int(ts // 3600):02d}:{int((ts % 3600) // 60):02d}:{int(ts % 60):02d}"
        e.setdefault("video_file", STATE.video_path.name)
    _save_events(events)
    return jsonify({"ok": True, "count": len(events), "path": str(_events_path())})


_RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)")


@app.route("/video")
def video():
    """Stream the source MP4 with HTTP Range support."""
    path = STATE.video_path
    file_size = os.path.getsize(path)
    range_header = request.headers.get("Range", "")
    if not range_header:
        return Response(
            _iter_file(path, 0, file_size - 1),
            mimetype="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
            direct_passthrough=True,
        )
    m = _RANGE_RE.search(range_header)
    if not m:
        abort(416)
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else file_size - 1
    end = min(end, file_size - 1)
    if start > end:
        abort(416)
    length = end - start + 1
    rv = Response(
        _iter_file(path, start, end),
        status=206,
        mimetype="video/mp4",
        direct_passthrough=True,
    )
    rv.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    rv.headers["Accept-Ranges"] = "bytes"
    rv.headers["Content-Length"] = str(length)
    return rv


def _iter_file(path: Path, start: int, end: int, chunk_size: int = 1024 * 256):
    remaining = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        while remaining > 0:
            data = f.read(min(chunk_size, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Golden-data clip labeller (Flask)")
    p.add_argument("--video", required=True, type=Path,
                   help="Path to the stitched game MP4")
    p.add_argument("--detector-output", type=Path, default=None,
                   help="Optional path to event_log_*.json / point_timeline_*.json / segments JSON")
    p.add_argument("--player", type=int, default=None,
                   help="Target jersey number (display-only)")
    p.add_argument("--n-random", type=int, default=30,
                   help="Number of random sample windows to add (default 30)")
    p.add_argument("--window-pad", type=float, default=1.5,
                   help="Seconds of padding around each detector span (default 1.5)")
    p.add_argument("--window-size", type=float, default=6.0,
                   help="Length of each random sample window in seconds (default 6.0)")
    p.add_argument("--labels-root", type=Path, default=HERE / "labels",
                   help="Where to write manifest/labels (default ./labels)")
    p.add_argument("--rebuild", action="store_true",
                   help="Discard any existing manifest.json and rebuild")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.video.exists():
        log.error(f"Video not found: {args.video}")
        sys.exit(2)

    STATE.video_path = args.video.resolve()
    STATE.game_key = derive_game_key(STATE.video_path)
    STATE.player = args.player
    STATE.duration_s = ffprobe_duration(STATE.video_path)
    STATE.labels_dir = args.labels_root / STATE.game_key
    STATE.labels_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Video       : {STATE.video_path}")
    log.info(f"Game key    : {STATE.game_key}")
    log.info(f"Duration    : {STATE.duration_s:.1f}s ({STATE.duration_s/60:.1f} min)")
    log.info(f"Labels dir  : {STATE.labels_dir}")

    STATE.manifest = load_or_build_manifest(args)
    STATE.labels = load_labels()
    log.info(f"Clips       : {len(STATE.manifest)}  |  labeled: {len(STATE.labels)}")

    print()
    print("─" * 60)
    print(f"  Open in your browser:  http://localhost:{args.port}/")
    print("─" * 60)
    print()

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()

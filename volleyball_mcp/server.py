#!/usr/bin/env python3
"""
Volleyball rally tools — MCP server (stdio).

Runs inside WSL next to the footage and GPU. Heavy work is done by detached
jobs (rally_lab/pipeline.py, dji_stitch.py) so tool calls return in seconds
with short summaries; poll long jobs with job_status.

Launch (what the plugin's .mcp.json does):
    wsl -d Ubuntu -- bash -lc "cd <repo>/volleyball_mcp && ~/.local/bin/uv run --quiet server.py"
"""

import json
import os
import re
import signal
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from mcp.server.fastmcp import FastMCP

REPO = Path(__file__).resolve().parent.parent
VIDEO_DIR = Path(os.environ.get("VB_VIDEO_DIR", "/mnt/d/volleyball"))
WORK = Path(os.environ.get("VB_WORK_DIR", "/mnt/d/volleyball/work"))
JOBS = WORK / "jobs"
LABELER_PORT = int(os.environ.get("VB_LABELER_PORT", "8765"))
SYS_PY = "/usr/bin/python3"     # torch / ultralytics / opencv live here
DJI_RE = re.compile(r"DJI_(\d{8})(\d{6})_(\d+)", re.IGNORECASE)
KEY_RE = re.compile(r"^\d{8}_game\d{2}$")

mcp = FastMCP("volleyball")


# ── helpers ──────────────────────────────────────────────────────────────────

def _key(game: str) -> str:
    game = game.strip().removeprefix("stitched_").removesuffix(".mp4")
    if not KEY_RE.match(game):
        raise ValueError(f"game must look like 20261004_game02, got {game!r}")
    return game


def _env():
    return dict(os.environ, PATH=f"{Path.home()}/.local/bin:" + os.environ.get("PATH", ""),
                MALLOC_MMAP_THRESHOLD_="134217728", MALLOC_TRIM_THRESHOLD_="-1")


def _start_job(kind: str, game: str, cmd: list) -> dict:
    JOBS.mkdir(parents=True, exist_ok=True)
    jid = f"{kind}-{game}-{uuid.uuid4().hex[:6]}"
    log = JOBS / f"{jid}.log"
    meta = {"id": jid, "kind": kind, "game": game, "cmd": cmd,
            "started": datetime.now().isoformat(timespec="seconds")}
    with open(log, "w") as fh:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=_env(),
                             start_new_session=True)
    meta["pid"] = p.pid
    json.dump(meta, open(JOBS / f"{jid}.json", "w"))
    return {"job_id": jid, "status": "started", "hint": "poll job_status(job_id); detect takes ~1 h for a new game, seconds if cached"}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # zombie?
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def _ffprobe_duration(p: Path) -> float:
    cache_f = WORK / "durations.json"
    cache = json.load(open(cache_f)) if cache_f.exists() else {}
    k = f"{p}:{p.stat().st_size}"
    if k not in cache:
        r = subprocess.run(["bash", "-lc", f"ffprobe -v error -show_entries format=duration -of csv=p=0 '{p}'"],
                           capture_output=True, text=True, env=_env())
        try:
            cache[k] = float(r.stdout.strip())
        except ValueError:
            cache[k] = 0.0
        WORK.mkdir(parents=True, exist_ok=True)
        json.dump(cache, open(cache_f, "w"))
    return cache[k]


def _hms(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _event_stats(path: Path):
    if not path.exists():
        return None
    ev = json.load(open(path)).get("events", [])
    ends = [e for e in ev if e.get("event") == "rally_end"]
    return {"rallies": len(ends), "confirmed": sum(1 for e in ends if e.get("confirmed", not e.get("auto"))),
            "sets": len({e.get("set") for e in ends})}


# ── tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
def list_games() -> dict:
    """Raw DJI recordings grouped into games, stitched games, and how far each game has been analysed."""
    raw = []
    for f in sorted(VIDEO_DIR.glob("DJI_*.MP4")) + sorted(VIDEO_DIR.glob("DJI_*.mp4")):
        m = DJI_RE.match(f.name)
        if m:
            st = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            raw.append((st, f))
    groups, cur = [], None
    for st, f in sorted(raw):
        end = st + timedelta(seconds=_ffprobe_duration(f))
        if cur is None or st.date() != cur["start"].date() or (st - cur["end"]).total_seconds() > 120:
            cur = {"start": st, "end": end, "files": []}
            groups.append(cur)
        cur["files"].append(f.name)
        cur["end"] = max(cur["end"], end)
    per_date = {}
    recordings = []
    for gr in groups:
        d = gr["start"].strftime("%Y%m%d")
        per_date[d] = per_date.get(d, 0) + 1
        key = f"{d}_game{per_date[d]:02d}"
        recordings.append({"game": key, "start": gr["start"].strftime("%Y-%m-%d %H:%M"),
                           "length": _hms((gr["end"] - gr["start"]).total_seconds()),
                           "files": len(gr["files"]), "stitched": (VIDEO_DIR / f"stitched_{key}.mp4").exists()})
    stitched = []
    for p in sorted(VIDEO_DIR.glob("stitched_*.mp4")):
        key = p.stem.removeprefix("stitched_")
        wd = WORK / key
        lab = REPO / "labels" / key
        stitched.append({
            "game": key, "length": _hms(_ffprobe_duration(p)),
            "analysed": {"proxy": (wd / "proxy.mp4").exists(), "ball": (wd / "proxy" / "tracks").exists(),
                         "players": (wd / "players.npz").exists(), "livedead": (wd / "livedead.npz").exists()},
            "auto_events": _event_stats(lab / f"auto_events_{key}.json"),
            "event_log": _event_stats(lab / f"event_log_{key}.json"),
        })
    return {"raw_recordings": recordings, "stitched_games": stitched,
            "note": "raw recordings are grouped by gaps > 2 min between files; check they are real games before stitching"}


@mcp.tool()
def stitch_games(date: str) -> dict:
    """Stitch one day's raw DJI files into one MP4 per game (lossless, ~6 min per game). date = YYYYMMDD."""
    if not re.fullmatch(r"\d{8}", date):
        raise ValueError("date must be YYYYMMDD")
    return _start_job("stitch", date, [SYS_PY, "dji_stitch.py", "-i", str(VIDEO_DIR), "--match-date", date])


@mcp.tool()
def detect_rallies(game: str, set_scores: str = "") -> dict:
    """Find rallies, sets and point winners for a stitched game. Runs as a background job.

    game: e.g. 20261004_game02.
    set_scores: final score of each set as home-away (home = Tyrants), e.g. "19-25,25-18,15-10".
      Strongly recommended — fixing each set's rally count to its score is the biggest accuracy gain.
    Writes labels/<game>/auto_events_<game>.json (and event_log_<game>.json only if none exists yet).
    """
    key = _key(game)
    if set_scores and not re.fullmatch(r"\s*\d+-\d+(\s*,\s*\d+-\d+)*\s*", set_scores):
        raise ValueError("set_scores must look like 25-19,18-25,15-12")
    cmd = [SYS_PY, "rally_lab/pipeline.py", "detect", "--key", key]
    if set_scores:
        cmd += ["--set-scores", set_scores.replace(" ", "")]
    return _start_job("detect", key, cmd)


@mcp.tool()
def job_status(job_id: str = "") -> dict:
    """Progress of a background job (latest step and result). Empty job_id lists recent jobs."""
    if not job_id:
        metas = sorted(JOBS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]
        out = []
        for m in metas:
            meta = json.load(open(m))
            out.append({"job_id": meta["id"], "started": meta["started"],
                        "running": _alive(meta["pid"])})
        return {"jobs": out}
    meta_f = JOBS / f"{job_id}.json"
    if not meta_f.exists():
        raise ValueError(f"unknown job {job_id}")
    meta = json.load(open(meta_f))
    lines = (JOBS / f"{job_id}.log").read_text(errors="replace").splitlines()
    steps, result, error = {}, None, None
    for ln in lines:
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if isinstance(d, dict) and "step" in d:
            steps[d["step"]] = d["status"]
            if d["status"] == "failed":
                error = d.get("error")
            if d.get("summary"):
                result = d["summary"]
    running = _alive(meta["pid"])
    out = {"job_id": job_id, "kind": meta["kind"], "game": meta["game"], "started": meta["started"],
           "state": "running" if running else ("failed" if error else "finished"), "steps": steps}
    if result:
        out["result"] = result
    if error:
        out["error"] = error
    if not steps:
        out["log_tail"] = lines[-5:]
    return out


@mcp.tool()
def rally_summary(game: str, set_number: int = 0, limit: int = 60) -> dict:
    """Detected rallies for a game (start/end/winner), optionally one set. Prefers the auto-detected list."""
    key = _key(game)
    lab = REPO / "labels" / key
    src = lab / f"auto_events_{key}.json"
    if not src.exists():
        src = lab / f"event_log_{key}.json"
    if not src.exists():
        return {"game": key, "error": "no rally events yet — run detect_rallies"}
    ev = json.load(open(src))["events"]
    starts = {(e["set"], e["rally_in_set"]): e for e in ev if e["event"] == "rally_start"}
    rows = []
    for e in ev:
        if e["event"] == "rally_end" and (not set_number or e["set"] == set_number):
            s = starts.get((e["set"], e["rally_in_set"]), {})
            rows.append(f"S{e['set']} R{e['rally_in_set']:>2} {s.get('t', '?')}-{e['t']} "
                        f"{(e['t_s'] - s.get('t_s', e['t_s'])):4.1f}s won:{e.get('score_by') or '?'} "
                        f"{e.get('score_after', '')}")
    summary = WORK / key / "summary.json"
    return {"game": key, "source": src.name,
            "summary": json.load(open(summary)) if summary.exists() else None,
            "rallies": rows[:limit], "truncated": len(rows) > limit}


@mcp.tool()
def evaluate_game(game: str) -> dict:
    """Score the auto-detected rallies against the user's confirmed labels and/or scoreboard checkpoints."""
    key = _key(game)
    r = subprocess.run([SYS_PY, "rally_lab/pipeline.py", "evaluate", "--key", key], cwd=REPO,
                       capture_output=True, text=True, env=_env(), timeout=300)
    if r.returncode:
        return {"error": r.stderr[-600:]}
    return json.loads(r.stdout.strip().splitlines()[-1])


@mcp.tool()
def use_auto_events(game: str, force: bool = False) -> dict:
    """Copy the auto-detected rallies into the labeler's event log (backs up the old one).

    Refuses when the current event log already has confirmed rallies, unless force=True.
    """
    key = _key(game)
    lab = REPO / "labels" / key
    auto, log = lab / f"auto_events_{key}.json", lab / f"event_log_{key}.json"
    if not auto.exists():
        return {"error": "no auto events — run detect_rallies first"}
    st = _event_stats(log)
    if st and st["confirmed"] and not force:
        return {"error": f"event log has {st['confirmed']} confirmed rallies; pass force=True to replace it (a backup is kept)"}
    backup = None
    if log.exists():
        backup = log.with_name(f"{log.name}.bak.{datetime.now():%Y%m%d%H%M%S}")
        log.rename(backup)
    log.write_text(auto.read_text())
    return {"event_log": str(log), "backup": str(backup) if backup else None, **_event_stats(log)}


@mcp.tool()
def open_labeler(game: str) -> dict:
    """Start the labeling web app for a game (localhost only) and return its URL."""
    key = _key(game)
    video = VIDEO_DIR / f"stitched_{key}.mp4"
    if not video.exists():
        return {"error": f"{video.name} not found"}
    state_f = WORK / "labeler.json"
    if state_f.exists():
        st = json.load(open(state_f))
        if _alive(st["pid"]):
            if st["game"] == key:
                return {"url": st["url"], "status": "already running"}
            os.killpg(st["pid"], signal.SIGTERM)
            time.sleep(1)
    log = WORK / "labeler.log"
    with open(log, "w") as fh:
        p = subprocess.Popen([SYS_PY, "label_clips.py", "--video", str(video), "--host", "127.0.0.1",
                              "--port", str(LABELER_PORT)], cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                             env=_env(), start_new_session=True)
    url = f"http://localhost:{LABELER_PORT}/g/{key}"
    json.dump({"pid": p.pid, "game": key, "url": url}, open(state_f, "w"))
    for _ in range(40):
        time.sleep(0.5)
        if "Open in your browser" in log.read_text(errors="replace"):
            return {"url": url, "status": "started"}
        if p.poll() is not None:
            return {"error": "labeler exited", "log_tail": log.read_text(errors="replace")[-600:]}
    return {"url": url, "status": "starting (first launch builds clip manifest; may take a minute)"}


@mcp.tool()
def open_review(game: str = "") -> dict:
    """Start the rally review web app (watch each detected rally, fix start/end, game start/end) and return its URL.

    Saves go to labels/<game>/review_<game>.json and are exported to the event log.
    """
    key = _key(game) if game else ""
    port = int(os.environ.get("VB_REVIEW_PORT", "8770"))
    url = f"http://localhost:{port}/" + (f"g/{key}" if key else "")
    state_f = WORK / "review_app.json"
    if state_f.exists() and _alive(json.load(open(state_f))["pid"]):
        return {"url": url, "status": "already running"}
    log = WORK / "review_app.log"
    with open(log, "w") as fh:
        p = subprocess.Popen([SYS_PY, "review_app.py", "--host", "127.0.0.1", "--port", str(port)], cwd=REPO,
                             stdout=fh, stderr=subprocess.STDOUT, env=_env(), start_new_session=True)
    json.dump({"pid": p.pid, "port": port}, open(state_f, "w"))
    for _ in range(20):
        time.sleep(0.5)
        if "Running on" in log.read_text(errors="replace"):
            return {"url": url, "status": "started"}
        if p.poll() is not None:
            return {"error": "review app exited", "log_tail": log.read_text(errors="replace")[-600:]}
    return {"url": url, "status": "starting"}


def _py_json(args: list, timeout=300) -> dict:
    r = subprocess.run([SYS_PY, "-W", "ignore", *args], cwd=REPO, capture_output=True, text=True,
                       env=_env(), timeout=timeout)
    lines = [ln for ln in (r.stdout + "\n" + r.stderr).splitlines() if ln.startswith("{")]
    if lines:
        return json.loads(lines[-1])
    return {"error": (r.stderr or r.stdout)[-600:]}


@mcp.tool()
def youtube_chapters(game: str, source: str = "event_log", show_scores: bool = False) -> dict:
    """Build YouTube chapters (one per rally, for the description) and a rally caption track. Local only.

    source: "event_log" (the user's reviewed labels, default) or "auto" (raw detections).
    show_scores: add running scores to labels even if the rallies are not all confirmed.
    """
    key = _key(game)
    if source not in ("event_log", "auto"):
        raise ValueError("source must be event_log or auto")
    args = ["rally_lab/youtube_chapters.py", "--key", key, "--source", source]
    if show_scores:
        args.append("--scores")
    return _py_json(args)


@mcp.tool()
def youtube_find_video(query: str) -> dict:
    """Find the user's uploaded videos whose title contains `query` (needs the edit authorization)."""
    return _py_json(["rally_lab/youtube_publish.py", "find", "--query", query], timeout=120)


@mcp.tool()
def youtube_publish(game: str, video_id: str, source: str = "event_log", show_scores: bool = False,
                    captions: bool = True, apply: bool = False) -> dict:
    """Put rally chapters into a YouTube video's description and upload a "Rallies" caption track.

    Dry run unless apply=True. Only set apply=True after the user has seen the dry-run result and
    explicitly said to publish — this changes their public-facing video. Text outside the rally
    block in the description is kept; re-publishing replaces just the block.
    """
    key = _key(game)
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("video_id must be the 11-character YouTube id")
    args = ["rally_lab/youtube_publish.py", "publish", "--key", key, "--video", video_id, "--source", source]
    if show_scores:
        args.append("--scores")
    if not captions:
        args.append("--no-captions")
    if apply:
        args.append("--apply")
    return _py_json(args, timeout=180)


if __name__ == "__main__":
    mcp.run()

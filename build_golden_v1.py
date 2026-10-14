#!/usr/bin/env python3
"""
First-pass "golden" event log from ball tracks — meant to be hand-corrected
in label_clips.py's events page, not trusted as-is.

Inputs (from fast-volleyball-tracking-inference, run on a 1280x720 30fps proxy
whose timeline matches the stitched video):
  tracks/track_*.json   ball tracks with rally_classification
  whistles JSON         {"whistles": [t_s, ...]}  (weak cue: neighbouring courts whistle too)

Logic:
  rallies   = rally-classified ball tracks (plus long, busy tracks the
              classifier missed), merged across short ball-lost gaps
  sets      = split on long breaks between rallies
  serve     = near/far side of the camera, from where the ball first appears
              and which way it moves (near serve flies away = y decreases)
  score_by  = side-out rule: whoever serves rally k+1 won rally k; last rally
              of a set goes to whoever is ahead
  teams     = home is the near side in set 1; sides swap every set

Output: labels/<game_key>/event_log_<game_key>.json in label_clips format,
every rally auto=True, confirmed=False.

Usage:
    python3 build_golden_v1.py --tracks /mnt/d/volleyball/work/out/proxy_20261004_game02/tracks \
        --whistles /mnt/d/volleyball/work/g2_whistles.json \
        --video-file stitched_20261004_game02.mp4 --game-key 20261004_game02
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

MERGE_GAP_S   = 2.5    # ball lost for less than this mid-rally -> same rally
MIN_RALLY_S   = 2.0    # aces / serve errors are short
PRE_ROLL_S    = 1.0    # serve toss happens just before the ball track starts
POST_ROLL_S   = 1.5
SET_BREAK_S   = 150.0  # gap between rallies that marks a new set
WHISTLE_SNAP_S = 2.0


def fmt(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def load_tracks(tracks_dir: Path):
    out = []
    for p in sorted(tracks_dir.glob("track_*.json")):
        d = json.load(open(p))
        pos = sorted(((int(f), float(x), float(y)) for (x, y), f in d["positions"]))
        if len(pos) < 3:
            continue
        fps = float(d.get("fps") or 30.0)
        rf = d.get("rally_features", {})
        rc = d.get("rally_classification", {})
        busy = (rf.get("duration_sec", 0) >= 4 and rf.get("path_len_px", 0) >= 1500
                and rf.get("coverage", 0) >= 0.5)
        # short rallies (ace, serve error, quick kill): ball crosses most of the court
        short = (rf.get("duration_sec", 0) >= 2 and rf.get("x_range_px", 0) >= 400
                 and rf.get("path_len_px", 0) >= 850)
        if not (rc.get("is_rally") or busy or short):
            continue
        out.append({"start": pos[0][0] / fps, "end": pos[-1][0] / fps, "pos": pos, "fps": fps,
                    "conf": float(rc.get("rally_confidence", 0.0)),
                    "strict": bool(rc.get("is_rally") or busy)})
    return out


def merge(tracks):
    merged = []
    for t in sorted(tracks, key=lambda t: t["start"]):
        if merged and t["start"] - merged[-1]["end"] <= MERGE_GAP_S:
            m = merged[-1]
            m["end"] = max(m["end"], t["end"])
            m["pos"] += t["pos"]
            m["conf"] = max(m["conf"], t["conf"])
            m["strict"] = m["strict"] or t["strict"]
        else:
            merged.append(dict(t))
    return [m for m in merged if m["end"] - m["start"] >= MIN_RALLY_S]


def serve_side(r, frame_w=1280):
    """'near' or 'far' (relative to the camera) plus a 0..1 confidence.

    Camera sits behind the near end line, left of where the near server
    stands: a near serve enters at the right frame edge and flies left fast.
    A far serve starts mid-frame at the far end line (toss rises) or drops in
    from the top. Tracks that start mid-rally (serve missed) also look 'far',
    hence the low confidence there.
    """
    fps = r["fps"]
    pos = sorted(r["pos"])
    f0 = pos[0][0]
    early = [p for p in pos if p[0] - f0 <= 1.0 * fps]
    if len(early) < 4:
        return None, 0.0
    x0 = statistics.median(p[1] for p in early[:3]) / frame_w
    vx = (early[-1][1] - early[0][1]) / frame_w / max((early[-1][0] - early[0][0]) / fps, 1e-3)
    if x0 > 0.85 and vx < -0.3:
        return "near", 0.9
    if x0 > 0.8:
        return "near", 0.5
    return "far", 0.6 if min(p[2] for p in early[:3]) < 30 or early[-1][2] < early[0][2] else 0.3


def snap(t, whistles, before=True):
    cands = [w for w in whistles if (t - WHISTLE_SNAP_S <= w <= t if before else t <= w <= t + WHISTLE_SNAP_S)]
    if not cands:
        return t
    return max(cands) if before else min(cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--whistles")
    ap.add_argument("--video-file", required=True)
    ap.add_argument("--game-key", required=True)
    ap.add_argument("--out", help="default labels/<game_key>/event_log_<game_key>.json")
    args = ap.parse_args()

    whistles = json.load(open(args.whistles))["whistles"] if args.whistles else []
    rallies = merge(load_tracks(Path(args.tracks)))

    # Sets: split on long breaks between confident rallies only -- warm-up
    # hitting between sets passes the short-rally rule and would bridge them
    spans, cur = [], None
    for r in (r for r in rallies if r["strict"]):
        if cur and r["start"] - cur[1] > SET_BREAK_S:
            spans.append(cur)
            cur = None
        cur = [cur[0] if cur else r["start"], r["end"]]
    if cur:
        spans.append(cur)
    sets = [[r for r in rallies if a - 5 <= r["start"] and r["end"] <= b + 5] for a, b in spans]
    rallies = [r for s in sets for r in s]

    events = []

    def add(ev, t, **kw):
        events.append({"event": ev, "t_s": round(t, 3), "t": fmt(t),
                       "video_file": args.video_file, **kw})

    add("game_start", rallies[0]["start"] - PRE_ROLL_S - 2, set=None, rally_in_set=None, score_by=None)
    for si, srs in enumerate(sets, 1):
        near_team = "home" if si % 2 == 1 else "away"     # sides swap every set
        far_team = "away" if near_team == "home" else "home"
        sides = [serve_side(r) for r in srs]
        add("set_start", srs[0]["start"] - PRE_ROLL_S, set=si, rally_in_set=None, score=None)
        home = away = 0
        for k, r in enumerate(srs):
            start = snap(r["start"] - PRE_ROLL_S, whistles, before=True)
            end = snap(r["end"] + POST_ROLL_S, whistles, before=False)
            if k + 1 < len(srs) and sides[k + 1][0]:
                side, conf = sides[k + 1]
                score_by = near_team if side == "near" else far_team
            else:
                score_by = "home" if home >= away else "away"
                conf = 0.0
            home += score_by == "home"
            away += score_by == "away"
            common = dict(set=si, rally_in_set=k + 1, auto=True, confirmed=False)
            add("rally_start", start, score_by=None, serve_side=sides[k][0],
                ball_conf=round(r["conf"], 2), needs_check=not r["strict"], **common)
            add("rally_end", end, score_by=score_by, score_conf=round(conf, 2),
                score_after=f"{home}-{away}", **common)
        add("set_end", srs[-1]["end"] + POST_ROLL_S + 1, set=si, rally_in_set=None, score_by=None)

    for i, e in enumerate(events):
        e["seq"] = i

    out = Path(args.out) if args.out else HERE / "labels" / args.game_key / f"event_log_{args.game_key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        bak = out.with_suffix(f".json.bak.{datetime.now():%Y%m%d%H%M%S}")
        out.rename(bak)
        print(f"Existing log backed up to {bak.name}")
    json.dump({"game": args.game_key, "generated_at": datetime.now().isoformat(timespec="seconds"),
               "generator": "build_golden_v1", "event_count": len(events), "events": events},
              open(out, "w"), indent=2)

    print(f"{len(rallies)} rallies in {len(sets)} set(s) -> {out}")
    for si, srs in enumerate(sets, 1):
        ends = [e for e in events if e["event"] == "rally_end" and e["set"] == si]
        print(f"  set {si}: {len(srs)} rallies {fmt(srs[0]['start'])}-{fmt(srs[-1]['end'])}"
              f"  final {ends[-1]['score_after']} (home-away)")


if __name__ == "__main__":
    main()

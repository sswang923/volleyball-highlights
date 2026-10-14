#!/usr/bin/env python3
"""
YouTube chapters + caption track from a rally event log (no network, stdlib only).

Chapters (pasted into the video description) follow YouTube's rules: the first
starts at 0:00, there are at least 3, each is at least 10 s long. One chapter
per rally, starting just before the serve, plus warm-up and set breaks.

The caption track (SRT) shows "Set 1 · Rally 7 · Tyrants 4–3" while each rally
is live — a toggleable on-video annotation.

Scores are included only when the event log has been reviewed (every rally
confirmed) unless --scores is forced; auto point winners are too unreliable.

    python3 rally_lab/youtube_chapters.py --key 20261004_game02 [--source auto|event_log] [--scores]
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEAD_S = 2.0          # chapter starts this long before the rally (serve toss)
MIN_CHAPTER_S = 10.0  # YouTube ignores all chapters if any is shorter
BLOCK_START = "▶ Rallies (auto-generated)"
BLOCK_END = "◀ end rallies"


def ts(t):
    t = int(max(0, t))
    h, m, s = t // 3600, t % 3600 // 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def srt_ts(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def load(key, source):
    lab = REPO / "labels" / key
    path = lab / (f"auto_events_{key}.json" if source == "auto" else f"event_log_{key}.json")
    ev = json.load(open(path))["events"]
    starts = {(e["set"], e["rally_in_set"]): e for e in ev if e["event"] == "rally_start"}
    rallies = []
    for e in ev:
        if e["event"] == "rally_end" and (e["set"], e["rally_in_set"]) in starts:
            s = starts[(e["set"], e["rally_in_set"])]
            rallies.append({"set": e["set"], "n": e["rally_in_set"], "start": s["t_s"], "end": e["t_s"],
                            "by": e.get("score_by"), "confirmed": bool(e.get("confirmed", not e.get("auto")))})
    rallies.sort(key=lambda r: r["start"])
    return path, rallies


def with_scores(rallies):
    score = {}
    for r in rallies:
        h, a = score.get(r["set"], (0, 0))
        h += r["by"] == "home"
        a += r["by"] == "away"
        score[r["set"]] = (h, a)
        r["score"] = (h, a)
    return rallies


def chapters(rallies, team, show_scores):
    out = [(0.0, "Warm-up")]
    prev_set = None
    for r in rallies:
        t = max(0.0, r["start"] - LEAD_S)
        if r["set"] != prev_set and prev_set is not None:
            out.append((max(out[-1][0] + MIN_CHAPTER_S, t - 30), f"Set break — Set {r['set']} next"))
        label = f"Set {r['set']} · Rally {r['n']}"
        if show_scores and r.get("score"):
            h, a = r["score"]
            label += f" · {team} {h}–{a}"
        out.append((t, label))
        prev_set = r["set"]
    # enforce the 10 s minimum: fold a too-close rally into the previous chapter ("Rallies 8–9")
    kept = [out[0]]
    for t, label in out[1:]:
        if t - kept[-1][0] >= MIN_CHAPTER_S:
            kept.append((t, label))
        elif " · Rally " in label and " · Rally " in kept[-1][1]:
            prev = kept[-1][1]
            first = prev.split(" · Rall")[1].split(" · ")[0].split()[-1].split("–")[0]
            last = label.split(" · Rally ")[1].split(" · ")[0]
            kept[-1] = (kept[-1][0], f"{prev.split(' · Rall')[0]} · Rallies {first}–{last}"
                        + "".join(f" · {x}" for x in label.split(" · ")[3:]))
    return kept


def srt(rallies, team, show_scores):
    lines = []
    for i, r in enumerate(rallies, 1):
        text = f"Set {r['set']} · Rally {r['n']}"
        if show_scores and r.get("score"):
            h, a = r["score"]
            text += f" · {team} {h}–{a}"
        lines += [str(i), f"{srt_ts(r['start'])} --> {srt_ts(r['end'])}", text, ""]
    return "\n".join(lines)


def build(key, source="event_log", scores=None, team="Tyrants"):
    path, rallies = load(key, source)
    reviewed = bool(rallies) and all(r["confirmed"] for r in rallies)
    show = reviewed if scores is None else scores
    with_scores(rallies)
    ch = chapters(rallies, team, show)
    block = "\n".join([BLOCK_START] + [f"{ts(t)} {label}" for t, label in ch] + [BLOCK_END])
    out_dir = REPO / "labels" / key
    (out_dir / f"youtube_chapters_{key}.txt").write_text(block + "\n", encoding="utf-8")
    (out_dir / f"youtube_captions_{key}.srt").write_text(srt(rallies, team, show), encoding="utf-8")
    return {"game": key, "source": path.name, "reviewed": reviewed, "scores_shown": show,
            "rallies": len(rallies), "chapters": len(ch), "description_chars": len(block),
            "chapters_file": str(out_dir / f"youtube_chapters_{key}.txt"),
            "captions_file": str(out_dir / f"youtube_captions_{key}.srt"),
            "preview": block.splitlines()[:8]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--source", choices=["auto", "event_log"], default="event_log")
    ap.add_argument("--scores", action="store_true", default=None, help="show scores even if not reviewed")
    ap.add_argument("--team", default="Tyrants")
    a = ap.parse_args()
    print(json.dumps(build(a.key, a.source, a.scores, a.team), ensure_ascii=False))


if __name__ == "__main__":
    main()

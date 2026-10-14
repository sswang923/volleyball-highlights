#!/usr/bin/env python3
"""
Ball Tracker — RT-DETR-X edition

Detects the volleyball frame-by-frame and classifies its state:
  - IN AIR  : ball detected, moving, not overlapping any player
  - HELD    : ball center is inside a player bounding box
  - GROUND  : ball detected but stationary (near-zero velocity), not held

Also reads whistle_start / whistle_end events from the rally JSON and
combines everything into one chronological SRT file for QA in VLC.

Detection uses RT-DETR-X (COCO class 32 = sports ball, class 0 = person)
in a single inference pass per frame — no second model needed.

Usage:
    python ball_tracker.py -v D:\\Volleyball\\game1\\game1_stitched.mp4 ^
                           -r D:\\Volleyball\\game1\\ball_rallies.json

    python ball_tracker.py -v ... -r ... --max-rallies 10 --gpu

    # Tighter confidence / denser sampling:
    python ball_tracker.py -v ... -r ... --conf 0.4 --interval 0.1 --gpu
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

BALL_CLASS   = 32   # COCO sports ball
PERSON_CLASS = 0    # COCO person

# Ball is "held" when its centre falls inside a player box expanded by this fraction
HELD_PADDING = 0.08   # 8% of player box size on each side

# Ball is "on ground" when normalised speed drops below this value
# (units: fraction of frame width per second)
GROUND_SPEED_THRESH = 0.015


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ms_to_srt(ms: int) -> str:
    h  = ms // 3_600_000; ms %= 3_600_000
    m  = ms //    60_000; ms %=    60_000
    s  = ms //     1_000; ms %=     1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def load_rallies(json_path: Path) -> list:
    with open(json_path) as f:
        data = json.load(f)
    return data.get("rallies", [])


def _ball_state(
    cx_norm: float, cy_norm: float,
    persons: list,                  # list of (px1n, py1n, px2n, py2n) normalised
    speed: Optional[float],         # normalised units/sec, None if no prior position
) -> str:
    """Classify ball state from its position, nearby players, and velocity."""
    # Check overlap with any player box (expanded by HELD_PADDING)
    for px1n, py1n, px2n, py2n in persons:
        pw = px2n - px1n
        ph = py2n - py1n
        ex1 = px1n - pw * HELD_PADDING
        ey1 = py1n - ph * HELD_PADDING
        ex2 = px2n + pw * HELD_PADDING
        ey2 = py2n + ph * HELD_PADDING
        if ex1 <= cx_norm <= ex2 and ey1 <= cy_norm <= ey2:
            return "HELD"

    # Stationary and not held → on the ground
    if speed is not None and speed < GROUND_SPEED_THRESH:
        return "GROUND"

    return "IN AIR"


# ---------------------------------------------------------------------------
# Core tracking
# ---------------------------------------------------------------------------

def track_balls(
    video_path: Path,
    rallies: list,
    conf: float = 0.35,
    scan_interval: float = 0.25,
    max_rallies: int = 0,
    gpu: bool = False,
) -> list:
    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit(f"\nMissing dependency: {e}\nInstall: pip install opencv-python ultralytics\n")

    torch.set_num_threads(1)
    device = "cuda:0" if gpu else "cpu"

    print(f"[+] Loading RT-DETR-X (device={device}) ...")
    model = YOLO("rtdetr-x.pt")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(fps * scan_interval))

    if max_rallies:
        rallies = rallies[:max_rallies]

    print(f"[+] Processing {len(rallies)} rallies  "
          f"(fps={fps:.2f}, interval={scan_interval}s, step={frame_step})")

    results_by_rally = []
    total_detections = 0
    total_frames     = 0

    for rally in rallies:
        rally_id = rally.get("rally", rally.get("point", "?"))
        start_t  = rally["start_t"]
        end_t    = rally["end_t"]
        start_f  = int(start_t * fps)
        end_f    = int(end_t   * fps)

        rally_frames = []
        prev_cx_norm: Optional[float] = None
        prev_cy_norm: Optional[float] = None
        prev_t_s:     Optional[float] = None

        frame_idx = start_f
        while frame_idx <= end_f:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            t_s  = frame_idx / fps
            fh, fw = frame.shape[:2]
            total_frames += 1

            # Single pass: detect ball + persons together
            preds = model(
                frame,
                classes=[BALL_CLASS, PERSON_CLASS],
                conf=conf,
                device=device,
                verbose=False,
            )

            balls   = []
            persons = []  # (px1n, py1n, px2n, py2n)

            for r in preds:
                for box in r.boxes:
                    cls  = int(box.cls[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    if cls == PERSON_CLASS:
                        persons.append((x1/fw, y1/fh, x2/fw, y2/fh))
                    elif cls == BALL_CLASS:
                        cx   = (x1 + x2) / 2.0
                        cy   = (y1 + y2) / 2.0
                        balls.append({
                            "cx_norm": round(cx / fw, 4),
                            "cy_norm": round(cy / fh, 4),
                            "cx_px":   round(cx, 1),
                            "cy_px":   round(cy, 1),
                            "w_px":    round(x2 - x1, 1),
                            "h_px":    round(y2 - y1, 1),
                            "conf":    round(float(box.conf[0]), 3),
                        })

            # Classify each ball detection and compute velocity
            detections = []
            for b in balls:
                cx_n = b["cx_norm"]
                cy_n = b["cy_norm"]

                speed = None
                if prev_cx_norm is not None and prev_t_s is not None:
                    dt = t_s - prev_t_s
                    if dt > 0:
                        speed = math.hypot(cx_n - prev_cx_norm,
                                           cy_n - prev_cy_norm) / dt

                state = _ball_state(cx_n, cy_n, persons, speed)

                detections.append({
                    **b,
                    "state":       state,
                    "speed_norm":  round(speed, 4) if speed is not None else None,
                    "players_nearby": len(persons),
                })

            if detections:
                total_detections += len(detections)
                best = max(detections, key=lambda d: d["conf"])
                prev_cx_norm = best["cx_norm"]
                prev_cy_norm = best["cy_norm"]
                prev_t_s     = t_s

            rally_frames.append({
                "frame":      frame_idx,
                "t_s":        round(t_s, 3),
                "detections": detections,
            })
            frame_idx += frame_step

        frames_with_ball = sum(1 for f in rally_frames if f["detections"])
        pct = frames_with_ball / len(rally_frames) * 100 if rally_frames else 0

        state_counts = {"IN AIR": 0, "HELD": 0, "GROUND": 0}
        for f in rally_frames:
            for d in f["detections"]:
                state_counts[d["state"]] = state_counts.get(d["state"], 0) + 1

        print(
            f"  Rally {rally_id:>3}: {frames_with_ball}/{len(rally_frames)} frames "
            f"({pct:.0f}%)  "
            f"AIR={state_counts['IN AIR']}  "
            f"HELD={state_counts['HELD']}  "
            f"GROUND={state_counts['GROUND']}"
        )

        results_by_rally.append({
            "rally":              rally_id,
            "start_t":            start_t,
            "end_t":              end_t,
            "duration_s":         round(end_t - start_t, 2),
            "whistle_events":     rally.get("events", []),
            "frames_scanned":     len(rally_frames),
            "frames_with_ball":   frames_with_ball,
            "ball_detection_pct": round(pct, 1),
            "state_counts":       state_counts,
            "frames":             rally_frames,
        })

    cap.release()

    overall_pct = (
        sum(r["frames_with_ball"] for r in results_by_rally) /
        max(1, sum(r["frames_scanned"] for r in results_by_rally)) * 100
    )
    total_air    = sum(r["state_counts"]["IN AIR"] for r in results_by_rally)
    total_held   = sum(r["state_counts"]["HELD"]   for r in results_by_rally)
    total_ground = sum(r["state_counts"]["GROUND"] for r in results_by_rally)

    print(f"\n[+] Done.")
    print(f"    Rallies processed : {len(results_by_rally)}")
    print(f"    Frames scanned    : {total_frames}")
    print(f"    Total detections  : {total_detections}")
    print(f"    Overall det %     : {overall_pct:.1f}%")
    print(f"    IN AIR / HELD / GROUND : {total_air} / {total_held} / {total_ground}")

    return results_by_rally


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

STATE_LABEL = {
    "IN AIR": "IN AIR",
    "HELD":   "HELD  ",
    "GROUND": "GROUND",
}


def write_outputs(results: list, video_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # --- JSON ---
    json_path = output_dir / f"{stem}_ball_positions.json"
    with open(json_path, "w") as f:
        json.dump(
            {"video": str(video_path), "model": "rtdetr-x",
             "total_rallies": len(results), "rallies": results},
            f, indent=2,
        )
    print(f"    JSON → {json_path}")

    # --- Combined SRT (whistle + ball in play + ball in air) ---
    # Each entry: (t_ms, end_ms, label)
    events = []

    for rally in results:
        rid = rally["rally"]

        # [WHISTLE] rally start / end
        for ev in rally.get("whistle_events", []):
            t_ms = int(ev["t"] * 1000)
            etype = ev["type"]
            if etype == "whistle_start":
                label = f"[WHISTLE] Rally {rid} START"
            elif etype == "whistle_end":
                label = f"[WHISTLE] Rally {rid} END  ({rally['duration_s']:.0f}s)"
            else:
                label = f"[WHISTLE] {etype}"
            events.append((t_ms, t_ms + 1500, label))

        # [BALL IN PLAY] — one subtitle spanning the whole rally
        play_start_ms = int(rally["start_t"] * 1000)
        play_end_ms   = int(rally["end_t"]   * 1000)
        events.append((play_start_ms, play_end_ms, f"[BALL IN PLAY] Rally {rid}"))

        # [BALL IN AIR] and [BALL DEAD] — consecutive-run subtitles per state
        cur_state:    str | None = None
        seg_start_ms: int | None = None
        seg_end_ms:   int | None = None

        def _flush_seg(state, start_ms, end_ms):
            if state == "IN AIR":
                events.append((start_ms, end_ms, f"[BALL IN AIR] Rally {rid}"))
            else:  # HELD or GROUND — ball is not flying
                events.append((start_ms, end_ms, f"[BALL DEAD] Rally {rid}"))

        for frame_data in rally["frames"]:
            if not frame_data["detections"]:
                if cur_state is not None:
                    _flush_seg(cur_state, seg_start_ms, seg_end_ms)
                    cur_state = seg_start_ms = seg_end_ms = None
                continue
            best  = max(frame_data["detections"], key=lambda d: d["conf"])
            t_ms  = int(frame_data["t_s"] * 1000)
            state = best["state"]   # "IN AIR", "HELD", or "GROUND"
            if state == cur_state:
                seg_end_ms = t_ms + 250
            else:
                if cur_state is not None:
                    _flush_seg(cur_state, seg_start_ms, seg_end_ms)
                cur_state    = state
                seg_start_ms = t_ms
                seg_end_ms   = t_ms + 250
        if cur_state is not None:
            _flush_seg(cur_state, seg_start_ms, seg_end_ms)

    # Sort by timestamp
    events.sort(key=lambda e: e[0])

    srt_lines = []
    for i, (t_ms, end_ms, label) in enumerate(events, 1):
        srt_lines.append(f"{i}\n{ms_to_srt(t_ms)} --> {ms_to_srt(end_ms)}\n{label}\n")

    srt_path = output_dir / f"{stem}_events.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
    print(f"    SRT  → {srt_path}  ({len(events)} events)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Track volleyball with RT-DETR-X — ball states + whistle events"
    )
    parser.add_argument("-v", "--video",    required=True, help="Path to stitched MP4")
    parser.add_argument("-r", "--rallies",  required=True,
                        help="Path to ball_rallies.json or event_log.json")
    parser.add_argument("-o", "--output",   default=None,
                        help="Output directory (default: same folder as video)")
    parser.add_argument("--conf",           type=float, default=0.35,
                        help="Detection confidence (default: 0.35)")
    parser.add_argument("--interval",       type=float, default=0.25,
                        help="Seconds between scanned frames (default: 0.25)")
    parser.add_argument("--max-rallies",    type=int,   default=0,
                        help="Only process first N rallies (0 = all)")
    parser.add_argument("--gpu",            action="store_true", help="Use CUDA GPU")
    args = parser.parse_args()

    video_path   = Path(args.video)
    rallies_path = Path(args.rallies)
    output_dir   = Path(args.output) if args.output else video_path.parent

    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")
    if not rallies_path.exists():
        sys.exit(f"Rallies JSON not found: {rallies_path}")

    rallies = load_rallies(rallies_path)
    print(f"[+] Loaded {len(rallies)} rallies from {rallies_path.name}")

    results = track_balls(
        video_path    = video_path,
        rallies       = rallies,
        conf          = args.conf,
        scan_interval = args.interval,
        max_rallies   = args.max_rallies,
        gpu           = args.gpu,
    )

    write_outputs(results, video_path, output_dir)


if __name__ == "__main__":
    main()

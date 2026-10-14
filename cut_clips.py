#!/usr/bin/env python3
"""
Clip cutter — extract one or more clips from a game video.

Times accept HH:MM:SS, MM:SS, or plain seconds (fractions ok: 1:23:45.5).

Usage:
    # One clip: 1:12:30 to 1:13:05
    python3 cut_clips.py -v /mnt/d/volleyball/stitched_20260620_game04.mp4 \
        -c 1:12:30-1:13:05

    # Several clips at once (each becomes its own file)
    python3 cut_clips.py -v /mnt/d/volleyball/stitched_20260620_game04.mp4 \
        -c 12:30-13:05 -c 25:10-25:48 -c 1:02:00-1:02:30

    # Name a clip (default name is <video>_<start>-<end>.mp4)
    python3 cut_clips.py -v game.mp4 -c 12:30-13:05:great_block

    # Frame-accurate cut (re-encodes; slower but starts exactly on time)
    python3 cut_clips.py -v game.mp4 -c 12:30-13:05 --accurate

Default is lossless stream copy: instant, but the start snaps to the nearest
keyframe before your start time, so the clip may begin up to a second or two
early. Use --accurate when the exact start matters.
"""

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_CLIP_RE = re.compile(r"^([\d:.]+)-([\d:.]+)(?::(.+))?$")


def parse_time(t: str) -> float:
    """'1:23:45.5' / '23:45' / '90' -> seconds."""
    parts = t.split(":")
    if len(parts) > 3 or not all(p for p in parts):
        raise ValueError(f"Bad time: {t!r}")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


def fmt_time(secs: float) -> str:
    s = int(secs)
    return f"{s // 3600:02d}{s % 3600 // 60:02d}{s % 60:02d}"


def cut(video: Path, start: float, end: float, output: Path, accurate: bool) -> Path:
    if accurate:
        # -ss after -i decodes from the start of the video: exact but slower.
        # Re-encode video, copy audio.
        codec = ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "copy"]
        cmd = ["ffmpeg", "-y", "-i", str(video),
               "-ss", str(start), "-to", str(end), *codec, str(output)]
    else:
        # -ss before -i seeks by keyframe: instant, lossless stream copy.
        cmd = ["ffmpeg", "-y", "-ss", str(start), "-to", str(end),
               "-i", str(video), "-c", "copy",
               "-avoid_negative_ts", "make_zero", str(output)]

    log.info(f"Cutting {output.name}  ({end - start:.1f}s{', re-encode' if accurate else ', stream copy'}) ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(result.stderr[-2000:])
        raise RuntimeError(f"ffmpeg failed for {output.name}")
    log.info(f"  -> {output}  ({output.stat().st_size / 1e6:.1f} MB)")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Extract clips from a game video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", "-v", required=True, help="Source video file")
    parser.add_argument(
        "--clip", "-c", action="append", required=True, dest="clips",
        help="START-END[:name], e.g. 12:30-13:05 or 1:02:00-1:02:30:great_block. Repeatable.",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Output directory (default: same as the source video)",
    )
    parser.add_argument(
        "--accurate", action="store_true",
        help="Frame-accurate start (re-encodes video; slower)",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"Video not found: {video}")

    output_dir = Path(args.output_dir) if args.output_dir else video.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for spec in args.clips:
        m = _CLIP_RE.match(spec)
        if not m:
            sys.exit(f"Bad clip spec: {spec!r}  (expected START-END or START-END:name)")
        start, end = parse_time(m.group(1)), parse_time(m.group(2))
        if end <= start:
            sys.exit(f"Clip end must be after start: {spec!r}")
        name = m.group(3) or f"{video.stem}_{fmt_time(start)}-{fmt_time(end)}"
        jobs.append((start, end, output_dir / f"{name}.mp4"))

    for start, end, output in jobs:
        cut(video, start, end, output, accurate=args.accurate)

    log.info(f"Done: {len(jobs)} clip(s) in {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DJI Stitcher — group DJI footage into games and concatenate each game into one MP4.

Groups files by the gap between one file's end (start timestamp + duration)
and the next file's start: DJI auto-split files are ~0s apart (same game), a
larger gap means the camera was stopped and restarted (new game). Stream-copy
concat: lossless and fast.

Usage:
    # Stitch all games in a directory
    python dji_stitch.py -i "D:/volleyball"

    # Stitch only one date's games
    python dji_stitch.py -i "D:/volleyball" --match-date 20260317

    # Stitch games on/after a date
    python dji_stitch.py -i "D:/volleyball" --min-date 20260619

    # Use a larger gap threshold in seconds (default: 120)
    python dji_stitch.py -i "D:/volleyball" --gap-seconds 300
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# DJI filename: DJI_YYYYMMDDHHMMSS_NNNN_D.MP4
# Group 1 = date YYYYMMDD, group 2 = time HHMMSS, group 3 = sequence number NNNN
_DJI_RE = re.compile(r"DJI_(\d{8})(\d{6})_(\d+)", re.IGNORECASE)


def _safe_duration(f: Path) -> float:
    """Duration in seconds, or 0.0 if ffprobe can't read it (e.g. junk file)."""
    try:
        return get_video_duration(f)
    except Exception:
        return 0.0


def discover_videos(input_dir: str, gap_seconds: int = 120) -> Dict[str, List[Path]]:
    """
    Find all MP4/MOV files and group them into individual games using each
    file's start timestamp (encoded in the DJI filename) plus its duration.

    DJI auto-splits a continuous recording into ~17 GB files; the next file's
    start timestamp lands within a second or two of the previous file's end
    (start + duration). When the camera is stopped and restarted for a new
    game there is a much larger gap. So a file whose start is more than
    gap_seconds after the previous file's end — or on a different date —
    begins a new game.

    Returns dict keyed by game label, e.g.:
        {'20251018_game01': [Path, ...], '20251018_game02': [...], ...}
    """
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    all_files = sorted(
        [
            f for f in p.iterdir()
            if f.suffix.upper() in (".MP4", ".MOV", ".AVI")
            and not f.name.startswith("._")          # skip macOS resource fork sidecars
            and not f.name.lower().startswith("stitched_")  # skip our own outputs
        ],
        key=lambda f: f.name,
    )
    if not all_files:
        raise ValueError(f"No video files found in {input_dir}")

    # Parse start datetime for every file; end = start + duration.
    parsed = []
    for f in all_files:
        m = _DJI_RE.match(f.name)
        if m:
            date = m.group(1)
            start = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            end = start + timedelta(seconds=_safe_duration(f))
        else:
            date, start, end = "unknown", None, None
        parsed.append({"file": f, "date": date, "start": start, "end": end})

    # Walk in filename order; start a new game on date change, missing
    # timestamp, or a start-to-previous-end gap larger than gap_seconds.
    groups: Dict[str, List[Path]] = {}
    game_idx = 1
    prev = None
    current_key = None

    for r in parsed:
        if prev is None or r["date"] != prev["date"]:
            new_game = True
        elif r["start"] is None or prev["end"] is None:
            new_game = True
        else:
            new_game = (r["start"] - prev["end"]).total_seconds() > gap_seconds

        if new_game:
            if prev is None or r["date"] != prev["date"]:
                game_idx = 1      # reset per date
            else:
                game_idx += 1
            current_key = f"{r['date']}_game{game_idx:02d}"

        groups.setdefault(current_key, []).append(r["file"])
        prev = r

    log.info(f"Found {len(all_files)} files → {len(groups)} game(s)  (gap={gap_seconds}s):")
    for game_key, files in sorted(groups.items()):
        size_gb = sum(f.stat().st_size for f in files) / 1e9
        seq_range = f"{files[0].name} … {files[-1].name}" if len(files) > 1 else files[0].name
        log.info(f"  {game_key}: {len(files)} file(s), {size_gb:.1f} GB  [{seq_range}]")

    return groups


def get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def stitch_videos(video_files: List[Path], output_path: Path) -> Path:
    """
    Concatenate MP4 files using ffmpeg concat demuxer (stream copy = lossless + fast).
    Input files must have compatible codecs (DJI footage from same camera: yes).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_path = f.name
        for v in video_files:
            f.write(f"file '{Path(v).as_posix()}'\n")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            "-video_track_timescale", "90000",  # normalize DJI timecodes → smooth playback
            "-avoid_negative_ts", "make_zero",  # shift any negative timestamps to 0
            str(output_path),
        ]
        log.info(f"Stitching {len(video_files)} files → {output_path.name} ...")
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        log.info(f"Stitch complete: {output_path}")
    finally:
        os.unlink(list_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Group DJI footage into games and stitch each game into one MP4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input-dir", "-i", required=True, help="Directory of DJI MP4 files")
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for stitched files (default: same as input-dir)",
    )
    parser.add_argument(
        "--match-date",
        default=None,
        help="Only stitch games from this date (YYYYMMDD). Omit to process all dates.",
    )
    parser.add_argument(
        "--min-date",
        default=None,
        help="Only stitch games on/after this date (YYYYMMDD).",
    )
    parser.add_argument(
        "--gap-seconds",
        type=int,
        default=120,
        help="Gap (s) between one file's end and the next file's start that marks "
             "a new game (default: 120). Auto-split files are ~0s apart.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit("ffmpeg not found. Install it and make sure it's on PATH.")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    game_groups = discover_videos(args.input_dir, gap_seconds=args.gap_seconds)

    if args.match_date:
        game_groups = {k: v for k, v in game_groups.items() if k.startswith(args.match_date)}
        if not game_groups:
            sys.exit(f"No games found for date {args.match_date}.")

    if args.min_date:
        game_groups = {k: v for k, v in game_groups.items() if k[:8] >= args.min_date}
        if not game_groups:
            sys.exit(f"No games found on/after {args.min_date}.")

    for game_key, video_files in sorted(game_groups.items()):
        if len(video_files) == 1:
            log.info(f"Skipping {game_key}: only one file, nothing to stitch.")
            continue
        output_path = output_dir / f"stitched_{game_key}.mp4"
        stitch_videos(video_files, output_path)
        duration = get_video_duration(output_path)
        h, m, s = int(duration // 3600), int((duration % 3600) // 60), int(duration % 60)
        log.info(f"  Duration: {h:02d}:{m:02d}:{s:02d}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Volleyball Highlight Reel Generator

Scans DJI MP4 footage to:
  1. Group files into individual games by detecting gaps in DJI sequence numbers
     (e.g. files _0002_/_0003_ are one game; a jump to _0006_ starts a new game)
  2. Optionally stitch each game into one continuous video (stream copy, lossless)
  3. Detect a specific player by jersey number using YOLO + EasyOCR
  4. Build a timestamped JSON/CSV index of every detected appearance
  5. Create a highlight reel MP4 of all their plays

Requirements:
    pip install opencv-python ultralytics easyocr numpy
    ffmpeg must be installed and on PATH (used for stitching and clip extraction)

Quick stitch command (no Python needed):
    # Stitch two or more specific MP4s together:
    printf "file '%s'\\n" file1.mp4 file2.mp4 file3.mp4 > /tmp/list.txt
    ffmpeg -f concat -safe 0 -i /tmp/list.txt -c copy output.mp4

Usage examples:

    # Scan all footage for player #12 (auto-groups into games by sequence number)
    python volleyball_highlights.py -i "/var/storage/volleyball videos" -p 12

    # Stitch each detected game then build highlights for player #7
    python volleyball_highlights.py -i "/var/storage/volleyball videos" -p 7 --stitch

    # Only process a specific date
    python volleyball_highlights.py -i "/var/storage/volleyball videos" -p 12 \\
        --match-date 20251018

    # Adjust sequence-number gap that triggers a new game (default: 2)
    python volleyball_highlights.py -i "/var/storage/volleyball videos" -p 12 \\
        --seq-gap 3

    # Index only, no highlight video, faster scan
    python volleyball_highlights.py -i "/var/storage/volleyball videos" -p 12 \\
        --no-highlights --scan-interval 2.0
"""

import argparse
import csv
import json
import logging
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single frame where the target player's jersey number was read."""
    video_file: str
    timestamp: float          # seconds into that file
    absolute_time: float      # seconds since first file of the day
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h of player bounding box in frame


@dataclass
class Segment:
    """
    A contiguous run of detections — one 'play' or on-court stint.
    start_time/end_time are relative to the source video_file.
    abs_start/abs_end are relative to the whole day's footage.
    """
    video_file: str
    start_time: float
    end_time: float
    abs_start: float
    abs_end: float
    detections: List[Detection] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# ─────────────────────────────────────────────────────────────────────────────
# Video file discovery and grouping
# ─────────────────────────────────────────────────────────────────────────────

# DJI filename: DJI_YYYYMMDDHHMMSS_NNNN_D.MP4
# Group 1 = date YYYYMMDD, group 2 = sequence number NNNN
_DJI_RE = re.compile(r"DJI_(\d{8})\d{6}_(\d+)", re.IGNORECASE)


def discover_videos(input_dir: str, seq_gap: int = 2) -> Dict[str, List[Path]]:
    """
    Find all MP4/MOV files and group them into individual games.

    DJI sequence numbers (the _NNNN_ field) increase within a recording session.
    A jump larger than seq_gap signals a new game (camera turned off between sets).
    Files on different dates always start a new game.

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
            and not f.name.startswith("._")   # skip macOS resource fork sidecars
        ],
        key=lambda f: f.name,
    )
    if not all_files:
        raise ValueError(f"No video files found in {input_dir}")

    # Parse (date, seq_num) for every file
    parsed = []
    for f in all_files:
        m = _DJI_RE.match(f.name)
        if m:
            parsed.append((m.group(1), int(m.group(2)), f))
        else:
            parsed.append(("unknown", -1, f))

    # Walk in filename order; start a new game on date change or large seq gap
    groups: Dict[str, List[Path]] = {}
    game_idx = 1
    prev_date, prev_seq = None, None
    current_key = None

    for date, seq, f in parsed:
        new_game = (
            date != prev_date
            or prev_seq is None
            or (seq != -1 and prev_seq != -1 and seq - prev_seq > seq_gap)
        )
        if new_game:
            if date != prev_date:
                game_idx = 1      # reset per date
            elif current_key is not None:
                game_idx += 1
            current_key = f"{date}_game{game_idx:02d}"
            prev_date = date

        groups.setdefault(current_key, []).append(f)
        if seq != -1:
            prev_seq = seq

    log.info(f"Found {len(all_files)} files → {len(groups)} game(s)  (seq-gap={seq_gap}):")
    for game_key, files in sorted(groups.items()):
        size_gb = sum(f.stat().st_size for f in files) / 1e9
        seq_range = f"{files[0].name} … {files[-1].name}" if len(files) > 1 else files[0].name
        log.info(f"  {game_key}: {len(files)} file(s), {size_gb:.1f} GB  [{seq_range}]")

    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Club detection
# ─────────────────────────────────────────────────────────────────────────────

# Map of lowercase keywords → canonical club name.
# Order matters: more specific phrases first.
CLUB_KEYWORDS = {
    "bay to bay":            "Bay to Bay",
    "baytobay":              "Bay to Bay",
    "b2b":                   "Bay to Bay",
    "academy volleyball":    "Academy Volleyball Club",
    "academy":               "Academy Volleyball Club",
    "tyrants":               "Tyrants Volleyball Club",
    "mvvc":                  "MVVC",
    "sf elite":              "SF Elite",
    "sfelite":               "SF Elite",
}


def detect_club(
    first_video: Path,
    reader,                    # EasyOCR Reader instance (reuse from PlayerDetector)
    sample_every_s: float = 30.0,
    window_s: float = 300.0,   # how many seconds worth of frames to sample
) -> str:
    """
    Sample frames from the middle of the video and run OCR looking for known
    club name text in banners, scoreboards, or jerseys.

    Sampling from the middle avoids warmup/setup footage at the start and
    dead-time at the end, giving the best chance of catching sideline banners
    and scorer's table signage during active play.

    Returns the best-matched club name, or 'unknown'.
    """
    try:
        import cv2
    except ImportError:
        return "unknown"

    cap = cv2.VideoCapture(str(first_video))
    if not cap.isOpened():
        return "unknown"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps

    # Center the sampling window in the middle of the file
    mid_s = duration_s / 2.0
    start_s = max(0.0, mid_s - window_s / 2.0)
    end_s   = min(duration_s, mid_s + window_s / 2.0)
    frame_step = max(1, int(fps * sample_every_s))

    log.info(
        f"  Scanning for club name: {_fmt_time(start_s)} → {_fmt_time(end_s)} "
        f"(middle of {_fmt_time(duration_s)} file)"
    )

    from collections import Counter
    votes: Counter = Counter()

    frame_idx = int(start_s * fps)
    end_frame = int(end_s * fps)

    while frame_idx < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Full-frame OCR — club banners appear anywhere: sidelines,
        # scorer's table, scoreboards, jersey text.
        ocr_hits = reader.readtext(frame, detail=1)
        full_text = " ".join(r[1] for r in ocr_hits).lower()
        for keyword, club in CLUB_KEYWORDS.items():
            if keyword in full_text:
                votes[club] += 1
                log.debug(f"  Club keyword {keyword!r} at t={frame_idx/fps:.0f}s")

        frame_idx += frame_step

    cap.release()

    if votes:
        best, count = votes.most_common(1)[0]
        log.info(f"  Club detected: {best!r}  (votes={count})")
        return best

    log.info("  Club: not detected in middle frames")
    return "unknown"


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


# ─────────────────────────────────────────────────────────────────────────────
# Player detection
# ─────────────────────────────────────────────────────────────────────────────

class PlayerDetector:
    """
    Detects a jersey number in a frame using:
      1. YOLOv8 to find person bounding boxes
      2. EasyOCR on the upper-body crop of each person to read numbers

    Notes on accuracy with DJI drone footage:
      - Works best when players are at least ~80px tall in frame
      - High-angle shots may show jersey backs (also numbered, so that's fine)
      - Very wide/high drone shots may not resolve numbers — lower --scan-interval
        won't help; you need closer footage for OCR to work reliably
    """

    def __init__(
        self,
        player_number: int,
        confidence_threshold: float = 0.4,
        gpu: bool = False,
    ):
        self.target = str(player_number)
        self.conf_thresh = confidence_threshold
        self.gpu = gpu
        self._load_models()

    def _load_models(self):
        missing = []
        try:
            from ultralytics import YOLO
            import easyocr
            import numpy as np
        except ImportError as e:
            sys.exit(
                f"\nMissing dependency: {e}\n"
                "Install all requirements with:\n"
                "  pip install opencv-python ultralytics easyocr numpy\n"
            )

        log.info("Loading YOLOv8n (person detection) ...")
        self.yolo = YOLO("yolov8n.pt")  # ~6MB, auto-downloads on first run

        log.info(f"Loading EasyOCR (gpu={self.gpu}) ...")
        self.reader = easyocr.Reader(["en"], gpu=self.gpu)

        import cv2, numpy as np
        self._cv2 = cv2
        self._np = np

    def detect(self, frame) -> List[Tuple[float, Tuple[int, int, int, int]]]:
        """
        Given a BGR frame (numpy array), return list of (confidence, bbox)
        for every person whose jersey number matches the target.
        bbox = (x, y, w, h) in pixel coords.
        """
        cv2, np = self._cv2, self._np
        results = self.yolo(frame, classes=[0], verbose=False)  # class 0 = person
        matches = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                person_h = y2 - y1
                if person_h < 30:
                    continue  # too small to read a number

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Jersey number lives in the upper 55% of the person crop
                jersey = crop[: int(person_h * 0.55), :]
                if jersey.size == 0:
                    continue

                # Scale up small crops so OCR can read them
                scale = max(1, 90 // jersey.shape[0])
                if scale > 1:
                    jersey = cv2.resize(
                        jersey,
                        (jersey.shape[1] * scale, jersey.shape[0] * scale),
                        interpolation=cv2.INTER_CUBIC,
                    )

                # Grayscale + Otsu threshold → clean digits
                gray = cv2.cvtColor(jersey, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                ocr_hits = self.reader.readtext(thresh, allowlist="0123456789")
                for _, text, conf in ocr_hits:
                    cleaned = re.sub(r"\s+", "", text)
                    if cleaned == self.target and conf >= self.conf_thresh:
                        matches.append((conf, (x1, y1, x2 - x1, y2 - y1)))
                        break  # one match per person bbox

        return matches


def scan_video(
    video_path: Path,
    detector: PlayerDetector,
    scan_interval: float = 1.0,
    time_offset: float = 0.0,
    start_s: float = 0.0,
    end_s: Optional[float] = None,
) -> List[Detection]:
    """
    Sample one frame every scan_interval seconds from video_path.
    start_s/end_s optionally restrict the scan to a time window (for parallel workers).
    time_offset: seconds to add to each timestamp for absolute positioning.
    Returns all frames where the target player was detected.
    """
    try:
        import cv2
    except ImportError:
        sys.exit("opencv-python is required: pip install opencv-python")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    start_frame = int(start_s * fps)
    end_frame = int((end_s * fps) if end_s is not None else total_frames)
    end_frame = min(end_frame, total_frames)
    frame_step = max(1, int(fps * scan_interval))
    chunk_duration = (end_frame - start_frame) / fps

    log.info(
        f"Scanning {video_path.name}  "
        f"[{_fmt_time(start_s)} → {_fmt_time(end_s or duration)}]  "
        f"({_fmt_time(chunk_duration)}, {fps:.1f}fps, every {scan_interval}s)"
    )

    detections: List[Detection] = []
    frame_idx = start_frame
    last_progress_log = -1

    while frame_idx < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        hits = detector.detect(frame)

        for conf, bbox in hits:
            det = Detection(
                video_file=video_path.name,
                timestamp=timestamp,
                absolute_time=time_offset + timestamp,
                confidence=conf,
                bbox=bbox,
            )
            detections.append(det)
            log.debug(
                f"  [{video_path.name}] {_fmt_time(timestamp)} "
                f"player #{detector.target} conf={conf:.2f}"
            )
            break  # one detection per sampled frame

        # Log progress every 2 minutes of video
        progress_bucket = int(timestamp // 120)
        if progress_bucket != last_progress_log and timestamp > 0:
            elapsed = timestamp - start_s
            pct = elapsed / chunk_duration * 100 if chunk_duration else 0
            log.info(
                f"  {video_path.name} [{_fmt_time(start_s)}-{_fmt_time(end_s or duration)}]: "
                f"{pct:.0f}%  detections: {len(detections)}"
            )
            last_progress_log = progress_bucket

        frame_idx += frame_step

    cap.release()
    log.info(f"  → {len(detections)} detection(s) in {video_path.name}")
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Parallel scanning
# ─────────────────────────────────────────────────────────────────────────────

# Module-level detector instance used by worker processes
_worker_detector: Optional["PlayerDetector"] = None


def _init_worker(player_number: int, confidence: float, gpu: bool):
    """Initialise a PlayerDetector in each worker process (called once per worker)."""
    global _worker_detector
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [W%(process)d] %(message)s", datefmt="%H:%M:%S")
    _worker_detector = PlayerDetector(player_number, confidence, gpu)


def _scan_chunk(args: tuple) -> List[Detection]:
    """Worker entry point: scan one time chunk of a video."""
    video_path, start_s, end_s, scan_interval, time_offset = args
    return scan_video(
        Path(video_path), _worker_detector, scan_interval, time_offset, start_s, end_s
    )


def parallel_scan(
    video_path: Path,
    player_number: int,
    confidence: float,
    gpu: bool,
    scan_interval: float,
    time_offset: float,
    n_workers: int,
) -> List[Detection]:
    """
    Split video_path into n_workers equal time chunks and scan them in parallel.
    Each worker gets its own PlayerDetector (YOLO + EasyOCR) instance.
    """
    from concurrent.futures import ProcessPoolExecutor

    duration = get_video_duration(video_path)
    chunk_s = duration / n_workers

    chunk_args = [
        (
            str(video_path),
            i * chunk_s,
            (i + 1) * chunk_s if i < n_workers - 1 else None,
            scan_interval,
            time_offset + i * chunk_s,
        )
        for i in range(n_workers)
    ]

    log.info(
        f"Parallel scan: {n_workers} workers × {_fmt_time(chunk_s)} chunks  "
        f"({video_path.name})"
    )

    all_detections: List[Detection] = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(player_number, confidence, gpu),
    ) as executor:
        for chunk_result in executor.map(_scan_chunk, chunk_args):
            all_detections.extend(chunk_result)

    return sorted(all_detections, key=lambda d: d.absolute_time)


# ─────────────────────────────────────────────────────────────────────────────
# Segment / index building
# ─────────────────────────────────────────────────────────────────────────────

def build_segments(
    detections: List[Detection],
    gap_threshold: float = 8.0,
    clip_padding: float = 4.0,
) -> List[Segment]:
    """
    Merge nearby detections into play segments.

    gap_threshold: detections farther apart than this (seconds) start a new segment.
    clip_padding:  seconds of context to include before/after each segment.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.absolute_time)
    segments: List[Segment] = []

    group_start = sorted_dets[0]
    group_end = sorted_dets[0]
    group_dets = [sorted_dets[0]]

    for det in sorted_dets[1:]:
        if det.absolute_time - group_end.absolute_time <= gap_threshold:
            group_end = det
            group_dets.append(det)
        else:
            segments.append(_finalize_segment(group_start, group_end, group_dets, clip_padding))
            group_start = group_end = det
            group_dets = [det]

    segments.append(_finalize_segment(group_start, group_end, group_dets, clip_padding))
    return segments


def _finalize_segment(
    first: Detection,
    last: Detection,
    dets: List[Detection],
    pad: float,
) -> Segment:
    return Segment(
        video_file=first.video_file,
        start_time=max(0.0, first.timestamp - pad),
        end_time=last.timestamp + pad,
        abs_start=max(0.0, first.absolute_time - pad),
        abs_end=last.absolute_time + pad,
        detections=dets,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Highlight reel creation
# ─────────────────────────────────────────────────────────────────────────────

def _get_video_dimensions(video_path: Path) -> Tuple[int, int]:
    """Return (width, height) of video via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-select_streams", "v:0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def _zoom_crop_filter(seg: Segment, frame_w: int, frame_h: int, zoom_factor: float) -> Optional[str]:
    """
    Compute an ffmpeg crop= filter string that zooms into the average player
    position across all detections in the segment.

    zoom_factor: fraction of frame width to use as crop window (e.g. 0.35 = 35%).
    Returns None if no bbox data is available (falls back to full frame).
    """
    bboxes = [d.bbox for d in seg.detections if d.bbox]
    if not bboxes:
        return None

    # Average player center across all detections
    avg_cx = int(sum(x + w / 2 for x, y, w, h in bboxes) / len(bboxes))
    avg_cy = int(sum(y + h / 2 for x, y, w, h in bboxes) / len(bboxes))

    # Crop window: zoom_factor of frame width, 16:9 aspect
    cw = int(frame_w * zoom_factor)
    cw = cw - (cw % 2)  # must be even for h264
    ch = cw * 9 // 16
    ch = ch - (ch % 2)

    # Center crop on player, clamped to frame bounds
    cx = max(0, min(avg_cx - cw // 2, frame_w - cw))
    cy = max(0, min(avg_cy - ch // 2, frame_h - ch))

    return f"crop={cw}:{ch}:{cx}:{cy},scale={frame_w}:{frame_h}"


def create_highlight_reel(
    video_map: Dict[str, Path],   # filename → full path
    segments: List[Segment],
    output_path: Path,
    player_number: int,
    crf: int = 22,
    zoom: bool = False,
    zoom_factor: float = 0.35,
) -> Optional[Path]:
    """
    Extract each play segment from its source file and concatenate into a reel.
    Adds an unobtrusive text overlay with player number and absolute timestamp.
    When zoom=True, crops and scales each clip to focus on the detected player.
    """
    if not segments:
        log.warning("No segments — highlight reel not created.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-fetch dimensions from first available source (all same resolution)
    frame_dims: Optional[Tuple[int, int]] = None
    if zoom:
        for seg in segments:
            src = video_map.get(seg.video_file)
            if src:
                try:
                    frame_dims = _get_video_dimensions(src)
                    log.info(f"Video dimensions: {frame_dims[0]}x{frame_dims[1]}, zoom factor: {zoom_factor}")
                except Exception:
                    pass
                break

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_paths: List[Path] = []

        for i, seg in enumerate(segments):
            src = video_map.get(seg.video_file)
            if src is None:
                log.warning(f"Source not found for segment: {seg.video_file}")
                continue

            clip_out = Path(tmpdir) / f"clip_{i:04d}.mp4"
            duration = seg.end_time - seg.start_time

            # Build video filter chain: optional zoom crop
            filters = []
            if zoom and frame_dims:
                crop_filter = _zoom_crop_filter(seg, frame_dims[0], frame_dims[1], zoom_factor)
                if crop_filter:
                    filters.append(crop_filter)

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(seg.start_time),
                "-i", str(src),
                "-t", str(duration),
            ]
            if filters:
                cmd += ["-vf", ",".join(filters)]
            cmd += [
                "-c:v", "libx264",
                "-crf", str(crf),
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "128k",
                str(clip_out),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log.warning(f"Clip {i+1} extraction failed:\n{result.stderr[-600:]}")
                continue

            clip_paths.append(clip_out)
            log.info(
                f"  [{i+1}/{len(segments)}] {seg.video_file}  "
                f"{_fmt_time(seg.start_time)} → {_fmt_time(seg.end_time)}  "
                f"({duration:.1f}s)"
            )

        if not clip_paths:
            log.error("No clips extracted — highlight reel aborted.")
            return None

        # Concatenate clips (stream copy since they're all same encoder settings)
        concat_list = Path(tmpdir) / "concat.txt"
        with open(concat_list, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{Path(cp).as_posix()}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)

    log.info(f"Highlight reel → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Compiled full-game video with chapters
# ─────────────────────────────────────────────────────────────────────────────

def create_compiled_video(
    game_groups: Dict[str, List[Path]],
    output_path: Path,
) -> Optional[Path]:
    """
    Stitch all games into one video (stream copy, lossless) and embed MP4
    chapter markers so any player can jump between games.

    Dead time between games (camera off = sequence number gaps) is eliminated
    because we only include the actual recorded files.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Measure each game's duration to place chapter timestamps
    chapters: List[Tuple[str, float, float]] = []   # (title, start_s, end_s)
    ordered_files: List[Path] = []
    cumulative_s = 0.0

    for game_key in sorted(game_groups.keys()):
        files = game_groups[game_key]
        game_start = cumulative_s
        log.info(f"  Measuring {game_key} ({len(files)} file(s)) ...")
        game_duration = sum(get_video_duration(f) for f in files)
        chapters.append((game_key, game_start, game_start + game_duration))
        ordered_files.extend(files)
        cumulative_s += game_duration
        log.info(f"    → {_fmt_time(game_duration)}  (starts at {_fmt_time(game_start)})")

    log.info(f"Total compiled duration: {_fmt_time(cumulative_s)}  ({len(chapters)} game(s))")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write concat list
        concat_list = Path(tmpdir) / "concat.txt"
        with open(concat_list, "w") as f:
            for v in ordered_files:
                f.write(f"file '{Path(v).as_posix()}'\n")

        # Write ffmetadata chapter file
        meta_path = Path(tmpdir) / "chapters.txt"
        with open(meta_path, "w") as f:
            f.write(";FFMETADATA1\n\n")
            for title, start_s, end_s in chapters:
                f.write(
                    f"[CHAPTER]\n"
                    f"TIMEBASE=1/1000\n"
                    f"START={int(start_s * 1000)}\n"
                    f"END={int(end_s * 1000)}\n"
                    f"title={title}\n\n"
                )

        # Concat + inject chapter metadata in one pass
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-f", "ffmetadata", "-i", str(meta_path),
            "-map", "0",
            "-map_metadata", "1",
            "-c", "copy",
            "-video_track_timescale", "90000",
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]
        log.info(f"Building compiled video → {output_path.name} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"Compile failed:\n{result.stderr[-800:]}")
            return None

    log.info(f"Compiled video → {output_path}")
    log.info("Chapters:")
    for title, start_s, _ in chapters:
        log.info(f"  {_fmt_time(start_s)}  {title}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Rally detection + point timeline
# ─────────────────────────────────────────────────────────────────────────────

_SCORE_RE = re.compile(
    r'\b(\d{1,2})\s*[-:]\s*(\d{1,2})\b'   # "12-8" or "12:8"
    r'|\b(\d{1,2})\s{1,3}(\d{1,2})\b'     # "12 8"
)


def _parse_volleyball_score(texts: List[str]) -> Optional[Tuple[int, int]]:
    """Extract a volleyball score (two numbers 0–30) from OCR text lines."""
    combined = " ".join(texts)
    m = _SCORE_RE.search(combined)
    if m:
        a = int(m.group(1) or m.group(3))
        b = int(m.group(2) or m.group(4))
        if 0 <= a <= 30 and 0 <= b <= 30:
            return (a, b)
    return None


def auto_detect_scoreboard_roi(
    video_path: Path,
    reader,
    n_samples: int = 15,
    n_strips: int = 6,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Sample frames from the middle of the video, divide the frame into
    horizontal strips, and find the strip that most consistently contains
    a volleyball score (two numbers 0–30).
    Returns (x, y, w, h) or None.
    """
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_f = total // 4
    end_f   = 3 * total // 4
    step    = max(1, (end_f - start_f) // n_samples)

    strip_h = frame_h // n_strips
    strips  = [(i, i * strip_h, strip_h) for i in range(n_strips)]

    votes: Counter = Counter()
    log.info(f"  Auto-detecting scoreboard region ({n_strips} horizontal strips) ...")

    frame_idx = start_f
    while frame_idx < end_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        for idx, y0, h in strips:
            strip = frame[y0: y0 + h, :]
            texts = reader.readtext(strip, detail=0, allowlist="0123456789-: ")
            if _parse_volleyball_score(texts):
                votes[idx] += 1
        frame_idx += step

    cap.release()

    if not votes:
        log.info("  Scoreboard not auto-detected — scores will be unknown")
        log.info("  Tip: use --scoreboard-roi x,y,w,h to specify the region manually")
        return None

    best_idx, count = votes.most_common(1)[0]
    y0 = best_idx * strip_h
    log.info(f"  Scoreboard detected in strip {best_idx} (y={y0}–{y0+strip_h}) "
             f"({count}/{n_samples} frames)")
    return (0, y0, frame_w, strip_h)


def _read_score(frame, reader, roi: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
    """OCR a single frame region for a volleyball score."""
    try:
        import cv2
    except ImportError:
        return None
    x, y, w, h = roi
    region = frame[y: y + h, x: x + w]
    if region.size == 0:
        return None
    scale = max(1, 60 // region.shape[0])
    if scale > 1:
        region = cv2.resize(region, (region.shape[1] * scale, region.shape[0] * scale))
    texts = reader.readtext(region, detail=0, allowlist="0123456789-: ")
    return _parse_volleyball_score(texts)


def _read_score_consensus(
    cap: "cv2.VideoCapture",
    fps: float,
    duration: float,
    reader,
    roi: Tuple[int, int, int, int],
    end_s: float,
) -> Optional[Tuple[int, int]]:
    """Sample 4 frames in [end_s+0.5, end_s+2.5] and return the majority score.

    Requires at least 2 frames to agree; returns None when no consensus exists.
    """
    try:
        import cv2
    except ImportError:
        return None
    sample_times = [end_s + 0.5 + i * (2.0 / 3.0) for i in range(4)]
    votes: List[Tuple[int, int]] = []
    for t in sample_times:
        t = min(t, duration - 0.1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret:
            continue
        score = _read_score(frame, reader, roi)
        if score is not None:
            votes.append(score)
    if not votes:
        return None
    counts = Counter(votes)
    best, best_n = counts.most_common(1)[0]
    return best if best_n >= 2 else None


def detect_whistles(video_path: Path) -> List[float]:
    """Detect referee whistle events in a video's audio track.

    Uses ffmpeg to pipe 16kHz mono PCM, then applies a sliding 50ms FFT
    window.  A frame is flagged when the 2–4.5 kHz band carries >= 35% of
    total energy AND the band energy exceeds the 95th percentile.
    Consecutive flagged frames within 0.15 s are merged into events;
    the peak timestamp of each event is returned.

    Pure numpy — no scipy dependency.  Returns [] if video has no audio.
    """
    try:
        import numpy as np
    except ImportError:
        log.info("  Whistle detection: numpy not available — skipping")
        return []

    log.info(f"  Whistle detection: {video_path.name} ...")

    SR  = 16000
    WIN = int(SR * 0.05)   # 800 samples = 50 ms window
    HOP = WIN // 2         # 50% overlap

    ffmpeg_bin = "ffmpeg"
    for candidate in [os.path.expanduser("~/.local/bin/ffmpeg"), "ffmpeg"]:
        if os.path.isfile(candidate):
            ffmpeg_bin = candidate
            break

    pipe_cmd = [
        ffmpeg_bin, "-i", str(video_path),
        "-vn",
        "-ar", str(SR), "-ac", "1",
        "-f", "s16le", "pipe:1",
    ]
    proc = subprocess.Popen(pipe_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw = proc.stdout.read()
    proc.wait()

    if not raw:
        log.info("  Whistle detection: no audio track — skipping")
        return []

    audio   = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hann    = np.hanning(WIN)
    freqs   = np.fft.rfftfreq(WIN, d=1.0 / SR)
    lo_bin  = int(np.searchsorted(freqs, 2000.0))
    hi_bin  = int(np.searchsorted(freqs, 4500.0))

    band_energies: List[float] = []
    timestamps:    List[float] = []
    ratios:        List[float] = []

    pos = 0
    while pos + WIN <= len(audio):
        window   = audio[pos: pos + WIN] * hann
        spectrum = np.abs(np.fft.rfft(window)) ** 2
        total_e  = float(spectrum.sum()) + 1e-12
        band_e   = float(spectrum[lo_bin:hi_bin].sum())
        band_energies.append(band_e)
        ratios.append(band_e / total_e)
        timestamps.append((pos + WIN / 2) / SR)
        pos += HOP

    if not band_energies:
        return []

    band_arr  = np.array(band_energies, dtype=np.float32)
    abs_thresh = float(np.percentile(band_arr, 95))
    ratio_thresh = 0.35

    # Merge consecutive flagged frames into events
    whistle_times: List[float] = []
    in_event     = False
    event_start  = 0.0
    event_peak_e = 0.0
    event_peak_t = 0.0

    for t, band_e, ratio in zip(timestamps, band_energies, ratios):
        flagged = band_e >= abs_thresh and ratio >= ratio_thresh
        if flagged and not in_event:
            in_event = True
            event_start = event_peak_t = t
            event_peak_e = band_e
        elif flagged and in_event:
            if band_e > event_peak_e:
                event_peak_e = band_e
                event_peak_t = t
        elif not flagged and in_event:
            duration = t - event_start
            if 0.15 <= duration <= 2.0:
                whistle_times.append(event_peak_t)
                log.debug(f"    Whistle @ {_fmt_time(event_peak_t)}  dur={duration:.2f}s  ratio={ratio:.2f}")
            in_event = False

    if in_event and timestamps:
        duration = timestamps[-1] - event_start
        if 0.15 <= duration <= 2.0:
            whistle_times.append(event_peak_t)

    log.info(f"  Whistle detection: {len(whistle_times)} event(s) found")
    return whistle_times


def detect_game_start(
    video_path: Path,
    reader,
    scoreboard_roi: Tuple[int, int, int, int],
    sample_interval: float = 10.0,
    max_scan_s: float = 2700.0,   # scan up to first 45 min
) -> float:
    """
    Find the timestamp where the game officially starts by scanning for the
    first scoreboard reading of 0-0.

    Returns the timestamp in seconds, or 0.0 if not detected.
    """
    try:
        import cv2
    except ImportError:
        return 0.0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps
    end_s        = min(max_scan_s, duration)
    frame_step   = max(1, int(fps * sample_interval))

    log.info(f"  Auto-detecting game start (scanning first {_fmt_time(end_s)} for 0-0 score) ...")

    frame_idx = 0
    while frame_idx < int(end_s * fps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        score = _read_score(frame, reader, scoreboard_roi)
        if score == (0, 0):
            t = frame_idx / fps
            log.info(f"  Game start detected at {_fmt_time(t)} (score 0-0)")
            cap.release()
            return t
        frame_idx += frame_step

    cap.release()
    log.info("  Game start not auto-detected — scanning from beginning")
    return 0.0


def filter_rallies_by_ball(
    video_map: Dict[str, Path],
    point_timeline: List[dict],
    yolo_model,
    conf: float = 0.2,
    min_ball_pct: float = 0.15,
    sample_interval: float = 0.5,
) -> List[dict]:
    """
    Filter the point timeline by requiring the ball (YOLO class 32) to appear
    in at least min_ball_pct of sampled frames within each rally segment.

    Segments with no ball visible are almost certainly false positives
    (camera panning, warmup, timeouts, etc.).
    """
    try:
        import cv2
    except ImportError:
        log.warning("Ball filter: opencv not available — skipping")
        return point_timeline

    BALL_CLASS = 32

    # Cache open VideoCaptures per source file
    caps: Dict[str, "cv2.VideoCapture"] = {}

    def _get_cap(filename: str) -> Optional["cv2.VideoCapture"]:
        if filename not in caps:
            src = video_map.get(filename)
            if src is None:
                return None
            caps[filename] = cv2.VideoCapture(str(src))
        return caps[filename]

    log.info(f"  Ball confirmation: checking {len(point_timeline)} segments ...")
    filtered  = []
    n_removed = 0

    for pt in point_timeline:
        cap = _get_cap(pt["video_file"])
        if cap is None or not cap.isOpened():
            filtered.append(pt)   # can't check — keep it
            continue

        fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step = max(1, int(fps * sample_interval))
        start_f    = int(pt["start_s"] * fps)
        end_f      = int(pt["end_s"]   * fps)

        frames_checked    = 0
        frames_with_ball  = 0
        frame_idx         = start_f

        while frame_idx <= end_f:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            preds = yolo_model(frame, classes=[BALL_CLASS], conf=conf, verbose=False)
            for r in preds:
                if len(r.boxes) > 0:
                    frames_with_ball += 1
                    break
            frames_checked += 1
            frame_idx += frame_step

        ball_pct = frames_with_ball / frames_checked if frames_checked > 0 else 0.0

        if ball_pct >= min_ball_pct:
            filtered.append(pt)
            log.debug(f"    Point {pt['point']:>3}: KEEP   ball={ball_pct:.0%}")
        else:
            n_removed += 1
            log.info(f"    Point {pt['point']:>3}: REMOVE  ball={ball_pct:.0%} < {min_ball_pct:.0%}")

    for cap in caps.values():
        cap.release()

    log.info(f"  Ball filter: kept {len(filtered)}/{len(point_timeline)}, removed {n_removed}")
    return filtered


def detect_rallies(
    video_path: Path,
    motion_threshold: float = 3.0,
    scan_interval: float = 0.5,
    min_rally_s: float = 3.0,
    min_dead_s: float = 3.0,
    pre_pad: float = 1.5,
    post_pad: float = 2.0,
    reader=None,
    scoreboard_roi: Optional[Tuple[int, int, int, int]] = None,
    time_offset: float = 0.0,
    start_offset_s: float = 0.0,    # seconds to skip at start (pregame/warmup)
    court_roi: Optional[Tuple[int, int, int, int]] = None,
    whistle_gate_s: float = 0.0,    # >0: drop segments with no whistle within this many s of the start
) -> Tuple[List[Tuple[float, float]], List[dict]]:
    """
    Detect volleyball rally periods using frame-to-frame motion.

    motion_threshold: 90th-percentile pixel diff (on 320×180 downsampled frame)
                      that separates active play from dead time.
    court_roi: 'x,y,w,h' in source pixels — motion is measured only inside this
               region, so bench/spectator/adjacent-court movement is ignored.
    whistle_gate_s: rallies start on a serve whistle; if > 0, segments whose
               start has no detected whistle within the window are dropped.
    Returns (segments, point_timeline) where timestamps are relative to this file.
    point_timeline entries also carry abs_start using time_offset.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        log.error("opencv-python required for rally detection")
        return [], []

    # Use cv2 only to get video metadata (fps / duration)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps
    cap.release()  # motion scan uses ffmpeg pipe below, not OpenCV

    out_fps = 1.0 / scan_interval   # e.g. 2.0 fps for 0.5s interval
    scan_duration = duration - start_offset_s
    n_expected = int(scan_duration * out_fps)
    scan_from  = f" from {_fmt_time(start_offset_s)}" if start_offset_s > 0 else ""
    log.info(
        f"  Detecting rallies: {video_path.name} ({_fmt_time(duration)}{scan_from}, "
        f"every {scan_interval}s ≈ {n_expected} samples)"
    )

    # Use ffmpeg to extract downsampled grayscale frames at the sample rate.
    # This is far faster than OpenCV random-seeking into a large SMB file:
    # ffmpeg reads sequentially, hardware-decodes internally, and outputs only
    # the frames we need at 320×180 grayscale.
    ffmpeg_bin = "ffmpeg"
    for candidate in [os.path.expanduser("~/.local/bin/ffmpeg"), "ffmpeg"]:
        if os.path.isfile(candidate):
            ffmpeg_bin = candidate
            break

    vf = f"fps={out_fps}"
    if court_roi:
        cx, cy, cw, ch = court_roi
        vf += f",crop={cw}:{ch}:{cx}:{cy}"
        log.info(f"  Court ROI: motion measured in {cw}x{ch}+{cx}+{cy} only")
    vf += ",scale=320:180"

    pipe_cmd = [
        ffmpeg_bin,
        *(["-ss", str(start_offset_s)] if start_offset_s > 0 else []),
        "-i", str(video_path),
        "-vf", vf,
        "-pix_fmt", "gray",
        "-vcodec", "rawvideo",
        "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.Popen(pipe_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    motion_data: List[Tuple[float, float]] = []
    prev_arr = None
    frame_count = 0
    FRAME_BYTES = 320 * 180
    last_log = -1

    while True:
        raw = proc.stdout.read(FRAME_BYTES)
        if not raw or len(raw) < FRAME_BYTES:
            break
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        timestamp = frame_count / out_fps + start_offset_s
        if prev_arr is not None:
            diff = np.abs(arr - prev_arr)
            motion = float(np.percentile(diff, 90))
            motion_data.append((timestamp, motion))
        prev_arr = arr
        frame_count += 1

        bucket = int(timestamp // 300)
        if bucket != last_log:
            pct = (timestamp - start_offset_s) / max(1, scan_duration) * 100
            log.info(f"  Rally scan: {pct:.0f}%  ({_fmt_time(timestamp)})")
            last_log = bucket

    proc.wait()

    if not motion_data:
        return [], []

    # 3-sample rolling average to smooth noise
    smoothed = []
    for i, (t, m) in enumerate(motion_data):
        window = motion_data[max(0, i - 1): i + 2]
        smoothed.append((t, sum(x[1] for x in window) / len(window)))

    # Build raw active segments from smoothed motion
    raw_segs: List[Tuple[float, float]] = []
    in_seg = False
    seg_start = 0.0
    for t, m in smoothed:
        if m > motion_threshold and not in_seg:
            seg_start = t
            in_seg = True
        elif m <= motion_threshold and in_seg:
            raw_segs.append((seg_start, t))
            in_seg = False
    if in_seg:
        raw_segs.append((seg_start, smoothed[-1][0]))

    # Pad and merge nearby raw segments, tracking the total ACTIVE
    # (above-threshold) time inside each merged segment. A merged segment
    # qualifies as a rally only if its active time reaches min_rally_s:
    # padding can no longer promote a half-second motion blip to a rally,
    # while a real rally split by a brief motion dip is still bridged.
    groups: List[List[float]] = []   # [padded_start, padded_end, active_s]
    for s, e in sorted(raw_segs):
        ps = max(start_offset_s, s - pre_pad)
        pe = min(duration, e + post_pad)
        if groups and ps < groups[-1][1] + min_dead_s:
            groups[-1][1] = max(groups[-1][1], pe)
            groups[-1][2] += e - s
        else:
            groups.append([ps, pe, e - s])

    segments = [(ps, pe) for ps, pe, active in groups if active >= min_rally_s]
    if len(groups) != len(segments):
        log.info(
            f"  Dropped {len(groups) - len(segments)} low-activity segment(s) "
            f"(< {min_rally_s}s of motion)"
        )

    log.info(f"  → {len(segments)} rally segment(s) detected")

    # Detect whistle events once for the whole file (used for boundary
    # snapping and, if enabled, whistle gating)
    whistle_times = detect_whistles(video_path)

    # Whistle gate: every real rally starts on a serve whistle. Drop segments
    # whose start has no whistle nearby (huddles, timeouts, warmup motion).
    if whistle_gate_s > 0:
        if whistle_times:
            gated = [
                (s, e) for s, e in segments
                if any(abs(wt - (s + pre_pad)) <= whistle_gate_s for wt in whistle_times)
            ]
            log.info(
                f"  Whistle gate (±{whistle_gate_s:.0f}s): kept {len(gated)}/{len(segments)} segment(s)"
            )
            segments = gated
        else:
            log.warning("  Whistle gate requested but no whistles detected — gate skipped")

    def _snap_to_whistle(t: float, window: float = 1.5) -> Optional[float]:
        best: Optional[float] = None
        best_d = window + 1.0
        for wt in whistle_times:
            d = abs(wt - t)
            if d <= window and d < best_d:
                best_d = d
                best = wt
        return best

    # Build point timeline, optionally reading score after each rally
    point_timeline: List[dict] = []
    if segments:
        cap2 = cv2.VideoCapture(str(video_path))
        set_number  = 1
        prev_score: Optional[Tuple[int, int]] = None

        for i, (start_s, end_s) in enumerate(segments):
            # ── Whistle snapping ──────────────────────────────────────────────
            whistle_start_t = _snap_to_whistle(start_s)
            whistle_end_t   = _snap_to_whistle(end_s)
            snapped_start   = whistle_start_t if whistle_start_t is not None else start_s
            snapped_end     = whistle_end_t   if whistle_end_t   is not None else end_s

            # ── Consensus score reading ───────────────────────────────────────
            score: Optional[Tuple[int, int]] = None
            if reader and scoreboard_roi:
                score = _read_score_consensus(
                    cap2, fps, duration, reader, scoreboard_roi, snapped_end
                )

            # ── Score monotonicity, set tracking, delta validation ────────────
            score_valid = False
            if score is not None:
                if prev_score is None:
                    # First readable score in this file — accept unconditionally
                    score_valid = True
                else:
                    a0, b0 = prev_score
                    a1, b1 = score
                    delta_a, delta_b = a1 - a0, b1 - b0
                    if (delta_a == 1 and delta_b == 0) or (delta_a == 0 and delta_b == 1):
                        # Exactly +1 on one side — clean point
                        score_valid = True
                    elif a1 <= 2 and b1 <= 2 and (a0 > 2 or b0 > 2):
                        # Score reset to near 0 → new set
                        set_number += 1
                        score_valid = True
                        log.info(f"    ↑ Set {set_number} detected at {_fmt_time(snapped_start)}")
                    # else: score decreased (not a reset) or jumped — OCR noise, reject

                if score_valid:
                    prev_score = score

            score_str = f"{score[0]}-{score[1]}" if score else "unknown"
            flag = "" if score_valid else " ⚠"
            log.info(
                f"    [S{set_number} Pt{i+1:>3}] {_fmt_time(snapped_start)} → {_fmt_time(snapped_end)}"
                f"  ({snapped_end - snapped_start:.0f}s)  score: {score_str}{flag}"
                + (f"  W↑{_fmt_time(whistle_start_t)}" if whistle_start_t is not None else "")
                + (f"  W↓{_fmt_time(whistle_end_t)}"   if whistle_end_t   is not None else "")
            )
            point_timeline.append({
                "point":           i + 1,
                "set":             set_number,
                "video_file":      video_path.name,
                "start":           _fmt_time(snapped_start),
                "end":             _fmt_time(snapped_end),
                "abs_start":       _fmt_time(time_offset + snapped_start),
                "abs_end":         _fmt_time(time_offset + snapped_end),
                "start_s":         round(snapped_start, 2),
                "end_s":           round(snapped_end, 2),
                "abs_start_s":     round(time_offset + snapped_start, 2),
                "abs_end_s":       round(time_offset + snapped_end, 2),
                "duration_s":      round(snapped_end - snapped_start, 1),
                "score":           score_str,
                "score_valid":     score_valid,
                "whistle_start_t": round(whistle_start_t, 3) if whistle_start_t is not None else None,
                "whistle_end_t":   round(whistle_end_t,   3) if whistle_end_t   is not None else None,
            })
        cap2.release()

    return segments, point_timeline


def _ass_time(secs: float) -> str:
    """Seconds → ASS timestamp H:MM:SS.cc"""
    cs = int(round(secs * 100))
    return f"{cs // 360000}:{cs % 360000 // 6000:02d}:{cs % 6000 // 100:02d}.{cs % 100:02d}"


def write_score_ass(events: List[Tuple[float, float, str]], ass_path: Path) -> Path:
    """
    Write an ASS subtitle file showing the set/score label during each rally.
    events: (start_s, end_s, label) in compiled-video time.
    Styled as a boxed overlay in the top-left corner.
    """
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\nPlayResY: 1080\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Score,DejaVu Sans,48,&H00FFFFFF,&H00FFFFFF,&H00000000,"
        "&H60000000,-1,0,0,0,100,100,0,0,3,8,0,7,40,40,30,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Text\n"
    )
    with open(ass_path, "w") as f:
        f.write(header)
        for start_s, end_s, label in events:
            f.write(
                f"Dialogue: 0,{_ass_time(start_s)},{_ass_time(end_s)},"
                f"Score,,0,0,0,{label}\n"
            )
    return ass_path


def create_rally_compilation(
    video_map: Dict[str, Path],
    point_timeline: List[dict],
    output_path: Path,
    crf: int = 22,
    score_overlay: bool = False,
) -> Optional[Path]:
    """
    Cut every rally segment from its source file, concatenate into a single
    video, and embed one MP4 chapter per point (labelled with score if known).
    Dead time between points is completely removed.

    score_overlay=True burns a set/score label into the video (one re-encode
    pass with the libass subtitles filter). Otherwise a sidecar .ass file with
    the same labels is written next to the MP4 (VLC auto-loads it) — no
    re-encode needed.
    """
    if not point_timeline:
        log.warning("No rally segments — compilation not created.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_paths: List[Path] = []
        compiled_s = 0.0
        chapter_data: List[Tuple[float, str]] = []   # (start_in_compiled, title)
        score_events: List[Tuple[float, float, str]] = []  # (start, end, label) in compiled time
        display_score = "0-0"   # score *entering* each rally (OCR reads it after the point)
        cur_set: Optional[int] = None

        for pt in point_timeline:
            src = video_map.get(pt["video_file"])
            if src is None:
                log.warning(f"Source not found: {pt['video_file']}")
                continue

            clip_out = Path(tmpdir) / f"rally_{pt['point']:04d}.mp4"
            dur = pt["end_s"] - pt["start_s"]

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(pt["start_s"]),
                "-i", str(src),
                "-t", str(dur),
                "-c", "copy",
                str(clip_out),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log.warning(f"Rally {pt['point']} extract failed")
                continue

            score_label = f" ({pt['score']})" if pt["score"] != "unknown" else ""
            chapter_data.append((compiled_s, f"Point {pt['point']}{score_label}  {pt['abs_start']}"))

            if pt["set"] != cur_set:
                cur_set = pt["set"]
                display_score = "0-0"
            score_events.append((compiled_s, compiled_s + dur, f"Set {cur_set}   {display_score}"))
            if pt.get("score_valid") and pt["score"] != "unknown":
                display_score = pt["score"]

            compiled_s += dur
            clip_paths.append(clip_out)

        if not clip_paths:
            log.error("No rally clips extracted.")
            return None

        # Concatenate all clips
        concat_list = Path(tmpdir) / "concat.txt"
        with open(concat_list, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{Path(cp).as_posix()}'\n")

        tmp_concat = Path(tmpdir) / "raw.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy",
            "-video_track_timescale", "90000",
            "-avoid_negative_ts", "make_zero",
            str(tmp_concat),
        ], check=True, stderr=subprocess.DEVNULL)

        # Score overlay: burn the set/score labels into the video, or write
        # them as a sidecar .ass that VLC auto-loads (no re-encode).
        if score_overlay and score_events:
            ass_path = write_score_ass(score_events, Path(tmpdir) / "score.ass")
            # Quote the path for the filtergraph parser (tmpdir has no quotes)
            burned = Path(tmpdir) / "burned.mp4"
            log.info(f"  Burning score overlay ({len(score_events)} rallies, re-encoding) ...")
            result = subprocess.run([
                "ffmpeg", "-y", "-i", str(tmp_concat),
                "-vf", f"subtitles=filename='{ass_path.as_posix()}'",
                "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
                "-c:a", "copy",
                str(burned),
            ], capture_output=True, text=True)
            if result.returncode == 0:
                tmp_concat = burned
            else:
                log.warning("Score overlay burn failed; writing sidecar .ass instead")
                log.debug(result.stderr[-400:])
                write_score_ass(score_events, output_path.with_suffix(".ass"))
        elif score_events:
            sidecar = write_score_ass(score_events, output_path.with_suffix(".ass"))
            log.info(f"  Score subtitles → {sidecar.name}  (VLC auto-loads; use --score-overlay to burn in)")

        # Build chapter metadata
        meta_path = Path(tmpdir) / "chapters.txt"
        with open(meta_path, "w") as f:
            f.write(";FFMETADATA1\n\n")
            for idx, (start_s, title) in enumerate(chapter_data):
                end_s = chapter_data[idx + 1][0] if idx + 1 < len(chapter_data) else compiled_s
                f.write(
                    f"[CHAPTER]\nTIMEBASE=1/1000\n"
                    f"START={int(start_s * 1000)}\nEND={int(end_s * 1000)}\n"
                    f"title={title}\n\n"
                )

        # Inject chapters — try two approaches for compatibility
        chapter_ok = False
        for cmd in [
            # Approach 1: explicit ffmetadata format flag
            ["ffmpeg", "-y", "-i", str(tmp_concat),
             "-f", "ffmetadata", "-i", str(meta_path),
             "-map", "0", "-map_metadata", "1", "-c", "copy", str(output_path)],
            # Approach 2: no format flag (auto-detect from file content)
            ["ffmpeg", "-y", "-i", str(tmp_concat),
             "-i", str(meta_path),
             "-map", "0", "-map_metadata", "1", "-c", "copy", str(output_path)],
        ]:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                chapter_ok = True
                break
            log.debug(f"Chapter injection attempt failed: {result.stderr[-200:]}")

        if not chapter_ok:
            log.warning("Chapter injection failed; saving without chapters")
            log.debug(result.stderr[-400:])
            import shutil
            shutil.copy(str(tmp_concat), str(output_path))

    log.info(f"Rally compilation → {output_path}  ({len(clip_paths)} points, {_fmt_time(compiled_s)})")
    return output_path


def export_point_timeline(
    game_key: str,
    point_timeline: List[dict],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write point_timeline JSON + CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"point_timeline_{game_key}.json"
    csv_path  = output_dir / f"point_timeline_{game_key}.csv"

    score_readable    = sum(1 for p in point_timeline if p["score"] != "unknown")
    score_valid_count = sum(1 for p in point_timeline if p.get("score_valid", False))
    data = {
        "game":              game_key,
        "generated_at":      datetime.now().isoformat(),
        "point_count":       len(point_timeline),
        "score_readable":    score_readable,
        "score_valid_count": score_valid_count,
        "points":            point_timeline,
    }
    json_path.write_text(json.dumps(data, indent=2))

    if point_timeline:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(point_timeline[0].keys()))
            writer.writeheader()
            writer.writerows(point_timeline)

    log.info(
        f"Point timeline → {json_path}  "
        f"({score_readable}/{len(point_timeline)} scores read, "
        f"{score_valid_count} score-delta valid)"
    )
    return json_path, csv_path


def build_event_log(
    point_timeline: List[dict],
    game_duration_s: float = 0.0,
) -> List[dict]:
    """
    Convert a flat point_timeline into a chronological event log.

    Event types (in order of emission):
        game_start   — beginning of the game footage (t=0)
        set_start    — first point of each set begins
        point_start  — a rally begins
        point_end    — a rally ends (score updated)
        set_end      — last point of a set ends
        game_end     — end of game footage

    Each event dict:
        seq          sequential index (0-based)
        event        event type string
        t_s          absolute game-relative timestamp in seconds
        t            HH:MM:SS formatted
        set          set number (None for game_start/game_end)
        point_in_set point index within the current set (None for set/game events)
        score        score string "X-Y" at this moment (None if unavailable)
        video_file   source file name (None for game_start/game_end when unambiguous)
    """
    if not point_timeline:
        return []

    events: List[dict] = []

    def _ev(event, t_s, set_=None, point_in_set=None, score=None, video_file=None):
        events.append({
            "event":        event,
            "t_s":          round(t_s, 2),
            "t":            _fmt_time(t_s),
            "set":          set_,
            "point_in_set": point_in_set,
            "score":        score,
            "video_file":   video_file,
        })

    # game_start at the very beginning of the footage
    _ev("game_start", 0.0, video_file=point_timeline[0]["video_file"])

    current_set: Optional[int] = None
    point_in_set = 0
    last_pt: Optional[dict] = None

    for pt in point_timeline:
        set_num = pt["set"]

        # Detect set transition
        if set_num != current_set:
            if current_set is not None and last_pt is not None:
                _ev("set_end", last_pt["abs_end_s"],
                    set_=current_set, score=last_pt["score"],
                    video_file=last_pt["video_file"])
            current_set = set_num
            point_in_set = 0
            _ev("set_start", pt["abs_start_s"],
                set_=set_num, video_file=pt["video_file"])

        point_in_set += 1

        _ev("point_start", pt["abs_start_s"],
            set_=set_num, point_in_set=point_in_set,
            score=pt["score"], video_file=pt["video_file"])

        _ev("point_end", pt["abs_end_s"],
            set_=set_num, point_in_set=point_in_set,
            score=pt["score"], video_file=pt["video_file"])

        last_pt = pt

    # Close the final set
    if last_pt is not None:
        _ev("set_end", last_pt["abs_end_s"],
            set_=current_set, score=last_pt["score"],
            video_file=last_pt["video_file"])

    # game_end at end of last point (or provided game duration)
    game_end_t = game_duration_s if game_duration_s > last_pt["abs_end_s"] else last_pt["abs_end_s"]
    _ev("game_end", game_end_t,
        score=last_pt["score"], video_file=last_pt["video_file"])

    _ORDER = {"game_start": 0, "set_start": 1, "point_start": 2,
              "point_end": 3, "set_end": 4, "game_end": 5}
    events.sort(key=lambda e: (e["t_s"], _ORDER.get(e["event"], 9)))
    for i, e in enumerate(events):
        e["seq"] = i

    return events


def export_event_log(
    game_key: str,
    events: List[dict],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write event log JSON + CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"event_log_{game_key}.json"
    csv_path  = output_dir / f"event_log_{game_key}.csv"

    data = {
        "game":         game_key,
        "generated_at": datetime.now().isoformat(),
        "event_count":  len(events),
        "events":       events,
    }
    json_path.write_text(json.dumps(data, indent=2))

    if events:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(events[0].keys()))
            writer.writeheader()
            writer.writerows(events)

    set_count   = len({e["set"] for e in events if e["event"] == "set_start"})
    point_count = sum(1 for e in events if e["event"] == "point_start")
    log.info(
        f"Event log → {json_path}  "
        f"({set_count} set(s), {point_count} point(s), {len(events)} total events)"
    )
    return json_path, csv_path


def export_event_srt(
    game_key: str,
    events: List[dict],
    output_dir: Path,
    game_label: str = "",
) -> Path:
    """
    Write a SubRip (.srt) subtitle file from an event log.

    Subtitle layout:
      - game_start / game_end   : 5-second display, centred label
      - set_start  / set_end    : 4-second display
      - point_start → point_end : spans the full rally duration
      - point_end               : 2-second "score flash" right after each rally
    """
    def _srt_ts(t_s: float) -> str:
        ms  = int(round((t_s % 1) * 1000))
        s   = int(t_s)
        h, r = divmod(s, 3600)
        m, sec = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

    # Build a lookup: for each point_start find its matching point_end
    by_set_pt: dict = {}
    for e in events:
        if e["event"] in ("point_start", "point_end"):
            key = (e["set"], e["point_in_set"])
            by_set_pt.setdefault(key, {})[e["event"]] = e

    entries: List[Tuple[float, float, str]] = []   # (start_s, end_s, text)

    # One entry per rally spanning its full duration
    for key, pair in by_set_pt.items():
        ps = pair.get("point_start")
        pe = pair.get("point_end")
        if ps and pe:
            score_part = f"  |  {pe['score']}" if pe["score"] and pe["score"] != "unknown" else ""
            label = f"Set {ps['set']}  ·  Point {ps['point_in_set']}{score_part}"
            entries.append((ps["t_s"], pe["t_s"], label))
            # Brief score flash for 2 s after rally ends
            if pe["score"] and pe["score"] != "unknown":
                entries.append((pe["t_s"], pe["t_s"] + 2.0, pe["score"]))

    # Set and game boundary markers
    for e in events:
        if e["event"] == "game_start":
            title = game_label if game_label else game_key
            entries.append((e["t_s"], e["t_s"] + 5.0, f"GAME START\n{title}"))
        elif e["event"] == "set_start":
            entries.append((e["t_s"], e["t_s"] + 4.0, f"SET {e['set']}"))
        elif e["event"] == "set_end":
            score_part = f"  {e['score']}" if e["score"] and e["score"] != "unknown" else ""
            entries.append((e["t_s"], e["t_s"] + 4.0, f"END OF SET {e['set']}{score_part}"))
        elif e["event"] == "game_end":
            score_part = f"  {e['score']}" if e["score"] and e["score"] != "unknown" else ""
            entries.append((e["t_s"], e["t_s"] + 5.0, f"GAME OVER{score_part}"))

    entries.sort(key=lambda x: x[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    srt_path = output_dir / f"event_log_{game_key}.srt"

    lines: List[str] = []
    for idx, (start_s, end_s, text) in enumerate(entries, start=1):
        lines.append(str(idx))
        lines.append(f"{_srt_ts(start_s)} --> {_srt_ts(end_s)}")
        lines.append(text)
        lines.append("")

    srt_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Event SRT  → {srt_path}  ({len(entries)} subtitle entries)")
    return srt_path


# ─────────────────────────────────────────────────────────────────────────────
# Index export
# ─────────────────────────────────────────────────────────────────────────────

def export_index(
    player_number: int,
    match_date: str,
    club: str,
    videos_scanned: List[str],
    total_footage_s: float,
    segments: List[Segment],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write JSON + CSV index files. Returns (json_path, csv_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total_play_s = sum(s.duration for s in segments)

    rows = [
        {
            "play": i + 1,
            "video_file": s.video_file,
            "start": _fmt_time(s.start_time),
            "end": _fmt_time(s.end_time),
            "duration_s": round(s.duration, 1),
            "abs_start": _fmt_time(s.abs_start),
            "abs_end": _fmt_time(s.abs_end),
            "detection_count": len(s.detections),
            "avg_confidence": round(
                sum(d.confidence for d in s.detections) / len(s.detections), 3
            ) if s.detections else 0.0,
        }
        for i, s in enumerate(segments)
    ]

    data = {
        "player_number": player_number,
        "game": match_date,
        "club": club,
        "generated_at": datetime.now().isoformat(),
        "videos_scanned": videos_scanned,
        "total_footage_seconds": round(total_footage_s, 1),
        "total_footage_formatted": _fmt_time(total_footage_s),
        "total_play_seconds": round(total_play_s, 1),
        "total_play_formatted": _fmt_time(total_play_s),
        "segment_count": len(segments),
        "segments": rows,
    }

    # Sanitize club name for use in filenames (spaces → underscores, drop special chars)
    club_slug = re.sub(r"[^\w]", "_", club).strip("_").lower()
    base = f"player_{player_number}_{match_date}_{club_slug}"
    json_path = output_dir / f"{base}_index.json"
    csv_path = output_dir / f"{base}_index.csv"

    json_path.write_text(json.dumps(data, indent=2))
    log.info(f"Index (JSON) → {json_path}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["play", "video_file", "start", "end", "duration_s",
                        "abs_start", "abs_end", "detection_count", "avg_confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Index (CSV)  → {csv_path}")

    return json_path, csv_path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit("ffmpeg not found. Install it: sudo apt install ffmpeg")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Volleyball player highlight reel generator for DJI footage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help='Directory of DJI MP4 files',
    )
    parser.add_argument(
        "--player-number", "-p",
        type=int,
        default=None,
        help="Jersey number of the player to track (e.g. 12). Optional when only --rally-cut is used.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./volleyball_output",
        help="Output directory for index files and highlight reel (default: ./volleyball_output)",
    )
    parser.add_argument(
        "--match-date",
        default=None,
        help="Only process games from this date (YYYYMMDD). Omit to process all dates.",
    )
    parser.add_argument(
        "--club",
        default=None,
        help=(
            "Manually specify the club name instead of auto-detecting from video. "
            "Use this when the banner/text isn't visible in the footage. "
            "Example: --club \"Academy Volleyball Club\""
        ),
    )
    parser.add_argument(
        "--seq-gap",
        type=int,
        default=2,
        help=(
            "DJI sequence-number gap that marks the start of a new game (default: 2). "
            "E.g. files _0003_ and _0006_ have a gap of 3, which triggers a new game "
            "when --seq-gap is 2."
        ),
    )
    parser.add_argument(
        "--stitch",
        action="store_true",
        help=(
            "Concatenate each game's files into one MP4 before scanning. "
            "Fast (stream copy, no re-encode) but produces large output files."
        ),
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=1.0,
        help="Seconds between sampled frames (default: 1.0). Higher = faster but may miss plays.",
    )
    parser.add_argument(
        "--clip-padding",
        type=float,
        default=4.0,
        help="Seconds of context around each detected play (default: 4.0)",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=8.0,
        help="Max seconds between detections before splitting into a new segment (default: 8.0)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Min OCR confidence [0-1] to accept a jersey number match (default: 0.4)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel scan workers (default: 1). "
            "Each worker loads its own YOLO+EasyOCR and scans a different chunk of the video. "
            "Recommended: 4 on an 8-core machine. Only applies when --stitch is used."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for EasyOCR inference (requires CUDA)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Build a single compiled video of ALL games with chapter markers. "
            "Dead time between games is cut; each game becomes a chapter you can jump to. "
            "Stream copy (lossless, fast). Can be combined with player scanning."
        ),
    )
    parser.add_argument(
        "--rally-cut",
        action="store_true",
        help=(
            "Detect rally periods via motion and cut dead time between points. "
            "Produces a rally compilation video with one chapter per point, "
            "plus a JSON/CSV point timeline. Optionally reads score from the "
            "scoreboard if visible (auto-detected or set via --scoreboard-roi)."
        ),
    )
    parser.add_argument(
        "--score-overlay",
        action="store_true",
        help=(
            "Burn a 'Set N  X-Y' label into the rally compilation video "
            "(re-encodes; slower). Without this flag the same labels are "
            "written as a sidecar .ass subtitle file that VLC auto-loads."
        ),
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=15.0,
        help=(
            "90th-percentile pixel-diff threshold (on 320×180 downsampled frame) "
            "to classify a frame as active play (default: 15.0). "
            "Raise if dead time is triggering false rallies; lower if rallies are missed. "
            "Typical range for volleyball drone footage: 10–25."
        ),
    )
    parser.add_argument(
        "--court-roi",
        default=None,
        help=(
            "Court region as 'x,y,w,h' pixels in the source frame. Motion is "
            "measured only inside it, so benches, spectators, and adjacent "
            "courts don't trigger false rallies. Omit to use the full frame."
        ),
    )
    parser.add_argument(
        "--whistle-gate",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Drop rally segments whose start has no detected whistle within "
            "this many seconds (rallies start on the serve whistle). "
            "Good starting value: 4. Default 0 = off."
        ),
    )
    parser.add_argument(
        "--min-rally",
        type=float,
        default=3.0,
        help="Minimum rally duration in seconds to include (default: 3.0)",
    )
    parser.add_argument(
        "--min-dead",
        type=float,
        default=3.0,
        help="Minimum dead-time gap in seconds before starting a new rally segment (default: 3.0)",
    )
    parser.add_argument(
        "--pregame-skip",
        type=float,
        default=0,
        help=(
            "Skip this many minutes at the start of each game to ignore warmup/pregame. "
            "If 0 (default), game start is auto-detected from the first 0-0 scoreboard reading."
        ),
    )
    parser.add_argument(
        "--ball-confirm",
        action="store_true",
        help=(
            "After motion-based rally detection, filter out segments where the ball "
            "is not visible in at least --min-ball-pct of sampled frames. "
            "Reduces false positives from camera movement and dead-ball periods."
        ),
    )
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=0.2,
        help="Ball detection confidence threshold for --ball-confirm (default: 0.2)",
    )
    parser.add_argument(
        "--min-ball-pct",
        type=float,
        default=0.15,
        help=(
            "Minimum fraction of frames that must contain a ball to keep a rally segment "
            "when --ball-confirm is set (default: 0.15 = 15%%)"
        ),
    )
    parser.add_argument(
        "--scoreboard-roi",
        default=None,
        help=(
            "Scoreboard region as 'x,y,w,h' pixels for score OCR "
            "(e.g. '100,0,400,80'). Omit to auto-detect from top/bottom strips."
        ),
    )
    parser.add_argument(
        "--no-highlights",
        action="store_true",
        help="Skip creating highlight reel video (build index only)",
    )
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="Zoom each highlight clip into the detected player position",
    )
    parser.add_argument(
        "--zoom-factor",
        type=float,
        default=0.35,
        help="Fraction of frame width used as crop window when --zoom is set (default: 0.35)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=22,
        help="H.264 CRF for highlight reel quality (18=high, 28=low, default: 22)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    _check_ffmpeg()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover and group videos into games by sequence-number proximity
    game_groups = discover_videos(args.input_dir, seq_gap=args.seq_gap)

    # Filter to a specific date if requested (keeps all games on that date)
    if args.match_date:
        game_groups = {k: v for k, v in game_groups.items() if k.startswith(args.match_date)}
        if not game_groups:
            sys.exit(
                f"No games found for date {args.match_date}. "
                f"Available dates: {sorted({k[:8] for k in game_groups})}"
            )

    # 2. Optionally build compiled full-game video with chapters
    if args.compile:
        log.info("\nBuilding compiled video with chapters ...")
        compiled_path = output_dir / "compiled_all_games.mp4"
        create_compiled_video(game_groups, compiled_path)

    if args.player_number is not None:
        detector = PlayerDetector(
            args.player_number,
            confidence_threshold=args.confidence,
            gpu=args.gpu,
        )
    else:
        # No player tracking — load EasyOCR only (needed for scoreboard / club detection)
        try:
            import easyocr as _easyocr
        except ImportError:
            sys.exit("easyocr is required: pip install easyocr")

        class _MinimalDetector:
            """Stub that loads EasyOCR + YOLOv8n without player scanning."""
            def __init__(self, gpu: bool):
                log.info("Loading EasyOCR (no player scanning) ...")
                self.reader = _easyocr.Reader(["en"], gpu=gpu)
                log.info("Loading YOLOv8n (for ball confirmation) ...")
                from ultralytics import YOLO as _YOLO
                self.yolo = _YOLO("yolov8n.pt")

        detector = _MinimalDetector(gpu=args.gpu)

    # 3. Process each game independently for player highlights
    for game_key, video_files in sorted(game_groups.items()):
        log.info(f"\n{'='*60}")
        log.info(f"Processing game: {game_key}  ({len(video_files)} file(s))")
        log.info(f"{'='*60}")

        game_output = output_dir / game_key
        game_output.mkdir(parents=True, exist_ok=True)

        # filename → full path (for highlight extraction)
        video_map: Dict[str, Path] = {v.name: v for v in video_files}

        # 2a. Detect which club this game is for
        if args.club:
            club = args.club
            log.info(f"Club (manual): {club}")
        else:
            log.info(f"Detecting club from opening frames of {video_files[0].name} ...")
            club = detect_club(video_files[0], detector.reader)

        # Optionally stitch the game's files into one MP4 first
        if args.stitch:
            stitched = game_output / f"stitched_{game_key}.mp4"
            stitch_videos(video_files, stitched)
            scan_sources = [stitched]
            video_map[stitched.name] = stitched
        else:
            scan_sources = video_files

        # 3. Player scanning (skipped when --player-number not provided)
        segments: List[Segment] = []
        total_footage_s = 0.0

        if args.player_number is not None:
            all_detections: List[Detection] = []
            cumulative_offset = 0.0

            for src in scan_sources:
                if args.workers > 1:
                    dets = parallel_scan(
                        src,
                        player_number=args.player_number,
                        confidence=args.confidence,
                        gpu=args.gpu,
                        scan_interval=args.scan_interval,
                        time_offset=cumulative_offset,
                        n_workers=args.workers,
                    )
                else:
                    dets = scan_video(
                        src,
                        detector,
                        scan_interval=args.scan_interval,
                        time_offset=cumulative_offset,
                    )
                all_detections.extend(dets)
                try:
                    cumulative_offset += get_video_duration(src)
                except Exception:
                    pass

            total_footage_s = cumulative_offset

            # 4. Build segments
            segments = build_segments(
                all_detections,
                gap_threshold=args.gap_threshold,
                clip_padding=args.clip_padding,
            )

            log.info(
                f"\nPlayer #{args.player_number} in {game_key}: "
                f"{len(segments)} segment(s) / {len(all_detections)} detection(s)"
            )
            if segments:
                total_play = sum(s.duration for s in segments)
                log.info(
                    f"  Total detected play time: {_fmt_time(total_play)}  "
                    f"of {_fmt_time(total_footage_s)} footage"
                )
                for i, s in enumerate(segments):
                    log.info(
                        f"    [{i+1:>3}] {s.video_file}  "
                        f"{_fmt_time(s.start_time)} → {_fmt_time(s.end_time)}  "
                        f"({s.duration:.0f}s)  detections: {len(s.detections)}"
                    )
            else:
                log.info(
                    f"  Player #{args.player_number} NOT detected in {game_key}.\n"
                    f"  Possible reasons:\n"
                    f"    • Player sat out this game\n"
                    f"    • Camera angle too high / players too small for OCR\n"
                    f"    • Try --confidence 0.25 or --scan-interval 0.5"
                )

            # 5. Export index
            videos_scanned = [v.name for v in (scan_sources if args.stitch else video_files)]
            export_index(
                player_number=args.player_number,
                match_date=game_key,
                club=club,
                videos_scanned=videos_scanned,
                total_footage_s=total_footage_s,
                segments=segments,
                output_dir=game_output,
            )

            # 6. Create highlight reel
            if not args.no_highlights and segments:
                club_slug = re.sub(r"[^\w]", "_", club).strip("_").lower()
                reel_path = game_output / f"player_{args.player_number}_{game_key}_{club_slug}_highlights.mp4"
                create_highlight_reel(
                    video_map=video_map,
                    segments=segments,
                    output_path=reel_path,
                    player_number=args.player_number,
                    crf=args.crf,
                    zoom=args.zoom,
                    zoom_factor=args.zoom_factor,
                )

        # 7. Rally cut — detect rally periods, build point timeline + compilation
        if args.rally_cut:
            log.info(f"\nRally detection for {game_key} ...")

            court_roi: Optional[Tuple[int, int, int, int]] = None
            if args.court_roi:
                try:
                    court_roi = tuple(int(x) for x in args.court_roi.split(","))  # type: ignore
                    if len(court_roi) != 4:
                        raise ValueError
                except ValueError:
                    log.warning(f"  Invalid --court-roi '{args.court_roi}' (need x,y,w,h) — using full frame")
                    court_roi = None

            # Parse or auto-detect scoreboard ROI (once per game)
            scoreboard_roi: Optional[Tuple[int, int, int, int]] = None
            if args.scoreboard_roi:
                try:
                    scoreboard_roi = tuple(int(x) for x in args.scoreboard_roi.split(","))  # type: ignore
                    log.info(f"  Scoreboard ROI (manual): {scoreboard_roi}")
                except ValueError:
                    log.warning(f"  Invalid --scoreboard-roi '{args.scoreboard_roi}', will auto-detect")
            if scoreboard_roi is None:
                scoreboard_roi = auto_detect_scoreboard_roi(
                    scan_sources[0], detector.reader
                )

            # Determine pregame offset — manual override takes precedence,
            # otherwise auto-detect from the first 0-0 scoreboard reading.
            game_start_s = 0.0
            if args.pregame_skip > 0:
                game_start_s = args.pregame_skip * 60.0
                log.info(f"  Pregame skip: {args.pregame_skip} min ({_fmt_time(game_start_s)})")
            elif scoreboard_roi:
                game_start_s = detect_game_start(
                    scan_sources[0], detector.reader, scoreboard_roi
                )

            all_rally_segs: List[Tuple[float, float]] = []
            all_points: List[dict] = []
            cumulative_offset = 0.0

            for i, src in enumerate(scan_sources):
                # Only apply pregame skip to the first file; game is underway in subsequent files
                file_start_offset = game_start_s if i == 0 else 0.0
                segs, pts = detect_rallies(
                    video_path=src,
                    motion_threshold=args.motion_threshold,
                    scan_interval=0.5,
                    min_rally_s=args.min_rally,
                    min_dead_s=args.min_dead,
                    pre_pad=1.5,
                    post_pad=2.0,
                    reader=detector.reader,
                    scoreboard_roi=scoreboard_roi,
                    time_offset=cumulative_offset,
                    start_offset_s=file_start_offset,
                    court_roi=court_roi,
                    whistle_gate_s=args.whistle_gate,
                )
                # Re-number points across all files in this game
                for pt in pts:
                    pt["point"] = len(all_points) + pt["point"]
                all_rally_segs.extend(segs)
                all_points.extend(pts)
                try:
                    cumulative_offset += get_video_duration(src)
                except Exception:
                    pass

            # Optionally filter out false-positive segments with no ball visible
            if args.ball_confirm and all_points:
                log.info(f"\nFiltering {len(all_points)} segments with ball detection ...")
                all_points = filter_rallies_by_ball(
                    video_map=video_map,
                    point_timeline=all_points,
                    yolo_model=detector.yolo,
                    conf=args.ball_conf,
                    min_ball_pct=args.min_ball_pct,
                )
                # Re-number after filtering
                for idx, pt in enumerate(all_points):
                    pt["point"] = idx + 1

            export_point_timeline(game_key, all_points, game_output)

            event_log = build_event_log(all_points, game_duration_s=cumulative_offset)
            export_event_log(game_key, event_log, game_output)
            export_event_srt(
                game_key,
                event_log,
                game_output,
                game_label=f"{game_key}  ({club})" if 'club' in dir() else game_key,
            )

            rally_path = game_output / f"rally_compilation_{game_key}.mp4"
            create_rally_compilation(
                video_map=video_map,
                point_timeline=all_points,
                output_path=rally_path,
                crf=args.crf,
                score_overlay=args.score_overlay,
            )

    log.info("\nAll done.")
    log.info(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

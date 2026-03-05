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
    strip_height: int = 100,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Sample frames from the middle of the video and find which horizontal strip
    (top or bottom) most consistently contains a volleyball score.
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

    votes: Counter = Counter()
    log.info("  Auto-detecting scoreboard region ...")

    frame_idx = start_f
    while frame_idx < end_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        for label, y0 in [("top", 0), ("bottom", frame_h - strip_height)]:
            strip = frame[y0: y0 + strip_height, :]
            texts = reader.readtext(strip, detail=0, allowlist="0123456789-: ")
            if _parse_volleyball_score(texts):
                votes[label] += 1
        frame_idx += step

    cap.release()

    if not votes:
        log.info("  Scoreboard not auto-detected in top/bottom strips — scores will be unknown")
        return None

    best, count = votes.most_common(1)[0]
    y0 = 0 if best == "top" else frame_h - strip_height
    log.info(f"  Scoreboard detected in {best} strip ({count}/{n_samples} frames)")
    return (0, y0, frame_w, strip_height)


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
) -> Tuple[List[Tuple[float, float]], List[dict]]:
    """
    Detect volleyball rally periods using frame-to-frame motion.

    motion_threshold: mean absolute pixel diff (on 320×180 downsampled frame)
                      that separates active play from dead time.
    Returns (segments, point_timeline) where timestamps are relative to this file.
    point_timeline entries also carry abs_start using time_offset.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        log.error("opencv-python required for rally detection")
        return [], []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps
    frame_step   = max(1, int(fps * scan_interval))

    log.info(f"  Detecting rallies: {video_path.name} ({_fmt_time(duration)}, every {scan_interval}s)")

    motion_data: List[Tuple[float, float]] = []
    prev_gray = None
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (320, 180))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None:
            motion = float(np.mean(np.abs(gray - prev_gray)))
            motion_data.append((frame_idx / fps, motion))
        prev_gray = gray
        frame_idx += frame_step

    cap.release()

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

    # Pad, merge short gaps, filter short rallies
    padded = [(max(0.0, s - pre_pad), min(duration, e + post_pad)) for s, e in raw_segs]
    merged: List[List[float]] = []
    for s, e in sorted(padded):
        if e - s < min_rally_s:
            continue
        if merged and s < merged[-1][1] + min_dead_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    segments = [(s, e) for s, e in merged]

    log.info(f"  → {len(segments)} rally segment(s) detected")

    # Build point timeline, optionally reading score after each rally
    point_timeline: List[dict] = []
    if segments:
        cap2 = cv2.VideoCapture(str(video_path))
        for i, (start_s, end_s) in enumerate(segments):
            score: Optional[Tuple[int, int]] = None
            if reader and scoreboard_roi:
                # Read score just after the rally ends (score updates after point)
                check_t = min(end_s + 1.0, duration - 0.1)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, int(check_t * fps))
                ret, frame = cap2.read()
                if ret:
                    score = _read_score(frame, reader, scoreboard_roi)
            score_str = f"{score[0]}-{score[1]}" if score else "unknown"
            log.info(
                f"    [Point {i+1:>3}] {_fmt_time(start_s)} → {_fmt_time(end_s)}"
                f"  ({end_s-start_s:.0f}s)  score: {score_str}"
            )
            point_timeline.append({
                "point":      i + 1,
                "video_file": video_path.name,
                "start":      _fmt_time(start_s),
                "end":        _fmt_time(end_s),
                "abs_start":  _fmt_time(time_offset + start_s),
                "start_s":    round(start_s, 2),
                "end_s":      round(end_s, 2),
                "duration_s": round(end_s - start_s, 1),
                "score":      score_str,
            })
        cap2.release()

    return segments, point_timeline


def create_rally_compilation(
    video_map: Dict[str, Path],
    point_timeline: List[dict],
    output_path: Path,
    crf: int = 22,
) -> Optional[Path]:
    """
    Cut every rally segment from its source file, concatenate into a single
    video, and embed one MP4 chapter per point (labelled with score if known).
    Dead time between points is completely removed.
    """
    if not point_timeline:
        log.warning("No rally segments — compilation not created.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_paths: List[Path] = []
        compiled_s = 0.0
        chapter_data: List[Tuple[float, str]] = []   # (start_in_compiled, title)

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

        # Inject chapters
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", str(tmp_concat),
            "-f", "ffmetadata", "-i", str(meta_path),
            "-map", "0", "-map_metadata", "1",
            "-c", "copy",
            str(output_path),
        ], capture_output=True, text=True)

        if result.returncode != 0:
            log.warning("Chapter injection failed; saving without chapters")
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

    data = {
        "game":           game_key,
        "generated_at":   datetime.now().isoformat(),
        "point_count":    len(point_timeline),
        "score_readable": sum(1 for p in point_timeline if p["score"] != "unknown"),
        "points":         point_timeline,
    }
    json_path.write_text(json.dumps(data, indent=2))

    if point_timeline:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(point_timeline[0].keys()))
            writer.writeheader()
            writer.writerows(point_timeline)

    log.info(f"Point timeline → {json_path}  ({data['score_readable']}/{len(point_timeline)} scores read)")
    return json_path, csv_path


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
        required=True,
        help="Jersey number of the player to track (e.g. 12)",
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
        "--motion-threshold",
        type=float,
        default=3.0,
        help=(
            "Mean pixel-diff threshold (on 320×180 frame) to classify a frame "
            "as 'active play' for rally detection (default: 3.0). "
            "Raise if static backgrounds trigger false rallies; lower if rallies are missed."
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

    detector = PlayerDetector(
        args.player_number,
        confidence_threshold=args.confidence,
        gpu=args.gpu,
    )

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

        # 3. Scan, tracking cumulative time offset across files
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

            all_rally_segs: List[Tuple[float, float]] = []
            all_points: List[dict] = []
            cumulative_offset = 0.0

            for src in scan_sources:
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

            export_point_timeline(game_key, all_points, game_output)

            rally_path = game_output / f"rally_compilation_{game_key}.mp4"
            create_rally_compilation(
                video_map=video_map,
                point_timeline=all_points,
                output_path=rally_path,
                crf=args.crf,
            )

    log.info("\nAll done.")
    log.info(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

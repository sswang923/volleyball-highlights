#!/usr/bin/env python3
"""
Frame Extractor for YOLOv8 Training Data

Pulls evenly-spaced frames from DJI volleyball footage into a folder
ready for labeling in Roboflow or CVAT.

Usage:
    # Extract 1 frame every 2 seconds from all MP4s in a folder:
    python extract_frames.py -i "D:\\volleyball videos" -o frames_to_label

    # Extract more densely (1 frame/sec) — better for fast ball motion:
    python extract_frames.py -i "D:\\volleyball videos" -o frames_to_label --interval 1.0

    # Cap total frames to avoid labeling overload:
    python extract_frames.py -i "D:\\volleyball videos" -o frames_to_label --max-frames 800

    # Only extract from a specific date (DJI naming convention):
    python extract_frames.py -i "D:\\volleyball videos" -o frames_to_label --match-date 20251018

    # Skip blurry frames (useful for fast volleyball motion):
    python extract_frames.py -i "D:\\volleyball videos" -o frames_to_label --skip-blurry
"""

import argparse
import cv2
import os
import sys
import re
import random
from pathlib import Path


def is_blurry(frame, threshold=80.0):
    """Return True if the frame is too blurry to be useful for labeling."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def extract_frames(
    input_dir: str,
    output_dir: str,
    interval: float = 2.0,
    max_frames: int = 0,
    match_date: str = None,
    skip_blurry: bool = False,
    blur_threshold: float = 80.0,
    shuffle: bool = True,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all MP4s
    mp4_files = sorted(input_path.rglob("*.MP4")) + sorted(input_path.rglob("*.mp4"))

    if match_date:
        mp4_files = [f for f in mp4_files if match_date in f.name]

    if not mp4_files:
        print(f"[!] No MP4 files found in: {input_dir}")
        sys.exit(1)

    print(f"[+] Found {len(mp4_files)} MP4 file(s)")
    if match_date:
        print(f"    Filtered to date: {match_date}")

    if shuffle:
        random.shuffle(mp4_files)

    total_saved = 0
    total_skipped_blur = 0

    for mp4 in mp4_files:
        if max_frames and total_saved >= max_frames:
            break

        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            print(f"  [!] Could not open: {mp4.name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps
        frame_step = max(1, int(fps * interval))

        print(f"  Processing: {mp4.name}  ({duration_s:.0f}s, {fps:.1f}fps, step={frame_step})")

        frame_idx = 0
        file_saved = 0

        while True:
            if max_frames and total_saved >= max_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            if skip_blurry and is_blurry(frame, blur_threshold):
                total_skipped_blur += 1
                frame_idx += frame_step
                continue

            # Build output filename: <source_stem>_f<frame_idx>.jpg
            stem = re.sub(r"[^\w\-]", "_", mp4.stem)
            out_name = f"{stem}_f{frame_idx:07d}.jpg"
            out_path = output_path / out_name

            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            total_saved += 1
            file_saved += 1
            frame_idx += frame_step

        cap.release()
        print(f"    Saved {file_saved} frames from this file")

    print(f"\n[+] Done. Total frames saved : {total_saved}")
    if skip_blurry:
        print(f"    Blurry frames skipped    : {total_skipped_blur}")
    print(f"    Output directory         : {output_path.resolve()}")
    print()
    print("Next steps:")
    print("  1. Upload the frames folder to https://roboflow.com (free account)")
    print("  2. Create a new project → Object Detection")
    print("  3. Label class 'ball' on every volleyball visible in each frame")
    print("  4. Export dataset in 'YOLOv8' format and download the zip")
    print("  5. Run: python train_yolov8.py --data path/to/dataset.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="Extract labeled-training frames from DJI volleyball footage"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Directory containing DJI MP4 files"
    )
    parser.add_argument(
        "-o", "--output", default="frames_to_label",
        help="Output directory for extracted frames (default: frames_to_label)"
    )
    parser.add_argument(
        "--interval", type=float, default=2.0,
        help="Seconds between extracted frames (default: 2.0). "
             "Use 1.0 for denser coverage, 3.0+ for speed."
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after saving this many frames total (0 = unlimited)"
    )
    parser.add_argument(
        "--match-date", type=str, default=None,
        help="Only process files containing this string in the filename (e.g. 20251018)"
    )
    parser.add_argument(
        "--skip-blurry", action="store_true",
        help="Skip frames that are too blurry to label accurately"
    )
    parser.add_argument(
        "--blur-threshold", type=float, default=80.0,
        help="Laplacian variance below this = blurry (default: 80.0, lower = keep more)"
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Process files in alphabetical order instead of random order"
    )

    args = parser.parse_args()

    extract_frames(
        input_dir=args.input,
        output_dir=args.output,
        interval=args.interval,
        max_frames=args.max_frames,
        match_date=args.match_date,
        skip_blurry=args.skip_blurry,
        blur_threshold=args.blur_threshold,
        shuffle=not args.no_shuffle,
    )


if __name__ == "__main__":
    main()

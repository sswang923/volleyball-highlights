#!/bin/bash
# ============================================================
# Upload Jun 19-21 2026 games to YouTube (unlisted)
#
# Games detected by dji_stitch.py (timestamp grouping, >= 2026-06-19).
# Multi-file games are pre-stitched to stitched_*.mp4; single-file games
# upload their original DJI file directly.
# Titles are "Game 1".."Game 14" (rename later in YouTube Studio).
#
# Requires a valid YouTube token (run: python3 auth_youtube.py first).
# Usage:  bash upload_20260622.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# path | description  (title is "Game <index>")
GAMES=(
  "/mnt/d/volleyball/DJI_20260619075644_0075_D.MP4|Jun 19 2026 07:56-08:36"
  "/mnt/d/volleyball/DJI_20260619104804_0076_D.MP4|Jun 19 2026 10:48-11:27"
  "/mnt/d/volleyball/stitched_20260619_game03.mp4|Jun 19 2026 11:40-12:32"
  "/mnt/d/volleyball/DJI_20260619160221_0079_D.MP4|Jun 19 2026 16:02-16:35"
  "/mnt/d/volleyball/stitched_20260619_game05.mp4|Jun 19 2026 17:48-18:45"
  "/mnt/d/volleyball/stitched_20260619_game06.mp4|Jun 19 2026 20:12-21:11"
  "/mnt/d/volleyball/stitched_20260620_game01.mp4|Jun 20 2026 08:03-08:52"
  "/mnt/d/volleyball/stitched_20260620_game02.mp4|Jun 20 2026 10:56-11:41"
  "/mnt/d/volleyball/DJI_20260620115354_0088_D.MP4|Jun 20 2026 11:53-12:37"
  "/mnt/d/volleyball/stitched_20260620_game04.mp4|Jun 20 2026 14:59-16:02"
  "/mnt/d/volleyball/stitched_20260620_game05.mp4|Jun 20 2026 17:15-18:12"
  "/mnt/d/volleyball/stitched_20260620_game06.mp4|Jun 20 2026 20:45-21:44"
  "/mnt/d/volleyball/DJI_20260621100052_0095_D.MP4|Jun 21 2026 10:00-10:40"
  "/mnt/d/volleyball/stitched_20260621_game02.mp4|Jun 21 2026 12:39-13:29"
)

echo "========================================"
echo "  Volleyball Upload — Jun 19-21 2026 (${#GAMES[@]} games)"
echo "========================================"

n=0
for entry in "${GAMES[@]}"; do
  n=$((n+1))
  path="${entry%%|*}"
  desc="${entry##*|}"
  echo ""
  echo "[$n/${#GAMES[@]}] Uploading Game $n  ($desc)"
  # Wait up to 90 min for a still-stitching file to appear and stop growing.
  waited=0
  while [ ! -f "$path" ] && [ $waited -lt 5400 ]; do
    echo "    waiting for $path ..."; sleep 30; waited=$((waited+30))
  done
  if [ ! -f "$path" ]; then
    echo "    !! MISSING after wait: $path  — skipping"
    continue
  fi
  # ensure stitch finished writing: wait until size is stable
  last=-1; cur=$(stat -c%s "$path")
  while [ "$cur" != "$last" ]; do
    last=$cur; sleep 15; cur=$(stat -c%s "$path")
  done
  python3 "$SCRIPT_DIR/youtube_upload.py" \
    -v "$path" \
    --title "Game $n" \
    --description "$desc"
done

echo ""
echo "========================================"
echo "  Done — ${#GAMES[@]} games processed."
echo "========================================"

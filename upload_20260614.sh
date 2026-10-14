#!/bin/bash
# ============================================================
# Upload today's games (2026-06-14) to YouTube
#
# Team: Bay 11-1
# Games (each a single continuous file — no stitching needed):
#   Game 1: DJI_20260614090232_0072_D.MP4  (Bay 11-1 vs Bay 12 Summer)
#   Game 2: DJI_20260614110255_0073_D.MP4  (Bay 11-1 vs Bay 12-P)
#   Game 3: DJI_20260614133627_0074_D.MP4  (Bay 11-1 vs Bay 14 Summer)
#
# Usage:  bash upload_20260614.sh
# ============================================================
set -e

VIDEO_DIR="/mnt/d/volleyball"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  Volleyball Upload — 2026-06-14"
echo "========================================"

# ── Game 1 ──────────────────────────────────────────────────
echo ""
echo "[1/3] Uploading Game 1: Bay 11-1 vs Bay 12 Summer..."
python3 "$SCRIPT_DIR/youtube_upload.py" \
  -v "$VIDEO_DIR/DJI_20260614090232_0072_D.MP4" \
  --title "Bay 11-1 vs Bay 12 Summer — 2026-06-14 Game 1" \
  --description "Bay 11-1 vs Bay 12 Summer, June 14 2026"

# ── Game 2 ──────────────────────────────────────────────────
echo ""
echo "[2/3] Uploading Game 2: Bay 11-1 vs Bay 12-P..."
python3 "$SCRIPT_DIR/youtube_upload.py" \
  -v "$VIDEO_DIR/DJI_20260614110255_0073_D.MP4" \
  --title "Bay 11-1 vs Bay 12-P — 2026-06-14 Game 2" \
  --description "Bay 11-1 vs Bay 12-P, June 14 2026"

# ── Game 3 ──────────────────────────────────────────────────
echo ""
echo "[3/3] Uploading Game 3: Bay 11-1 vs Bay 14 Summer..."
python3 "$SCRIPT_DIR/youtube_upload.py" \
  -v "$VIDEO_DIR/DJI_20260614133627_0074_D.MP4" \
  --title "Bay 11-1 vs Bay 14 Summer — 2026-06-14 Game 3" \
  --description "Bay 11-1 vs Bay 14 Summer, June 14 2026"

echo ""
echo "========================================"
echo "  All 3 games uploaded!"
echo "========================================"

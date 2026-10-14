#!/bin/bash
# ============================================================
# Upload today's games (2026-04-12) to YouTube
#
# Games:
#   Game 1: DJI_20260412090355_0034_D.MP4  (Academy 13-1 vs Bay to Bay National)
#   Game 2: DJI_20260412095150_0035_D.MP4  (Academy 13-1 vs Eclipse 14/13)
#   Game 3: DJI_20260412123327_0036_D.MP4 + DJI_20260412124719_0037_D.MP4
#           (Academy 13-1 vs Academy 14-1 N) — needs merge first
#
# Usage:  bash upload_20260412.sh
# ============================================================
set -e

VIDEO_DIR="/mnt/d/volleyball"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MERGED_FILE="$VIDEO_DIR/stitched_20260412_game03.mp4"

echo "========================================"
echo "  Volleyball Upload — 2026-04-12"
echo "========================================"

# ── Step 1: Merge Game 3 (0036 + 0037) ──────────────────────
echo ""
echo "[1/4] Merging Game 3 files (0036_D + 0037_D)..."

CONCAT_LIST=$(mktemp /tmp/concat_XXXXX.txt)
cat > "$CONCAT_LIST" << EOF
file '$VIDEO_DIR/DJI_20260412123327_0036_D.MP4'
file '$VIDEO_DIR/DJI_20260412124719_0037_D.MP4'
EOF

ffmpeg -y \
  -f concat -safe 0 \
  -i "$CONCAT_LIST" \
  -c copy \
  -video_track_timescale 90000 \
  -avoid_negative_ts make_zero \
  "$MERGED_FILE"

rm -f "$CONCAT_LIST"

MERGED_SIZE=$(du -h "$MERGED_FILE" | cut -f1)
echo "[+] Merge complete: $MERGED_FILE ($MERGED_SIZE)"

# ── Step 2: Upload Game 1 ───────────────────────────────────
echo ""
echo "[2/4] Uploading Game 1: Academy 13-1 vs Bay to Bay National..."
python3 "$SCRIPT_DIR/youtube_upload.py" \
  -v "$VIDEO_DIR/DJI_20260412090355_0034_D.MP4" \
  --title "Academy 13-1 vs Bay to Bay National — 2026-04-12 Game 1" \
  --description "Academy 13-1 vs Bay to Bay National, April 12 2026"

# ── Step 3: Upload Game 2 ───────────────────────────────────
echo ""
echo "[3/4] Uploading Game 2: Academy 13-1 vs Eclipse 14/13..."
python3 "$SCRIPT_DIR/youtube_upload.py" \
  -v "$VIDEO_DIR/DJI_20260412095150_0035_D.MP4" \
  --title "Academy 13-1 vs Eclipse 14/13 — 2026-04-12 Game 2" \
  --description "Academy 13-1 vs Eclipse 14/13, April 12 2026"

# ── Step 4: Upload Game 3 ───────────────────────────────────
echo ""
echo "[4/4] Uploading Game 3: Academy 13-1 vs Academy 14-1 N..."
python3 "$SCRIPT_DIR/youtube_upload.py" \
  -v "$MERGED_FILE" \
  --title "Academy 13-1 vs Academy 14-1 N — 2026-04-12 Game 3" \
  --description "Academy 13-1 vs Academy 14-1 N, April 12 2026"

echo ""
echo "========================================"
echo "  All 3 games uploaded!"
echo "========================================"

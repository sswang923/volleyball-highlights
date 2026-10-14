#!/bin/bash
# ============================================================
# Upload Oct 4 2026 games (Tyrants 16 Blue) to YouTube (unlisted)
#
# Games detected by dji_stitch.py --match-date 20261004 (4 games, 2 files each).
# Requires a valid YouTube token (run: python3 auth_youtube.py first).
# Usage:  bash upload_20261004.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEAM="Tyrants 16 Blue"
DATE="Oct 4 2026"

# path | description  (title is "$TEAM — $DATE — Game <index>")
GAMES=(
  "/mnt/d/volleyball/stitched_20261004_game01.mp4|Tyrants 16 Blue — Oct 4 2026, started 11:09"
  "/mnt/d/volleyball/stitched_20261004_game02.mp4|Tyrants 16 Blue — Oct 4 2026, started 12:29"
  "/mnt/d/volleyball/stitched_20261004_game03.mp4|Tyrants 16 Blue — Oct 4 2026, started 14:50"
  "/mnt/d/volleyball/stitched_20261004_game04.mp4|Tyrants 16 Blue — Oct 4 2026, started 17:41"
)

echo "========================================"
echo "  Volleyball Upload — $TEAM $DATE (${#GAMES[@]} games)"
echo "========================================"

n=0
for entry in "${GAMES[@]}"; do
  n=$((n+1))
  path="${entry%%|*}"
  desc="${entry##*|}"
  next="${GAMES[$n]%%|*}"
  echo ""
  echo "[$n/${#GAMES[@]}] Uploading Game $n  ($desc)"
  # Wait up to 90 min for a still-stitching file to appear.
  waited=0
  while [ ! -f "$path" ] && [ $waited -lt 5400 ]; do
    echo "    waiting for $path ..."; sleep 30; waited=$((waited+30))
  done
  if [ ! -f "$path" ]; then
    echo "    !! MISSING after wait: $path  — skipping"
    continue
  fi
  # Stitch is done when the next game's file appears or dji_stitch exits.
  while pgrep -f "dji_stitch.py" >/dev/null && { [ -z "$next" ] || [ ! -f "$next" ]; }; do
    sleep 30
  done
  python3 "$SCRIPT_DIR/youtube_upload.py" \
    -v "$path" \
    --title "$TEAM — $DATE — Game $n" \
    --description "$desc"
done

echo ""
echo "========================================"
echo "  Done — ${#GAMES[@]} games processed."
echo "========================================"

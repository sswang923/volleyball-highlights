#!/usr/bin/env bash
# Wrapper that sets malloc env vars before Python starts.
# Required on WSL2 to prevent a glibc double-free crash when
# PyTorch and OpenCV are both loaded in the same process.
export MALLOC_MMAP_THRESHOLD_=134217728
export MALLOC_TRIM_THRESHOLD_=-1
exec python3 "$(dirname "$0")/ball_tracker.py" "$@"

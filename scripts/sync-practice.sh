#!/usr/bin/env bash
set -euo pipefail

# 1. Define external temp dirs and logging
export UNISON_TMPDIR=/tmp/unison-tmp
export TMPDIR="$UNISON_TMPDIR"
export UNISON_NUMBACKUPS=1
mkdir -p "$UNISON_TMPDIR"

LOG="$HOME/sync-practice-debug.log"
ROOT1="/home/alfrizz/my_practice"
ROOT2="/mnt/g/My Drive/Ingegneria/Data Science GD/My-Practice"

echo "=== [$(date '+%F %T')] Starting Unison sync ===" >> "$LOG"

exec /usr/local/bin/unison \
  -root      "$ROOT1" \
  -root      "$ROOT2" \
  -batch \
  -auto \
  -times \
  -perms     0 \
  -maxbackups 1 \
  -confirmmerge=false \
  -prefer    newer \
  -links     true \
  -fastcheck true \
  -repeat    7 \
  -watch \
  -debug     all \
  -logfile   "$LOG" \
  -ignore    "Name .Trash-0" \
  -ignore    "Name .unison*" \
  -ignore    "Path **/.unison*/**" \
  -ignore    "Name .git" \
  -ignore    "Path .git/**" \
  -ignore    "Path **/.git/**" \
  -ignore    "Name __pycache__" \
  -ignore    "Path **/__pycache__/**" \
  -ignore    "Name .ipynb_checkpoints" \
  -ignore    "Path **/.ipynb_checkpoints/**" \
  -ignore    "Path **/*.log" \
  -ignore    "Regex .*[Cc]heckpoint.*"

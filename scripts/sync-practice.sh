#!/usr/bin/env bash
set -euo pipefail

# 1. Define external temp dirs and logging
export UNISON_TMPDIR=/tmp/unison-tmp
export TMPDIR="$UNISON_TMPDIR"
export UNISON_NUMBACKUPS=1
mkdir -p "$UNISON_TMPDIR"

LOG="$HOME/sync-practice-debug.log"

echo "=== [$(date '+%F %T')] Starting Unison sync ===" >> "$LOG"

exec unison \
  -batch \
  -auto \
  -times \
  -maxbackups 1 \
  -confirmmerge=false \
  -prefer    newer \
  -links     true \
  -fastcheck true \
  -repeat    7 \
  -watch \
  -debug     all \
  -logfile   "$LOG" \
  my_practice

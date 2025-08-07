#!/usr/bin/env bash
set -euo pipefail

# --- Config
LOG="$HOME/sync-practice.log"
LOCK="/tmp/sync-practice.lock"
TARGET="/mnt/g/My Drive/Ingegneria/Data Science GD/My-Practice"

# --- Single instance lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] Already running; exiting." >> "$LOG"
  exit 0
fi

# --- Temp dirs for Unison
export UNISON_TMPDIR=/tmp/unison-tmp
export TMPDIR="$UNISON_TMPDIR"
export UNISON_NUMBACKUPS=1
mkdir -p "$UNISON_TMPDIR"

echo "=== [$(date '+%F %T')] Boot trigger: starting Unison ===" >> "$LOG"

# --- Wait for Windows drive to be ready (up to 60s)
for i in {1..60}; do
  if mountpoint -q /mnt/g && [ -d "$TARGET" ]; then
    echo "[$(date '+%F %T')] G: ready and target exists." >> "$LOG"
    break
  fi
  (( i % 5 == 0 )) && echo "[$(date '+%F %T')] Waiting for G:... (${i}s)" >> "$LOG"
  if (( i == 60 )); then
    echo "[$(date '+%F %T')] Timeout: G: not ready. Exiting." >> "$LOG"
    exit 1
  fi
  sleep 1
done

# --- Run Unison
/usr/local/bin/unison \
  -batch \
  -auto \
  -times \
  -maxbackups 1 \
  -confirmmerge=false \
  -prefer newer \
  -links true \
  -fastcheck true \
  -repeat 7 \
  -logfile "$LOG" \
  my_practice

echo "[$(date '+%F %T')] Unison exited." >> "$LOG"

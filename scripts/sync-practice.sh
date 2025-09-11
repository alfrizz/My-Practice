#!/usr/bin/env bash
set -euo pipefail

LOCK="/tmp/sync-practice.lock"

TARGET="/mnt/g/My Drive/Ingegneria/Data Science GD/My-Practice"
LOG="$TARGET/scripts/sync-practice.log"

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

# ONE-TIME cleanup of any leftover Unison scratch files
rm -rf "${UNISON_TMPDIR}/"*

#   suppress “Permission denied” so find can’t kill the script
find "$HOME/my_practice" "$TARGET" \
  -type f -name '.unison.*.unison.tmp' -delete \
  2>/dev/null || true

echo "=== [$(date '+%F %T')] Boot trigger: starting Unison ===" >> "$LOG"

# --- Wait for Windows drive to be ready (up to 30sec)
for i in {1..30}; do
  if mountpoint -q /mnt/g && [ -d "$TARGET" ]; then
    echo "[$(date '+%F %T')] G: ready and target exists." >> "$LOG"
    break
  fi
  (( i % 5 == 0 )) && echo "[$(date '+%F %T')] Waiting for G:... (${i}s)" >> "$LOG"
  if (( i == 30 )); then
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
  -repeat 30 \
  -logfile "$LOG" \
  my_practice
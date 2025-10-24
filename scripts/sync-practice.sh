#!/usr/bin/env bash
set -euo pipefail

LOG="$HOME/my_practice/scripts/sync-practice.log"
> "$LOG"    # reset logfile each time

LOCK="/tmp/sync-practice.lock"

TARGET="/mnt/g/My Drive/Ingegneria/Data Science GD/My-Practice"

# -- Single instance lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] Already running; exiting." >> "$LOG"
  exit 0
fi

echo "[$(date '+%F %T')] script invoked" >> "$LOG"

# --- Temp dirs for Unison
export UNISON_TMPDIR=/tmp/unison-tmp
export TMPDIR="$UNISON_TMPDIR"
export UNISON_NUMBACKUPS=1
mkdir -p "$UNISON_TMPDIR"
export UNISON_DIR="$HOME/my_practice/scripts"

# ONE-TIME cleanup of any leftover Unison scratch files
rm -rf "${UNISON_TMPDIR}/"*
find "$HOME/my_practice" "$TARGET" \
  -type f -name '.unison.*.unison.tmp' -delete 2>/dev/null || true

# --- Wait for Windows drive to be ready (up 5 minutes)
for i in {1..300}; do
  if mountpoint -q /mnt/g && [ -d "$TARGET" ]; then
    echo "[$(date '+%F %T')] G: ready and target exists." >> "$LOG"
    break
  fi
  (( i % 5 == 0 )) && echo "[$(date '+%F %T')] Waiting for G:... (${i}s)" >> "$LOG"
  if (( i == 300 )); then
    echo "[$(date '+%F %T')] Timeout: G: not ready. Exiting." >> "$LOG"
    exit 1
  fi
  sleep 1
done

# — Now that /mnt/g and $TARGET are ready, record the Boot trigger
echo "=== [$(date '+%F %T')] Boot trigger: starting Unison ===" >> "$LOG"

# --- Run Unison
unison \
  "$HOME/my_practice" \
  "$TARGET" \
  -fat \
  -perms 0 \
  -batch \
  -auto \
  -times \
  -maxbackups 1 \
  -confirmmerge=false \
  -prefer newer \
  -links true \
  -fastcheck true \
  -silent \
  -repeat 30 \
  -ignore 'Name .git' \
  -ignore 'Path .git/**' \
  -ignore 'Name build' \
  -ignore 'Path **/dist/**' \
  -ignore 'Regex .*[Cc]heckpoint.*' \
  -ignore 'Path **/.ipynb_checkpoints/**' \
  -ignore 'Name __pycache__' \
  -ignore 'Path **/__pycache__/**' \
  -ignore 'Name *.sw?' \
  -ignore 'Name *~' \
  -ignore 'Name .DS_Store' \
  -ignore 'Name .Trash-0' \
  -ignore 'Name sync-practice.log' \
  -logfile "$LOG"

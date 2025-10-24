#!/usr/bin/env bash
set -euo pipefail

LOG="$HOME/scripts/sync-practice.log"

LOCK="/tmp/sync-practice.lock"

TARGET="/mnt/g/My Drive/Ingegneria/Data Science GD/My-Practice"

# -- Single instance lock
exec 9>"$LOCK"
if ! flock -n 9; then
  exit 0
fi


# --- Temp dirs for Unison
export UNISON_TMPDIR=/tmp/unison-tmp
export TMPDIR="$UNISON_TMPDIR"
export UNISON_NUMBACKUPS=1
mkdir -p "$UNISON_TMPDIR"
export UNISON_DIR="$HOME/.unison"

# ONE-TIME cleanup of any leftover Unison scratch files
rm -rf "${UNISON_TMPDIR}/"*
find "$HOME" "$TARGET" \
  -type f -name '.unison.*.unison.tmp' -delete 2>/dev/null || true

# --- Wait for Windows drive to be ready (up 5 minutes)
for i in {1..300}; do
  if mountpoint -q /mnt/g && [ -d "$TARGET" ]; then
    break
  fi
  if (( i == 300 )); then
    exit 1
  fi
  sleep 1
done

# — Now that /mnt/g and $TARGET are ready, record the Boot trigger

# --- Run Unison
unison \
  -root "$HOME" \
  -root "$TARGET" \
  -fat \
  -perms 0 \
  -batch \
  -auto \
  -times \
  -maxbackups 1 \
  -confirmmerge=false \
  -prefer newer \
  -links false \
  -fastcheck true \
  -repeat 10 \
  -ignore 'Name .*' \
  -ignore 'Path jlenv/**' \
  -ignore 'Path snap/**' \
  -ignore 'Path bin/**' \
  -ignore 'Name sync-practice.log' \
  -logfile "$LOG"
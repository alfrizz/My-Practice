#!/usr/bin/env bash
set -euo pipefail

# 1) Force all temp dirs into /tmp
export TMPDIR=/tmp
export TEMP=/tmp
export TMP=/tmp
mkdir -p /tmp

# 1.a) Delete anything in /tmp older than a day (1440 minutes)
find /tmp -mindepth 1 -mmin +1440 -delete || true

# 1.b) Delete Jupyter checkpoint directories if /workspace exists
if [ -d /workspace ]; then
  find /workspace -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} + 2>/dev/null || true
  echo "=== Removed all .ipynb_checkpoints (if any) ==="
else
  echo "=== /workspace not present; skipping checkpoint cleanup ==="
fi

# --- Wait for /workspace to be a host bind (not tmpfs) before starting Jupyter
# Default MAX_WAIT is 30 seconds. Set MAX_WAIT=0 to wait indefinitely.
MAX_WAIT="${MAX_WAIT:-30}"
SLEEP=2
ELAPSED=0

# --- Wait for /workspace to contain your data before starting Jupyter
echo "Checking /workspace for data; waiting up to ${MAX_WAIT}s..."
while true; do
  # Check if the directory exists AND contains the 'my_models' folder
  if [ -d "/workspace/my_models" ]; then
    echo "Detected data in /workspace. Starting..."
    break
  fi

  if [ "${MAX_WAIT:-0}" -ne 0 ] && [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "Error: /workspace/my_models not found after ${MAX_WAIT}s."
    exit 1
  fi

  sleep $SLEEP
  ELAPSED=$((ELAPSED + SLEEP))
done

# 2) Hand off to the real NVIDIA entrypoint (preserves CLI args)
exec /opt/nvidia/nvidia_entrypoint.sh "$@"

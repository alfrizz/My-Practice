#!/usr/bin/env bash
set -e

# 1) Force all temp‐dirs into /tmp
export TMPDIR=/tmp
export TEMP=/tmp
export TMP=/tmp
mkdir -p /tmp

# 1.a) Delete anything in /tmp older than a day (1440 minutes)
find /tmp -mindepth 1 -mmin +1440 -delete

# 1.b) Delete Jupyter checkpoint directories
find /workspace -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
echo "=== Removed all .ipynb_checkpoints ==="

# 2) Hand off to the real NVIDIA entrypoint (preserves CLI args)
exec /opt/nvidia/nvidia_entrypoint.sh "$@"

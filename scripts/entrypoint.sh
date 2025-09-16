#!/usr/bin/env bash
set -e

# 0) Install any new or updated packages from requirements.txt
pip install --upgrade \
    --upgrade-strategy only-if-needed \
    -r /workspace/scripts/requirements.txt

# 1) Force all temp‐dirs into /tmp
export TMPDIR=/tmp
export TEMP=/tmp
export TMP=/tmp
mkdir -p /tmp

# 1.a) Delete anything in /tmp older than three hours (180 minutes)
find /tmp -mindepth 1 -mmin +180 -delete

# 1.b) Delete Jupyter checkpoint directories
find /workspace -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
echo "=== Removed all .ipynb_checkpoints ==="

# 2) Hand off to the real NVIDIA entrypoint (preserves CLI args)
exec /opt/nvidia/nvidia_entrypoint.sh "$@"



#### after changes ####
# chmod +x entrypoint.sh
# rebuild container
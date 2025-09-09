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

# 1.a) Delete anything in /tmp older than one week (60*24*7 minutes)
find /tmp -mindepth 1 -mmin +10080 -delete

# 2) Hand off to the real NVIDIA entrypoint (preserves CLI args)
exec /opt/nvidia/nvidia_entrypoint.sh "$@"



#### after changes ####
# rebuild container
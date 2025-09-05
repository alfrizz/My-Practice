#!/usr/bin/env bash
set -e

# 1) Force temp‐dirs into /tmp
export TMPDIR=/tmp
export TEMP=/tmp
export TMP=/tmp
mkdir -p /tmp

# 1.a) Delete any files older than one week (60*24*7) in /tmp
find /tmp -mindepth 1 -mmin +10080 -delete

# 2) Hand off to the real NVIDIA entrypoint
exec /opt/nvidia/nvidia_entrypoint.sh "$@"

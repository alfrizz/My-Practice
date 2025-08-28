#!/usr/bin/env bash
set -e

# 1) Force temp files into /tmp
export TMPDIR=/tmp
export TEMP=/tmp
export TMP=/tmp
mkdir -p /tmp

# 2) Delegate to NVIDIA entrypoint
exec /opt/nvidia/nvidia_entrypoint.sh "$@"

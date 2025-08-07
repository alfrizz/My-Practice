#!/usr/bin/env bash
set -e
echo "🏁 entrypoint START at $(date)"

exec /opt/nvidia/nvidia_entrypoint.sh "$@"

#!/usr/bin/env bash
set -e
echo "🏁 entrypoint START at $(date)"

# 1) Clean stale WSL2 bind-mounts (your existing logic)
# …

# 2) From /workspace/my_models, ensure dfs_training under EVERY dir
ROOT=/workspace/my_models
find "$ROOT" -type d \
  -exec mkdir -p "{}/dfs_training" \; \
  -exec chmod 777 "{}/dfs_training" \; \
  -print | sed 's|$|/dfs_training created|'

# 3) Delegate to NVIDIA’s entrypoint + JupyterLab
exec /opt/nvidia/nvidia_entrypoint.sh "$@"

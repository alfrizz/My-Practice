#!/usr/bin/env bash
set -e

# 1) Clean up any stale WSL2‐bind‐mount dirs that Docker left behind
#    (adjust the glob to match your distro name, e.g. 'Ubuntu-20.04')
for d in /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/*/*; do
  if [ -d "$d" ] && ! mountpoint -q "$d"; then
    echo "⚙️  Removing stale mount: $d"
    rm -rf "$d"
  fi
done

# 2) Ensure your notebook save path really exists inside the container
mkdir -p /workspace/my_models/Trading/_Stock_Analysis_/dfs_training
chmod 777 /workspace/my_models/Trading/_Stock_Analysis_/dfs_training

# 3) Finally hand off to whatever CMD was in the Dockerfile
exec "$@"

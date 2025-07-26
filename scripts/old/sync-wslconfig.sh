#!/usr/bin/env bash
set -euo pipefail

WIN_USER="alfrizz"
WIN_CONF="/mnt/c/Users/$WIN_USER/.wslconfig"
SRC="$(pwd)/.wslconfig"

echo "Copying $SRC → $WIN_CONF"
cp "$SRC" "$WIN_CONF"

echo "Restarting WSL…"
powershell.exe -NoProfile -Command "wsl --shutdown"

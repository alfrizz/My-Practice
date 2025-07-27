#!/usr/bin/env bash
set -euo pipefail

# Adjust path if your scripts folder moves
SRC="$(pwd)/.wslconfig"
DST="/mnt/c/Users/alfri/.wslconfig"

echo "Copying $SRC → $DST"
cp "$SRC" "$DST"

echo "Restarting WSL..."
powershell.exe -NoProfile -Command "wsl --shutdown"

#!/usr/bin/env bash
: "${HOME:=/home/alfrizz}"
export HOME
set -euo pipefail

LOG="$HOME/scripts/sync-practice.log"

# Ensure log exists
mkdir -p "$(dirname "$LOG")"
: >"$LOG"

# Boot marker
echo "$(date) sync-practice.sh started (pid $$)" >>"$LOG"

LOCK="/tmp/sync-practice.lock"
TARGET="/mnt/g/My Drive/Ingegneria/Data Science GD/My-Practice"
PS_CMD='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'

# Single-instance lock
exec 9>"$LOCK"
if ! flock -n 9; then
  exit 0
fi

# Ensure scripts are executable (one-shot)
chmod u+x "$HOME/scripts/sync-practice.sh" "$HOME/scripts/entrypoint.sh" || true

# Unison temp dirs and cleanup
export UNISON_TMPDIR=/tmp/unison-tmp
export TMPDIR="$UNISON_TMPDIR"
export UNISON_NUMBACKUPS=1
mkdir -p "$UNISON_TMPDIR" "$HOME/.unison"
rm -rf "${UNISON_TMPDIR}/"*
find "$HOME" "$TARGET" -type f -name '.unison.*.unison.tmp' -delete 2>/dev/null || true

# Wait for G: mount (up to 5 minutes)
for i in {1..300}; do
  if mountpoint -q /mnt/g && [ -d "$TARGET" ]; then
    echo "$(date) mount ready: $TARGET" >>"$LOG"
    break
  fi
  if (( i == 300 )); then
    echo "$(date) mount timeout waiting for $TARGET" >>"$LOG"
    exit 1
  fi
  sleep 1
done

# Start Unison in background (ignore transient manual log and runlog)
unison \
  -root "$HOME" -root "$TARGET" -fat -perms 0 -batch -auto -times \
  -maxbackups 1 -confirmmerge=false -prefer newer -links false -fastcheck true \
  -repeat 10 -silent \
  -ignore 'Name .*' \
  -ignore 'Path jlenv/**' \
  -ignore 'Path snap/**' \
  -ignore 'Path bin/**' \
  -ignore 'Name docker_tmp' \
  -ignore 'Name __pycache__' \
  -ignore 'Name *.pyc' \
  -ignore 'Name sync-practice.log' \
  -ignore 'Name sync-practice-manual.log' \
  -ignore 'Name sync-practice.runlog' \
  -ignore 'Name *.pth' \
  -ignore 'Name *.lnk' \
  -logfile "$LOG" &

chmod -R u+rwX "$HOME/scripts" || true

# Docker and Jupyter handling (simple)
MAX_DOCKER_WAIT=300
SLEEP=2
echo "$(date) Waiting up to ${MAX_DOCKER_WAIT}s for Docker daemon..." >>"$LOG"

for _ in $(seq 1 $((MAX_DOCKER_WAIT / SLEEP))); do
  if docker info >/dev/null 2>&1; then
    echo "$(date) Docker available; ensuring gpu-jl is running." >>"$LOG"

    if docker ps -a --format '{{.Names}}' | grep -xq gpu-jl; then
      docker start gpu-jl >>"$LOG" 2>&1 || echo "$(date) Failed to start gpu-jl" >>"$LOG"
    else
      docker-compose -f "$HOME/scripts/docker-compose.yml" up -d --no-deps jupyter >>"$LOG" 2>&1 || echo "$(date) Failed to create/start gpu-jl via docker-compose" >>"$LOG"
    fi

    # Wait for entrypoint to detect the host mount, then wait for Jupyter and open localhost in Windows
    for _ in $(seq 1 60); do
      if docker logs --tail 200 gpu-jl 2>/dev/null | grep -q 'Detected host mount for /workspace'; then
        # Wait until Jupyter responds, then open browser
        for _ in $(seq 1 60); do
          if curl -sSf http://localhost:8888/lab >/dev/null 2>&1; then

            # Minimal, safe: wait for Windows explorer to be present before invoking Start-Process
            for try in $(seq 1 60); do
              if [ -x "$PS_CMD" ] && "$PS_CMD" -NoProfile -Command "if (Get-Process -Name explorer -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >/dev/null 2>&1; then
                break
              fi
              sleep 2
            done

            # harmless readiness marker
            touch /tmp/jupyter-ready

            if [ -x "$PS_CMD" ]; then
              "$PS_CMD" -NoProfile -ExecutionPolicy Bypass -Command "Start-Process 'http://localhost:8888/lab'" >>"$LOG" 2>&1 && echo "$(date) powershell invocation succeeded" >>"$LOG" || echo "$(date) powershell invocation failed" >>"$LOG"
            else
              echo "$(date) powershell not found at $PS_CMD" >>"$LOG"
            fi

            break
          fi
          sleep 2
        done
        break
      fi
      sleep 2
    done

    break
  fi
  sleep $SLEEP
done


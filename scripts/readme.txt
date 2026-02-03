This repository contains scripts and configuration files to build, run, and synchronize a Dockerized Jupyter Lab environment on WSL2 with GPU support and Unison-backed workspace syncing:  

-----------------------------------------------

pipreq          Bash helper to install Python packages into the running gpu-jl container, update requirements.txt, and keep it sorted/unique.
                **Behavior note:** pipreq rebuilds the jupyter image, tags/pushes `alfrizz/gpu-jl:latest`, and restarts the jupyter service via docker-compose. It will stop/remove any existing `gpu-jl` container and start the freshly built service.

Dockerfile      Defines the Docker image: starts from an NVIDIA PyTorch base, installs Python deps, and sets up the Jupyter Lab entrypoint.

sysctl.conf     System-level kernel tuning (present at /etc/sysctl.conf).  
                **Note:** the distro `/etc/sysctl.conf` is authoritative and is used by the system.

.wslconfig      WSL2 global settings: caps RAM/CPU, configures swap size & location, and enables localhost forwarding.

wsl.conf        WSL2 per-distro automount and boot-time commands; mounts Windows drives with metadata and launches `sync-practice.sh` at boot.  
                **Boot flow (documented):** WSL boot command waits for the G: path, then launches `sync-practice.sh` (Unison + docker handling).

fstab           Automatically mounts G: into WSL at `/mnt/g`.

entrypoint.sh   Container entrypoint: forces all temporary container files into /tmp and deletes them on a schedule, removes `.ipynb_checkpoints` if `/workspace` exists, **waits for `/workspace` to be a host bind (not tmpfs)** before starting Jupyter, then delegates to NVIDIA’s default entrypoint for Jupyter startup.  
                **Environment:** honors `MAX_WAIT` (seconds). `MAX_WAIT=0` means wait indefinitely for a host bind. `docker-compose.yml` sets `MAX_WAIT=0` for the jupyter service.

docker-compose.yml  Orchestrates the gpu-jl Jupyter service: builds from Dockerfile, mounts the workspace, assigns GPUs, CPU/memory limits, ports. Sets `MAX_WAIT=0` in the container environment so the entrypoint waits for the host bind.

requirements.txt    Pinned Python dependencies for reproducible builds: Jupyter Lab, torchmetrics, pandas, plotly, Optuna, and more.
                    **Note:** keep pinned versions (e.g., `jupyterlab==4.3.5`) to avoid ambiguity during image builds.

shrink-wsl.ps1      WSL maintenance: prunes old `/tmp` files and Jupyter checkpoints, runs fstrim, compacts Ubuntu `ext4.vhdx` and logs each step along with C: free-space and VHDX size.  
                    **Caution:** must be run from an elevated Windows PowerShell session. It stops Docker Desktop and host services (VMCompute, LxssManager) and shuts down WSL as part of the compaction flow. Close running containers and save work before running.  
                    Run manually from Windows (elevated):  
                    `PowerShell.exe -NoProfile -ExecutionPolicy Bypass -File "G:\My Drive\Ingegneria\Data Science GD\My-Practice\scripts\shrink-wsl.ps1"`

shrink-wsl.log      Timestamped record of every action, disk-free and VHDX-size before/after maintenance, and compaction output.  

sync-practice.sh    Bash script run at WSL boot: waits for G: drive, then runs Unison in batch mode to sync `/home/alfrizz` ↔ G: folder. After starting Unison it ensures Docker is available, starts or creates the `gpu-jl` container, and performs a small restart if the container started before the host bind.  
                    **Detailed behavior:**  
                      - Waits for `/mnt/g/My Drive/.../My-Practice` to appear (up to configured timeout).  
                      - Starts Unison in background to keep files in sync.  
                      - If `gpu-jl` exists, attempts `docker start`; otherwise `docker run` is used.  
                      - If the container is running but `/workspace` is not a host bind, the script restarts the container once to allow Docker to attach the host bind.  
                      - Waits for the container entrypoint to detect the host bind (`Detected host mount for /workspace` in container logs), then polls `http://localhost:8888/lab` until Jupyter responds.  
                      - Before invoking PowerShell to open the browser, the script waits for Windows `explorer` to be present to avoid the Start-Process race.  
                      - Touches `/tmp/jupyter-ready` as a harmless readiness marker before calling PowerShell.  
                      - Logs `powershell invocation succeeded` or `powershell invocation failed` to `sync-practice.log`.

sync-practice.log   Logs of syncs and the autostart sequence; contains entries such as mount readiness, container start/restart, `/workspace` mount detection, and `powershell invocation` results.

-----------------------------------------------
C:\Users\alfri\.wslconfig
-----------------------------------------------
[wsl2]
# Maximum RAM WSL2 can consume (leave the rest GB for Windows)
memory=56GB

# Number of virtual processors (leave at least 4 for host OS)
processors=20

# Swap settings – large enough to avoid OOM, small enough to not thrash
swap=16GB

# Place the swap file alongside your user profile for fast NVMe access
swapFile=C:\\Users\\alfri\\AppData\\Local\\Docker\\wsl\\swap\\wsl2_swap.vhdx

# Enable localhost forwarding so Windows ↔ WSL networking is seamless
localhostForwarding=true


-----------------------------------------------
\\wsl.localhost\Ubuntu-22.04\etc\wsl.conf
-----------------------------------------------
[automount]
enabled       = true
root          = /mnt/
options       = "metadata,uid=1000,gid=1000,umask=022"
mountFsTab    = true

[boot]
command = /bin/su - alfrizz -c "/bin/bash -lc '/home/alfrizz/scripts/sync-practice.sh &'"

[user]
default = alfrizz



-----------------------------------------------
\\wsl.localhost\Ubuntu-22.04\etc\fstab
-----------------------------------------------
G: /mnt/g drvfs metadata,uid=1000,gid=1000,fmask=111,umask=022 0 0


-----------------------------------------------
\\wsl.localhost\Ubuntu-22.04\etc\sysctl.conf
-----------------------------------------------
# Delays swapping until RAM is under heavy pressure.
vm.swappiness = 20
# Disables heuristic overcommit; enforces allocation limits.
vm.overcommit_memory = 2
# Allows processes to allocate up to 80 % of total RAM + swap.
vm.overcommit_ratio = 80


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# WSL2 VHDX File Locations

- **Ubuntu-22.04 root filesystem**  
  Path:  
  `C:\Users\alfri\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu22.04LTS_…\LocalState\ext4.vhdx`  
  Contains live Linux distro, all installed packages and home directory.

- **Docker “utility” distro root FS**  
  Path:  
  `C:\Users\alfri\AppData\Local\Docker\wsl\main\ext4.vhdx`  
  The internal lightweight distro Docker Desktop uses to orchestrate its WSL back end.

- **Docker Desktop data disk**  
  Path:  
  `C:\Users\alfri\AppData\Local\Docker\wsl\disk\docker_data.vhdx`  
  Holds container images, volumes, and layer data for Docker Desktop.

- **WSL2 swap file**  
  Path:  
  `C:\Users\alfri\AppData\Local\Docker\wsl\swap\wsl2_swap.vhdx`  
  Virtual memory used by WSL2 (configured in Windows `.wslconfig`). 



------------------------------
to print all files from wsl:
------------------------------

files=(
  "$HOME/scripts/pipreq"
  "$HOME/scripts/Dockerfile"
  "/mnt/c/Users/alfri/.wslconfig"
  "/etc/wsl.conf"
  "/etc/fstab"
  "/etc/sysctl.conf"
  "$HOME/scripts/entrypoint.sh"
  "$HOME/scripts/docker-compose.yml"
  "$HOME/scripts/requirements.txt"
  "$HOME/scripts/shrink-wsl.ps1"
  "$HOME/scripts/sync-practice.sh"
)
for f in "${files[@]}"; do
  echo "===== START: $f ====="
  if [ -f "$f" ] && [ -r "$f" ]; then
    sed -n '1,20000p' "$f"
  else
    echo "[missing or not readable]"
  fi
  echo "===== END: $f ====="
done


------------------------------
Notes, caveats, and recommended safeguards
------------------------------

1. **Removed unused helper**  
   - `open-jupyter-incognito.ps1` was removed from the repository because the boot flow already opens the browser from WSL (`sync-practice.sh` → PowerShell `Start-Process`). Keep a Windows-side helper only if you prefer Chrome/Edge incognito and want to run it from Windows startup.

2. **.docker_ignore**  
   - The README previously referenced `.docker_ignore`. That file was not present; the reference has been removed. If you want to keep build contexts small, add a `.dockerignore` with patterns to exclude large or sensitive files.

3. **sysctl.conf location**  
   - The per-distro `/etc/sysctl.conf` is present and used. The README no longer references a non-existent `scripts/sysctl.conf`. If you prefer to track the same settings in the repo, add a `scripts/sysctl.conf` and commit it.

4. **Entrypoint host-bind wait**  
   - `entrypoint.sh` waits for `/workspace` to be a host bind (not tmpfs) and honors `MAX_WAIT`. `docker-compose.yml` sets `MAX_WAIT=0` so the container will wait indefinitely for the host bind. This prevents Jupyter from starting against an empty tmpfs and avoids a stuck entrypoint.

5. **sync-practice.sh restart behavior**  
   - The boot script will restart the container once if it detects the container started before the host bind. This is intentional and deterministic; document it so users understand why a restart may occur during boot.

6. **Readiness marker**  
   - The script writes `/tmp/jupyter-ready` before invoking PowerShell. This file is a harmless indicator you can check to confirm the opener step ran.

7. **Explorer wait before Start-Process**  
   - To avoid the race where `Start-Process` fails if Windows explorer is not yet running, the script waits for `explorer` before invoking PowerShell. This is the minimal, safe change that made autostart reliable.

8. **pipreq side effects**  
   - `pipreq` rebuilds the image, tags/pushes `alfrizz/gpu-jl:latest`, and restarts the service. Document these side effects so users know `pipreq` is not a purely local installer.

9. **shrink-wsl.ps1 warnings**  
   - The compaction script stops Docker Desktop and host services and requires admin rights. Warn users to close running containers and save work before running it.

10. **Health checks to run after reboot**  
   - Use these read-only commands to verify autostart succeeded:
     ```bash
     tail -n 80 ~/scripts/sync-practice.log
     docker ps --filter "name=gpu-jl" --format "table {{.Names}}\t{{.Status}}"
     docker exec gpu-jl sh -c 'mount | grep /workspace || echo "/workspace not mounted"'
     grep -nH "powershell invocation" ~/scripts/sync-practice.log || true
     ```
   - If these show the mount ready, container Up, `/workspace` as a drvfs bind, and `powershell invocation succeeded`, the autostart completed successfully.

11. **Common external failure modes (rare)**  
    - Docker Desktop fails to start on Windows.  
    - G: network/drive not mounted early (or drive letter changes).  
    - Antivirus, EFS/BitLocker, or permission issues blocking access to the VHDX or network drive.  
    - Major Windows or Docker updates that change WSL/Docker behavior.

12. **Recommended persistent settings**  
    - Keep `localhostForwarding=true` in `.wslconfig`.  
    - Ensure Docker Desktop autostarts and WSL integration for `Ubuntu-22.04` is enabled.  
    - Keep `MAX_WAIT=0` in the container environment so the entrypoint waits for the host bind.  
    - Keep the explorer-wait block in `sync-practice.sh` — it is small and prevents the Start-Process race.


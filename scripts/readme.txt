This repository contains scripts and configuration files to build, run, and synchronize a Dockerized Jupyter Lab environment on WSL2 with GPU support and Unison-backed workspace syncing.

-----------------------------------------------

pipreq			Bash helper to install Python packages into the running gpu-jl container, update requirements.txt, and keep it sorted/unique.

Dockerfile		Defines the Docker image: starts from an NVIDIA PyTorch base, installs Python deps, and sets up the Jupyter Lab entrypoint.

.wslconfig		WSL2 global settings: caps RAM/CPU, configures swap size & location, and enables localhost forwarding.
wsl.conf		WSL2 per-distro automount and boot-time commands; mounts Windows drives with metadata and launches sync-practice.sh at boot 
                        (it´s executed when wsl runs, eg when docker desktop invokes it "Enable integration with additional distros: Ubuntu-22.04")
fstab                   automatically mounts G:

entrypoint.sh		Container entrypoint: forces all temporary container files into /tmp and deletes them on a schedule, 
                        then delegates to NVIDIA’s default entrypoint for Jupyter startup.

docker-compose.yml	Orchestrates the gpu-jl Jupyter service: builds from Dockerfile, mounts the workspace, assigns GPUs, CPU/memory limits, ports.
.docker_ignore		Lists files/folders to exclude from Docker build context, while preserving only the necessary scripts/ assets for image build.
requirements.txt	Pinned Python dependencies for reproducible builds: Jupyter Lab, torchmetrics, pandas, plotly, Optuna, and more.

shrink-wsl.ps1          WSL maintenance: prunes old `/tmp` files and Jupyter checkpoints, runs fstrim, compacts Ubuntu `ext4.vhdx`and logs each step along with C: free-space and VHDX size 
                        Manually execute on an elevated Powershell: PowerShell.exe -NoProfile -ExecutionPolicy Bypass -File "G:\My Drive\Ingegneria\Data Science GD\My-Practice\scripts\shrink-wsl.ps1"
shrink-wsl.log          Timestamped record of every action, disk-free and VHDX-size before/after maintenance, and compaction output.  

sync-practice.sh	Bash script run at WSL boot: waits for G: drive, then runs Unison in batch mode to sync /home/alfrizz/my_practice ↔ G: folder.
sync-practice.log       Logs of syncs between wsl and G:

-----------------------------------------------
C:\Users\alfri\.wslconfig
-----------------------------------------------

[wsl2]
# Maximum RAM WSL2 can consume (leave the rest GB for Windows)
memory=26GB

# Number of virtual processors (leave at least 4 for host OS)
processors=16

# Swap settings – large enough to avoid OOM, small enough to not thrash
swap=24GB

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
options       = "metadata,uid=1000,gid=1000,umask=022,fmask=111"
networkDrives = true
mountFsTab    = true

[boot]
command = su - alfrizz -c "/home/alfrizz/my_practice/scripts/sync-practice.sh >> /home/alfrizz/my_practice/scripts/sync-practice.log 2>&1 &"


-----------------------------------------------
\\wsl.localhost\Ubuntu-22.04\etc\fstab
-----------------------------------------------

G: /mnt/g drvfs metadata,uid=1000,gid=1000,umask=022,fmask=111 0 0


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



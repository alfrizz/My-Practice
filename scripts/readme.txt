This repository contains scripts and configuration files to build, run, and synchronize a Dockerized Jupyter Lab environment on WSL2 with GPU support and Unison-backed workspace syncing.


File			Description

pipreq			Bash helper to install Python packages into the running gpu-jl container, update requirements.txt, and keep it sorted/unique.
Dockerfile		Defines the Docker image: starts from an NVIDIA PyTorch base, installs Python deps, and sets up the Jupyter Lab entrypoint.
.wslconfig		WSL2 global settings: caps RAM/CPU, configures swap size & location, and enables localhost forwarding.
wsl.conf		WSL2 per-distro automount and boot-time commands; mounts Windows drives with metadata and launches sync-practice.sh at boot.
sync-practice.sh	Bash script run at WSL boot: waits for G: drive, then runs Unison in batch mode to sync /home/alfrizz/my_practice ↔ G: folder.
entrypoint.sh		Container entrypoint: forces all temporary files into /tmp, then delegates to NVIDIA’s default entrypoint for Jupyter startup.
my_practice.prf		Unison profile defining two sync roots (~/my_practice and the mounted G: workspace), ignore rules, and non-interactive flags.
docker-compose.yml	Orchestrates the gpu-jl Jupyter service: builds from Dockerfile, mounts the workspace, assigns GPUs, CPU/memory limits, ports.
.docker_ignore		Lists files/folders to exclude from Docker build context, while preserving only the necessary scripts/ assets for image build.
requirements.txt	Pinned Python dependencies for reproducible builds: Jupyter Lab, torchmetrics, pandas, plotly, Optuna, and more.
shrink-wsl.ps1	        PowerShell script to stop Docker Desktop, shut down WSL, compact the Ubuntu ext4.vhdx, detach the VHDX, restart WSL, and log each step (scheduled in taskschd.msc)
shrink-wsl.log         	Timestamped log file capturing the output and free-space before/after each shrink-and-recovery run


-------------


The files .wslconfig and wsl.conf are just shortcuts, so their full content is copied also here below (N.B. double check if it´s the most up to date):


.wslconfig:

[wsl2]
# Maximum RAM WSL2 can consume (leave the rest GB for Windows)
memory=24GB

# Number of virtual processors (leave at least 4 for host OS)
processors=16

# Swap settings – large enough to avoid OOM, small enough to not thrash
swap=24GB

# Place the swap file alongside your user profile for fast NVMe access
swapFile=C:\\Users\\alfri\\wsl2_swap.vhdx

# Enable localhost forwarding so Windows ↔ WSL networking is seamless
localhostForwarding=true



wsl.conf:

[automount]
root = /mnt/
options = "metadata,uid=1000,gid=1000,umask=022,fmask=111"

[boot]
command = /usr/bin/su - alfrizz -c "/usr/bin/bash -lc 'nohup /home/alfrizz/my_practice/scripts/sync-practice.sh >/dev/null 2>&1 &'"


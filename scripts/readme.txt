File/Folder	        Purpose
.wslconfig	        Defines WSL2 resource limits (memory, CPUs, swap)
sync-wslconfig.sh	Copies .wslconfig into your Windows profile and restarts WSL
.dockerignore	        Lists files/folders to exclude from Docker build context
Dockerfile	        Specifies how to build your Docker image (OS base, Python, dependencies)
docker-compose.yml	Orchestrates containers and can set container‐level resource limits (CPU, memory)
entrypoint.sh	        Launches Jupyter Lab inside the Docker container
requirements.txt	Enumerates Python packages to install in the image
pipreq	                Script that regenerates requirements.txt based on imports in your notebooks/scripts
GPU benchmark.ipynb	Jupyter Notebook for measuring GPU performance
readme.txt	        Legacy notes and quick reminders
.ipynb_checkpoints/	Auto‐saved notebook checkpoints (can be ignored by Docker via .dockerignore)
old/	                Archived scripts and previous configurations
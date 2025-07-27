
File/Folder	Purpose

.wslconfig	Defines WSL2 resource limits (memory, CPU cores, swap file location) to optimize performance; includes localhost forwarding and networking tweaks (NB it´s a link to the original one).

Dockerfile	Specifies the base OS image, installs system and Python dependencies, sets up environment variables, creates a non-root user, and defines the build steps.

docker-compose.yml	Orchestrates multiple services, configures container-level CPU/memory limits, mounts volumes, defines networks, and wires together Jupyter Lab and its deps.

entrypoint.sh	Bootstraps the container environment (e.g. activates venv, fixes permissions), runs migrations or setup tasks, and launches Jupyter Lab with proper flags.

requirements.txt	Enumerates all Python packages with pinned versions for a reproducible image build; can be hand-edited or auto-regenerated as needed.

pipreq	Wrapper script around pipreqs that scans project files for imports, regenerates requirements.txt, and skips dev/test modules (run pipreq instead of e.g. pip3, pipx).

.dockerignore	Lists files and folders (e.g. .git, __pycache__, large data dumps) to exclude from Docker build context, keeping images lean and builds fast.

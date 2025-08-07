
File/Folder	Purpose

.wslconfig	Defines WSL2 resource limits (memory, CPU cores, swap file location) to optimize performance; includes localhost forwarding and networking tweaks (NB it´s a link to the original one).

Dockerfile	Specifies the base OS image, installs system and Python dependencies, sets up environment variables, creates a non-root user, and defines the build steps.

docker-compose.yml	Orchestrates multiple services, configures container-level CPU/memory limits, mounts volumes, defines networks, and wires together Jupyter Lab and its deps.

entrypoint.sh	very first script that runs inside  container or WSL environment before JupyterLab launches. Its job is to prepare the workspace so every piece of code that follows has what it needs.

requirements.txt	Enumerates all Python packages with pinned versions for a reproducible image build; can be hand-edited or auto-regenerated as needed.

pipreq	Wrapper script around pipreqs that scans project files for imports, regenerates requirements.txt, and skips dev/test modules (run pipreq instead of e.g. pip3, pipx).

.dockerignore	Lists files and folders (e.g. .git, __pycache__, large data dumps) to exclude from Docker build context, keeping images lean and builds fast.

sync-practice.sh – An executable Bash wrapper on ext4 workspace (~/my_practice/scripts/sync-practice.sh) that runs unison mypractice -batch -auto -fastcheck -confirmbigdel to kick off the two-way sync.

my_practice.prf – The Unison profile (saved under ~/.unison/mypractice.prf) that defines the two roots, the ignore rule for .git, and all the non-interactive flags (auto, batch, fastcheck, confirmbigdel).

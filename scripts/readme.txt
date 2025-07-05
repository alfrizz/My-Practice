End-to-End Overview: GPU-Optimized JupyterLab via Docker
This guide outlines the high-level steps required to run a GPU-accelerated JupyterLab environment—automatically rebuilding when new libraries are added and maximizing hardware utilization. The same pattern works whether the base image is NVIDIA’s PyTorch container or its TensorFlow counterpart.

1. Prerequisites
• Host machine with Docker and the NVIDIA Container Toolkit enabled (or native Linux with NVIDIA-Docker). • WSL2 enabled on Windows, or a Linux host with up-to-date NVIDIA drivers. • Project folder cloned at a known path (e.g. ~/workspace/project).

2. Define Project Structure
Inside the project directory, create a docker_files/ folder containing:

Dockerfile: specifies the GPU base image (PyTorch or TensorFlow) and copies in dependencies.

requirements.txt: lists Python packages to install at build time.

3. Build the GPU Image
A launcher script (jl) examines requirements.txt. On first run or whenever this file changes, it triggers a docker build to produce a new image tagged (for example) gpu-jl-image. This image includes CUDA, cuDNN, mixed-precision support, and any pinned libraries.

4. Install New Libraries Immediately
A companion helper (pipreq) takes package names as arguments, appends them to requirements.txt, and—if a GPU container is already running—executes pip install inside it. This delivers the new library without restarting the container yet ensures persistence on the next launch.

5. Launching JupyterLab
Invoking the single jl command does three things in sequence:

Detects changes in requirements.txt and rebuilds the image if needed.

Cleans up any stale container instance by name.

Runs a fresh Docker container with GPU access, mounting the project folder and starting JupyterLab on http://localhost:8888.

6. GPU-Tuning Best Practices
Enable cuDNN’s auto-tuner at runtime (torch.backends.cudnn.benchmark = True).

Use pinned host memory and non-blocking transfers for CPU→GPU copy tests to saturate PCIe.

Leverage mixed-precision APIs (torch.cuda.amp or TensorFlow’s mixed_precision).

Monitor utilization via nvidia-smi dmon -s u to ensure SM units stay near 100%.

This workflow yields a reproducible, on-demand GPU environment. The base image can later be swapped between NVIDIA’s PyTorch or TensorFlow containers to compare raw performance on LSTM or other deep-learning workloads.
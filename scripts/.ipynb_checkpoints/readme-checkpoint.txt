SCRIPTS FOLDER OVERVIEW

All files live under My-Practice/scripts/ and work together to build and run the GPU-enabled JupyterLab.

FILES .dockerignore Excludes large or unwanted files (models, data, caches) from the Docker build context.

Dockerfile Defines a build-arg BASE_IMAGE (TensorFlow vs PyTorch), installs requirements.txt, copies the code, and launches JupyterLab.

requirements.txt Pinned Python dependencies. Used at build time by Dockerfile.

jl Launcher script that: 1. Prompts to choose TensorFlow or PyTorch base image 2. Builds “gpu-jl:custom” from Dockerfile 3. Runs the container with: • /workspace volume mount • auto-install missing packages loop • JupyterLab on port 8888

pipreq Development helper that: 1. Installs a new package into the running “gpu-jl” container 2. Appends “package==version” to requirements.txt 3. Keeps dependencies in sync with the imports

TYPICAL WORKFLOW

First run or after dependency changes cd My-Practice/scripts ./jl • Prompts for framework choice • Uses .dockerignore to slim the build context • Builds and tags “gpu-jl:custom” • Mounts the project, installs any missing packages, starts JupyterLab

Add a new import in the code import foo Then pin and install it by running: ./pipreq foo

Rebuild with updated requirements ./jl • Docker reuses cached layers for speed • New dependencies from requirements.txt are baked into the image

The custom image now contains exactly the libraries your notebooks need, starts quickly, and remains reproducible.
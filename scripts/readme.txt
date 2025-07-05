Files

.dockerignore Excludes unwanted files (models, data, caches) from the Docker build context.

Dockerfile Blueprint for your custom image. Reads requirements.txt and sets up JupyterLab.

requirements.txt Pinned Python dependencies used during image build.

jl Launcher script that:

Prompts for TensorFlow or PyTorch base image

Builds gpu-jl:custom via Dockerfile

Runs the container with an auto-install loop and JupyterLab

pipreq Dev-time helper that:

Installs a new package into the running container

Appends package==version to requirements.txt

Keeps your dependency list in sync

Typical Workflow

First run / after changes

bash
cd My-Practice/scripts
./jl
Uses .dockerignore to slim the build

Builds and tags gpu-jl:custom with your chosen base image

Mounts your project, installs missing packages, launches JupyterLab

Add a new import In your code:

python
import foo
Then pin and install it:

bash
./pipreq foo
Rebuild & restart Rerun ./jl to bake updated requirements.txt into a fresh image.
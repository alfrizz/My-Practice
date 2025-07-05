# ─────────────────────────────────────────────────────
# 0) Let the builder pick the base image at build time
# ─────────────────────────────────────────────────────
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.02-py3
FROM ${BASE_IMAGE}

# ─────────────────────────────────────────────────────
# 1) Set up workspace & deps
# ─────────────────────────────────────────────────────
WORKDIR /workspace
COPY scripts/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────
# 2) Copy your code & launch JupyterLab
# ─────────────────────────────────────────────────────
COPY . /workspace
CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]

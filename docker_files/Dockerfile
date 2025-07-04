# 1) Base with CUDA‐enabled PyTorch
FROM nvcr.io/nvidia/pytorch:24.07-py3

# 2) Install dos2unix, Node.js & npm (for JupyterLab extension manager)
RUN apt-get update \
 && apt-get install -y dos2unix nodejs npm \
 && rm -rf /var/lib/apt/lists/*

# 3) Bring in the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN dos2unix /entrypoint.sh \
 && chmod +x /entrypoint.sh

# 4) Copy & install Python requirements (cached unless requirements.txt changes)
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# 5) Set the working root and entrypoint
WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]

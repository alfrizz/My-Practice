FROM nvcr.io/nvidia/pytorch:24.07-py3

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . /workspace

# Launch JupyterLab
CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]

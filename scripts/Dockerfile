ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.02-py3
FROM ${BASE_IMAGE}

WORKDIR /workspace

# 1) Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 2) Copy and set up entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
 && chmod +x /usr/local/bin/entrypoint.sh

# 3) Don’t pre-create my_models—Docker Compose will mount it for you

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["jupyter","lab","--no-browser","--ip=0.0.0.0","--allow-root","--ServerApp.token=","--ServerApp.root_dir=/workspace"]

# after changes:
# docker-compose up -d --build


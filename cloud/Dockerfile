FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# Basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential ninja-build python3-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Copy repo
COPY . /workspace

# Install uv and create venv
# Install uv; make sure it's on PATH for subsequent layers
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"
RUN uv --version && uv venv --python=3.11

# Install runtime dependencies and the local package (no deps to preserve CUDA libs)
RUN bash -lc 'source .venv/bin/activate && \
    uv pip install -r cloud/requirements-runpod.txt && \
    pip install -e . --no-deps'

# Start the serverless handler
CMD ["/bin/bash","-lc","source .venv/bin/activate && python -u cloud/handler.py | cat"]



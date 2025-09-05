FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# Basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy repo
COPY . /workspace

# Install uv and create venv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    bash -lc 'source $HOME/.cargo/env && uv --version' && \
    uv venv --python=3.11

# Install runtime dependencies and the local package (no deps to preserve CUDA libs)
RUN bash -lc 'source .venv/bin/activate && \
    uv pip install -r cloud/requirements-runpod.txt && \
    pip install -e . --no-deps'

# Start the serverless handler
CMD ["/bin/bash","-lc","source .venv/bin/activate && python -u cloud/handler.py | cat"]



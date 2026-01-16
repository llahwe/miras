FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

SHELL ["/bin/bash", "-lc"]

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git curl ca-certificates jq rclone \
  && rm -rf /var/lib/apt/lists/*

# Install uv (fast python package manager). Vast images often already include python.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

# Default entrypoint just opens a shell; Vast onstart-cmd should call vast/entrypoint.sh.
CMD ["bash"]


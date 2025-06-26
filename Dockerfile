# Use official PyTorch base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

# Copy pixi files first (for better caching)
COPY pixi.toml pixi.lock* pyproject.toml ./

# Install pixi dependencies
RUN pixi install

# Copy the rest of your repository
COPY . /workspace

RUN mkdir -p /data/input \
    /data/processed \
    /data/organized \
    /data/segmentations \
    /data/outputs

# Run all pixi tasks/commands
# Option 1: Run all tasks defined in pixi.toml
# RUN pixi run 

# Option 2: Run specific pixi commands in sequence
# RUN pixi run preprocess && \
#     pixi run train && \
#     pixi run evaluate

# Option 3: Create an entrypoint script
# RUN echo '#!/bin/bash\npixi run "$@"' > /entrypoint.sh && \
#     chmod +x /entrypoint.sh

# # Set entrypoint to use pixi
# ENTRYPOINT ["/entrypoint.sh"]
# CMD ["default"]
WORKDIR "/workspace"
CMD ["/bin/bash"]
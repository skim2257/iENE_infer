# Use official PyTorch base image
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

# Copy pixi files first (for better caching)
COPY pixi.toml pixi.lock* ./

# Install pixi dependencies
RUN pixi install

# Copy the rest of your repository
COPY . /workspace

# Run all pixi tasks/commands
# Option 1: Run all tasks defined in pixi.toml
RUN pixi run --all

# Option 2: Run specific pixi commands in sequence
# RUN pixi run preprocess && \
#     pixi run train && \
#     pixi run evaluate

# Option 3: Create an entrypoint script
RUN echo '#!/bin/bash\npixi run "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Set entrypoint to use pixi
ENTRYPOINT ["/entrypoint.sh"]
CMD ["default"]
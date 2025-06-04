FROM ubuntu:22.04


# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

# Copy project files
COPY pixi.toml pyproject.toml ./
COPY src/ ./src/
COPY LICENSE README.md ./

# Create necessary directories for model checkpoints
RUN mkdir -p src/models \
    && mkdir -p src/nnunet/nnUNet_results/Dataset001_Larynx/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0 \
    && mkdir -p src/nnunet/nnUNet_raw \
    && mkdir -p src/nnunet/nnUNet_preprocessed

# Install dependencies using pixi
RUN pixi install

# Create directories for data volumes
RUN mkdir -p /data/input \
    && mkdir -p /data/processed \
    && mkdir -p /data/organized \
    && mkdir -p /data/segmentations \
    && mkdir -p /data/outputs

# Set environment variables for nnUNet
ENV nnUNet_raw=/app/src/nnunet/nnUNet_raw
ENV nnUNet_results=/app/src/nnunet/nnUNet_results
ENV nnUNet_preprocessed=/app/src/nnunet/nnUNet_preprocessed

# Default command - run the full pipeline
CMD ["bash", "-c", "\
    echo 'Starting iENE inference pipeline...' && \
    pixi run process || echo 'Skipping DICOM processing...' && \
    pixi run prepare && \
    pixi run -e nnunet larynx && \
    pixi run predict && \
    pixi run average && \
    echo 'Pipeline completed successfully!'"]
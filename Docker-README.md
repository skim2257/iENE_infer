# Docker Setup for iENE Inference (CPU-only)

This setup allows you to run the iENE inference pipeline using Docker with CPU-only execution.

## Quick Start

1. **Setup environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your actual paths
   nano .env
   ```

2. **Download model checkpoints:**
   - Download from: https://drive.google.com/drive/folders/1U_js4aYkxT5EgMaD3sM22_50VOkjxiVw?usp=drive_link
   - Place `fold_1.ckpt` through `fold_4.ckpt` in the directory specified by `HOST_IENE_MODELS`
   - Place `checkpoint_best.pth` at the path specified by `HOST_NNUNET_CHECKPOINT`

3. **Run the pipeline:**
   ```bash
   # Using the helper script
   ./docker-run.sh run

   # Or using docker-compose directly
   docker-compose up iene-inference
   ```

## Available Commands

### Using the helper script (`docker-run.sh`):

```bash
# Build Docker image
./docker-run.sh build

# Run full pipeline (DICOM → NIfTI → Segmentation → Prediction)
./docker-run.sh run

# Run only DICOM preprocessing
./docker-run.sh preprocess

# Run only prediction (skip segmentation)
./docker-run.sh predict

# Open interactive shell
./docker-run.sh shell

# Run health check
./docker-run.sh health

# Skip building (use existing image)
./docker-run.sh run --no-build
```

### Using docker-compose directly:

```bash
# Full pipeline
docker-compose up iene-inference

# Only preprocessing
docker-compose --profile preprocess up iene-preprocess

# Only prediction
docker-compose --profile predict-only up iene-predict-only

# Interactive shell
docker-compose run --rm iene-inference bash
```

## Environment Variables

Configure these in your `.env` file:

| Variable | Description | Example |
|----------|-------------|---------|
| `HOST_DATA_ROOT` | Raw DICOM dataset directory | `/home/user/data/dicom` |
| `HOST_PROCESSED_PATH` | Processed NIfTI output directory | `/home/user/data/processed` |
| `HOST_ORGANIZED_PATH` | nnUNet-ready data directory | `/home/user/data/organized` |
| `HOST_SEGMENTATION_PATH` | Segmentation results directory | `/home/user/data/segmentations` |
| `HOST_OUTPUT_PATH` | Final predictions directory | `/home/user/data/outputs` |
| `HOST_IENE_MODELS` | iENE model checkpoints directory | `/home/user/models/iene` |
| `HOST_NNUNET_CHECKPOINT` | nnUNet checkpoint file | `/home/user/models/checkpoint_best.pth` |

## Pipeline Steps

The Docker setup runs the following pipeline:

1. **Process** (optional): Convert DICOM to NIfTI
   ```bash
   pixi run process
   ```

2. **Prepare**: Organize data for nnUNet
   ```bash
   pixi run prepare
   ```

3. **Larynx**: Run larynx segmentation
   ```bash
   pixi run -e nnunet larynx
   ```

4. **Predict**: Run iENE prediction
   ```bash
   pixi run predict
   ```

5. **Average**: Average predictions
   ```bash
   pixi run average
   ```

## Resource Requirements

- **CPU**: 2-4 cores recommended (configurable in docker-compose.yaml)
- **Memory**: 8-16GB recommended
- **Storage**: Ensure sufficient disk space for data processing

## Data Structure

### Input (DICOM):
```
dataset/
├── patient_one/
│   └── CT/
│       └── CT.nii.gz
└── patient_two/
    └── CT/
        └── CT.nii.gz
```

### Organized (nnUNet ready):
```
organized/
├── patient_one_0000.nii.gz
├── patient_two_0000.nii.gz
└── patient_three_0000.nii.gz
```

### Output:
```
outputs/
├── predictions_1_None.csv
├── predictions_1_x+.csv
├── ...
└── predictions_AVERAGE.csv
```

## Troubleshooting

### Common Issues:

1. **Permission errors**: Ensure Docker has access to your data directories
2. **Out of memory**: Reduce batch size or increase Docker memory limits
3. **Missing models**: Verify model checkpoints are downloaded and paths are correct

### Debug commands:

```bash
# Check container logs
docker-compose logs iene-inference

# Run health check
./docker-run.sh health

# Interactive debugging
./docker-run.sh shell
```

### Performance optimization:

1. **CPU cores**: Adjust `deploy.resources.limits.cpus` in docker-compose.yaml
2. **Memory**: Adjust `deploy.resources.limits.memory` in docker-compose.yaml
3. **Batch size**: Modify `--batch_size` parameter in pixi.toml tasks
4. **Workers**: Adjust `--num_workers` parameter for data loading

## Development

To modify the pipeline:

1. Edit source files in `src/`
2. Update `pixi.toml` for new dependencies or tasks
3. Rebuild the Docker image: `./docker-run.sh build`

## Support

For issues related to:
- **Model setup**: Check the main README.md
- **Docker configuration**: Check this Docker-README.md
- **Pipeline execution**: Check container logs and health check output
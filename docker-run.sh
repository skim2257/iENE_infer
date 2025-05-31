#!/bin/bash

# iENE Inference Docker Runner Script
# This script helps run the iENE inference pipeline using Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() { echo -e "${BLUE}INFO:${NC} $1"; }
print_success() { echo -e "${GREEN}SUCCESS:${NC} $1"; }
print_warning() { echo -e "${YELLOW}WARNING:${NC} $1"; }
print_error() { echo -e "${RED}ERROR:${NC} $1"; }

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build         Build the Docker image"
    echo "  run           Run the full pipeline"
    echo "  preprocess    Run only DICOM preprocessing"
    echo "  predict       Run only prediction (skip segmentation)"
    echo "  shell         Open interactive shell in container"
    echo "  health        Run health check"
    echo ""
    echo "Options:"
    echo "  --no-build    Skip building image (use existing)"
    echo "  --help        Show this help message"
    echo ""
    echo "Environment:"
    echo "  Copy .env.example to .env and configure your paths"
}

# Check if .env file exists
check_env_file() {
    if [ ! -f .env ]; then
        print_error ".env file not found!"
        print_info "Copy .env.example to .env and configure your paths:"
        print_info "  cp .env.example .env"
        print_info "  nano .env  # Edit the paths"
        exit 1
    fi
}

# Check if model checkpoints exist
check_models() {
    if [ ! -f .env ]; then
        return
    fi
    
    source .env
    
    if [ ! -d "$HOST_IENE_MODELS" ]; then
        print_warning "iENE model checkpoints directory not found: $HOST_IENE_MODELS"
        print_info "Download fold_1.ckpt through fold_4.ckpt from Google Drive"
    fi
    
    if [ ! -f "$HOST_NNUNET_CHECKPOINT" ]; then
        print_warning "nnUNet checkpoint not found: $HOST_NNUNET_CHECKPOINT"
        print_info "Download checkpoint_best.pth from Google Drive"
    fi
}

# Build Docker image
build_image() {
    print_info "Building Docker image..."
    docker build --platform linux/amd64 -t iene-inference:latest .
    print_success "Docker image built successfully"
}

# Run health check
run_health_check() {
    print_info "Running health check..."
    docker-compose run --rm iene-inference pixi run health-check
}

# Main execution
main() {
    local command=$1
    local no_build=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-build)
                no_build=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                if [ -z "$command" ]; then
                    command=$1
                fi
                shift
                ;;
        esac
    done
    
    # Default command
    if [ -z "$command" ]; then
        command="run"
    fi
    
    case $command in
        build)
            build_image
            ;;
        run)
            check_env_file
            check_models
            if [ "$no_build" = false ]; then
                build_image
            fi
            print_info "Running full iENE inference pipeline..."
            docker-compose up iene-inference
            print_success "Pipeline completed"
            ;;
        preprocess)
            check_env_file
            if [ "$no_build" = false ]; then
                build_image
            fi
            print_info "Running DICOM preprocessing only..."
            docker-compose --profile preprocess up iene-preprocess
            ;;
        predict)
            check_env_file
            check_models
            if [ "$no_build" = false ]; then
                build_image
            fi
            print_info "Running prediction only..."
            docker-compose --profile predict-only up iene-predict-only
            ;;
        shell)
            check_env_file
            print_info "Opening interactive shell..."
            docker-compose run --rm iene-inference bash
            ;;
        health)
            if [ "$no_build" = false ]; then
                build_image
            fi
            run_health_check
            ;;
        *)
            print_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
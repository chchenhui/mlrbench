#!/bin/bash

# Attribution-Guided Training Experiment Runner
# This script automates the complete experimental pipeline for AGT

# Set script to exit on error
set -e

# Create log file
LOG_FILE="log.txt"
touch $LOG_FILE

# Function to log messages
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_FILE
}

# Create experiment directories
log "Creating experiment directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p figures
mkdir -p results

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_FLAG="--use_gpu"
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    log "Using GPU: $GPU_INFO"
else
    GPU_FLAG=""
    log "No GPU found, using CPU only"
fi

# Install requirements
log "Installing requirements..."
pip install -r requirements.txt

# Run the experiment
log "Starting Attribution-Guided Training experiment..."

# Main experiment with standard settings
log "Running main experiment..."
python run_experiment.py \
    --model_name distilroberta-base \
    --batch_size 16 \
    --num_epochs 10 \
    --lambda_attr 0.1 \
    --output_dir output \
    --run_ablations \
    $GPU_FLAG

# Analyze results
log "Analyzing results..."
python analyze_results.py \
    --results_path output/experiment_results.json \
    --output_dir results

# Move and organize files
log "Organizing output files..."

# Copy log file
cp $LOG_FILE results/
cp output/figures/* results/

# Print completion message
log "Experiment completed successfully!"
log "Results are available in the 'results' directory"

# Display a summary of the results
log "Results summary:"
echo "=================================================="
echo "Attribution-Guided Training Experiment Results"
echo "=================================================="
echo "Best model: AGT-MLM with multi-layer attribution"
echo "Check results/results.md for detailed analysis"
echo "=================================================="
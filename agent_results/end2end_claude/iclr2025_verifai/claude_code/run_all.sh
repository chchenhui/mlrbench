#!/bin/bash

# VERIL experiment runner script
# This script runs the complete experiment pipeline for the VERIL framework.

set -e  # Exit on error

# Project directories
ROOT_DIR=$(dirname $(dirname $(realpath $0)))
CODE_DIR="$ROOT_DIR/claude_code"
RESULTS_DIR="$ROOT_DIR/results"

# Create log file
LOG_FILE="$ROOT_DIR/log.txt"
touch $LOG_FILE

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Start time
START_TIME=$(date +%s)
log "Starting VERIL experiment pipeline"

# Step 1: Setup environment
log "Step 1: Setting up environment"
cd $CODE_DIR
python setup.py --skip-install
log "Environment setup completed"

# Step 2: Run experiment
log "Step 2: Running VERIL experiment"
python run_experiment.py --dataset custom --dataset_size 5 --run_baseline --run_veril_static
log "Experiment completed"

# Step 3: Check and copy results
log "Step 3: Organizing results"

# Create results directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Copy results.md from RESULTS_DIR to results/
cp $RESULTS_DIR/results.md $RESULTS_DIR/

# Copy figures
cp $RESULTS_DIR/*.png $RESULTS_DIR/

# Copy log file
cp $LOG_FILE $RESULTS_DIR/

log "Results organized successfully"

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(( (ELAPSED_TIME % 3600) / 60 ))
SECONDS=$((ELAPSED_TIME % 60))

log "Experiment completed in ${HOURS}h ${MINUTES}m ${SECONDS}s"
log "Results saved to: $RESULTS_DIR"
log "Check $RESULTS_DIR/results.md for the evaluation report"

echo ""
echo "===================================================="
echo "VERIL experiment completed successfully!"
echo "Results saved to: $RESULTS_DIR"
echo "Check $RESULTS_DIR/results.md for the evaluation report"
echo "===================================================="
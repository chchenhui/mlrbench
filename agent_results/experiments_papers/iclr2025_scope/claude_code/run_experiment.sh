#!/bin/bash

# Run experiment script
# This script runs the full experiment with simulated results for demo purposes

# Set up logging
LOG_FILE="results/log.txt"
mkdir -p results

echo "Starting experiment at $(date)" | tee -a $LOG_FILE

# Run the main experiment with a smaller setup for demonstration
echo "Running main experiment with simulated data..." | tee -a $LOG_FILE
python main.py --model dsrsq --dataset nq --max_samples 100 --num_epochs 3 --debug --seed 42 | tee -a $LOG_FILE

echo "Experiment completed at $(date)" | tee -a $LOG_FILE

# Move results to the required location
echo "Moving results to the parent results folder..." | tee -a $LOG_FILE
cp -r results/* ../results/

echo "Done!" | tee -a $LOG_FILE
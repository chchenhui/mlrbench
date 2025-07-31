#!/bin/bash
# Run all Benchmark Cards experiments

# Create log file
LOG_FILE="log.txt"
RESULTS_DIR="results"

echo "Starting Benchmark Cards experiments at $(date)" | tee -a $LOG_FILE

# Make scripts executable
chmod +x *.py

# Create results directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Test Benchmark Card implementation
echo "Testing Benchmark Card implementation..." | tee -a $LOG_FILE
python test_benchmark_card.py 2>&1 | tee -a $LOG_FILE

# Generate benchmark cards
echo "Generating benchmark cards..." | tee -a $LOG_FILE
python generate_benchmark_cards.py --output-dir $RESULTS_DIR/benchmark_cards 2>&1 | tee -a $LOG_FILE

# Run experiments on multiple datasets
echo "Running experiments on multiple datasets..." | tee -a $LOG_FILE
python run_experiments.py --results-dir $RESULTS_DIR 2>&1 | tee -a $LOG_FILE

# Create additional visualizations
echo "Creating additional visualizations..." | tee -a $LOG_FILE
for dataset in adult diabetes credit-g; do
    python visualize_results.py --results-dir $RESULTS_DIR --dataset $dataset 2>&1 | tee -a $LOG_FILE
done

# Organize results
echo "Organizing results..." | tee -a $LOG_FILE
mkdir -p $RESULTS_DIR/../results
cp -r $RESULTS_DIR/* $RESULTS_DIR/../results/
cp $LOG_FILE $RESULTS_DIR/../results/

echo "All experiments completed at $(date)" | tee -a $LOG_FILE
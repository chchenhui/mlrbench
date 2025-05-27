#!/bin/bash

# Create required directories
mkdir -p logs
mkdir -p data
mkdir -p results

# Install the package in development mode
pip install -e .

# Run the simplified experiment
echo "Running AEB experiment..."
python run_simplified.py --config experiments/minimal_config.yaml

echo "Experiment completed!"
echo "Results are available in the 'results' directory."

# Copy results to the required location
echo "Copying results to /home/chenhui/mlr-bench/pipeline_gemini/iclr2025_mldpr/results"
mkdir -p /home/chenhui/mlr-bench/pipeline_gemini/iclr2025_mldpr/results
cp -r /home/chenhui/mlr-bench/pipeline_gemini/iclr2025_mldpr/claude_code/results/* /home/chenhui/mlr-bench/pipeline_gemini/iclr2025_mldpr/results/
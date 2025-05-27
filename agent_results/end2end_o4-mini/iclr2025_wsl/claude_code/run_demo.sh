#!/bin/bash

# Run the minimal experiment demo to generate placeholder results
echo "Starting minimal experiment demo..."

# Ensure proper Python environment (modify if needed)
# If you need to use a specific Python environment or virtual environment, uncomment and modify:
# source /path/to/your/venv/bin/activate

# Go to script directory
cd "$(dirname "$0")"

# Create results folder if it doesn't exist
mkdir -p ../results/figures

# Run the minimal demo script
python -m scripts.run_minimal

echo "Demo experiment completed!"
echo "Results are available in the 'results' folder."
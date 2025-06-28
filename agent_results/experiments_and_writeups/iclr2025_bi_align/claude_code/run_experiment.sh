#!/bin/bash

# Script to run the Dynamic Human-AI Co-Adaptation experiment
# and organize the results

# Create log directories if they don't exist
mkdir -p ../results

# Start logging
exec > >(tee -a ../results/log.txt)
exec 2>&1

echo "===== Starting Dynamic Human-AI Co-Adaptation Experiment ====="
echo "Date: $(date)"
echo "Machine: $(hostname)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Using CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU(s): $(python -c 'import torch; print(torch.cuda.device_count())')"
if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    echo "GPU Model: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo "================================================"

# Run the main experiment
echo "Running main experiment..."
python main.py

# Check if experiment was successful
if [ $? -eq 0 ]; then
    echo "Experiment completed successfully!"
    
    # Create results directory if it doesn't exist
    mkdir -p ../results
    
    # Move all result files to the results directory
    echo "Organizing results..."
    
    # Move results.md to the root of results directory
    cp ../results/results.md ../results.md
    
    echo "Results have been organized in the 'results' directory."
    echo "See 'results.md' for a comprehensive report."
else
    echo "Experiment failed. Please check the logs for details."
fi

echo "===== Experiment process completed at $(date) ====="
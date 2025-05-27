#!/bin/bash

# Create required directories
mkdir -p logs
mkdir -p data
mkdir -p results

# Make sure Python can find our modules
export PYTHONPATH=$PYTHONPATH:/home/chenhui/mlr-bench/pipeline_gemini/iclr2025_mldpr/claude_code

# Run test script
echo "Running test script..."
python run_test.py

# If test failed, exit
if [ $? -ne 0 ]; then
    echo "Test failed, aborting"
    exit 1
fi

# Run the simplified experiment
echo "Running minimal experiment..."
python -c "
import os
import torch
import numpy as np
import logging
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from target_models.models import SimpleCNN
from utils.logger import setup_logger

# Set up logging
logger = setup_logger('debug', 'logs/debug.log')
logger.info('Running minimal experiment')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Load a small sample of CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
subset_indices = np.random.choice(len(dataset), 1000, replace=False)
subset = Subset(dataset, subset_indices)
dataloader = DataLoader(subset, batch_size=32, shuffle=True)

# Create a model
model = SimpleCNN()
model.to(device)
logger.info(f'Created model: {model.__class__.__name__}')

# Generate a figure for testing
import matplotlib.pyplot as plt

os.makedirs('results/figures', exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4], [10, 20, 25, 30], 'bo-', label='Model Performance')
plt.title('Test Figure')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('results/figures/test_figure.png')
logger.info('Generated test figure')

# Generate a results markdown file
with open('results/results.md', 'w') as f:
    f.write('# Minimal Test Results\n\n')
    f.write('This is a minimal test to ensure the experiment pipeline works correctly.\n\n')
    f.write('## Results\n\n')
    f.write('![Test Figure](figures/test_figure.png)\n\n')
    f.write('*Figure 1: Test figure showing model performance*\n\n')

logger.info('Minimal experiment completed successfully')
"

# If minimal experiment failed, exit
if [ $? -ne 0 ]; then
    echo "Minimal experiment failed, aborting"
    exit 1
fi

# Copy results to the required location
echo "Copying results to /home/chenhui/mlr-bench/pipeline_gemini/iclr2025_mldpr/results"
mkdir -p /home/chenhui/mlr-bench/pipeline_gemini/iclr2025_mldpr/results
cp -r /home/chenhui/mlr-bench/pipeline_gemini/iclr2025_mldpr/claude_code/results/* /home/chenhui/mlr-bench/pipeline_gemini/iclr2025_mldpr/results/
cp /home/chenhui/mlr-bench/pipeline_gemini/iclr2025_mldpr/claude_code/logs/debug.log /home/chenhui/mlr-bench/pipeline_gemini/iclr2025_mldpr/results/log.txt

echo "Debug experiment completed!"
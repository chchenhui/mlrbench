# Experimental Automation for Composite vs Accuracy Selection

This directory contains scripts to run an automated experiment comparing model selection based on single accuracy vs composite scoring (accuracy + F1) on the breast cancer dataset.

## Setup

Dependencies (should already be available in environment):
- Python 3
- numpy
- pandas
- scikit-learn
- torch
- matplotlib

## Running the Experiment

1. Navigate to this directory:
   ```bash
   cd codex_experiments/iclr2025_mldpr/codex
   ```
2. Ensure dependencies are installed (e.g., via pip):
   ```bash
   pip install numpy pandas scikit-learn torch matplotlib
   ```
3. Run the experiment script:
   ```bash
   python3 run_experiments.py
   ```

Results (CSV, JSON), log file (`log.txt`), and plots will be saved under `results/` and `figures/` respectively.

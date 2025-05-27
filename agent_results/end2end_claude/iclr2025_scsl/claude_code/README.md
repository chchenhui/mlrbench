# Causally-Informed Multi-Modal Representation Learning (CIMRL)

This repository contains the implementation of Causally-Informed Multi-Modal Representation Learning (CIMRL), a framework to discover and mitigate shortcut learning in multi-modal models without requiring explicit annotation of spurious features.

## Overview

Large Multi-Modal Models (LMMs) are vulnerable to shortcut learning, where models rely on spurious correlations rather than causal relationships. This is especially problematic in multi-modal settings where spurious correlations can exist across modalities.

CIMRL leverages three key innovations:
1. A contrastive invariance mechanism that identifies features that remain stable across intentionally perturbed inputs
2. A modality disentanglement component that separates shared causal features from modality-specific spurious ones
3. An intervention-based fine-tuning approach where the model is trained to maintain predictions when spurious features are manipulated

## Repository Structure

- `models/`: Model implementations
  - `cimrl.py`: CIMRL model implementation
  - `baselines.py`: Baseline models for comparison
- `data/`: Data loading and processing utilities
  - `dataloader.py`: DataLoader implementation for multi-modal data
  - `synthetic_dataset.py`: Synthetic dataset with controlled spurious correlations
  - `waterbirds.py`: Waterbirds dataset implementation
- `utils/`: Utility functions
  - `training.py`: Training and evaluation utilities
  - `metrics.py`: Metrics computation
  - `visualization.py`: Visualization utilities
- `configs/`: Configuration files
  - `default.json`: Default configuration
  - `waterbirds.json`: Configuration for Waterbirds dataset
  - `experiments.json`: Experiment configuration
- `main.py`: Main script for running individual experiments
- `run_experiments.py`: Script for running all experiments

## Requirements

- Python 3.8+
- PyTorch 1.10+
- transformers
- scikit-learn
- matplotlib
- seaborn
- tqdm
- numpy
- pandas

## Installation

```bash
pip install torch torchvision transformers scikit-learn matplotlib seaborn tqdm numpy pandas
```

## Running Experiments

### Single Experiment

To run a single experiment:

```bash
python main.py --config configs/default.json --model cimrl --seed 42
```

Parameters:
- `--config`: Path to configuration file (default: configs/default.json)
- `--model`: Model to train (choices: cimrl, standard, groupdro, jtt, ccr)
- `--seed`: Random seed (default: 42)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--no_cuda`: Disable CUDA

### All Experiments

To run all experiments as specified in the experiment configuration:

```bash
python run_experiments.py --experiment_config configs/experiments.json --seed 42
```

Parameters:
- `--experiment_config`: Path to experiment configuration (default: configs/experiments.json)
- `--seed`: Random seed (default: 42)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--no_cuda`: Disable CUDA

## Customizing Experiments

### Configuration Files

The configuration files control all aspects of the experiments, including:
- Dataset parameters
- Model architecture
- Training parameters
- Evaluation metrics

You can create custom configuration files by modifying the existing ones.

### Experiment Configuration

The experiment configuration file (`configs/experiments.json`) specifies which models, configurations, and seeds to use for the experiments:

```json
{
  "models": ["cimrl", "standard", "groupdro", "jtt", "ccr"],
  "configs": ["configs/default.json", "configs/waterbirds.json"],
  "seeds": [42, 43, 44]
}
```

### Adding New Datasets

To add a new dataset:
1. Create a dataset implementation in the `data/` directory
2. Update the `get_dataloaders` function in `data/dataloader.py`
3. Create a configuration file for the dataset in the `configs/` directory

### Adding New Models

To add a new baseline model:
1. Implement the model in the `models/` directory
2. Update the `get_model` function in `main.py`

## Results

After running experiments, the results will be saved in the `results/` directory:
- `results_summary.json`: Summary of all experiment results
- `figures/`: Visualizations and plots
- `results.md`: Markdown file summarizing the results
- `log.txt`: Experiment log

## License

This project is licensed under the MIT License - see the LICENSE file for details.
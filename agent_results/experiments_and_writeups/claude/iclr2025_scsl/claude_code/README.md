# AIFS: Adaptive Invariant Feature Extraction using Synthetic Interventions

This repository contains the implementation of the AIFS method and experimental framework for addressing spurious correlations in deep learning models.

## Overview

Adaptive Invariant Feature Extraction using Synthetic Interventions (AIFS) is a novel method that automatically discovers and neutralizes hidden spurious correlations without requiring explicit supervision or group annotations. It works by applying structured interventions in the latent space and training models to be invariant to these interventions.

The repository includes:

- Implementation of the AIFS method
- Implementation of baseline methods for comparison
- Datasets with spurious correlations
- Evaluation metrics for measuring robustness to spurious correlations
- Visualization utilities for analyzing results

## Installation

Clone the repository and install the required dependencies:

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Repository Structure

```
claude_code/
├── model.py            # AIFS model implementation
├── baselines.py        # Baseline methods implementation
├── datasets.py         # Dataset utilities
├── evaluation.py       # Evaluation metrics and utilities
├── visualization.py    # Visualization utilities
├── run_experiments.py  # Main script to run experiments
├── simulate_results.py # Script to simulate results for demonstration
├── README.md           # This file
└── results/            # Directory for storing experimental results
```

## Running Experiments

### Full Experiment

To run the full experiment, executing all baselines and the AIFS method:

```bash
python run_experiments.py --train-all --dataset spurious_cifar10 --output-dir ./results
```

By default, this will:
1. Load or create the spurious CIFAR-10 dataset
2. Train all models (Standard ERM, Group DRO, Domain Adversarial, Reweighting, AIFS)
3. Evaluate the models on test data
4. Generate visualizations and summary of results

### Custom Experiments

To run specific models or customize the experiment:

```bash
python run_experiments.py --train-standard --train-aifs --dataset spurious_cifar10 --epochs 30 --batch-size 64
```

### Available Datasets

- `spurious_cifar10`: CIFAR-10 with colored borders as spurious features
- `spurious_adult`: Adult Census Income dataset with amplified demographic biases

### Available Models

- `--train-standard`: Standard Empirical Risk Minimization (ERM)
- `--train-group-dro`: Group Distributionally Robust Optimization (Group DRO)
- `--train-dann`: Domain Adversarial Neural Network (DANN)
- `--train-reweighting`: Reweighting based on class and group
- `--train-aifs`: Our proposed Adaptive Invariant Feature Extraction using Synthetic Interventions
- `--train-all`: Train all models

### Important Parameters

- `--dataset`: Dataset to use (choices: spurious_cifar10, spurious_adult)
- `--spurious-ratio`: Ratio of spurious correlation in the dataset (default: 0.95)
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Batch size for training (default: 64)
- `--learning-rate`: Learning rate (default: 0.001)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output-dir`: Directory to save results (default: ./results)

For AIFS-specific parameters:
- `--num-masks`: Number of intervention masks for AIFS (default: 5)
- `--mask-ratio`: Proportion of dimensions to include in each mask (default: 0.2)
- `--lambda-sens`: Weight for sensitivity loss in AIFS (default: 0.1)
- `--intervention-prob`: Probability of applying intervention during training (default: 0.5)

## Simulation Mode

For demonstration purposes or when computational resources are limited, you can run the simulation mode:

```bash
python simulate_results.py
```

This will generate simulated results and visualizations based on expected model behaviors, allowing you to see the output format and visualization styles without running the full training.

## Evaluating Results

After running experiments, results will be saved in the specified output directory:

- Training metrics: `training_metrics.json`
- Evaluation results: `evaluation_results.json` 
- Summary visualizations in the `plots/` subdirectory
- A comprehensive summary in `results.md`

## Extending the Framework

### Adding a New Dataset

To add a new dataset with spurious correlations:

1. Create a new dataset class in `datasets.py` that inherits from `torch.utils.data.Dataset`
2. Implement the necessary methods: `__init__`, `__len__`, `__getitem__`
3. Add the dataset to the `get_dataloaders` function
4. Update the `get_model_config` function in `run_experiments.py` to support the new dataset

### Adding a New Baseline Method

To add a new baseline method:

1. Implement the model class in `baselines.py` that inherits from `nn.Module`
2. Implement a training function for the model
3. Add the model and training function to `run_experiments.py`

## Citation

If you use this code in your research, please cite our work:

```
@article{author2025aifs,
  title={Adaptive Invariant Feature Extraction using Synthetic Interventions},
  author={Author, A.},
  journal={Conference on Spurious Correlation and Shortcut Learning (SCSL)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This work was supported by the Workshop on Spurious Correlation and Shortcut Learning: Foundations and Solutions (SCSL) at ICLR 2025.
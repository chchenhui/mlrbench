# SpurGen: A Synthetic Multimodal Benchmark for Spurious Correlation Detection

This repository contains the implementation of SpurGen, a synthetic multimodal benchmark for detecting and mitigating spurious correlations in machine learning models. SpurGen generates paired data (images and captions) with configurable spurious channels to enable systematic evaluation of model robustness to spurious correlations.

## Features

- **Synthetic Data Generation**: Create controllable image-text pairs with configurable spurious correlations
- **Evaluation Metrics**: Measures for quantifying a model's reliance on spurious features
  - Spurious Sensitivity Score (SSS)
  - Invariance Gap (IG)
  - Worst-group Accuracy
- **Robustification Methods**: Implementation of several methods to mitigate spurious correlations
  - Empirical Risk Minimization (ERM - baseline)
  - Invariant Risk Minimization (IRM)
  - Group Distributionally Robust Optimization (Group-DRO)
  - Adversarial Feature Debiasing
  - Contrastive Augmentation

## Installation

Clone the repository and install the dependencies:

```bash
# Navigate to the project directory
cd claude_code

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision matplotlib numpy pillow
```

## Usage

### Generating Synthetic Data

To generate a synthetic dataset with spurious correlations:

```bash
python scripts/run_experiment.py --generate_data --data_dir data --num_classes 10 --num_samples 10000
```

This will create a dataset with 10 classes and 10,000 samples, where each class has spurious correlations with background, color, and shape attributes.

### Running Experiments

To run experiments with different robustification methods:

```bash
python scripts/run_experiment.py --data_dir data --modality image --methods erm irm group_dro adversarial contrastive --exp_name spurgen_experiment
```

You can specify different modalities (`image`, `text`, or `multimodal`) and choose which robustification methods to evaluate.

### Command-line Arguments

- `--data_dir`: Directory to save/load the dataset
- `--generate_data`: Whether to generate new data or use existing data
- `--num_classes`: Number of classes in the dataset
- `--num_samples`: Number of samples to generate
- `--modality`: Data modality to use (`image`, `text`, or `multimodal`)
- `--feature_dim`: Dimension of feature vectors
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of training epochs
- `--lr`: Learning rate
- `--methods`: Robustification methods to evaluate
- `--irm_lambda`: IRM penalty weight
- `--adv_lambda`: Adversarial penalty weight
- `--cont_lambda`: Contrastive loss weight
- `--exp_name`: Experiment name
- `--save_dir`: Directory to save results
- `--seed`: Random seed
- `--gpu`: GPU device ID

## Project Structure

```
claude_code/
├── data/               # Data generation and loading
│   ├── generator.py    # SpurGen dataset generator
│   └── dataset.py      # PyTorch datasets and data loaders
├── models/             # Model architectures
│   ├── models.py       # Base models for different modalities
│   └── robustification.py  # Robustification method implementations
├── utils/              # Utility functions
│   ├── metrics.py      # Evaluation metrics
│   └── training.py     # Training loops and trainers
├── scripts/            # Experiment scripts
│   └── run_experiment.py  # Main experiment runner
└── README.md           # This file
```

## Examples

### Generate a small dataset for testing

```bash
python scripts/run_experiment.py --generate_data --data_dir data --num_classes 5 --num_samples 1000
```

### Run a quick experiment with only ERM and IRM

```bash
python scripts/run_experiment.py --data_dir data --modality image --methods erm irm --num_epochs 5 --exp_name quick_test
```

### Evaluate all methods on multimodal data

```bash
python scripts/run_experiment.py --data_dir data --modality multimodal --methods erm irm group_dro adversarial contrastive --num_epochs 10 --exp_name multimodal_test
```

## Results

The experiment results are saved in the specified `--save_dir` directory (default: `../results`). The following files are generated:

- `results.md`: Summary of the experiment results
- `[exp_name]_results.json`: Raw results in JSON format
- `[exp_name]_comparison.png`: Bar chart comparing all methods
- `[exp_name]_[method]_training_curves.png`: Training curves for each method
- `[exp_name]_[method]_sss.png`: Spurious Sensitivity Scores for each method
- `sample_visualization.png`: Visualization of sample data points

## Citation

If you use this code in your research, please cite our paper:

```
@inproceedings{spurgen2025,
  title={SpurGen: A Synthetic Multimodal Benchmark for Detecting and Mitigating Spurious Correlations},
  author={SpurGen Team},
  booktitle={ICLR Workshop on Spurious Correlation and Shortcut Learning},
  year={2025}
}
```
# WeightNet: Permutation-Invariant Transformer for Model Property Prediction

This repository contains the implementation of WeightNet, a Transformer-based architecture designed to predict neural network properties directly from their weights. The model is specifically designed to handle permutation symmetry in neural network weights, making it robust to different neuron orderings.

## Overview

Neural network weights possess inherent permutation symmetries - neurons in a layer can be reordered without changing functionality. WeightNet is a novel architecture that can predict high-level properties (e.g., accuracy, robustness, generalization gap) directly from the weights while being invariant to these permutations.

The project includes:
- **WeightNet**: A permutation-invariant transformer architecture
- **Baseline Models**: Simple MLP and statistical baselines for comparison
- **Synthetic Dataset Generation**: Utilities to create a model zoo with varied architectures
- **Ablation Studies**: Experiments to analyze the importance of different components

## Features

- **Permutation Invariance**: Handles permutation symmetry in neural network weights through specialized attention mechanisms and/or canonicalization
- **Cross-Architecture Generalization**: Can handle models with different architectures and sizes
- **Multi-Property Prediction**: Simultaneously predicts multiple model properties
- **Comprehensive Evaluation**: Includes baselines and ablation studies for thorough evaluation
- **Visualization Tools**: Extensive visualization utilities for analyzing results

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd claude_code

# Install required packages
pip install -r requirements.txt
```

## Usage

### Running Experiments

The main experiment script (`run_experiments.py`) can be used to run the experiments:

```bash
# Run the main WeightNet experiment
python run_experiments.py --config configs/weightnet_experiment.yaml

# Run baseline experiments
python run_experiments.py --config configs/baseline_experiment.yaml

# Run ablation studies
python run_experiments.py --config configs/ablation_experiment.yaml

# Run a specific model from a configuration
python run_experiments.py --config configs/weightnet_experiment.yaml --model weightnet_main

# Resume training from a checkpoint
python run_experiments.py --config configs/weightnet_experiment.yaml --resume experiments/weightnet_main/weightnet_main_epoch_10.pth

# Skip training and only evaluate
python run_experiments.py --config configs/weightnet_experiment.yaml --skip-training
```

### Configuration

The experiments are controlled through YAML configuration files located in the `configs` directory:

- `base_config.yaml`: Base configuration with default settings
- `weightnet_experiment.yaml`: Configuration for the main WeightNet experiment
- `baseline_experiment.yaml`: Configuration for baseline models
- `ablation_experiment.yaml`: Configuration for ablation studies

You can modify these files to change experimental settings, model parameters, and more.

### Dataset Generation

The system can generate a synthetic model zoo for training and evaluation. By default, this happens automatically during experiment runs, but you can also generate it separately:

```python
from data.dataset import create_model_zoo

create_model_zoo(
    output_dir="data/raw",
    num_models_per_architecture=25,
    architectures=["resnet18", "resnet34", "resnet50", "vgg11", "vgg16", "mobilenet_v2", "densenet121"],
    generate_variations=True,
    random_seed=42
)
```

## Project Structure

```
claude_code/
├── configs/                  # Configuration files
│   ├── base_config.yaml
│   ├── weightnet_experiment.yaml
│   ├── baseline_experiment.yaml
│   └── ablation_experiment.yaml
├── data/                     # Data processing modules
│   ├── data_utils.py         # Utilities for data processing
│   └── dataset.py            # Dataset creation and loading
├── models/                   # Model implementations
│   └── weight_net.py         # WeightNet and baseline models
├── utils/                    # Utility functions
│   ├── trainer.py            # Training and evaluation pipeline
│   ├── visualization.py      # Visualization utilities
│   └── config.py             # Configuration utilities
├── experiments/              # Experiment outputs (created during runs)
├── logs/                     # Log files (created during runs)
├── figures/                  # Visualizations (created during runs)
├── requirements.txt          # Required packages
├── run_experiments.py        # Main experiment script
└── README.md                 # Project documentation
```

## Experiment Results

After running experiments, results are saved in the following locations:

- **Model Checkpoints**: `experiments/<model_name>/`
- **Training Logs**: `logs/<experiment_name>/`
- **Visualizations**: `figures/<model_name>/`
- **Results Summary**: `results/results.md`

The `results/results.md` file contains a comprehensive summary of the experiment results, including model performance, visualizations, and analysis.

## Models

### WeightNet

WeightNet is a permutation-invariant transformer designed to predict model properties from weights. It has two main components:

1. **Intra-Layer Attention**: Permutation-invariant attention within each layer of the neural network
2. **Cross-Layer Attention**: Attention between different layers to capture global structure

The model handles permutation symmetry through specialized attention mechanisms and/or canonicalization preprocessing.

### Baselines

Two baseline models are included for comparison:

1. **MLP Baseline**: A simple multi-layer perceptron that takes flattened weights as input
2. **Stats Baseline**: A model that extracts statistical features from weights and uses an MLP for prediction

## Customization

### Adding New Models

To add a new model architecture, you can extend the model implementations in `models/weight_net.py` and update the `create_model` function in `run_experiments.py`.

### Adding New Properties

To predict different properties, modify the `model_properties` list in the configuration files and update the dataset generation code if needed.

### Custom Datasets

While the system includes synthetic dataset generation, you can also use your own pre-trained models by placing their weights in the `data/raw` directory and creating a corresponding metadata file.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{weightnet2025,
  title={Permutation-Invariant Transformer for Cross-Architecture Model Property Prediction from Weights},
  author={Your Name},
  journal={Workshop on Neural Network Weights as a New Data Modality},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
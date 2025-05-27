# Neural Weight Archeology (NWA) Experiments

This repository contains the code for experiments on Neural Weight Archeology, a framework for decoding model behaviors from weight patterns.

## Overview

Neural Weight Archeology (NWA) treats neural networks as archeological artifacts, extracting insights directly from weight structures without requiring inference runs. The framework uses graph neural networks and attention mechanisms to capture weight connectivity patterns and identify important weight clusters.

The implementation includes:

- **Dataset Generation**: Creates a dataset of neural network models with labeled properties.
- **Baseline Models**: Simple statistical and PCA-based approaches for weight analysis.
- **NWPA Model**: Neural Weight Pattern Analyzer using graph neural networks and attention.
- **Evaluation Framework**: Metrics and visualizations for comparing methods.

## Requirements

The experiments require the following Python packages:

- PyTorch >= 1.10.0
- NumPy
- Matplotlib
- Pandas
- scikit-learn
- tqdm
- seaborn

GPU acceleration is used if available.

## Directory Structure

```
claude_code/
├── data/
│   └── data_generator.py   # Dataset generation
├── models/
│   ├── baseline_models.py  # Baseline models implementation
│   └── nwpa.py             # NWPA model implementation
├── utils/
│   ├── evaluation.py       # Evaluation utilities
│   └── visualization.py    # Visualization utilities
├── main.py                 # Main experiment framework
├── run_experiment.py       # Script to run all experiments
└── README.md               # This file
```

## How to Run

### Quick Start

To run all experiments with default settings:

```bash
cd /path/to/claude_code
python run_experiment.py
```

This will:
1. Generate a dataset of neural network models
2. Train and evaluate all model types (Statistics, PCA, NWPA)
3. Create visualizations and result tables
4. Save all outputs to the `results` directory

### Custom Configuration

You can customize the experiments by directly modifying parameters in `run_experiment.py` or by using the following command-line arguments:

```bash
python main.py --model_type [nwpa|statistics|pca|all] --num_models 100 --epochs 30 --batch_size 32
```

Key parameters:

- `--model_type`: Model to use (`nwpa`, `statistics`, `pca`, or `all`)
- `--num_models`: Number of models to generate for the dataset
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--hidden_dim`: Hidden dimension size in models
- `--learning_rate`: Learning rate for optimization
- `--weight_decay`: Weight decay for regularization
- `--use_attention`: Enable attention in GNN (for NWPA model)

For a complete list of options:

```bash
python main.py --help
```

## Output

The experiments produce the following outputs in the `results` directory:

- **Log File**: `log.txt` containing the full execution log
- **Result Files**: JSON files with detailed metrics for each model
- **Training Curves**: Plots of training and validation loss over epochs
- **Performance Comparisons**: Bar charts comparing classification and regression performance
- **Weight Pattern Visualizations**: 2D projections of weight patterns using PCA and t-SNE
- **Summary**: `results.md` with a comprehensive analysis of all results

## Customizing Experiments

### Dataset Size

You can adjust the number of models in the dataset:

```bash
python main.py --num_models 200
```

### Model Configuration

To customize the NWPA model:

```bash
python main.py --model_type nwpa --hidden_dim 256 --num_layers 4 --use_attention
```

### Training Parameters

To adjust training settings:

```bash
python main.py --epochs 50 --batch_size 64 --lr 0.0005 --weight_decay 1e-4
```

## License

This project is provided for research purposes only.

## Acknowledgments

This implementation is based on the research proposal "Neural Weight Archeology: A Framework for Decoding Model Behaviors from Weight Patterns."
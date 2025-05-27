# Permutation-Equivariant Graph Embeddings of Neural Weights

This repository contains the implementation of "Permutation-Equivariant Graph Embeddings of Neural Weights for Model Retrieval and Synthesis" as described in the proposal. The code provides a framework for learning symmetry-aware embeddings of neural network weights that are invariant to neuron permutations and scaling.

## Project Structure

```
claude_code/
├── data/                 # Data handling and processing
│   └── dataset.py        # Dataset classes and model zoo management
├── models/               # Model implementations
│   └── models.py         # GNN architecture and baseline models
├── utils/                # Utility functions
│   ├── training.py       # Training and evaluation managers
│   └── metrics.py        # Evaluation metrics and visualization tools
├── scripts/              # Experiment scripts
│   └── train.py          # Main training script
├── results/              # Experiment results (will be created)
│   ├── figures/          # Generated figures
│   ├── results.md        # Analysis of experiment results
│   └── log.txt           # Experiment log
├── run_experiment.py     # Runner script for experiments
└── README.md             # This file
```

## Requirements

The implementation requires the following packages:

- PyTorch (1.9.0+)
- PyTorch Geometric (for GNNs)
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- tqdm
- pandas

You can install these dependencies using the following command:

```bash
pip install torch torch-geometric numpy matplotlib seaborn scikit-learn tqdm pandas
```

## Quick Start

To run the experiments with default parameters:

```bash
python run_experiment.py
```

For a smaller test run with reduced dataset size and training epochs:

```bash
python run_experiment.py --small
```

To utilize GPU for training (if available):

```bash
python run_experiment.py --gpu
```

To specify a custom experiment name and random seed:

```bash
python run_experiment.py --experiment_name custom_experiment --seed 123 --gpu
```

## Experimental Pipeline

The experimental pipeline consists of the following steps:

1. **Data Generation**: Create a synthetic model zoo with diverse architectures and accuracies.
2. **Model Training**: Train the permutation-equivariant GNN model and baseline models.
3. **Evaluation**: Evaluate models on downstream tasks:
   - Model retrieval (Recall@k, MRR)
   - Zero-shot performance prediction (R², MSE, Spearman correlation)
   - Model merging via embedding interpolation
4. **Visualization**: Generate visualizations of embeddings, retrieval performance, and accuracy prediction.
5. **Analysis**: Summarize results in a comprehensive report.

## Advanced Configuration

For advanced configuration, you can modify the parameters in the `scripts/train.py` script:

```bash
python -m scripts.train --help
```

Key parameters include:

- `--num_models`: Number of synthetic models to generate
- `--model_type`: Type of model to use (`gnn`, `pca_mlp`, or `mlp`)
- `--hidden_dim`: Hidden dimension size
- `--global_dim`: Global embedding dimension
- `--message_passing_steps`: Number of message passing steps
- `--batch_size`: Batch size
- `--num_epochs`: Number of epochs
- `--learning_rate`: Learning rate
- `--temperature`: Temperature for contrastive loss

## Understanding the Results

After running an experiment, the results will be organized in the `results/` directory:

- `log.txt`: Contains the experiment execution logs
- `results.md`: Summary of results and analysis
- `figures/`: Contains visualizations including:
  - Embedding visualizations by architecture and accuracy
  - Retrieval performance comparison
  - Accuracy prediction performance
  - Model interpolation results

The results analyze the effectiveness of the permutation-equivariant embedding approach compared to baseline methods, focusing on three key tasks:

1. **Model Retrieval**: How well the embeddings preserve similarity between models
2. **Performance Prediction**: Ability to predict model accuracy directly from weight embeddings
3. **Model Merging**: Benefits of interpolating in the embedding space for creating hybrid models

## Extending the Framework

The framework is designed to be extensible. You can add:

- New model architectures in `models/models.py`
- Additional evaluation metrics in `utils/metrics.py`
- Different dataset generation strategies in `data/dataset.py`

To use real neural network weights instead of synthetic models, modify the `ModelZooManager` class in `data/dataset.py` to load weights from files.

## Citation

If you find this code useful for your research, please cite:

```
@article{weightembeddings2025,
  title={Permutation-Equivariant Graph Embeddings of Neural Weights for Model Retrieval and Synthesis},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```
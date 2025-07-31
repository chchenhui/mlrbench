# Permutation-Equivariant Contrastive Embeddings for Model Zoo Retrieval

This repository contains the implementation of "Permutation-Equivariant Contrastive Embeddings for Model Zoo Retrieval," an approach for retrieving semantically similar neural networks directly from their weight tensors through permutation-equivariant graph neural networks.

## Overview

As model repositories grow past a million entries, practitioners need effective ways to discover pre-trained networks suited to new tasks. This project develops a novel framework for retrieving semantically similar neural networks through:

1. **Permutation-equivariant GNN Encoder**: A graph-neural architecture that respects neuron permutation and parameter scaling symmetries inherent in weight tensors.
2. **Contrastive Learning with Symmetry-Preserving Augmentations**: A training paradigm that explicitly encodes functional equivalence by preserving invariance under weight matrix transformations.
3. **Model Retrieval System**: A k-nearest-neighbor database enabling efficient discovery of transferable models from heterogeneous architecture classes.

## Repository Structure

```
claude_code/
├── config.py                  # Configuration parameters
├── data_prep.py               # Data preparation utilities
├── weight_to_graph.py         # Weight tensor to graph conversion
├── gnn_encoder.py             # Permutation-equivariant GNN encoder
├── contrastive_learning.py    # Contrastive learning framework
├── baselines.py               # Baseline methods implementation
├── evaluation.py              # Evaluation metrics and utilities
├── visualization.py           # Visualization utilities
├── dataloader.py              # Dataset and dataloader utilities
├── run_experiment.py          # Main experiment script
└── README.md                  # This file
```

## Requirements

The code requires the following dependencies:

- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric
- NumPy
- Pandas
- Matplotlib
- Seaborn
- tqdm
- scikit-learn
- UMAP (optional, for embedding visualization)

You can install the required packages using:

```bash
pip install torch==1.12.0 torch-geometric==2.1.0 numpy pandas matplotlib seaborn tqdm scikit-learn umap-learn
```

## Running the Experiment

### Quick Start

To run the complete experiment with default settings:

```bash
python run_experiment.py
```

This will:
1. Prepare a synthetic model zoo dataset
2. Train the permutation-equivariant GNN encoder
3. Evaluate against baseline methods
4. Generate visualizations and results

### Command-line Arguments

The main script accepts the following command-line arguments:

- `--seed`: Random seed for reproducibility (default: 42)
- `--epochs`: Number of training epochs (default: 50)
- `--skip-training`: Skip model training and use random model weights

Example:

```bash
python run_experiment.py --seed 123 --epochs 100
```

### Step-by-Step Execution

If you prefer to run the experiment step-by-step:

1. **Data Preparation**:
   ```
   # Inside Python
   from data_prep import ModelZooDataset
   model_zoo = ModelZooDataset()
   model_zoo.create_synthetic_model_zoo()
   ```

2. **Convert Weights to Graphs**:
   ```
   # Inside Python
   from weight_to_graph import WeightToGraph
   converter = WeightToGraph()
   graphs = converter.convert_model_to_graphs(weights)
   ```

3. **Train the Model**:
   ```
   # Inside Python
   from gnn_encoder import ModelEmbedder
   from contrastive_learning import ContrastiveLearningFramework
   
   model = ModelEmbedder()
   framework = ContrastiveLearningFramework(model)
   history, best_model = framework.train(train_loader, val_loader, num_epochs=50, optimizer=optimizer)
   ```

4. **Evaluate and Visualize**:
   ```
   # Inside Python
   from evaluation import RetrievalEvaluator
   from visualization import ResultsVisualizer
   
   evaluator = RetrievalEvaluator()
   metrics = evaluator.evaluate_knn_retrieval(embeddings, task_labels)
   
   visualizer = ResultsVisualizer()
   visualizer.plot_retrieval_metrics(metrics, save_as="retrieval_metrics")
   ```

## Results

After running the experiment, the results will be saved in the `results` directory:

- `results.md`: Markdown summary of the experimental results
- `results.json`: Raw results in JSON format
- `log.txt`: Execution log with timing information
- Various figures showing model performance metrics

Key metrics reported include:
- Precision@k, Recall@k, F1@k for retrieval performance
- Mean Average Precision (mAP)
- Transfer learning performance
- Symmetry robustness metrics
- Clustering quality metrics

## Extending the Code

### Adding New Datasets

To add a new dataset, modify the `data_prep.py` file to load and process your own model weights.

### Implementing New Baselines

To add a new baseline method, implement your encoder in `baselines.py` following the existing pattern.

### Custom Augmentations

To add new symmetry-preserving augmentations, extend the `SymmetryAugmenter` class in `contrastive_learning.py`.

## Citation

If you use this code in your research, please cite:

```
@inproceedings{permutation-equivariant-embeddings,
  title={Permutation-Equivariant Contrastive Embeddings for Model Zoo Retrieval},
  author={Your Name},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

## Acknowledgments

This work builds on advances in graph neural networks, contrastive learning, and weight space learning.
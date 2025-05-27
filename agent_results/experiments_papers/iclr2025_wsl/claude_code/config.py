"""
Configuration file for the model zoo retrieval experiment.
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = BASE_DIR.parent / "results"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
FIGURES_DIR = BASE_DIR / "figures"

# Ensure directories exist
for dir_path in [RESULTS_DIR, MODELS_DIR, DATA_DIR, LOGS_DIR, FIGURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "vision_models": {
        "num_models": 50,  # Reduced from 30k for experiment purposes
        "architectures": ["resnet18", "resnet34", "vgg16", "mobilenet_v2", "efficientnet_b0"],
        "tasks": ["classification", "detection"],
        "datasets": ["imagenet", "cifar10"]
    },
    "nlp_models": {
        "num_models": 30,  # Reduced from 20k for experiment purposes
        "architectures": ["bert-base-uncased", "distilbert-base-uncased", "roberta-base"],
        "tasks": ["classification", "summarization"],
        "datasets": ["sst2", "mnli"]
    },
    "scientific_models": {
        "num_models": 20,  # Reduced from 5k for experiment purposes
        "architectures": ["mlp", "cnn"],
        "tasks": ["regression", "forecasting"],
        "datasets": ["physics", "chemistry"]
    }
}

# Model configuration
MODEL_CONFIG = {
    "gnn_encoder": {
        "edge_dim": 16,
        "node_dim": 64,
        "hidden_dim": 128,
        "output_dim": 256,
        "num_layers": 3,
        "dropout": 0.2,
        "readout": "attention"  # "mean", "sum", "max", "attention"
    },
    "transformer_encoder": {  # Baseline model
        "hidden_dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "dropout": 0.1,
        "output_dim": 256
    },
    "pca_encoder": {  # Baseline model
        "n_components": 256
    },
    "hypernetwork": {  # Baseline model
        "hidden_dim": 128,
        "output_dim": 256,
        "num_layers": 3,
        "dropout": 0.2
    }
}

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 16,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "lambda_contrastive": 0.8,  # Weight for contrastive loss
    "temperature": 0.07,  # Temperature parameter for contrastive loss
    "num_negatives": 16,  # Number of negative samples per anchor
    "early_stopping_patience": 10,
    "augmentation": {
        "permutation_prob": 0.15,  # Probability of applying permutation to a layer
        "scaling_range": (0.5, 2.0),  # Range for dynamic scaling
        "dropout_prob": 0.05  # Probability for DropConnect
    }
}

# Evaluation configuration
EVAL_CONFIG = {
    "k_values": [1, 5, 10],  # For Precision@k and mAP@k
    "num_folds": 5,  # For cross-validation
    "finetuning_budgets": [10, 50, 100],  # Number of finetuning steps for transfer learning
    "num_symmetry_tests": 20,  # Number of symmetry robustness tests
}

# Visualization configuration
VIZ_CONFIG = {
    "embedding_vis": {
        "n_components": 2,  # For dimensionality reduction (t-SNE, UMAP)
        "perplexity": 30,  # For t-SNE
        "n_neighbors": 15,  # For UMAP
    },
    "figure_size": (10, 8),
    "save_format": "png",
    "dpi": 300
}

# Logging configuration
LOG_CONFIG = {
    "log_level": "INFO",
    "log_file": os.path.join(LOGS_DIR, "experiment.log"),
}
"""
Configuration settings for UAD experiments.
"""
import os
from pathlib import Path
import torch

# Paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    # Using a smaller model for faster experimentation
    "default": {
        "name": "facebook/opt-350m",  # Smaller model
        "cache_dir": str(MODELS_DIR),
    },
    "small": {
        "name": "distilgpt2",
        "cache_dir": str(MODELS_DIR),
    },
    "medium": {
        "name": "facebook/opt-1.3b",  # Medium-sized model, if more compute is available
        "cache_dir": str(MODELS_DIR),
    },
}

# Dataset configurations
DATASET_CONFIGS = {
    "squad": {
        "name": "squad_v2",
        "split": "validation[:100]",  # Using a very small subset for faster experimentation
        "cache_dir": str(DATA_DIR),
    },
    "xsum": {
        "name": "xsum",
        "split": "test[:50]",  # Using a very small subset for faster experimentation
        "cache_dir": str(DATA_DIR),
    },
}

# Experiment configurations
EXPERIMENT_CONFIGS = {
    "baseline": {
        "decoding_method": "greedy",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_length": 50,  # Reduced max length
        "num_beams": 1,
    },
    "uad_entropy": {
        "decoding_method": "uad",
        "uncertainty_method": "entropy",
        "intervention_strategy": "rerank",
        "threshold_init": 0.5,  # Initial threshold value
        "threshold_alpha": 0.1,  # Learning rate for dynamic thresholding
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,  # Using top_k for re-ranking
        "max_length": 50,  # Reduced max length
    },
}

# Evaluation configurations
EVAL_CONFIGS = {
    "metrics": ["bleu", "rouge", "hallucination_rate"],
    "hallucination_threshold": 0.7,  # Threshold for classifying a token as a hallucination
    "num_samples": 50,  # Reduced number of samples to evaluate due to compute constraints
}

# Training configurations
TRAIN_CONFIGS = {
    "batch_size": 8,
    "num_epochs": 3,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "logging_steps": 100,
    "eval_steps": 500,
    "save_steps": 1000,
}

# Hardware configurations
HARDWARE_CONFIGS = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_workers": 4,
    "fp16": torch.cuda.is_available(),  # Use fp16 if GPU is available
}

# Logging configurations
LOGGING_CONFIGS = {
    "log_level": "INFO",
    "use_tensorboard": True,
}

# Seed for reproducibility
SEED = 42
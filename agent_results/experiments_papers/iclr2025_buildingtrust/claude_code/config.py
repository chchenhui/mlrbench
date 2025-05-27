"""
Configuration file for the Self-Correcting Language Model experiment.
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = PROJECT_ROOT.parent / "results"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
LOG_PATH = RESULTS_DIR / "log.txt"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create directories if they don't exist
for dir_path in [DATA_DIR, CACHE_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Model Configuration
MODEL_CONFIGS = {
    "base_model": {
        "name": "llama-3.1-8b",
        "huggingface_id": "meta-llama/Meta-Llama-3.1-8B",
        "revision": "main"
    },
    "falcon": {
        "name": "falcon-7b",
        "huggingface_id": "tiiuae/falcon-7b",
        "revision": "main"
    },
    "mistral": {
        "name": "mistral-7b",
        "huggingface_id": "mistralai/Mistral-7B-v0.1",
        "revision": "main" 
    }
}

# API-based models
API_MODELS = {
    "gpt-4o-mini": {
        "name": "gpt-4o-mini",
        "provider": "openai",
        "context_length": 32768
    },
    "claude-3.7-sonnet": {
        "name": "claude-3-7-sonnet-20250219",  # Corrected format with hyphens instead of dots
        "provider": "anthropic",
        "context_length": 200000
    }
}

# Default model to use for self-correction
DEFAULT_MODEL = "mistral"
USE_API_MODEL = True  # Set to True to use API models, False to use local models
DEFAULT_API_MODEL = "claude-3.7-sonnet"

# SCLM Configuration
SCLM_CONFIG = {
    "confidence_threshold": 0.85,  # Threshold for confidence scoring
    "max_iterations": 5,  # Maximum number of correction iterations
    "retrieval_k": 5,  # Number of documents to retrieve
    "chunk_size": 512,  # Size of chunks for processing
    "attention_weight_layers": [0.1, 0.1, 0.2, 0.3, 0.3],  # Weights for each layer's attention
}

# Experiment Configuration
EXPERIMENT_CONFIG = {
    "seed": 42,
    "batch_size": 32,
    "max_samples": 100,  # Limit samples for quicker experiments
    "num_workers": 4,
    "use_gpu": True,  # Whether to use GPU
}

# Dataset Configuration
DATASET_CONFIG = {
    "truthfulqa": {
        "name": "truthfulqa",
        "huggingface_id": "truthful_qa",
        "config": "generation",  # Specify the config
        "split": "validation",
    },
    "fever": {
        "name": "fever",
        "huggingface_id": "fever",
        "config": "v1.0",  # Specify the config
        "split": "validation",
        "trust_remote_code": True,  # Allow running custom code
    }
}

# Evaluation Configuration
EVAL_CONFIG = {
    "metrics": ["accuracy", "f1", "hallucination_rate", "latency", "bleu", "rouge"],
    "human_eval": False,  # Set to True to include human evaluation
    "save_predictions": True,  # Save model predictions
}

# Logging Configuration
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
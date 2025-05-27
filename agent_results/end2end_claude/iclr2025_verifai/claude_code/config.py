"""
Configuration file for the VERIL (Verification-Enriched Recursive Improvement Learning) experiment.
"""
import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_DIR = ROOT_DIR / "claude_code"
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = CODE_DIR / "data"
CHECKPOINTS_DIR = CODE_DIR / "checkpoints"
LOG_FILE = ROOT_DIR / "log.txt"

# Make sure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset configurations
DATASET_NAME = "HumanEval"  # Options: "HumanEval", "APPS", or "custom"
DATASET_SIZE = 5  # Number of examples to use (for smaller experiment)
SAMPLE_GENERATIONS = 1  # Number of code generations per problem

# Verification configurations
VERIFICATION_TOOLS = {
    "static": True,    # Static analysis tools (linters, type checkers)
    "dynamic": True,   # Dynamic testing (test case execution)
    "formal": False,   # Formal verification (SMT solvers) - computationally intensive
}

# Model configurations
MODELS = {
    "baseline": {
        "name": "gpt-4o-mini",  # Use API models for quick testing
        "use_verification": False,
        "max_new_tokens": 512,
        "temperature": 0.7,
    },
    "veril_static": {
        "name": "gpt-4o-mini",
        "use_verification": True,
        "verification_types": ["static"],
        "max_new_tokens": 512,
        "temperature": 0.7,
    },
    "veril_dynamic": {
        "name": "gpt-4o-mini",
        "use_verification": True,
        "verification_types": ["dynamic"],
        "max_new_tokens": 512,
        "temperature": 0.7,
    },
    "veril_full": {
        "name": "gpt-4o-mini",
        "use_verification": True,
        "verification_types": ["static", "dynamic"],
        "max_new_tokens": 512,
        "temperature": 0.7,
    },
}

# Training configurations
TRAINING = {
    "num_iterations": 2,  # Number of recursive improvement iterations (reduced for testing)
    "learning_rate": 1e-5,
    "batch_size": 4,
    "max_epochs": 2,
    "gradient_accumulation_steps": 2,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
}

# Error taxonomy configurations
ERROR_TAXONOMY = {
    "syntax": {
        "severity": 0.5,
        "examples": ["syntax error", "invalid syntax", "unexpected token"],
    },
    "type": {
        "severity": 0.7,
        "examples": ["type error", "unexpected type", "incompatible types"],
    },
    "logic": {
        "severity": 0.8,
        "examples": ["logic error", "incorrect algorithm", "wrong implementation"],
    },
    "semantic": {
        "severity": 0.9,
        "examples": ["incorrect functionality", "wrong output", "failed test case"],
    },
    "security": {
        "severity": 1.0,
        "examples": ["security vulnerability", "insecure code", "injection risk"],
    },
}

# Evaluation metrics
METRICS = {
    "pass_at_k": [1, 3, 5],  # k values for pass@k
    "error_rate": True,
    "veripass_at_k": [1, 3, 5],
    "error_types": True,
    "learning_curve": True,
    "resource_usage": True,
}

# Experiment configurations
EXPERIMENT = {
    "run_baseline": True,
    "run_veril_static": True,
    "run_veril_dynamic": False,  # Reduce testing scope
    "run_veril_full": False,  # Full version is more computationally intensive
    "ablation_studies": False,  # Run ablation studies
    "num_trials": 1,  # Number of trials for each experiment (reduced for testing)
    "seed": 42,  # Random seed for reproducibility
}

# API keys for closed-source models (retrieve from environment)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# GPU settings
USE_GPU = True
NUM_GPUS = 1  # Number of GPUs to use
GPU_IDS = [0]  # IDs of GPUs to use
"""
Configuration file for the Reasoning Uncertainty Networks (RUNs) experiment.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset configurations
DATASET_CONFIG = {
    "scientific": {
        "name": "sciq",
        "split": "train[:1000]",  # Use a subset for experimentation
        "reasoning_type": "scientific"
    },
    "legal": {
        "name": "legal_reasoning",
        "source": "custom",  # Will create a custom dataset
        "size": 100,
        "reasoning_type": "legal"
    },
    "medical": {
        "name": "medical_reasoning",
        "source": "custom",  # Will create a custom dataset
        "size": 100,
        "reasoning_type": "medical"
    }
}

# LLM configurations
LLM_CONFIG = {
    "primary_model": {
        "name": "claude-3.7-sonnet",
        "provider": "anthropic",
        "max_tokens": 1000,
        "temperature": 0.2
    },
    "embedding_model": {
        "name": "all-MiniLM-L6-v2",
        "provider": "sentence-transformers",
        "dimension": 384
    }
}

# RUNs framework configuration
RUNS_CONFIG = {
    "reasoning_graph": {
        "max_nodes": 20,
        "prompt_template": "Given [problem], please reason step by step:\nStep 1: [assertion]\nStep 2: [assertion]\n...\nConclusion: [final answer]"
    },
    "uncertainty_initializer": {
        "use_llm_confidence": True,
        "use_semantic_similarity": True,
        "use_knowledge_verification": True,
        "num_variations": 5,
        "similarity_threshold": 0.85
    },
    "belief_propagation": {
        "max_iterations": 10,
        "convergence_threshold": 1e-4,
        "edge_weight_default": 0.5
    },
    "hallucination_detection": {
        "confidence_threshold": 0.7,
        "uncertainty_increase_threshold": 0.1,
        "gamma": 2.0,  # Weight for uncertainty increase
        "delta": 1.5,  # Weight for logical inconsistency
    }
}

# Baseline configurations
BASELINE_CONFIG = {
    "selfcheckgpt": {
        "num_samples": 10,
        "temperature": 0.7,
        "similarity_threshold": 0.8
    },
    "multidim_uq": {
        "num_dimensions": 3,
        "num_responses": 10
    },
    "calibration": {
        "method": "temperature_scaling",
        "validation_size": 100
    },
    "hudex": {
        "explanation_model": "gpt-4o-mini",
        "threshold": 0.65
    },
    "metaqa": {
        "num_mutations": 5,
        "similarity_threshold": 0.75
    }
}

# Evaluation configurations
EVAL_CONFIG = {
    "metrics": ["precision", "recall", "f1", "auroc", "auprc", "ece", "brier"],
    "num_test_examples": 100,
    "hallucination_types": ["factual", "logical", "numerical"],
    "significance_test": "wilcoxon",
    "random_seed": 42,
    "num_runs": 3  # Number of runs for averaging results
}

# Visualization configurations
VIZ_CONFIG = {
    "figures": {
        "format": "png",
        "dpi": 300,
        "figsize": (10, 8)
    },
    "colors": {
        "runs": "#1f77b4",
        "selfcheckgpt": "#ff7f0e",
        "multidim_uq": "#2ca02c",
        "calibration": "#d62728",
        "hudex": "#9467bd",
        "metaqa": "#8c564b"
    }
}

# Experiment configurations
EXPERIMENT_CONFIG = {
    "name": "runs_hallucination_detection",
    "description": "Evaluation of Reasoning Uncertainty Networks for hallucination detection",
    "save_model": True,
    "log_level": "INFO",
    "use_gpu": True,
    "batch_size": 8,
    "verbose": True
}
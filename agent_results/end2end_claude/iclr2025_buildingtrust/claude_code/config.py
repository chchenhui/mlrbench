"""
Configuration file for TrustPath experiment.
Contains settings for model access, data paths, and experiment parameters.
"""

import os
import json
from pathlib import Path

# Base directories
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLAUDE_CODE_DIR = ROOT_DIR / "claude_code"
DATA_DIR = CLAUDE_CODE_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
LOG_FILE = RESULTS_DIR / "log.txt"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "n_samples": 100,  # Number of samples to generate for evaluation
    "domains": ["science", "history", "current_events"],  # Domains to cover
    "error_types": ["factual", "logical", "vague"],  # Types of errors to annotate
}

# LLM API configuration
LLM_CONFIG = {
    "model": "claude-3-7-sonnet-20250219",  # Model to use for experiments
    "temperature": 0.3,  # Lower temperature for more predictable outputs
    "max_tokens": 1000,  # Maximum tokens for responses
}

# Self-verification module configuration
SELF_VERIFICATION_CONFIG = {
    "confidence_threshold": 0.7,  # Threshold below which statements are marked as potential errors
    "verification_prompt_template": """
    Carefully review your previous response to the question "{question}". Your response was: "{response}"
    For each statement in your response:
    1. Assess your confidence in its accuracy (0-100%)
    2. If confidence is below 80%, explain why you're uncertain
    3. Provide alternative formulations that might be more accurate
    4. Identify any statements that should be verified with external sources
    """,
}

# Factual consistency checker configuration
FACTUAL_CHECKER_CONFIG = {
    "verification_threshold": 0.6,  # Threshold below which claims are marked as potentially erroneous
    "max_documents": 3,  # Maximum number of documents to retrieve per claim
    "knowledge_sources": [
        "wikipedia",  # Use Wikipedia as a knowledge source
        "academic_papers",  # Use academic papers as a knowledge source
    ],
}

# Human feedback simulation configuration
HUMAN_FEEDBACK_CONFIG = {
    "feedback_probability": 0.8,  # Probability of receiving feedback on an error detection
    "feedback_accuracy": 0.9,  # Accuracy of simulated human feedback
}

# Evaluation configuration
EVAL_CONFIG = {
    "metrics": ["precision", "recall", "f1", "bleu", "rouge"],
    "n_runs": 1,  # Number of runs for experiments
    "random_seed": 42,  # Random seed for reproducibility
}

# Baseline methods configuration
BASELINE_CONFIG = {
    "methods": [
        "simple_fact_checking",
        "uncertainty_estimation",
        "standard_correction",
    ],
}

# Visualization configuration
VIZ_CONFIG = {
    "confidence_colors": {
        "high": "#4CAF50",  # Green for high confidence
        "medium": "#FFC107",  # Yellow for medium confidence
        "low": "#F44336",  # Red for low confidence
    },
    "font_size": 12,
    "line_width": 2,
    "figsize": (10, 6),
}

# Save configuration as JSON for reference
def save_config():
    """Save the configuration to a JSON file."""
    config = {
        "dataset": DATASET_CONFIG,
        "llm": LLM_CONFIG,
        "self_verification": SELF_VERIFICATION_CONFIG,
        "factual_checker": FACTUAL_CHECKER_CONFIG,
        "human_feedback": HUMAN_FEEDBACK_CONFIG,
        "evaluation": EVAL_CONFIG,
        "baseline": BASELINE_CONFIG,
        "visualization": VIZ_CONFIG,
    }
    
    config_path = CLAUDE_CODE_DIR / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return config_path

if __name__ == "__main__":
    # Save configuration to JSON file
    config_path = save_config()
    print(f"Configuration saved to {config_path}")
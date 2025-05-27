"""
Utility functions for the VERIL experiment.
"""

import logging
import os
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import LOG_FILE, RESULTS_DIR

# Setup logging
def setup_logging(log_file: Path = LOG_FILE, level=logging.INFO) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("veril")
    logger.setLevel(level)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Set formatters
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create logger
logger = setup_logging()

def log_exception(e: Exception) -> None:
    """Log an exception with traceback."""
    logger.error(f"Exception occurred: {str(e)}")
    logger.error(traceback.format_exc())

def time_function(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to run.")
        return result
    return wrapper

def save_json(data: Any, file_path: Union[str, Path]) -> None:
    """Save data as JSON."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(file_path: Union[str, Path]) -> Any:
    """Load data from JSON."""
    with open(file_path, 'r') as f:
        return json.load(f)

def setup_gpu(use_gpu: bool = True, gpu_ids: List[int] = [0]) -> torch.device:
    """Setup GPU device if available."""
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_ids[0]}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(gpu_ids[0])}")
        # Set visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

def plot_learning_curve(
    iterations: List[int], 
    metrics: Dict[str, List[float]], 
    title: str, 
    xlabel: str = "Iterations", 
    ylabel: str = "Value",
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot learning curves for multiple metrics.
    
    Args:
        iterations: List of iteration numbers
        metrics: Dictionary mapping metric names to their values at each iteration
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        output_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    for metric_name, values in metrics.items():
        plt.plot(iterations, values, marker='o', label=metric_name)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved learning curve plot to {output_path}")
    
    return plt.gcf()

def plot_bar_comparison(
    model_names: List[str], 
    metrics: Dict[str, List[float]], 
    title: str, 
    xlabel: str = "Models", 
    ylabel: str = "Performance",
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a bar plot comparing different models on multiple metrics.
    
    Args:
        model_names: Names of the models to compare
        metrics: Dictionary mapping metric names to values for each model
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        output_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    num_models = len(model_names)
    num_metrics = len(metrics)
    bar_width = 0.8 / num_metrics
    
    plt.figure(figsize=(12, 6))
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        x = np.arange(num_models)
        offset = i * bar_width - (num_metrics - 1) * bar_width / 2
        plt.bar(x + offset, values, bar_width, label=metric_name)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(num_models), model_names, rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved bar comparison plot to {output_path}")
    
    return plt.gcf()

def create_results_table(
    model_names: List[str],
    metrics: Dict[str, List[float]],
    title: str
) -> str:
    """
    Create a Markdown table from the results.
    
    Args:
        model_names: Names of the models
        metrics: Dictionary mapping metric names to values for each model
        title: Table title
        
    Returns:
        Markdown table as string
    """
    # Create header
    header = f"### {title}\n\n| Model | " + " | ".join(metrics.keys()) + " |\n"
    separator = "| --- | " + " | ".join(["---"] * len(metrics)) + " |\n"
    
    # Create rows
    rows = ""
    for i, model in enumerate(model_names):
        row_values = [f"{metrics[metric][i]:.4f}" for metric in metrics.keys()]
        rows += f"| {model} | " + " | ".join(row_values) + " |\n"
    
    return header + separator + rows

def extract_code_from_response(response: str) -> str:
    """
    Extract code from a model response, handling various formats.
    
    Args:
        response: The raw model response text
        
    Returns:
        Extracted code as string
    """
    # Try to extract code from between ```python and ``` markers
    import re
    
    # Pattern for Python code blocks
    python_pattern = r"```(?:python)?\s*([\s\S]*?)```"
    matches = re.findall(python_pattern, response)
    
    if matches:
        # Return the longest code block if multiple are found
        return max(matches, key=len).strip()
    
    # If no code blocks, return the entire response
    return response.strip()

def calculate_pass_at_k(n_samples: int, n_correct: int, k: int) -> float:
    """
    Calculate the pass@k metric.
    
    Args:
        n_samples: Total number of samples
        n_correct: Number of correct samples
        k: The k in pass@k
        
    Returns:
        pass@k score
    """
    if n_samples < k:
        return float(n_correct > 0)
    
    # Calculate the probability of getting at least one correct in k samples
    if n_correct == 0:
        return 0.0
    elif n_correct >= k:
        return 1.0
    else:
        return 1.0 - np.prod(1.0 - np.arange(1, k+1) / n_samples)

def calculate_metrics(results: Dict[str, Any], pass_at_k: List[int] = [1, 3, 5]) -> Dict[str, float]:
    """
    Calculate evaluation metrics from results.
    
    Args:
        results: Dictionary containing evaluation results
        pass_at_k: List of k values for the pass@k metric
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Calculate pass@k
    for k in pass_at_k:
        if k <= results["total_samples"]:
            metrics[f"pass@{k}"] = calculate_pass_at_k(
                results["total_samples"], 
                results["correct_samples"], 
                k
            )
    
    # Calculate error rates
    if "error_counts" in results:
        total_errors = sum(results["error_counts"].values())
        metrics["error_rate"] = total_errors / results["total_samples"] if results["total_samples"] > 0 else 0.0
        
        # Calculate error type rates
        for error_type, count in results["error_counts"].items():
            metrics[f"{error_type}_error_rate"] = count / results["total_samples"] if results["total_samples"] > 0 else 0.0
    
    # Calculate verification pass rates
    if "verification_results" in results:
        veri_pass = sum(1 for v in results["verification_results"] if v)
        metrics["veri_pass_rate"] = veri_pass / results["total_samples"] if results["total_samples"] > 0 else 0.0
    
    return metrics
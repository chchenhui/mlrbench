"""
Utility functions for the adaptive code assistant experiment.
"""

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import random
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("adaptive_code_assistant")

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def save_json(data: Union[Dict, List], file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Data saved to {file_path}")

def load_json(file_path: str) -> Union[Dict, List]:
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Data loaded from {file_path}")
    return data

def ensure_dir(dir_path: str) -> None:
    """Ensure that a directory exists."""
    os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Ensured directory exists: {dir_path}")

def plot_learning_curves(
    metrics: Dict[str, Dict[str, List[float]]],
    save_path: str,
    title: str = "Learning Curves",
    xlabel: str = "Interaction Steps",
) -> None:
    """
    Plot learning curves for different methods.
    
    Args:
        metrics: Dictionary with method names as keys and dictionaries of metrics as values
        save_path: Path to save the plot
        title: Plot title
        xlabel: X-axis label
    """
    plt.figure(figsize=(12, 8))
    
    for method_name, method_metrics in metrics.items():
        for metric_name, values in method_metrics.items():
            plt.plot(values, label=f"{method_name}: {metric_name}")
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Learning curves saved to {save_path}")

def plot_comparison_bar(
    metrics: Dict[str, Dict[str, float]],
    save_path: str,
    title: str = "Method Comparison",
    ylabel: str = "Metric Value",
) -> None:
    """
    Plot a bar chart comparing different methods.
    
    Args:
        metrics: Dictionary with method names as keys and dictionaries of metrics as values
        save_path: Path to save the plot
        title: Plot title
        ylabel: Y-axis label
    """
    methods = list(metrics.keys())
    metric_names = list(metrics[methods[0]].keys())
    
    plt.figure(figsize=(12, 8))
    
    # Set position of bars on X axis
    x = np.arange(len(methods))
    width = 0.8 / len(metric_names)
    
    for i, metric_name in enumerate(metric_names):
        metric_values = [metrics[method][metric_name] for method in methods]
        plt.bar(x + i * width, metric_values, width, label=metric_name)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x + width * (len(metric_names) - 1) / 2, methods)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Comparison bar chart saved to {save_path}")

def calculate_statistics(results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics for experimental results.
    
    Args:
        results: Dictionary with metric names as keys and lists of values as values
        
    Returns:
        Dictionary with metric names as keys and dictionaries of statistics as values
    """
    stats = {}
    for metric, values in results.items():
        values_array = np.array(values)
        stats[metric] = {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "median": float(np.median(values_array))
        }
    
    return stats

def create_results_table(metrics: Dict[str, Dict[str, float]]) -> str:
    """
    Create a markdown table from metrics.
    
    Args:
        metrics: Dictionary with method names as keys and dictionaries of metrics as values
        
    Returns:
        Markdown formatted table
    """
    methods = list(metrics.keys())
    metric_names = list(metrics[methods[0]].keys())
    
    table = "| Method | " + " | ".join(metric_names) + " |\n"
    table += "| --- | " + " | ".join(["---"] * len(metric_names)) + " |\n"
    
    for method in methods:
        row = f"| {method} | "
        row += " | ".join([f"{metrics[method][metric]:.4f}" for metric in metric_names])
        row += " |\n"
        table += row
    
    return table

def timer(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to run")
        return result
    return wrapper
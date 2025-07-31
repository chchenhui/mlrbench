"""
Utility functions for the Self-Correcting Language Model experiment.
"""
import time
import torch
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

from config import RESULTS_DIR, FIGURES_DIR, logger

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def log_experiment_config(config: Dict[str, Any]) -> None:
    """Log experiment configuration."""
    logger.info("Experiment Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
        
def time_function(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Log execution time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        # If result is a dictionary, add execution time to it
        if isinstance(result, dict) and "metrics" in result:
            result["metrics"]["latency"] = execution_time
            
        return result
    return wrapper

def save_json(data: Any, file_path: Union[str, Path]) -> None:
    """Save data as JSON."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
        
def load_json(file_path: Union[str, Path]) -> Any:
    """Load data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_metric_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot comparison of metrics across different models.
    
    Args:
        results: Dictionary mapping model names to their metrics
        metric: The metric to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Extract model names and metric values
    models = list(results.keys())
    values = [results[model].get(metric, 0) for model in models]
    
    # Create bar plot
    colors = sns.color_palette("muted", len(models))
    bars = plt.bar(models, values, color=colors)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Customize plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, max(values) * 1.15)  # Add 15% padding to the top
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    plt.close()

def plot_convergence(
    iterations: List[int],
    metrics: Dict[str, List[float]],
    title: str,
    xlabel: str = "Iteration",
    save_path: Optional[Path] = None
) -> None:
    """
    Plot metrics across iterations.
    
    Args:
        iterations: List of iteration numbers
        metrics: Dictionary mapping metric names to lists of values
        title: Plot title
        xlabel: X-axis label
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Plot each metric
    for metric_name, values in metrics.items():
        plt.plot(iterations, values, marker='o', label=metric_name)
    
    # Customize plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Value")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    plt.close()

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: 2D array with confusion matrix values
        class_names: List of class names
        title: Plot title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    # Customize plot
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    plt.close()

def create_results_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    caption: str = "Results"
) -> str:
    """
    Create a markdown table from results.
    
    Args:
        results: Dictionary mapping model names to their metrics
        metrics: List of metrics to include in the table
        caption: Table caption
    
    Returns:
        Markdown-formatted table as a string
    """
    # Create table header
    table = f"### {caption}\n\n"
    table += "| Model | " + " | ".join(metrics) + " |\n"
    table += "| --- | " + " | ".join(["---"] * len(metrics)) + " |\n"
    
    # Add rows for each model
    for model, model_metrics in results.items():
        values = []
        for metric in metrics:
            if metric in model_metrics:
                # Format according to the metric type
                if isinstance(model_metrics[metric], float):
                    values.append(f"{model_metrics[metric]:.3f}")
                else:
                    values.append(str(model_metrics[metric]))
            else:
                values.append("N/A")
        
        table += f"| {model} | " + " | ".join(values) + " |\n"
    
    return table

def make_table_from_data(
    data: List[Dict[str, Any]],
    columns: List[str],
    caption: str = "Data Table"
) -> str:
    """
    Create a markdown table from a list of dictionaries.
    
    Args:
        data: List of dictionaries containing the data
        columns: List of column keys to include
        caption: Table caption
    
    Returns:
        Markdown-formatted table as a string
    """
    # Create table header
    table = f"### {caption}\n\n"
    table += "| " + " | ".join(columns) + " |\n"
    table += "| " + " | ".join(["---"] * len(columns)) + " |\n"
    
    # Add rows
    for item in data:
        row = []
        for col in columns:
            if col in item:
                row.append(str(item[col]))
            else:
                row.append("N/A")
        table += "| " + " | ".join(row) + " |\n"
    
    return table
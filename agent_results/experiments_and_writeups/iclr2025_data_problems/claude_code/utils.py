import os
import json
import logging
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple, Optional, Union, Any

# Set up logging
def setup_logging(log_file: str = "log.txt") -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("influence_space")
    logger.setLevel(logging.INFO)
    
    # Create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set device for PyTorch
def get_device() -> torch.device:
    """Get PyTorch device (GPU if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save and load JSON files
def save_json(data: Any, file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(file_path: str) -> Any:
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

# Save and load PyTorch models
def save_model(model: torch.nn.Module, file_path: str) -> None:
    """Save PyTorch model to a file."""
    torch.save(model.state_dict(), file_path)

def load_model(model: torch.nn.Module, file_path: str) -> torch.nn.Module:
    """Load PyTorch model from a file."""
    model.load_state_dict(torch.load(file_path))
    return model

# Visualization utilities
def plot_loss_curves(train_losses: List[float], val_losses: List[float], 
                   title: str = "Loss Curves", save_path: Optional[str] = None) -> None:
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_metric_curves(metrics: Dict[str, List[float]], 
                     title: str = "Metric Curves", save_path: Optional[str] = None) -> None:
    """Plot multiple metric curves."""
    plt.figure(figsize=(10, 6))
    
    for name, values in metrics.items():
        plt.plot(values, label=name)
    
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_cluster_distribution(cluster_sizes: List[int], 
                            title: str = "Cluster Size Distribution", 
                            save_path: Optional[str] = None) -> None:
    """Plot distribution of cluster sizes."""
    plt.figure(figsize=(12, 6))
    
    # Sort cluster sizes for better visualization
    sorted_sizes = sorted(cluster_sizes, reverse=True)
    
    plt.bar(range(len(sorted_sizes)), sorted_sizes)
    plt.title(title)
    plt.xlabel("Cluster Index (sorted by size)")
    plt.ylabel("Cluster Size")
    plt.grid(True, axis='y')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_influence_distribution(influence_scores: List[float], 
                              title: str = "Influence Score Distribution", 
                              save_path: Optional[str] = None) -> None:
    """Plot distribution of influence scores."""
    plt.figure(figsize=(10, 6))
    
    plt.hist(influence_scores, bins=50)
    plt.title(title)
    plt.xlabel("Influence Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_method_comparison(methods: Dict[str, Dict[str, float]], 
                         metric_name: str, 
                         title: str = "Method Comparison", 
                         save_path: Optional[str] = None) -> None:
    """Plot comparison of different methods on a specific metric."""
    plt.figure(figsize=(10, 6))
    
    # Extract metric values for each method
    methods_names = list(methods.keys())
    metric_values = [methods[method][metric_name] for method in methods_names]
    
    # Create bar chart
    plt.bar(methods_names, metric_values)
    plt.title(title)
    plt.xlabel("Method")
    plt.ylabel(metric_name)
    plt.grid(True, axis='y')
    
    # Add values on top of bars
    for i, value in enumerate(metric_values):
        plt.text(i, value + 0.01, f"{value:.4f}", ha='center')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_fairness_metrics(demographic_gaps: Dict[str, Dict[str, float]], 
                        title: str = "Demographic Performance Gaps", 
                        save_path: Optional[str] = None) -> None:
    """Plot fairness metrics across demographic groups."""
    methods = list(demographic_gaps.keys())
    demographics = list(demographic_gaps[methods[0]].keys())
    
    # Create a figure with subplots for each demographic attribute
    fig, axs = plt.subplots(1, len(demographics), figsize=(15, 6))
    
    # If only one demographic, axs won't be an array
    if len(demographics) == 1:
        axs = [axs]
    
    for i, demo in enumerate(demographics):
        # Extract gaps for this demographic across all methods
        gaps = [demographic_gaps[method][demo] for method in methods]
        
        # Create bar chart
        axs[i].bar(methods, gaps)
        axs[i].set_title(f"{demo} Gap")
        axs[i].set_xlabel("Method")
        axs[i].set_ylabel("Performance Gap")
        axs[i].grid(True, axis='y')
        
        # Add values on top of bars
        for j, value in enumerate(gaps):
            axs[i].text(j, value + 0.01, f"{value:.4f}", ha='center')
    
    plt.tight_layout()
    plt.suptitle(title, y=1.05)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# Evaluation metrics for image-caption retrieval
def compute_recalls(similarity_matrix: np.ndarray, ks: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute Recall@K metrics for image-caption retrieval.
    
    Args:
        similarity_matrix: Matrix of shape (n_images, n_captions) with similarity scores
        ks: List of K values for Recall@K
        
    Returns:
        Dictionary with Recall@K metrics
    """
    n = similarity_matrix.shape[0]
    recall_metrics = {}
    
    # Image to text retrieval
    ranks = np.zeros(n)
    for i in range(n):
        # Get descending ranks of similarities for this image
        inds = np.argsort(similarity_matrix[i])[::-1]
        # Find where the ground truth index (i) appears
        ranks[i] = np.where(inds == i)[0][0]
    
    # Compute Recall@K for image to text
    for k in ks:
        recall_metrics[f"i2t_recall@{k}"] = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
    
    # Text to image retrieval
    ranks = np.zeros(n)
    for i in range(n):
        # Get descending ranks of similarities for this caption
        inds = np.argsort(similarity_matrix[:, i])[::-1]
        # Find where the ground truth index (i) appears
        ranks[i] = np.where(inds == i)[0][0]
    
    # Compute Recall@K for text to image
    for k in ks:
        recall_metrics[f"t2i_recall@{k}"] = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
    
    # Compute average Recall@K
    for k in ks:
        recall_metrics[f"avg_recall@{k}"] = (recall_metrics[f"i2t_recall@{k}"] + 
                                           recall_metrics[f"t2i_recall@{k}"]) / 2
    
    return recall_metrics

# Timers for performance profiling
class Timer:
    """Simple timer for performance profiling."""
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed_time = 0
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.elapsed_time = (datetime.now() - self.start_time).total_seconds()
        print(f"{self.name} took {self.elapsed_time:.2f} seconds")
        
    def get_elapsed_time(self) -> float:
        return self.elapsed_time
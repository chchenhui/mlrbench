"""
Utility functions for the LASS (LLM-Assisted Spuriousity Scout) framework.
"""

import os
import random
import logging
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Tuple, Any, Optional, Union

# Set up logging
def setup_logger(log_file: str = None) -> logging.Logger:
    """
    Set up a logger for the experiment.
    
    Args:
        log_file: Path to the log file.
        
    Returns:
        logger: Logger object.
    """
    logger = logging.getLogger("LASS")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    
    # Create file handler if log file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Visualization utilities
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], 
                         title: str = "Confusion Matrix", cmap: str = "Blues",
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix for model predictions.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        classes: Class names.
        title: Plot title.
        cmap: Colormap for the plot.
        save_path: Path to save the figure.
        
    Returns:
        fig: Figure object.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_feature_embeddings(embeddings: np.ndarray, labels: np.ndarray, 
                          cluster_labels: Optional[np.ndarray] = None,
                          title: str = "Feature Embeddings",
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 2D visualization of feature embeddings using t-SNE.
    
    Args:
        embeddings: Feature embeddings.
        labels: True labels for points.
        cluster_labels: Cluster assignments (if available).
        title: Plot title.
        save_path: Path to save the figure.
        
    Returns:
        fig: Figure object.
    """
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot based on true labels
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label="True Class")
    
    # If cluster labels are provided, create a separate plot
    if cluster_labels is not None:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        scatter2 = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter2, label="Cluster")
        plt.title(f"{title} (Clusters)")
        
        if save_path:
            # Modify save path for the cluster plot
            base, ext = os.path.splitext(save_path)
            plt.savefig(f"{base}_clusters{ext}", bbox_inches='tight')
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_learning_curves(train_metrics: Dict[str, List[float]], val_metrics: Dict[str, List[float]],
                       title: str = "Learning Curves", save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
    """
    Plot learning curves for model training.
    
    Args:
        train_metrics: Dictionary of training metrics (loss, accuracy, etc.).
        val_metrics: Dictionary of validation metrics.
        title: Plot title prefix.
        save_path: Path prefix to save figures.
        
    Returns:
        figs: Dictionary of figure objects.
    """
    figs = {}
    
    for metric_name in train_metrics.keys():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot training metric
        plt.plot(train_metrics[metric_name], label=f'Train {metric_name}')
        
        # Plot validation metric if available
        if metric_name in val_metrics:
            plt.plot(val_metrics[metric_name], label=f'Validation {metric_name}')
        
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f"{title} - {metric_name}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure if path is provided
        if save_path:
            metric_save_path = f"{save_path}_{metric_name.lower()}.png"
            plt.savefig(metric_save_path, bbox_inches='tight')
        
        figs[metric_name] = fig
    
    return figs

def plot_group_performances(group_metrics: Dict[str, Dict[str, float]],
                          title: str = "Group Performance Comparison",
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot performance metrics across different groups.
    
    Args:
        group_metrics: Dictionary mapping group names to their metrics.
        title: Plot title.
        save_path: Path to save the figure.
        
    Returns:
        fig: Figure object.
    """
    groups = list(group_metrics.keys())
    metrics = list(next(iter(group_metrics.values())).keys())
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        values = [group_metrics[group][metric] for group in groups]
        
        # Create bar plot
        ax = axes[i]
        bars = ax.bar(groups, values, color=sns.color_palette("husl", len(groups)))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        ax.set_xlabel('Groups')
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Group")
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def cluster_embeddings(embeddings: np.ndarray, method: str = 'kmeans', 
                     n_clusters: int = 5, **kwargs) -> np.ndarray:
    """
    Cluster feature embeddings to identify potential error groups.
    
    Args:
        embeddings: Feature embeddings.
        method: Clustering method ('kmeans' or 'dbscan').
        n_clusters: Number of clusters (for K-means).
        **kwargs: Additional arguments for clustering algorithms.
        
    Returns:
        cluster_labels: Cluster assignments for each sample.
    """
    if method.lower() == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
    elif method.lower() == 'dbscan':
        clusterer = DBSCAN(**kwargs)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    cluster_labels = clusterer.fit_predict(embeddings)
    return cluster_labels

def save_model(model: torch.nn.Module, path: str, metadata: Optional[Dict] = None) -> None:
    """
    Save trained model and optional metadata.
    
    Args:
        model: PyTorch model.
        path: Path to save the model.
        metadata: Optional metadata to save with the model.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), path)
    
    # Save metadata if provided
    if metadata:
        metadata_path = f"{os.path.splitext(path)[0]}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

def load_model(model: torch.nn.Module, path: str) -> Tuple[torch.nn.Module, Optional[Dict]]:
    """
    Load trained model and metadata if available.
    
    Args:
        model: PyTorch model instance (uninitialized).
        path: Path to the saved model.
        
    Returns:
        model: Loaded PyTorch model.
        metadata: Metadata dictionary if available, None otherwise.
    """
    # Load model state dict
    model.load_state_dict(torch.load(path))
    
    # Try to load metadata if it exists
    metadata = None
    metadata_path = f"{os.path.splitext(path)[0]}_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata

def compute_worst_group_accuracy(outputs: torch.Tensor, labels: torch.Tensor, 
                               group_labels: torch.Tensor) -> Tuple[float, int]:
    """
    Compute worst-group accuracy from model outputs.
    
    Args:
        outputs: Model output logits.
        labels: True class labels.
        group_labels: Group labels for each sample.
        
    Returns:
        worst_acc: Accuracy of the worst performing group.
        worst_group: ID of the worst performing group.
    """
    preds = torch.argmax(outputs, dim=1)
    unique_groups = torch.unique(group_labels).cpu().numpy()
    
    group_accuracies = {}
    for group in unique_groups:
        group_mask = (group_labels == group)
        if not torch.any(group_mask):
            continue
            
        group_correct = torch.sum((preds == labels)[group_mask]).item()
        group_total = torch.sum(group_mask).item()
        group_acc = group_correct / group_total
        group_accuracies[int(group)] = group_acc
    
    worst_group = min(group_accuracies.items(), key=lambda x: x[1])
    return worst_group[1], worst_group[0]

def save_results(results: Dict, path: str) -> None:
    """
    Save experiment results to a file.
    
    Args:
        results: Dictionary of experiment results.
        path: Path to save the results.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

def load_results(path: str) -> Dict:
    """
    Load experiment results from a file.
    
    Args:
        path: Path to the results file.
        
    Returns:
        results: Dictionary of experiment results.
    """
    with open(path, 'r') as f:
        results = json.load(f)
    
    return results
"""
Visualization utilities for Neural Weight Archeology experiments.

This module provides functions for creating visualizations of model weight patterns,
training curves, and evaluation results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_training_curves(
    train_losses: List[float], 
    val_losses: List[float], 
    title: str = "Training Curves",
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        title: Plot title
        save_path: Path to save the figure (if None, figure is displayed)
    """
    plt.figure(figsize=(10, 5))
    
    epochs = list(range(1, len(train_losses) + 1))
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_comparison_bar_chart(
    categories: List[str], 
    values_list: List[List[float]], 
    labels: List[str],
    title: str = "Performance Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Create a bar chart to compare performance metrics across models
    
    Args:
        categories: List of category names (e.g., model names)
        values_list: List of lists of values to plot (one list per metric)
        labels: Labels for each list of values
        title: Plot title
        save_path: Path to save the figure (if None, figure is displayed)
    """
    plt.figure(figsize=(12, 6))
    
    num_categories = len(categories)
    num_metrics = len(values_list)
    
    # Set width of bars
    bar_width = 0.8 / num_metrics
    
    # Set positions of bars on x-axis
    positions = np.arange(num_categories)
    
    # Create bars
    for i, (values, label) in enumerate(zip(values_list, labels)):
        offset = (i - num_metrics / 2 + 0.5) * bar_width
        plt.bar(positions + offset, values, bar_width, label=label)
    
    # Add labels, title, and legend
    plt.xlabel('Models')
    plt.ylabel('Values')
    plt.title(title)
    plt.xticks(positions, categories)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Add value labels on top of bars
    for i, values in enumerate(values_list):
        offset = (i - num_metrics / 2 + 0.5) * bar_width
        for j, value in enumerate(values):
            plt.text(j + offset, value + 0.01, f"{value:.2f}", 
                    ha='center', va='bottom', rotation=0, fontsize=8)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_property_correlation(
    test_results: Dict, 
    title: str = "Property Correlations",
    save_path: Optional[str] = None
) -> None:
    """
    Create a visualization of property correlations
    
    Args:
        test_results: Dictionary of test results
        title: Plot title
        save_path: Path to save the figure (if None, figure is displayed)
    """
    # Check if regression results are available
    if 'regression' not in test_results:
        print("No regression results available for correlation plot")
        return
    
    # Create a simple bar chart of R² values for each target
    r2_scores = test_results['regression']['per_target']['r2_score']
    
    # If available, use the target names from test_results, otherwise use index numbers
    target_names = test_results.get('regression_target_names', [f"Target {i+1}" for i in range(len(r2_scores))])
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(target_names, r2_scores, color='cornflowerblue')
    
    plt.title(title)
    plt.xlabel('Regression Targets')
    plt.ylabel('R² Score')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar, value in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", 
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_weight_patterns(
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: str,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize weight patterns detected by the model
    
    Args:
        model: The model to analyze
        data_loader: DataLoader for the dataset
        device: Device to use for evaluation
        save_path: Path to save the figure (if None, figure is displayed)
    """
    model.eval()
    
    # Collect features from a subset of the data
    features = []
    model_types = []
    accuracies = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= 50:  # Limit to avoid too much data
                break
                
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get features
            features.append(batch['features'].cpu().numpy())
            
            # Get model types and accuracies if available
            if 'class_model_type' in batch:
                model_types.extend(batch['class_model_type'].cpu().numpy())
            
            if 'regression_targets' in batch:
                # Assuming the first regression target is accuracy
                accuracies.extend(batch['regression_targets'][:, 0].cpu().numpy())
    
    # Combine features
    features = np.vstack(features)
    
    # Create visualizations
    plt.figure(figsize=(16, 12))
    
    # Apply dimensionality reduction for visualization
    pca = PCA(n_components=2)
    
    # PCA projection
    pca_result = pca.fit_transform(features)
    
    # Check if we have enough samples for t-SNE (at least 5)
    n_samples = features.shape[0]
    use_tsne = n_samples >= 5
    
    if use_tsne:
        # t-SNE with adaptive perplexity based on sample size
        perplexity = min(5, n_samples - 1)  # ensure perplexity is valid
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_result = tsne.fit_transform(features)
    else:
        # If too few samples, duplicate PCA result as placeholder
        tsne_result = pca_result.copy()
    
    # Plot PCA
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=model_types if model_types else 'blue', cmap='viridis', alpha=0.7)
    plt.title('PCA Projection of Weight Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    if model_types:
        plt.colorbar(scatter, label='Model Type')
    
    # Plot t-SNE
    plt.subplot(2, 2, 2)
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=model_types if model_types else 'blue', cmap='viridis', alpha=0.7)
    plt.title('t-SNE Projection of Weight Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    if model_types:
        plt.colorbar(scatter, label='Model Type')
    
    # Plot PCA colored by accuracy
    plt.subplot(2, 2, 3)
    if accuracies:
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=accuracies, cmap='plasma', alpha=0.7)
        plt.colorbar(scatter, label='Accuracy')
    else:
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    plt.title('PCA Projection (Colored by Accuracy)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # Plot t-SNE colored by accuracy
    plt.subplot(2, 2, 4)
    if accuracies:
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=accuracies, cmap='plasma', alpha=0.7)
        plt.colorbar(scatter, label='Accuracy')
    else:
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7)
    plt.title('t-SNE Projection (Colored by Accuracy)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_attention_weights(
    attention_weights: np.ndarray,
    node_labels: List[str],
    title: str = "Attention Weights Visualization",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize attention weights between nodes
    
    Args:
        attention_weights: Matrix of attention weights of shape [N, N]
        node_labels: Labels for each node
        title: Plot title
        save_path: Path to save the figure (if None, figure is displayed)
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=node_labels,
        yticklabels=node_labels
    )
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
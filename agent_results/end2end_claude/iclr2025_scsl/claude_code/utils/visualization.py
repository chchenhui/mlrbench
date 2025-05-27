#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for CIMRL experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from .training import predict
from .metrics import compute_group_metrics


def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_path=None):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        save_path: Path to save the plot
        
    Returns:
        fig: Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(train_losses) + 1)
    
    # Plot loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot accuracy curves
    axes[0, 1].plot(epochs, train_metrics['accuracy'], 'b-', label='Training Accuracy')
    axes[0, 1].plot(epochs, val_metrics['accuracy'], 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot worst-group accuracy curves
    axes[1, 0].plot(epochs, train_metrics['worst_group_accuracy'], 'b-', label='Training Worst-Group Acc')
    axes[1, 0].plot(epochs, val_metrics['worst_group_accuracy'], 'r-', label='Validation Worst-Group Acc')
    axes[1, 0].set_title('Worst-Group Accuracy Curves')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Worst-Group Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot AUC curves
    axes[1, 1].plot(epochs, train_metrics['average_auc'], 'b-', label='Training AUC')
    axes[1, 1].plot(epochs, val_metrics['average_auc'], 'r-', label='Validation AUC')
    axes[1, 1].set_title('AUC Curves')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_performance_comparison(model_names, metrics_dict, metric_name='accuracy', groups=False, save_path=None):
    """
    Plot performance comparison between models.
    
    Args:
        model_names: List of model names
        metrics_dict: Dictionary of metrics for each model
        metric_name: Metric to plot
        groups: Whether to plot group-wise metrics
        save_path: Path to save the plot
        
    Returns:
        fig: Figure object
    """
    if groups:
        # Plot group-wise performance
        num_groups = len([k for k in metrics_dict[model_names[0]] if k.startswith('group_')])
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bar_width = 0.8 / len(model_names)
        index = np.arange(num_groups)
        
        for i, model_name in enumerate(model_names):
            group_metrics = [metrics_dict[model_name][f'group_{g}'][metric_name] for g in range(num_groups)]
            ax.bar(index + i * bar_width, group_metrics, bar_width, label=model_name)
        
        ax.set_xlabel('Group')
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f'Group-wise {metric_name.capitalize()} Comparison')
        ax.set_xticks(index + bar_width * (len(model_names) - 1) / 2)
        ax.set_xticklabels([f'Group {g}' for g in range(num_groups)])
        ax.legend()
        ax.grid(True, axis='y')
    
    else:
        # Plot overall performance
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = [metrics_dict[model_name][metric_name] for model_name in model_names]
        
        ax.bar(model_names, metrics)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f'{metric_name.capitalize()} Comparison')
        
        # Add values on top of bars
        for i, v in enumerate(metrics):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        ax.grid(True, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(conf_matrix, classes, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix
        classes: List of class names
        save_path: Path to save the plot
        
    Returns:
        fig: Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # Plot heatmap
    sns.heatmap(conf_matrix_norm, annot=conf_matrix, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(feature_importance, save_path=None):
    """
    Plot feature importance.
    
    Args:
        feature_importance: Dictionary of feature importance scores
        save_path: Path to save the plot
        
    Returns:
        fig: Figure object
    """
    # This is a placeholder for actual implementation
    # Would need to be customized based on the specific feature importance computation
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Example: plotting top-10 features
    features = list(feature_importance.keys())[:10]
    importances = [feature_importance[f] for f in features]
    
    ax.barh(features, importances)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importance')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_visualizations(model, dataloader, device, save_path=None):
    """
    Visualize feature representations.
    
    Args:
        model: Trained model
        dataloader: DataLoader for the dataset
        device: Device to use
        save_path: Path to save the plot
        
    Returns:
        fig: Figure object
    """
    # Get predictions and group labels
    targets, preds, probs, group_labels = predict(model, dataloader, device)
    
    # Get feature representations (this assumes the model outputs representations)
    representations = []
    for batch in dataloader:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            elif isinstance(batch[key], dict):
                for subkey in batch[key]:
                    if isinstance(batch[key][subkey], torch.Tensor):
                        batch[key][subkey] = batch[key][subkey].to(device)
        
        with torch.no_grad():
            outputs = model(batch, compute_loss=False)
            
            # Assuming the model outputs representations
            if 'representations' in outputs:
                batch_reps = outputs['representations']
                
                # Concatenate different types of representations if needed
                if isinstance(batch_reps, dict):
                    # Extract shared representation (most important for CIMRL)
                    if 'shared' in batch_reps:
                        batch_reps = batch_reps['shared']
                    else:
                        # Fallback to first representation type
                        batch_reps = list(batch_reps.values())[0]
                
                representations.append(batch_reps.cpu().numpy())
    
    # Concatenate representations
    representations = np.concatenate(representations, axis=0)
    
    # Apply dimensionality reduction
    if representations.shape[1] > 2:
        # First reduce with PCA to 50 dimensions if necessary
        if representations.shape[1] > 50:
            pca = PCA(n_components=50)
            representations = pca.fit_transform(representations)
        
        # Then apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        representations_2d = tsne.fit_transform(representations)
    else:
        representations_2d = representations
    
    # Create figure with multiple subplots for different visualizations
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot by class
    scatter = axes[0].scatter(
        representations_2d[:, 0],
        representations_2d[:, 1],
        c=targets,
        cmap='viridis',
        alpha=0.7,
        s=30
    )
    
    legend1 = axes[0].legend(*scatter.legend_elements(), title="Classes")
    axes[0].add_artist(legend1)
    axes[0].set_title('Feature Representations by Class')
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    
    # Plot by group (if available)
    if group_labels is not None:
        scatter = axes[1].scatter(
            representations_2d[:, 0],
            representations_2d[:, 1],
            c=group_labels,
            cmap='rainbow',
            alpha=0.7,
            s=30
        )
        
        legend2 = axes[1].legend(*scatter.legend_elements(), title="Groups")
        axes[1].add_artist(legend2)
        axes[1].set_title('Feature Representations by Group')
        axes[1].set_xlabel('Component 1')
        axes[1].set_ylabel('Component 2')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_group_performance_comparison(models_dict, dataset, metric='accuracy', save_path=None):
    """
    Plot group-wise performance comparison across models.
    
    Args:
        models_dict: Dictionary of model names and their results
        dataset: Dataset name
        metric: Metric to plot
        save_path: Path to save the plot
        
    Returns:
        fig: Figure object
    """
    # Load results for each model
    model_names = list(models_dict.keys())
    group_metrics = {}
    
    for model_name, results in models_dict.items():
        if 'group_metrics' in results:
            group_metrics[model_name] = results['group_metrics']
    
    if not group_metrics:
        return None
    
    # Determine number of groups
    first_model = next(iter(group_metrics.values()))
    groups = [g for g in first_model.keys() if g.startswith('group_')]
    
    # Extract metric for each group and model
    data = {model_name: [] for model_name in model_names}
    
    for group in groups:
        for model_name in model_names:
            if group in group_metrics[model_name]:
                if metric in group_metrics[model_name][group]:
                    data[model_name].append(group_metrics[model_name][group][metric])
                else:
                    data[model_name].append(np.nan)
            else:
                data[model_name].append(np.nan)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.8 / len(model_names)
    index = np.arange(len(groups))
    
    for i, model_name in enumerate(model_names):
        ax.bar(index + i * bar_width, data[model_name], bar_width, label=model_name)
    
    ax.set_xlabel('Group')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'Group-wise {metric.capitalize()} Comparison on {dataset}')
    ax.set_xticks(index + bar_width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([g.replace('group_', 'Group ') for g in groups])
    ax.legend()
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_learning_dynamics(train_metrics, val_metrics, metric_name='accuracy', save_path=None):
    """
    Plot learning dynamics for a specific metric.
    
    Args:
        train_metrics: Dictionary of training metrics for different models
        val_metrics: Dictionary of validation metrics for different models
        metric_name: Metric to plot
        save_path: Path to save the plot
        
    Returns:
        fig: Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = list(train_metrics.keys())
    
    for model_name in model_names:
        epochs = range(1, len(train_metrics[model_name][metric_name]) + 1)
        ax.plot(epochs, train_metrics[model_name][metric_name], '-', label=f'{model_name} (Train)')
        ax.plot(epochs, val_metrics[model_name][metric_name], '--', label=f'{model_name} (Val)')
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f'{metric_name.capitalize()} Learning Dynamics')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
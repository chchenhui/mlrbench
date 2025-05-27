"""
Visualization utilities for AIFS experiments.

This module provides visualization functions for analyzing and
presenting experimental results.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any


def set_plotting_style():
    """Set a consistent style for all plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16


def plot_training_history(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: str,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> str:
    """
    Plot training history for multiple models and metrics.
    
    Args:
        histories: Dictionary mapping model names to training histories
            Each history is a dict mapping metric names to list of values
        save_path: Path to save the figure
        metrics: List of metrics to plot (defaults to ['train_loss', 'val_loss', 'train_acc', 'val_acc'])
        figsize: Figure size as (width, height)
        
    Returns:
        Path to the saved figure
    """
    set_plotting_style()
    
    # Default metrics if not specified
    if metrics is None:
        metrics = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    
    # Determine number of rows and columns for subplots
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle case where there's only one metric
    if n_metrics == 1:
        axes = np.array([axes])
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot metric for each model
        for model_name, history in histories.items():
            if metric in history:
                ax.plot(history[metric], label=f"{model_name}")
        
        # Set labels and title
        metric_title = ' '.join(w.capitalize() for w in metric.split('_'))
        ax.set_title(metric_title)
        ax.set_xlabel('Epoch')
        
        # Set y-label based on metric type
        if 'loss' in metric:
            ax.set_ylabel('Loss')
        elif 'acc' in metric:
            ax.set_ylabel('Accuracy (%)')
        else:
            ax.set_ylabel(metric_title)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_group_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    metric: str = 'accuracy',
    figsize: Tuple[int, int] = (10, 6)
) -> str:
    """
    Plot performance comparison across groups for multiple models.
    
    Args:
        results: Dictionary mapping model names to results
            Each result is a dict with metrics for different groups
        save_path: Path to save the figure
        metric: Metric to compare (default: 'accuracy')
        figsize: Figure size as (width, height)
        
    Returns:
        Path to the saved figure
    """
    set_plotting_style()
    
    # Extract data
    models = list(results.keys())
    
    # Check if results have the required group metrics
    if not all(('aligned_accuracy' in results[model] and 
               'unaligned_accuracy' in results[model]) for model in models):
        raise ValueError("Results must contain 'aligned_accuracy' and 'unaligned_accuracy' for all models")
    
    # Prepare data for plotting
    aligned_values = [results[model]['aligned_accuracy'] for model in models]
    unaligned_values = [results[model]['unaligned_accuracy'] for model in models]
    overall_values = [results[model]['overall_accuracy'] for model in models]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set positions and width for the bars
    x = np.arange(len(models))
    width = 0.25
    
    # Create bars
    aligned_bars = ax.bar(x - width, aligned_values, width, label='Aligned Group')
    unaligned_bars = ax.bar(x, unaligned_values, width, label='Unaligned Group')
    overall_bars = ax.bar(x + width, overall_values, width, label='Overall')
    
    # Customize the plot
    ax.set_ylabel(f'{metric.capitalize()} (%)')
    ax.set_title(f'Group Performance Comparison ({metric.capitalize()})')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels above the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    add_labels(aligned_bars)
    add_labels(unaligned_bars)
    add_labels(overall_bars)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_disparity_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    figsize: Tuple[int, int] = (10, 6),
    sort_by_disparity: bool = True
) -> str:
    """
    Plot disparity comparison (difference between aligned and unaligned group performance)
    for multiple models.
    
    Args:
        results: Dictionary mapping model names to results
        save_path: Path to save the figure
        figsize: Figure size as (width, height)
        sort_by_disparity: Whether to sort models by disparity
        
    Returns:
        Path to the saved figure
    """
    set_plotting_style()
    
    # Extract data
    models = list(results.keys())
    
    # Check if results have the required disparity metric
    if not all('disparity' in results[model] for model in models):
        raise ValueError("Results must contain 'disparity' for all models")
    
    # Prepare data for plotting
    disparities = [results[model]['disparity'] for model in models]
    
    # Optionally sort by disparity
    if sort_by_disparity:
        sort_idx = np.argsort(disparities)
        models = [models[i] for i in sort_idx]
        disparities = [disparities[i] for i in sort_idx]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bars
    colors = ['green' if d < 0.1 else 'orange' if d < 0.3 else 'red' for d in disparities]
    bars = ax.barh(models, disparities, color=colors)
    
    # Customize the plot
    ax.set_xlabel('Disparity (Aligned - Unaligned Accuracy)')
    ax.set_title('Model Fairness Comparison')
    ax.set_xlim(0, max(disparities) * 1.1 + 0.05)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    # Add reference line for ideal case (zero disparity)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    # Add value labels next to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', va='center', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_confusion_matrices(
    results: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: str,
    figsize: Tuple[int, int] = (15, 10)
) -> str:
    """
    Plot confusion matrices for multiple models.
    
    Args:
        results: Dictionary mapping model names to confusion matrices
        class_names: List of class names
        save_path: Path to save the figure
        figsize: Figure size as (width, height)
        
    Returns:
        Path to the saved figure
    """
    set_plotting_style()
    
    # Determine number of models
    n_models = len(results)
    
    # Determine layout for subplots
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle case where there's only one model
    if n_models == 1:
        axes = np.array([axes])
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Plot confusion matrix for each model
    for i, (model_name, conf_matrix) in enumerate(results.items()):
        ax = axes[i]
        
        # Plot the confusion matrix
        im = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                          xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        # Set labels and title
        ax.set_title(f'{model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_feature_importance(
    feature_importances: Dict[str, np.ndarray],
    feature_names: List[str],
    save_path: str,
    figsize: Tuple[int, int] = (12, 8),
    top_n: Optional[int] = None
) -> str:
    """
    Plot feature importance for multiple models.
    
    Args:
        feature_importances: Dictionary mapping model names to feature importance arrays
        feature_names: List of feature names
        save_path: Path to save the figure
        figsize: Figure size as (width, height)
        top_n: Number of top features to display (None for all)
        
    Returns:
        Path to the saved figure
    """
    set_plotting_style()
    
    # Determine number of models
    n_models = len(feature_importances)
    
    # Determine layout for subplots
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle case where there's only one model
    if n_models == 1:
        axes = np.array([axes])
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Plot feature importance for each model
    for i, (model_name, importances) in enumerate(feature_importances.items()):
        ax = axes[i]
        
        # Ensure lengths match
        if len(importances) != len(feature_names):
            raise ValueError(f"Length mismatch: {len(importances)} importances vs {len(feature_names)} feature names")
        
        # Sort features by importance
        sorted_idx = np.argsort(importances)
        
        # Get top N features if specified
        if top_n is not None and top_n < len(sorted_idx):
            sorted_idx = sorted_idx[-top_n:]
        
        # Extract sorted names and values
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_importances = importances[sorted_idx]
        
        # Plot horizontal bars
        ax.barh(sorted_names, sorted_importances)
        
        # Set labels and title
        ax.set_title(f'{model_name}')
        ax.set_xlabel('Importance')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_latent_space_visualization(
    latent_features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    save_path: str,
    method: str = 'tsne',
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Latent Space Visualization'
) -> str:
    """
    Plot latent space visualization using dimensionality reduction techniques.
    
    Args:
        latent_features: Array of latent features [n_samples, n_features]
        labels: Array of class labels [n_samples]
        groups: Array of group labels [n_samples]
        save_path: Path to save the figure
        method: Dimensionality reduction method ('tsne' or 'pca')
        figsize: Figure size as (width, height)
        title: Plot title
        
    Returns:
        Path to the saved figure
    """
    set_plotting_style()
    
    # Perform dimensionality reduction
    if method.lower() == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(latent_features)
        
    elif method.lower() == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(latent_features)
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'tsne' or 'pca'.")
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique labels and groups
    unique_labels = np.unique(labels)
    unique_groups = np.unique(groups)
    
    # Create a colormap with a different color for each label
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))
    
    # Use different markers for different groups
    markers = ['o', 'x']
    
    # Plot each class-group combination
    for label_idx, label in enumerate(unique_labels):
        for group_idx, group in enumerate(unique_groups):
            mask = (labels == label) & (groups == group)
            if sum(mask) > 0:
                ax.scatter(
                    reduced_features[mask, 0], 
                    reduced_features[mask, 1],
                    c=[cmap(label_idx)],
                    marker=markers[group_idx if group_idx < len(markers) else 0],
                    label=f'Class {label}, Group {group}',
                    alpha=0.7
                )
    
    # Add legend and title
    ax.legend(title='Class, Group', loc='best')
    ax.set_title(title)
    ax.set_xlabel(f'{method.upper()} Dimension 1')
    ax.set_ylabel(f'{method.upper()} Dimension 2')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def create_summary_plots(
    training_histories: Dict[str, Dict[str, List[float]]],
    evaluation_results: Dict[str, Dict[str, float]],
    class_names: List[str],
    confusion_matrices: Optional[Dict[str, np.ndarray]] = None,
    feature_importances: Optional[Dict[str, np.ndarray]] = None,
    feature_names: Optional[List[str]] = None,
    latent_features: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    save_dir: str = 'plots'
) -> Dict[str, str]:
    """
    Create a comprehensive set of plots summarizing experimental results.
    
    Args:
        training_histories: Dictionary mapping model names to training histories
        evaluation_results: Dictionary mapping model names to evaluation results
        class_names: List of class names
        confusion_matrices: Optional dictionary mapping model names to confusion matrices
        feature_importances: Optional dictionary mapping model names to feature importance arrays
        feature_names: Optional list of feature names (required if feature_importances is provided)
        latent_features: Optional dictionary mapping model names to (features, labels, groups) tuples
        save_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot types to file paths
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a dictionary to store all plot paths
    plot_paths = {}
    
    # Plot training history
    history_path = os.path.join(save_dir, 'training_history.png')
    plot_paths['training_history'] = plot_training_history(
        training_histories, history_path
    )
    
    # Plot group comparison
    group_path = os.path.join(save_dir, 'group_comparison.png')
    plot_paths['group_comparison'] = plot_group_comparison(
        evaluation_results, group_path
    )
    
    # Plot disparity comparison
    disparity_path = os.path.join(save_dir, 'disparity_comparison.png')
    plot_paths['disparity_comparison'] = plot_disparity_comparison(
        evaluation_results, disparity_path
    )
    
    # Plot confusion matrices if provided
    if confusion_matrices is not None:
        conf_matrix_path = os.path.join(save_dir, 'confusion_matrices.png')
        plot_paths['confusion_matrices'] = plot_confusion_matrices(
            confusion_matrices, class_names, conf_matrix_path
        )
    
    # Plot feature importance if provided
    if feature_importances is not None and feature_names is not None:
        importance_path = os.path.join(save_dir, 'feature_importance.png')
        plot_paths['feature_importance'] = plot_feature_importance(
            feature_importances, feature_names, importance_path
        )
    
    # Plot latent space visualization if provided
    if latent_features is not None:
        for model_name, (features, labels, groups) in latent_features.items():
            latent_path = os.path.join(save_dir, f'latent_space_{model_name}.png')
            plot_paths[f'latent_space_{model_name}'] = plot_latent_space_visualization(
                features, labels, groups, latent_path,
                title=f'Latent Space Visualization - {model_name}'
            )
    
    return plot_paths


def save_results_to_json(
    results: Dict[str, Any],
    filepath: str
) -> str:
    """
    Save experimental results to a JSON file.
    
    Args:
        results: Dictionary of results to save
        filepath: Path to save the JSON file
        
    Returns:
        Path to the saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Clean results to ensure JSON serializable
    cleaned_results = {}
    
    for model_name, model_results in results.items():
        cleaned_model_results = {}
        
        for metric_name, metric_value in model_results.items():
            # Convert numpy types to Python native types
            if isinstance(metric_value, (np.integer, np.floating)):
                cleaned_model_results[metric_name] = float(metric_value)
            elif isinstance(metric_value, (list, np.ndarray)):
                # Convert list-like objects
                if len(metric_value) > 0 and isinstance(metric_value[0], (np.integer, np.floating)):
                    cleaned_model_results[metric_name] = [float(v) for v in metric_value]
                else:
                    cleaned_model_results[metric_name] = metric_value
            elif isinstance(metric_value, dict):
                # Handle nested dictionaries
                cleaned_nested = {}
                for k, v in metric_value.items():
                    if isinstance(v, (np.integer, np.floating)):
                        cleaned_nested[str(k)] = float(v)
                    else:
                        cleaned_nested[str(k)] = v
                cleaned_model_results[metric_name] = cleaned_nested
            else:
                cleaned_model_results[metric_name] = metric_value
        
        cleaned_results[model_name] = cleaned_model_results
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(cleaned_results, f, indent=2)
    
    return filepath


def create_results_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    filepath: str,
    format: str = 'csv'
) -> str:
    """
    Create a results table and save it as CSV or markdown.
    
    Args:
        results: Dictionary mapping model names to results
        metrics: List of metrics to include in the table
        filepath: Path to save the table
        format: Output format ('csv' or 'markdown')
        
    Returns:
        Path to the saved table
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Create a DataFrame from the results
    models = list(results.keys())
    table_data = {'Model': models}
    
    for metric in metrics:
        if all(metric in results[model] for model in models):
            values = [results[model][metric] for model in models]
            table_data[metric] = values
    
    df = pd.DataFrame(table_data)
    
    # Save the table in the specified format
    if format.lower() == 'csv':
        df.to_csv(filepath, index=False)
    elif format.lower() == 'markdown':
        # Convert DataFrame to markdown table
        markdown_table = df.to_markdown(index=False)
        with open(filepath, 'w') as f:
            f.write(markdown_table)
    else:
        raise ValueError(f"Unknown format: {format}. Choose 'csv' or 'markdown'.")
    
    return filepath
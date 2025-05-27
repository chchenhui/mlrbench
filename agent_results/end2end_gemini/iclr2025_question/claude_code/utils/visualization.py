"""
Visualization utilities for the AUG-RAG experiments.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Any, Optional
import logging
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

def set_plotting_style():
    """Set the plotting style for consistent visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['savefig.dpi'] = 100
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1

def compare_models_bar_chart(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str,
    title: str,
    output_path: str,
    higher_is_better: bool = True,
    sort_by_performance: bool = True
) -> str:
    """
    Create a bar chart comparing models on a specific metric.
    
    Args:
        metrics: Dictionary with model names as keys and metric dictionaries as values.
        metric_name: Name of the metric to plot.
        title: Title of the plot.
        output_path: Directory to save the plot to.
        higher_is_better: Whether higher values of the metric are better.
        sort_by_performance: Whether to sort models by performance.
    
    Returns:
        Path to the saved plot.
    """
    set_plotting_style()
    
    # Extract metric values
    model_names = []
    metric_values = []
    
    for model_name, model_metrics in metrics.items():
        if metric_name in model_metrics:
            model_names.append(model_name)
            metric_values.append(model_metrics[metric_name])
    
    if not model_names:
        logger.warning(f"No data for metric {metric_name}")
        return None
    
    # Sort if requested
    if sort_by_performance:
        # Sort in descending order if higher is better, otherwise ascending
        data = list(zip(model_names, metric_values))
        data.sort(key=lambda x: x[1], reverse=higher_is_better)
        model_names, metric_values = zip(*data)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Set color based on model name patterns
    colors = []
    for name in model_names:
        if 'aug_rag' in name.lower():
            colors.append('#1f77b4')  # Blue for AUG-RAG models
        elif 'rag' in name.lower():
            colors.append('#ff7f0e')  # Orange for RAG models
        else:
            colors.append('#2ca02c')  # Green for baseline models
    
    # Create bar chart
    bars = plt.bar(model_names, metric_values, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel(metric_name)
    plt.title(title)
    
    # Improve readability
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust y-axis to start from 0 or slightly lower
    y_min = 0
    if not higher_is_better and min(metric_values) > 0:
        y_min = max(0, min(metric_values) * 0.9)
    plt.ylim(y_min, max(metric_values) * 1.1)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_path, exist_ok=True)
    fig_path = os.path.join(output_path, f"{metric_name}_comparison.png")
    plt.savefig(fig_path)
    plt.close()
    
    logger.info(f"Saved bar chart to {fig_path}")
    return fig_path


def plot_uncertainty_threshold_experiment(
    thresholds: List[float],
    hallucination_rates: List[float],
    retrieval_frequencies: List[float],
    output_path: str
) -> str:
    """
    Plot the effect of uncertainty thresholds on hallucination rates 
    and retrieval frequencies.
    
    Args:
        thresholds: List of threshold values.
        hallucination_rates: List of hallucination rates for each threshold.
        retrieval_frequencies: List of retrieval frequencies for each threshold.
        output_path: Directory to save the plot to.
    
    Returns:
        Path to the saved plot.
    """
    set_plotting_style()
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot hallucination rate
    color = 'tab:red'
    ax1.set_xlabel('Uncertainty Threshold')
    ax1.set_ylabel('Hallucination Rate', color=color)
    ax1.plot(thresholds, hallucination_rates, marker='o', color=color, 
             linewidth=2, label='Hallucination Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create second y-axis for retrieval frequency
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Retrieval Frequency', color=color)
    ax2.plot(thresholds, retrieval_frequencies, marker='s', color=color, 
             linewidth=2, label='Retrieval Frequency')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title and grid
    plt.title('Effect of Uncertainty Threshold on Hallucination Rate and Retrieval Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_path, exist_ok=True)
    fig_path = os.path.join(output_path, "uncertainty_threshold_experiment.png")
    plt.savefig(fig_path)
    plt.close()
    
    logger.info(f"Saved threshold experiment plot to {fig_path}")
    return fig_path


def plot_calibration_curve(
    confidences: List[float],
    correctness: List[bool],
    output_path: str,
    num_bins: int = 10,
    model_name: str = "Model"
) -> str:
    """
    Plot a calibration curve showing expected vs observed accuracy.
    
    Args:
        confidences: List of model confidence scores.
        correctness: List of boolean values indicating correctness.
        output_path: Directory to save the plot to.
        num_bins: Number of bins for calibration analysis.
        model_name: Name of the model.
    
    Returns:
        Path to the saved plot.
    """
    set_plotting_style()
    
    # Convert to numpy arrays
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=int)
    
    # Create bins and bin assignments
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries[1:-1])
    
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    # Compute bin statistics
    for i in range(len(confidences)):
        bin_idx = bin_indices[i]
        bin_accuracies[bin_idx] += correctness[i]
        bin_confidences[bin_idx] += confidences[i]
        bin_counts[bin_idx] += 1
    
    # Compute averages
    for i in range(num_bins):
        if bin_counts[i] > 0:
            bin_accuracies[i] /= bin_counts[i]
            bin_confidences[i] /= bin_counts[i]
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Plot the diagonal (perfect calibration)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Plot the calibration curve
    valid_bins = bin_counts > 0
    plt.plot(bin_confidences[valid_bins], bin_accuracies[valid_bins], 
             marker='o', linewidth=2, label=f'{model_name} Calibration')
    
    # Add bin counts as annotations
    for i in range(num_bins):
        if bin_counts[i] > 0:
            plt.annotate(f'{int(bin_counts[i])}', 
                        (bin_confidences[i], bin_accuracies[i]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')
    
    # Add labels and title
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Calibration Curve for {model_name}')
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set equal aspect ratio and limits
    plt.axis('square')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Save figure
    os.makedirs(output_path, exist_ok=True)
    fig_path = os.path.join(output_path, f"{model_name.replace(' ', '_')}_calibration.png")
    plt.savefig(fig_path)
    plt.close()
    
    logger.info(f"Saved calibration curve to {fig_path}")
    return fig_path


def plot_uncertainty_histograms(
    correct_uncertainties: List[float],
    incorrect_uncertainties: List[float],
    output_path: str,
    model_name: str = "Model",
    metric_name: str = "Uncertainty"
) -> str:
    """
    Plot histograms of uncertainty values for correct and incorrect predictions.
    
    Args:
        correct_uncertainties: List of uncertainty values for correct predictions.
        incorrect_uncertainties: List of uncertainty values for incorrect predictions.
        output_path: Directory to save the plot to.
        model_name: Name of the model.
        metric_name: Name of the uncertainty metric.
    
    Returns:
        Path to the saved plot.
    """
    set_plotting_style()
    
    plt.figure(figsize=(12, 8))
    
    # Plot histograms
    bins = np.linspace(0, 1, 20)
    plt.hist(correct_uncertainties, bins=bins, alpha=0.7, 
             label=f'Correct Predictions (n={len(correct_uncertainties)})',
             color='green', edgecolor='black')
    plt.hist(incorrect_uncertainties, bins=bins, alpha=0.7, 
             label=f'Incorrect Predictions (n={len(incorrect_uncertainties)})',
             color='red', edgecolor='black')
    
    # Add labels and title
    plt.xlabel(metric_name)
    plt.ylabel('Count')
    plt.title(f'{metric_name} Distribution for Correct and Incorrect Predictions ({model_name})')
    
    # Add mean lines
    if correct_uncertainties:
        plt.axvline(np.mean(correct_uncertainties), color='green', linestyle='dashed', 
                   linewidth=2, label=f'Mean (Correct): {np.mean(correct_uncertainties):.3f}')
    if incorrect_uncertainties:
        plt.axvline(np.mean(incorrect_uncertainties), color='red', linestyle='dashed', 
                   linewidth=2, label=f'Mean (Incorrect): {np.mean(incorrect_uncertainties):.3f}')
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    os.makedirs(output_path, exist_ok=True)
    fig_path = os.path.join(output_path, f"{model_name.replace(' ', '_')}_{metric_name.lower()}_histogram.png")
    plt.savefig(fig_path)
    plt.close()
    
    logger.info(f"Saved uncertainty histogram to {fig_path}")
    return fig_path


def plot_retrieval_patterns(
    uncertainty_values: List[float],
    retrieval_triggered: List[bool],
    thresholds: List[float],
    output_path: str,
    model_name: str = "AUG-RAG"
) -> str:
    """
    Plot uncertainty values and retrieval patterns across a sequence.
    
    Args:
        uncertainty_values: List of uncertainty values at each step.
        retrieval_triggered: List of booleans indicating retrieval at each step.
        thresholds: List of threshold values at each step.
        output_path: Directory to save the plot to.
        model_name: Name of the model.
    
    Returns:
        Path to the saved plot.
    """
    set_plotting_style()
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Plot uncertainty values
    steps = list(range(len(uncertainty_values)))
    plt.plot(steps, uncertainty_values, marker='o', color='blue', 
             linewidth=2, label='Uncertainty')
    
    # Plot thresholds
    plt.plot(steps, thresholds, linestyle='--', color='green', 
             linewidth=2, label='Threshold')
    
    # Highlight retrieval steps
    retrieval_steps = [i for i, triggered in enumerate(retrieval_triggered) if triggered]
    retrieval_uncertainties = [uncertainty_values[i] for i in retrieval_steps]
    plt.scatter(retrieval_steps, retrieval_uncertainties, color='red', s=150, 
                marker='*', label='Retrieval Triggered')
    
    # Add labels and title
    plt.xlabel('Generation Step')
    plt.ylabel('Uncertainty Value')
    plt.title(f'Uncertainty Values and Retrieval Patterns ({model_name})')
    
    # Improve readability
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Make x-axis integers only
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save figure
    os.makedirs(output_path, exist_ok=True)
    fig_path = os.path.join(output_path, f"{model_name.replace(' ', '_')}_retrieval_pattern.png")
    plt.savefig(fig_path)
    plt.close()
    
    logger.info(f"Saved retrieval pattern plot to {fig_path}")
    return fig_path


def plot_ablation_results(
    ablation_type: str,
    ablation_values: List[Union[str, float]],
    metrics: Dict[str, List[float]],
    output_path: str
) -> str:
    """
    Plot results from ablation studies.
    
    Args:
        ablation_type: Type of ablation study (e.g., "uncertainty_methods").
        ablation_values: List of ablation values (e.g., method names).
        metrics: Dictionary with metric names as keys and lists of values as values.
        output_path: Directory to save the plot to.
    
    Returns:
        Path to the saved plot.
    """
    set_plotting_style()
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create a subplot for each metric
    num_metrics = len(metrics)
    num_cols = min(2, num_metrics)
    num_rows = (num_metrics + num_cols - 1) // num_cols
    
    for i, (metric_name, values) in enumerate(metrics.items(), 1):
        plt.subplot(num_rows, num_cols, i)
        
        # Convert ablation_values to string for consistent labels
        x_values = [str(val) for val in ablation_values]
        
        # Set colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
        
        # Create bar chart
        bars = plt.bar(x_values, values, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        # Add labels and title
        plt.xlabel(ablation_type.replace('_', ' ').title())
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} by {ablation_type.replace("_", " ").title()}')
        
        # Improve readability
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust y-axis to start from 0 or slightly lower
        plt.ylim(0, max(values) * 1.1)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_path, exist_ok=True)
    fig_path = os.path.join(output_path, f"{ablation_type}_ablation.png")
    plt.savefig(fig_path)
    plt.close()
    
    logger.info(f"Saved ablation study plot to {fig_path}")
    return fig_path


def plot_learning_curve(
    epochs: List[int],
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    output_path: str,
    model_name: str = "Model"
) -> Dict[str, str]:
    """
    Plot learning curves for training and validation metrics.
    
    Args:
        epochs: List of epoch numbers.
        train_metrics: Dictionary with metric names as keys and lists of training values.
        val_metrics: Dictionary with metric names as keys and lists of validation values.
        output_path: Directory to save the plots to.
        model_name: Name of the model.
    
    Returns:
        Dictionary mapping metric names to saved plot paths.
    """
    set_plotting_style()
    
    plot_paths = {}
    
    # Create a plot for each metric
    for metric_name in train_metrics.keys():
        if metric_name not in val_metrics:
            logger.warning(f"Metric {metric_name} not found in validation metrics")
            continue
        
        plt.figure(figsize=(12, 8))
        
        # Plot training and validation metrics
        plt.plot(epochs, train_metrics[metric_name], marker='o', linewidth=2,
                 label=f'Training {metric_name}')
        plt.plot(epochs, val_metrics[metric_name], marker='s', linewidth=2,
                 label=f'Validation {metric_name}')
        
        # Add labels and title
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{model_name} Learning Curve - {metric_name}')
        
        # Improve readability
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Make x-axis integers only
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Save figure
        os.makedirs(output_path, exist_ok=True)
        fig_path = os.path.join(output_path, f"{model_name.replace(' ', '_')}_{metric_name.lower()}_learning_curve.png")
        plt.savefig(fig_path)
        plt.close()
        
        logger.info(f"Saved learning curve for {metric_name} to {fig_path}")
        plot_paths[metric_name] = fig_path
    
    return plot_paths
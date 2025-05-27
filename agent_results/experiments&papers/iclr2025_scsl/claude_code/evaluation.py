"""
Evaluation metrics and utilities for AIFS experiments.

This module implements evaluation metrics and utilities for measuring robustness
to spurious correlations and comparing different methods.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def evaluate_group_fairness(
    model: nn.Module,
    test_loader: DataLoader,
    num_groups: int = 2,
    num_classes: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate model fairness across different groups.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader with group labels
        num_groups: Number of groups in the data
        num_classes: Number of classes (optional)
        
    Returns:
        Dictionary of fairness metrics
    """
    model = model.to(device)
    model.eval()
    
    # Initialize storage for predictions and targets
    all_preds = []
    all_labels = []
    all_groups = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) != 3:
                raise ValueError("Group fairness evaluation requires group labels")
            
            inputs, labels, groups = batch
            inputs, labels, groups = inputs.to(device), labels.to(device), groups.to(device)
            
            # Get predictions
            if hasattr(model, 'forward') and callable(getattr(model, 'forward')):
                if hasattr(model, 'base_model') and callable(getattr(model, 'base_model')):
                    # For models that wrap other models (e.g., GroupDRO, Reweighting)
                    outputs = model.base_model(inputs)
                else:
                    # Handle different return types
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        # For models that return multiple outputs (e.g., AIFS, DANN)
                        outputs = outputs[0]
            else:
                raise ValueError("Model does not have a callable forward method")
            
            # Get predicted class
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_groups.extend(groups.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_groups = np.array(all_groups)
    
    # Initialize metrics
    metrics = {
        "overall_accuracy": accuracy_score(all_labels, all_preds),
        "group_accuracy": {},
        "group_precision": {},
        "group_recall": {},
        "group_f1": {},
        "class_accuracy": {},
        "class_group_accuracy": {},
        "demographic_parity": {},
        "equalized_odds": {}
    }
    
    # Compute per-group metrics
    for g in range(num_groups):
        group_mask = (all_groups == g)
        if sum(group_mask) == 0:
            continue
            
        group_preds = all_preds[group_mask]
        group_labels = all_labels[group_mask]
        
        # Basic metrics
        metrics["group_accuracy"][g] = accuracy_score(group_labels, group_preds)
        
        # Multi-class metrics require averaging
        if num_classes is not None and num_classes > 2:
            metrics["group_precision"][g] = precision_score(
                group_labels, group_preds, average='macro', zero_division=0
            )
            metrics["group_recall"][g] = recall_score(
                group_labels, group_preds, average='macro', zero_division=0
            )
            metrics["group_f1"][g] = f1_score(
                group_labels, group_preds, average='macro', zero_division=0
            )
        else:
            # Binary metrics
            metrics["group_precision"][g] = precision_score(
                group_labels, group_preds, zero_division=0
            )
            metrics["group_recall"][g] = recall_score(
                group_labels, group_preds, zero_division=0
            )
            metrics["group_f1"][g] = f1_score(
                group_labels, group_preds, zero_division=0
            )
    
    # Compute worst-group accuracy
    metrics["worst_group_accuracy"] = min(metrics["group_accuracy"].values())
    
    # Compute per-class accuracy
    if num_classes is not None:
        for c in range(num_classes):
            class_mask = (all_labels == c)
            if sum(class_mask) == 0:
                continue
                
            class_preds = all_preds[class_mask]
            class_labels = all_labels[class_mask]
            
            metrics["class_accuracy"][c] = accuracy_score(class_labels, class_preds)
            
            # Compute per-class-per-group accuracy
            metrics["class_group_accuracy"][c] = {}
            for g in range(num_groups):
                class_group_mask = (all_labels == c) & (all_groups == g)
                if sum(class_group_mask) == 0:
                    continue
                    
                class_group_preds = all_preds[class_group_mask]
                class_group_labels = all_labels[class_group_mask]
                
                metrics["class_group_accuracy"][c][g] = accuracy_score(
                    class_group_labels, class_group_preds
                )
    
    # Compute fairness metrics
    if num_classes is not None and num_classes == 2:  # Binary classification
        # Demographic Parity (Equal acceptance rates across groups)
        group_selection_rates = {}
        for g in range(num_groups):
            group_mask = (all_groups == g)
            if sum(group_mask) == 0:
                continue
                
            group_preds = all_preds[group_mask]
            group_selection_rates[g] = np.mean(group_preds)
        
        # Compute demographic parity difference (max - min)
        if len(group_selection_rates) > 1:
            metrics["demographic_parity"] = max(group_selection_rates.values()) - min(group_selection_rates.values())
        
        # Equalized Odds (Equal true and false positive rates across groups)
        group_tpr = {}
        group_fpr = {}
        for g in range(num_groups):
            group_mask = (all_groups == g)
            if sum(group_mask) == 0:
                continue
                
            group_preds = all_preds[group_mask]
            group_labels = all_labels[group_mask]
            
            # True positive rate
            pos_mask = (group_labels == 1)
            if sum(pos_mask) > 0:
                group_tpr[g] = np.mean(group_preds[pos_mask])
            
            # False positive rate
            neg_mask = (group_labels == 0)
            if sum(neg_mask) > 0:
                group_fpr[g] = np.mean(group_preds[neg_mask])
        
        # Compute equalized odds difference (max - min for both TPR and FPR)
        if len(group_tpr) > 1:
            metrics["equalized_odds_tpr"] = max(group_tpr.values()) - min(group_tpr.values())
        if len(group_fpr) > 1:
            metrics["equalized_odds_fpr"] = max(group_fpr.values()) - min(group_fpr.values())
            
        # Overall equalized odds (average of TPR and FPR differences)
        if "equalized_odds_tpr" in metrics and "equalized_odds_fpr" in metrics:
            metrics["equalized_odds"] = (metrics["equalized_odds_tpr"] + metrics["equalized_odds_fpr"]) / 2
    
    return metrics


def evaluate_spurious_correlation_impact(
    model: nn.Module,
    test_loader: DataLoader
) -> Dict[str, float]:
    """
    Evaluate the impact of spurious correlations on the model's predictions.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader with group labels
        
    Returns:
        Dictionary of impact metrics
    """
    model = model.to(device)
    model.eval()
    
    # Initialize storage
    aligned_correct = 0
    aligned_total = 0
    unaligned_correct = 0
    unaligned_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) != 3:
                raise ValueError("Spurious correlation evaluation requires group labels")
            
            inputs, labels, groups = batch
            inputs, labels, groups = inputs.to(device), labels.to(device), groups.to(device)
            
            # Get predictions
            if hasattr(model, 'forward') and callable(getattr(model, 'forward')):
                if hasattr(model, 'base_model') and callable(getattr(model, 'base_model')):
                    # For models that wrap other models
                    outputs = model.base_model(inputs)
                else:
                    # Handle different return types
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        # For models that return multiple outputs
                        outputs = outputs[0]
            else:
                raise ValueError("Model does not have a callable forward method")
            
            # Get predicted class
            _, predicted = torch.max(outputs, 1)
            
            # Calculate metrics for aligned and unaligned groups
            # Group 0 is aligned (spurious feature correlates with label)
            # Group 1 is unaligned (spurious feature does not correlate with label)
            aligned_mask = (groups == 0)
            unaligned_mask = (groups == 1)
            
            # Count correct predictions for each group
            aligned_correct += (predicted[aligned_mask] == labels[aligned_mask]).sum().item()
            aligned_total += aligned_mask.sum().item()
            
            unaligned_correct += (predicted[unaligned_mask] == labels[unaligned_mask]).sum().item()
            unaligned_total += unaligned_mask.sum().item()
    
    # Calculate accuracies
    aligned_accuracy = aligned_correct / aligned_total if aligned_total > 0 else 0
    unaligned_accuracy = unaligned_correct / unaligned_total if unaligned_total > 0 else 0
    
    # Calculate disparity (difference in accuracy between aligned and unaligned)
    disparity = aligned_accuracy - unaligned_accuracy
    
    # Calculate overall accuracy
    overall_accuracy = (aligned_correct + unaligned_correct) / (aligned_total + unaligned_total)
    
    return {
        "aligned_accuracy": aligned_accuracy,
        "unaligned_accuracy": unaligned_accuracy,
        "disparity": disparity,
        "overall_accuracy": overall_accuracy
    }


def visualize_training_curves(
    metrics_dict: Dict[str, Dict[str, List[float]]],
    save_dir: str,
    filename: str = "training_curves.png"
) -> str:
    """
    Visualize training curves for multiple models.
    
    Args:
        metrics_dict: Dictionary of metrics for each model
            {model_name: {metric_name: [values]}}
        save_dir: Directory to save the plot
        filename: Name of the output file
        
    Returns:
        Path to the saved figure
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Define metrics to plot
    metrics_to_plot = ["train_loss", "val_loss", "train_acc", "val_acc"]
    titles = ["Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy"]
    y_labels = ["Loss", "Loss", "Accuracy (%)", "Accuracy (%)"]
    
    # Plot each metric
    for i, (metric, title, y_label) in enumerate(zip(metrics_to_plot, titles, y_labels)):
        ax = axes[i]
        
        for model_name, metrics in metrics_dict.items():
            if metric in metrics and len(metrics[metric]) > 0:
                ax.plot(metrics[metric], label=model_name)
        
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only add legend to the first plot to avoid redundancy
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def visualize_group_performance(
    results_dict: Dict[str, Dict[str, float]],
    save_dir: str,
    filename: str = "group_performance.png"
) -> str:
    """
    Visualize performance across different groups for multiple models.
    
    Args:
        results_dict: Dictionary of results for each model
            {model_name: {metric_name: value}}
        save_dir: Directory to save the plot
        filename: Name of the output file
        
    Returns:
        Path to the saved figure
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    # Extract group accuracies
    model_names = list(results_dict.keys())
    aligned_accs = [results_dict[name]["aligned_accuracy"] for name in model_names]
    unaligned_accs = [results_dict[name]["unaligned_accuracy"] for name in model_names]
    overall_accs = [results_dict[name]["overall_accuracy"] for name in model_names]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.25
    
    # Set position of bar on X axis
    r1 = np.arange(len(model_names))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Make the plot
    ax.bar(r1, aligned_accs, width=bar_width, label='Aligned Group', color='skyblue')
    ax.bar(r2, unaligned_accs, width=bar_width, label='Unaligned Group', color='salmon')
    ax.bar(r3, overall_accs, width=bar_width, label='Overall', color='lightgreen')
    
    # Add labels and title
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Performance Comparison Across Groups')
    ax.set_xticks([r + bar_width for r in range(len(model_names))])
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Create legend & Show graphic
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add value labels on top of bars
    for i, (r, acc) in enumerate(zip(r1, aligned_accs)):
        ax.text(r, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    for i, (r, acc) in enumerate(zip(r2, unaligned_accs)):
        ax.text(r, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    for i, (r, acc) in enumerate(zip(r3, overall_accs)):
        ax.text(r, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def visualize_disparity(
    results_dict: Dict[str, Dict[str, float]],
    save_dir: str,
    filename: str = "disparity.png"
) -> str:
    """
    Visualize disparity (difference between aligned and unaligned group performance)
    for multiple models.
    
    Args:
        results_dict: Dictionary of results for each model
            {model_name: {metric_name: value}}
        save_dir: Directory to save the plot
        filename: Name of the output file
        
    Returns:
        Path to the saved figure
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    # Extract disparities
    model_names = list(results_dict.keys())
    disparities = [results_dict[name]["disparity"] for name in model_names]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    # Sort by disparity (lowest disparity first)
    sorted_indices = np.argsort(disparities)
    sorted_model_names = [model_names[i] for i in sorted_indices]
    sorted_disparities = [disparities[i] for i in sorted_indices]
    
    # Define colors based on disparity value (lower is better)
    colors = ['green' if d < 0.1 else 'orange' if d < 0.2 else 'red' for d in sorted_disparities]
    
    # Create the horizontal bar chart
    bars = ax.barh(sorted_model_names, sorted_disparities, color=colors)
    
    # Add labels
    ax.set_xlabel('Disparity (Aligned - Unaligned Accuracy)')
    ax.set_title('Model Fairness Comparison')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{sorted_disparities[i]:.3f}', 
                va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def create_comparison_table(
    results_dict: Dict[str, Dict[str, float]],
    save_dir: str,
    filename: str = "comparison_table.csv"
) -> str:
    """
    Create a table comparing performance across different models.
    
    Args:
        results_dict: Dictionary of results for each model
            {model_name: {metric_name: value}}
        save_dir: Directory to save the table
        filename: Name of the output file
        
    Returns:
        Path to the saved table
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    # Extract metrics for table
    model_names = list(results_dict.keys())
    
    # Define metrics to include in the table
    metrics_to_include = [
        "overall_accuracy",
        "aligned_accuracy",
        "unaligned_accuracy",
        "disparity",
        "worst_group_accuracy"
    ]
    
    # Initialize table data
    table_data = {
        "Model": model_names
    }
    
    # Add metrics to table
    for metric in metrics_to_include:
        if all(metric in results_dict[name] for name in model_names):
            table_data[metric] = [results_dict[name][metric] for name in model_names]
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    
    return save_path


def generate_summary_plots(
    metrics_dict: Dict[str, Dict[str, List[float]]],
    results_dict: Dict[str, Dict[str, float]],
    save_dir: str
) -> Dict[str, str]:
    """
    Generate a set of summary plots visualizing experiment results.
    
    Args:
        metrics_dict: Dictionary of training metrics for each model
        results_dict: Dictionary of evaluation results for each model
        save_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot type to file path
    """
    # Create all visualization plots
    plot_paths = {}
    
    # Training curves
    plot_paths["training_curves"] = visualize_training_curves(
        metrics_dict, save_dir, "training_curves.png"
    )
    
    # Group performance comparison
    plot_paths["group_performance"] = visualize_group_performance(
        results_dict, save_dir, "group_performance.png"
    )
    
    # Disparity visualization
    plot_paths["disparity"] = visualize_disparity(
        results_dict, save_dir, "disparity.png"
    )
    
    # Comparison table
    plot_paths["comparison_table"] = create_comparison_table(
        results_dict, save_dir, "comparison_table.csv"
    )
    
    return plot_paths


# Example usage:
# group_fairness_metrics = evaluate_group_fairness(model, test_loader, num_groups=2, num_classes=10)
# impact_metrics = evaluate_spurious_correlation_impact(model, test_loader)
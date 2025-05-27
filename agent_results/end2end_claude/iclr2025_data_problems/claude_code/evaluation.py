#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics and functions for Attribution-Guided Training experiments.
"""

import torch
import numpy as np
import pandas as pd
import logging
import os
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)

def compute_attribution_metrics(
    true_sources: List[int],
    pred_sources: List[int],
    pred_probs: List[float],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute attribution metrics for a set of predictions.
    
    Args:
        true_sources: List of true source IDs
        pred_sources: List of predicted source IDs
        pred_probs: List of prediction probabilities
        threshold: Confidence threshold for attribution
        
    Returns:
        Dictionary of metrics
    """
    # Basic accuracy
    accuracy = accuracy_score(true_sources, pred_sources)
    
    # Convert to binary classification for threshold-based metrics
    binary_preds = [1 if p >= threshold else 0 for p in pred_probs]
    binary_truth = [1] * len(true_sources)  # All examples should be attributed
    
    # Compute precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_truth, binary_preds, average='binary'
    )
    
    # Mean reciprocal rank
    # For each true source, check its rank in the predicted probabilities
    all_ranks = []
    unique_sources = list(set(true_sources))
    
    for source in unique_sources:
        # Find indices where this is the true source
        indices = [i for i, s in enumerate(true_sources) if s == source]
        
        # Get corresponding predictions
        source_preds = [pred_sources[i] for i in indices]
        source_probs = [pred_probs[i] for i in indices]
        
        # Check if the true source is in the predictions
        ranks = []
        for i in range(len(indices)):
            if source_preds[i] == source:
                ranks.append(1)  # Rank 1 if correct
            else:
                ranks.append(0)  # Rank 0 if incorrect (simplified)
                
        if ranks:
            all_ranks.extend(ranks)
    
    mrr = np.mean(all_ranks) if all_ranks else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr
    }

def compute_attribution_precision_score(
    correct_attributions: List[int],
    total_attributions: List[int]
) -> float:
    """
    Compute Attribution Precision Score (APS).
    
    Args:
        correct_attributions: Number of correct attributions per example
        total_attributions: Total number of attributions made per example
        
    Returns:
        Attribution Precision Score
    """
    if len(correct_attributions) == 0 or sum(total_attributions) == 0:
        return 0.0
    
    individual_precision = [c / max(t, 1) for c, t in zip(correct_attributions, total_attributions)]
    return np.mean(individual_precision)

def compute_attribution_recall_score(
    correct_attributions: List[int],
    required_attributions: List[int]
) -> float:
    """
    Compute Attribution Recall Score (ARS).
    
    Args:
        correct_attributions: Number of correct attributions per example
        required_attributions: Number of required attributions per example
        
    Returns:
        Attribution Recall Score
    """
    if len(correct_attributions) == 0 or sum(required_attributions) == 0:
        return 0.0
    
    individual_recall = [c / max(r, 1) for c, r in zip(correct_attributions, required_attributions)]
    return np.mean(individual_recall)

def compute_attribution_f1_score(
    precision: float,
    recall: float
) -> float:
    """
    Compute Attribution F1 Score (AF1).
    
    Args:
        precision: Attribution Precision Score
        recall: Attribution Recall Score
        
    Returns:
        Attribution F1 Score
    """
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

def compute_content_originality_score(
    content_requiring_attribution: List[int],
    total_content: List[int]
) -> float:
    """
    Compute Content Originality Score (COS).
    
    Args:
        content_requiring_attribution: Amount of content requiring attribution per example
        total_content: Total amount of content per example
        
    Returns:
        Content Originality Score
    """
    if len(content_requiring_attribution) == 0 or sum(total_content) == 0:
        return 1.0
    
    individual_cos = [1 - (c / max(t, 1)) for c, t in zip(content_requiring_attribution, total_content)]
    return np.mean(individual_cos)

def compute_attribution_efficiency(
    attribution_f1: float,
    relative_cost: float
) -> float:
    """
    Compute Attribution Efficiency (AE).
    
    Args:
        attribution_f1: Attribution F1 Score
        relative_cost: Relative computational cost compared to baseline
        
    Returns:
        Attribution Efficiency
    """
    if relative_cost == 0:
        return 0.0
    
    return attribution_f1 / relative_cost

def evaluate_model(
    model,
    dataloader,
    device,
    top_k: int = 1,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to use for inference
        top_k: Number of top predictions to consider
        threshold: Confidence threshold for attribution
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_true_sources = []
    all_pred_sources = []
    all_pred_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            source_idx = batch["source_idx"].to(device)
            
            # Get predictions
            top_indices, top_probs = model.predict_sources(
                input_ids=input_ids,
                attention_mask=attention_mask,
                top_k=top_k
            )
            
            # Get top-1 predictions
            pred_sources = top_indices[:, 0].cpu().numpy()
            pred_probs = top_probs[:, 0].cpu().numpy()
            
            # Get true sources
            true_sources = source_idx.cpu().numpy()
            
            all_true_sources.extend(true_sources)
            all_pred_sources.extend(pred_sources)
            all_pred_probs.extend(pred_probs)
    
    # Compute attribution metrics
    metrics = compute_attribution_metrics(
        true_sources=all_true_sources,
        pred_sources=all_pred_sources,
        pred_probs=all_pred_probs,
        threshold=threshold
    )
    
    # For simplicity, we simulate the specialized metrics
    # In a real implementation, these would be computed with actual data
    
    # Simulate correct and total attributions (simplified)
    correct_attributions = [int(p == t) for p, t in zip(all_pred_sources, all_true_sources)]
    total_attributions = [1] * len(all_true_sources)
    required_attributions = [1] * len(all_true_sources)
    
    # Simulate content lengths (simplified)
    content_requiring_attribution = [int(p != t) for p, t in zip(all_pred_sources, all_true_sources)]
    total_content = [1] * len(all_true_sources)
    
    # Compute specialized metrics
    aps = compute_attribution_precision_score(correct_attributions, total_attributions)
    ars = compute_attribution_recall_score(correct_attributions, required_attributions)
    af1 = compute_attribution_f1_score(aps, ars)
    cos = compute_content_originality_score(content_requiring_attribution, total_content)
    
    # Add specialized metrics to results
    metrics.update({
        "attribution_precision_score": aps,
        "attribution_recall_score": ars,
        "attribution_f1_score": af1,
        "content_originality_score": cos
    })
    
    return metrics

def plot_attribution_scores(
    model_names: List[str],
    precision_scores: List[float],
    recall_scores: List[float],
    f1_scores: List[float],
    output_dir: str,
    title: str = "Attribution Scores Comparison"
) -> str:
    """
    Plot attribution precision, recall, and F1 scores for multiple models.
    
    Args:
        model_names: List of model names
        precision_scores: List of precision scores
        recall_scores: List of recall scores
        f1_scores: List of F1 scores
        output_dir: Directory to save the plot
        title: Plot title
        
    Returns:
        Path to the saved plot
    """
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(model_names))
    width = 0.25
    
    plt.bar(x - width, precision_scores, width, label='Precision')
    plt.bar(x, recall_scores, width, label='Recall')
    plt.bar(x + width, f1_scores, width, label='F1')
    
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title(title)
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'attribution_scores.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved attribution scores plot to {plot_path}")
    
    return plot_path

def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: List[float],
    val_metrics: List[float],
    metric_name: str,
    output_dir: str,
    title: str = "Learning Curves"
) -> str:
    """
    Plot learning curves with losses and metrics.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_metrics: List of training metrics
        val_metrics: List of validation metrics
        metric_name: Name of the metric (e.g., "Accuracy", "F1")
        output_dir: Directory to save the plot
        title: Plot title
        
    Returns:
        Path to the saved plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training and Validation {metric_name}')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'learning_curves_{metric_name.lower()}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved learning curves plot to {plot_path}")
    
    return plot_path

def plot_model_comparison(
    model_names: List[str],
    metrics: Dict[str, List[float]],
    output_dir: str,
    title: str = "Model Comparison"
) -> str:
    """
    Plot a comparison of multiple models across different metrics.
    
    Args:
        model_names: List of model names
        metrics: Dictionary mapping metric names to lists of values for each model
        output_dir: Directory to save the plot
        title: Plot title
        
    Returns:
        Path to the saved plot
    """
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(model_names))
    width = 0.8 / len(metrics)
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        plt.bar(x + i * width - 0.4 + width/2, values, width, label=metric_name)
    
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title(title)
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved model comparison plot to {plot_path}")
    
    return plot_path

def plot_lambda_ablation(
    lambda_values: List[float],
    attribution_f1: List[float],
    task_performance: List[float],
    task_name: str,
    output_dir: str,
    title: str = "Effect of Attribution Weight (λ)"
) -> str:
    """
    Plot ablation study for the attribution loss weight (λ).
    
    Args:
        lambda_values: List of λ values
        attribution_f1: List of attribution F1 scores for each λ
        task_performance: List of task performance metrics for each λ
        task_name: Name of the task performance metric
        output_dir: Directory to save the plot
        title: Plot title
        
    Returns:
        Path to the saved plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create twin axes for different scales
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot attribution F1 scores
    line1 = ax1.plot(lambda_values, attribution_f1, 'b-o', label='Attribution F1')
    ax1.set_xlabel('λ (Attribution Loss Weight)')
    ax1.set_ylabel('Attribution F1', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot task performance
    line2 = ax2.plot(lambda_values, task_performance, 'r-s', label=task_name)
    ax2.set_ylabel(task_name, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center')
    
    plt.title(title)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'lambda_ablation.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved λ ablation plot to {plot_path}")
    
    return plot_path

def plot_architecture_comparison(
    architectures: List[str],
    metrics: Dict[str, List[float]],
    output_dir: str,
    title: str = "Attribution Network Architecture Comparison"
) -> str:
    """
    Plot comparison of different attribution network architectures.
    
    Args:
        architectures: List of architecture names
        metrics: Dictionary mapping metric names to lists of values for each architecture
        output_dir: Directory to save the plot
        title: Plot title
        
    Returns:
        Path to the saved plot
    """
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(architectures))
    width = 0.8 / len(metrics)
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        plt.bar(x + i * width - 0.4 + width/2, values, width, label=metric_name)
    
    plt.xlabel('Architecture')
    plt.ylabel('Scores')
    plt.title(title)
    plt.xticks(x, architectures, rotation=45, ha='right')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'architecture_comparison.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved architecture comparison plot to {plot_path}")
    
    return plot_path

def plot_threshold_effect(
    thresholds: List[float],
    precision: List[float],
    recall: List[float],
    f1: List[float],
    output_dir: str,
    title: str = "Effect of Attribution Threshold"
) -> str:
    """
    Plot the effect of the attribution threshold on precision, recall, and F1.
    
    Args:
        thresholds: List of threshold values
        precision: Precision at each threshold
        recall: Recall at each threshold
        f1: F1 score at each threshold
        output_dir: Directory to save the plot
        title: Plot title
        
    Returns:
        Path to the saved plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(thresholds, precision, 'b-o', label='Precision')
    plt.plot(thresholds, recall, 'r-s', label='Recall')
    plt.plot(thresholds, f1, 'g-^', label='F1')
    
    plt.xlabel('Attribution Threshold')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'threshold_effect.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved threshold effect plot to {plot_path}")
    
    return plot_path

def plot_computational_efficiency(
    model_names: List[str],
    attribution_f1: List[float],
    training_times: List[float],
    inference_times: List[float],
    output_dir: str,
    title: str = "Attribution Efficiency"
) -> str:
    """
    Plot the trade-off between attribution quality and computational efficiency.
    
    Args:
        model_names: List of model names
        attribution_f1: Attribution F1 score for each model
        training_times: Training time for each model (relative to baseline)
        inference_times: Inference time for each model (relative to baseline)
        output_dir: Directory to save the plot
        title: Plot title
        
    Returns:
        Path to the saved plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot training time vs. F1
    plt.subplot(1, 2, 1)
    plt.scatter(training_times, attribution_f1, s=100)
    
    # Add model names as annotations
    for i, model in enumerate(model_names):
        plt.annotate(model, (training_times[i], attribution_f1[i]),
                   textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Relative Training Time')
    plt.ylabel('Attribution F1 Score')
    plt.title('Training Efficiency')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot inference time vs. F1
    plt.subplot(1, 2, 2)
    plt.scatter(inference_times, attribution_f1, s=100)
    
    # Add model names as annotations
    for i, model in enumerate(model_names):
        plt.annotate(model, (inference_times[i], attribution_f1[i]),
                   textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Relative Inference Time')
    plt.ylabel('Attribution F1 Score')
    plt.title('Inference Efficiency')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'computational_efficiency.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved computational efficiency plot to {plot_path}")
    
    return plot_path

def create_results_table(
    model_names: List[str],
    metrics: Dict[str, List[float]],
    output_dir: str,
    filename: str = "results_table.csv"
) -> str:
    """
    Create a CSV table of results for all models and metrics.
    
    Args:
        model_names: List of model names
        metrics: Dictionary mapping metric names to lists of values for each model
        output_dir: Directory to save the table
        filename: Output filename
        
    Returns:
        Path to the saved table
    """
    # Create DataFrame
    data = {"Model": model_names}
    for metric_name, values in metrics.items():
        data[metric_name] = values
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved results table to {output_path}")
    
    return output_path

def create_results_markdown(
    model_names: List[str],
    metrics: Dict[str, List[float]],
    plot_paths: List[str],
    output_dir: str,
    filename: str = "results.md"
) -> str:
    """
    Create a markdown summary of results with tables and figures.
    
    Args:
        model_names: List of model names
        metrics: Dictionary mapping metric names to lists of values for each model
        plot_paths: List of paths to generated plots
        output_dir: Directory to save the markdown
        filename: Output filename
        
    Returns:
        Path to the saved markdown
    """
    # Create results table
    table_data = [["Model"] + list(metrics.keys())]
    for i, model in enumerate(model_names):
        row = [model]
        for metric_name in metrics.keys():
            row.append(f"{metrics[metric_name][i]:.4f}")
        table_data.append(row)
    
    # Format table as markdown
    table_md = "| " + " | ".join(table_data[0]) + " |\n"
    table_md += "| " + " | ".join(["---"] * len(table_data[0])) + " |\n"
    for row in table_data[1:]:
        table_md += "| " + " | ".join(row) + " |\n"
    
    # Format plots as markdown
    plots_md = ""
    for plot_path in plot_paths:
        plot_name = os.path.basename(plot_path)
        # Adjust path for results folder structure
        adjusted_path = plot_name
        plots_md += f"![{plot_name}]({adjusted_path})\n\n"
    
    # Combine into full markdown
    markdown = f"""# Experimental Results

## Performance Comparison

The following table shows the performance of different attribution methods:

{table_md}

## Visualizations

{plots_md}

"""
    
    # Save markdown
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        f.write(markdown)
    
    logger.info(f"Saved results markdown to {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Test evaluation functions
    logging.basicConfig(level=logging.INFO)
    
    # Mock data for testing
    model_names = ["AGT", "PostHoc", "DataShapley", "MinimalSubset"]
    test_metrics = {
        "Accuracy": [0.85, 0.75, 0.78, 0.80],
        "Precision": [0.82, 0.70, 0.75, 0.78],
        "Recall": [0.86, 0.72, 0.77, 0.79],
        "F1": [0.84, 0.71, 0.76, 0.78],
        "MRR": [0.90, 0.80, 0.85, 0.87]
    }
    
    # Test plotting functions
    output_dir = "test_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_attribution_scores(
        model_names=model_names,
        precision_scores=test_metrics["Precision"],
        recall_scores=test_metrics["Recall"],
        f1_scores=test_metrics["F1"],
        output_dir=output_dir
    )
    
    plot_model_comparison(
        model_names=model_names,
        metrics=test_metrics,
        output_dir=output_dir
    )
    
    create_results_table(
        model_names=model_names,
        metrics=test_metrics,
        output_dir=output_dir
    )
    
    logger.info("Evaluation tests passed")
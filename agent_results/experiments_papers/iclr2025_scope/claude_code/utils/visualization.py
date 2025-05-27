#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for the proposed architecture.

This module provides functions to create visualizations of experimental results:
1. Loss curves
2. Performance metrics over time
3. Memory usage comparison
4. Throughput comparison
5. Token efficiency comparison
6. Latency comparison
7. Information retention comparison
8. Ablation study results
9. Baseline comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple, Any, Optional, Union

# Set default font size
matplotlib.rcParams.update({'font.size': 12})
# Use a high-contrast color scheme
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Training and Validation Loss"
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        title: Plot title
    
    Returns:
        fig: Matplotlib figure
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, 'o-', color=colors[0], label='Training Loss')
    ax.plot(epochs, val_losses, 'o-', color=colors[1], label='Validation Loss')
    
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis to log scale if loss values vary significantly
    if max(train_losses + val_losses) / min(filter(lambda x: x > 0, train_losses + val_losses)) > 10:
        ax.set_yscale('log')
    
    fig.tight_layout()
    
    return fig


def plot_performance_metrics(
    metrics_over_time: List[Dict[str, float]],
    metric_names: List[str] = ['f1', 'exact_match', 'rouge_l', 'bleu'],
    title: str = "Performance Metrics Over Time"
) -> plt.Figure:
    """
    Plot performance metrics over time.
    
    Args:
        metrics_over_time: List of dictionaries containing metric values at each time step
        metric_names: List of metric names to plot
        title: Plot title
    
    Returns:
        fig: Matplotlib figure
    """
    steps = range(1, len(metrics_over_time) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, metric in enumerate(metric_names):
        # Check if the metric exists in the data
        if metric in metrics_over_time[0]:
            values = [m.get(metric, 0) for m in metrics_over_time]
            ax.plot(steps, values, 'o-', color=colors[i % len(colors)], label=metric.replace('_', ' ').title())
    
    ax.set_title(title)
    ax.set_xlabel('Evaluation Step')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis range from 0 to 1 for metrics that are typically in that range
    if all(m in ['f1', 'exact_match', 'rouge_l', 'bleu'] for m in metric_names):
        ax.set_ylim(0, 1)
    
    fig.tight_layout()
    
    return fig


def plot_memory_usage(
    memory_data: List[Tuple[str, float]],
    title: str = "Memory Usage Comparison"
) -> plt.Figure:
    """
    Plot memory usage comparison across models.
    
    Args:
        memory_data: List of (model_name, memory_usage) tuples
        title: Plot title
    
    Returns:
        fig: Matplotlib figure
    """
    model_names = [m[0] for m in memory_data]
    memory_values = [m[1] for m in memory_data]
    
    # Sort by memory usage (ascending)
    sorted_indices = np.argsort(memory_values)
    model_names = [model_names[i] for i in sorted_indices]
    memory_values = [memory_values[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, memory_values, color=colors[:len(model_names)])
    
    ax.set_title(title)
    ax.set_xlabel('Model')
    ax.set_ylabel('Memory Usage (MB)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    fig.tight_layout()
    
    return fig


def plot_throughput(
    throughput_data: List[Tuple[str, float]],
    title: str = "Throughput Comparison"
) -> plt.Figure:
    """
    Plot throughput comparison across models.
    
    Args:
        throughput_data: List of (model_name, throughput) tuples
        title: Plot title
    
    Returns:
        fig: Matplotlib figure
    """
    model_names = [m[0] for m in throughput_data]
    throughput_values = [m[1] for m in throughput_data]
    
    # Sort by throughput (descending)
    sorted_indices = np.argsort(throughput_values)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    throughput_values = [throughput_values[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, throughput_values, color=colors[:len(model_names)])
    
    ax.set_title(title)
    ax.set_xlabel('Model')
    ax.set_ylabel('Throughput (tokens/s)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    fig.tight_layout()
    
    return fig


def plot_token_efficiency(
    efficiency_data: List[Tuple[str, float]],
    title: str = "Token Efficiency Comparison"
) -> plt.Figure:
    """
    Plot token efficiency comparison across models.
    
    Args:
        efficiency_data: List of (model_name, token_efficiency) tuples
        title: Plot title
    
    Returns:
        fig: Matplotlib figure
    """
    model_names = [m[0] for m in efficiency_data]
    efficiency_values = [m[1] for m in efficiency_data]
    
    # Sort by efficiency (ascending, lower is better)
    sorted_indices = np.argsort(efficiency_values)
    model_names = [model_names[i] for i in sorted_indices]
    efficiency_values = [efficiency_values[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, efficiency_values, color=colors[:len(model_names)])
    
    ax.set_title(title)
    ax.set_xlabel('Model')
    ax.set_ylabel('Token Efficiency (selected/total)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limits from 0 to 1
    ax.set_ylim(0, 1)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    fig.tight_layout()
    
    return fig


def plot_latency(
    latency_data: List[Tuple[str, float]],
    title: str = "Latency Comparison"
) -> plt.Figure:
    """
    Plot latency comparison across models.
    
    Args:
        latency_data: List of (model_name, latency) tuples
        title: Plot title
    
    Returns:
        fig: Matplotlib figure
    """
    model_names = [m[0] for m in latency_data]
    latency_values = [m[1] for m in latency_data]
    
    # Sort by latency (ascending, lower is better)
    sorted_indices = np.argsort(latency_values)
    model_names = [model_names[i] for i in sorted_indices]
    latency_values = [latency_values[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, latency_values, color=colors[:len(model_names)])
    
    ax.set_title(title)
    ax.set_xlabel('Model')
    ax.set_ylabel('Latency (s)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    fig.tight_layout()
    
    return fig


def plot_information_retention(
    retention_data: List[Tuple[str, float]],
    title: str = "Information Retention Comparison"
) -> plt.Figure:
    """
    Plot information retention comparison across models.
    
    Args:
        retention_data: List of (model_name, retention_score) tuples
        title: Plot title
    
    Returns:
        fig: Matplotlib figure
    """
    model_names = [m[0] for m in retention_data]
    retention_values = [m[1] for m in retention_data]
    
    # Sort by retention (descending, higher is better)
    sorted_indices = np.argsort(retention_values)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    retention_values = [retention_values[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, retention_values, color=colors[:len(model_names)])
    
    ax.set_title(title)
    ax.set_xlabel('Model')
    ax.set_ylabel('Information Retention Score')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limits from 0 to 1
    ax.set_ylim(0, 1)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    fig.tight_layout()
    
    return fig


def plot_ablation_results(
    ablation_data: Dict[str, float],
    title: str = "Ablation Study Results"
) -> plt.Figure:
    """
    Plot ablation study results.
    
    Args:
        ablation_data: Dictionary mapping setting names to performance values
        title: Plot title
    
    Returns:
        fig: Matplotlib figure
    """
    settings = list(ablation_data.keys())
    values = list(ablation_data.values())
    
    # Ensure 'full_model' comes first
    if 'full_model' in settings:
        full_idx = settings.index('full_model')
        settings = [settings[full_idx]] + [s for i, s in enumerate(settings) if i != full_idx]
        values = [values[full_idx]] + [v for i, v in enumerate(values) if i != full_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(settings, values, color=colors[:len(settings)])
    
    # Highlight the full model
    if 'full_model' in settings:
        bars[0].set_color('#ff7f0e')  # Orange
    
    ax.set_title(title)
    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('Performance Score')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    fig.tight_layout()
    
    return fig


def plot_baseline_comparison(
    baseline_data: Dict[str, Dict[str, float]],
    metrics: List[str] = ['f1', 'exact_match', 'rouge_l', 'bleu'],
    title: str = "Baseline Comparison"
) -> plt.Figure:
    """
    Plot comparison of task performance metrics across models.
    
    Args:
        baseline_data: Dictionary mapping model names to metric dictionaries
        metrics: List of metric names to include
        title: Plot title
    
    Returns:
        fig: Matplotlib figure
    """
    models = list(baseline_data.keys())
    
    # Check if our proposed model exists, put it first
    if 'dsrsq' in models:
        dsrsq_idx = models.index('dsrsq')
        models = [models[dsrsq_idx]] + [m for i, m in enumerate(models) if i != dsrsq_idx]
    
    # Filter metrics that exist in all models
    available_metrics = []
    for metric in metrics:
        if all(metric in baseline_data[model] for model in models):
            available_metrics.append(metric)
    
    if not available_metrics:
        # Fallback to whatever metrics are available
        all_metrics = set()
        for model in models:
            all_metrics.update(baseline_data[model].keys())
        available_metrics = list(all_metrics)[:4]  # Take up to 4
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bar_width = 0.8 / len(models)
    index = np.arange(len(available_metrics))
    
    for i, model in enumerate(models):
        model_metrics = [baseline_data[model].get(metric, 0) for metric in available_metrics]
        
        # Calculate position
        pos = index - 0.4 + (i + 0.5) * bar_width
        
        # Plot bars
        ax.bar(pos, model_metrics, width=bar_width, label=model, color=colors[i % len(colors)])
    
    ax.set_title(title)
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_xticks(index)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics])
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis range from 0 to 1 for metrics that are typically in that range
    if all(m in ['f1', 'exact_match', 'rouge_l', 'bleu'] for m in available_metrics):
        ax.set_ylim(0, 1)
    
    fig.tight_layout()
    
    return fig
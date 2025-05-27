#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization module for the LLM-TAC experiment.
This module provides functions for visualizing results and metrics.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

def plot_training_curve(stats: Dict[str, List[float]], save_path: str, title: str) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        stats: Dictionary containing training statistics
        save_path: Path to save the plot
        title: Title of the plot
    """
    logger.info(f"Plotting training curve to {save_path}")
    
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    if 'train_loss' in stats:
        plt.plot(stats['train_loss'], label='Training Loss', marker='o', linestyle='-', color='blue')
    
    # Plot validation loss
    if 'val_loss' in stats:
        plt.plot(stats['val_loss'], label='Validation Loss', marker='s', linestyle='--', color='red')
    
    # Plot training accuracy
    if 'train_accuracy' in stats:
        plt.plot(stats['train_accuracy'], label='Training Accuracy', marker='^', linestyle='-', color='green')
    
    # Plot validation accuracy
    if 'val_accuracy' in stats:
        plt.plot(stats['val_accuracy'], label='Validation Accuracy', marker='v', linestyle='--', color='purple')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_metrics_comparison(results: Dict[str, Any], save_path: str, title: str) -> None:
    """
    Plot comparison of metrics across different methods.
    
    Args:
        results: Dictionary containing results for different methods
        save_path: Path to save the plot
        title: Title of the plot
    """
    logger.info(f"Plotting metrics comparison to {save_path}")
    
    # Extract methods and metrics
    methods = ['LLM-TAC']
    if 'Baselines' in results and results['Baselines']:
        methods.extend(list(results['Baselines'].keys()))
    if 'Ablations' in results and results['Ablations']:
        methods.extend(list(results['Ablations'].keys()))
    
    metrics = [
        'tactic_accuracy', 
        'proof_completion_rate', 
        'reduction_in_manual_writing'
    ]
    
    metric_values = {
        metric: [results['LLM-TAC'].get(metric, 0)] + 
                [results['Baselines'].get(method, {}).get(metric, 0) for method in methods[1:] 
                 if method in results.get('Baselines', {})] +
                [results['Ablations'].get(method, {}).get(metric, 0) for method in methods[1:]
                 if method in results.get('Ablations', {})]
        for metric in metrics
    }
    
    # Normalize reduction_in_manual_writing to be between 0 and 1 for better visualization
    if 'reduction_in_manual_writing' in metric_values:
        metric_values['reduction_in_manual_writing'] = [v / 100 for v in metric_values['reduction_in_manual_writing']]
    
    # Plot metrics
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(methods))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width - width, metric_values[metric], width, 
                label=metric.replace('_', ' ').title())
    
    plt.title(title)
    plt.xlabel('Method')
    plt.ylabel('Value')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Create a separate plot for proof completion time
    plot_completion_time_comparison(results, save_path.replace('.png', '_time.png'), 
                                    title + ' - Completion Time')

def plot_completion_time_comparison(results: Dict[str, Any], save_path: str, title: str) -> None:
    """
    Plot comparison of proof completion time across different methods.
    
    Args:
        results: Dictionary containing results for different methods
        save_path: Path to save the plot
        title: Title of the plot
    """
    logger.info(f"Plotting completion time comparison to {save_path}")
    
    # Extract methods
    methods = ['LLM-TAC']
    if 'Baselines' in results and results['Baselines']:
        methods.extend(list(results['Baselines'].keys()))
    if 'Ablations' in results and results['Ablations']:
        methods.extend(list(results['Ablations'].keys()))
    
    # Extract completion times
    completion_times = [
        results['LLM-TAC'].get('proof_completion_time', 0)
    ]
    
    if 'Baselines' in results and results['Baselines']:
        completion_times.extend([
            results['Baselines'].get(method, {}).get('proof_completion_time', 0)
            for method in methods[1:] if method in results['Baselines']
        ])
    
    if 'Ablations' in results and results['Ablations']:
        completion_times.extend([
            results['Ablations'].get(method, {}).get('proof_completion_time', 0)
            for method in methods[1:] if method in results['Ablations']
        ])
    
    # Plot completion times
    plt.figure(figsize=(10, 6))
    
    plt.bar(methods, completion_times, color='skyblue')
    
    plt.title(title)
    plt.xlabel('Method')
    plt.ylabel('Completion Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    
    # Add values on top of bars
    for i, v in enumerate(completion_times):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_rl_progression(stats: Dict[str, List[float]], save_path: str, title: str) -> None:
    """
    Plot performance progression during reinforcement learning.
    
    Args:
        stats: Dictionary containing RL statistics
        save_path: Path to save the plot
        title: Title of the plot
    """
    logger.info(f"Plotting RL progression to {save_path}")
    
    plt.figure(figsize=(10, 6))
    
    # Plot tactic accuracy
    if 'tactic_accuracy' in stats:
        plt.plot(stats['tactic_accuracy'], label='Tactic Accuracy', marker='o', linestyle='-', color='blue')
    
    # Plot proof completion rate
    if 'proof_completion_rate' in stats:
        plt.plot(stats['proof_completion_rate'], label='Proof Completion Rate', marker='s', linestyle='--', color='red')
    
    # Plot reduction in manual writing
    if 'reduction_in_manual_writing' in stats:
        # Normalize to 0-1 scale
        normalized_reduction = [r / 100 for r in stats['reduction_in_manual_writing']]
        plt.plot(normalized_reduction, label='Reduction in Manual Writing', marker='^', linestyle='-', color='green')
    
    # Plot average reward
    if 'avg_reward' in stats:
        # Normalize rewards to 0-1 scale for better visualization
        max_reward = max(stats['avg_reward']) if stats['avg_reward'] else 1
        min_reward = min(stats['avg_reward']) if stats['avg_reward'] else 0
        range_reward = max(1, max_reward - min_reward)
        normalized_rewards = [(r - min_reward) / range_reward for r in stats['avg_reward']]
        plt.plot(normalized_rewards, label='Normalized Reward', marker='v', linestyle='--', color='purple')
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)
    
    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_per_domain_performance(results: Dict[str, Any], save_path: str, title: str) -> None:
    """
    Plot performance across different domains.
    
    Args:
        results: Dictionary containing results with domain-specific performance
        save_path: Path to save the plot
        title: Title of the plot
    """
    logger.info(f"Plotting per-domain performance to {save_path}")
    
    # Extract domain-specific metrics
    if 'per_domain_metrics' not in results:
        logger.warning("No per-domain metrics found, skipping domain performance plot")
        return
    
    domains = list(results['per_domain_metrics'].keys())
    tactic_accuracy = [results['per_domain_metrics'][domain]['tactic_accuracy'] for domain in domains]
    proof_completion_rate = [results['per_domain_metrics'][domain]['proof_completion_rate'] for domain in domains]
    reduction = [results['per_domain_metrics'][domain]['reduction_in_manual_writing'] / 100 for domain in domains]
    
    # Plot metrics
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(domains))
    width = 0.25
    
    plt.bar(x - width, tactic_accuracy, width, label='Tactic Accuracy')
    plt.bar(x, proof_completion_rate, width, label='Proof Completion Rate')
    plt.bar(x + width, reduction, width, label='Reduction in Manual Writing')
    
    plt.title(title)
    plt.xlabel('Domain')
    plt.ylabel('Value')
    plt.xticks(x, domains, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_per_difficulty_performance(results: Dict[str, Any], save_path: str, title: str) -> None:
    """
    Plot performance across different difficulty levels.
    
    Args:
        results: Dictionary containing results with difficulty-specific performance
        save_path: Path to save the plot
        title: Title of the plot
    """
    logger.info(f"Plotting per-difficulty performance to {save_path}")
    
    # Extract difficulty-specific metrics
    if 'per_difficulty_metrics' not in results:
        logger.warning("No per-difficulty metrics found, skipping difficulty performance plot")
        return
    
    difficulties = list(results['per_difficulty_metrics'].keys())
    tactic_accuracy = [results['per_difficulty_metrics'][diff]['tactic_accuracy'] for diff in difficulties]
    proof_completion_rate = [results['per_difficulty_metrics'][diff]['proof_completion_rate'] for diff in difficulties]
    reduction = [results['per_difficulty_metrics'][diff]['reduction_in_manual_writing'] / 100 for diff in difficulties]
    
    # Plot metrics
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(difficulties))
    width = 0.25
    
    plt.bar(x - width, tactic_accuracy, width, label='Tactic Accuracy')
    plt.bar(x, proof_completion_rate, width, label='Proof Completion Rate')
    plt.bar(x + width, reduction, width, label='Reduction in Manual Writing')
    
    plt.title(title)
    plt.xlabel('Difficulty')
    plt.ylabel('Value')
    plt.xticks(x, difficulties)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
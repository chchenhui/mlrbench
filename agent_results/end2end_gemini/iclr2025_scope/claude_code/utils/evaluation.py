"""
Evaluation utilities for MeLPA and baseline methods.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union


def compute_accuracy(model, dataloader, adapter_name, device):
    """
    Compute accuracy of a model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        adapter_name: Name of the adapter to use
        device: Device to run evaluation on
    
    Returns:
        Accuracy as a percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {k: v for k, v in batch.items() if k != 'labels'}
            
            outputs = model(inputs, adapter_name=adapter_name)
            pred = torch.argmax(outputs.logits, dim=1)
            correct += (pred == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    
    accuracy = 100.0 * correct / total
    return accuracy


def compute_continual_learning_metrics(all_tasks_metrics):
    """
    Compute metrics for continual learning evaluation.
    
    Args:
        all_tasks_metrics: Dictionary with metrics for all tasks across learning sequence
    
    Returns:
        Dictionary with computed metrics
    """
    # Extract task sequence from metrics
    if 'final' not in all_tasks_metrics or 'forgetting_metrics' in all_tasks_metrics:
        # Metrics already computed
        return all_tasks_metrics
    
    # Get task IDs
    task_ids = sorted([int(task_id) for task_id in all_tasks_metrics['final'].keys()])
    
    # Get final accuracies
    final_accuracies = [
        all_tasks_metrics['final'][str(task_id)]['accuracy'] 
        for task_id in task_ids
    ]
    
    # Get initial accuracies (accuracies right after learning each task)
    initial_accuracies = []
    for task_id in task_ids:
        # Find the point where this task was just learned
        metric_key = f'after_task_{task_id}'
        if metric_key in all_tasks_metrics:
            initial_accuracies.append(
                all_tasks_metrics[metric_key][str(task_id)]['accuracy']
            )
        else:
            # If not found, use the final accuracy as fallback
            initial_accuracies.append(
                all_tasks_metrics['final'][str(task_id)]['accuracy']
            )
    
    # Calculate Average Accuracy (ACC)
    acc = np.mean(final_accuracies)
    
    # Calculate Backward Transfer (BWT)
    bwt_values = []
    for i in range(len(task_ids) - 1):
        # Accuracy on task i after learning all tasks
        r_n_i = final_accuracies[i]
        # Accuracy on task i right after learning it
        r_i_i = initial_accuracies[i]
        bwt_values.append(r_n_i - r_i_i)
    
    bwt = np.mean(bwt_values) if bwt_values else 0.0
    
    # Calculate Forward Transfer (FWT) if possible
    # This requires baseline performances, which might not be available
    
    metrics = {
        'average_accuracy': acc,
        'backward_transfer': bwt,
        # 'forward_transfer': fwt  # Not calculated
    }
    
    return metrics


def plot_learning_curves(
    training_curves: Dict,
    save_path: str = None,
    title: str = 'Training Curves',
    show: bool = False
):
    """
    Plot training and validation loss curves.
    
    Args:
        training_curves: Dictionary mapping run/adapter names to metrics 
                        (with train_losses and val_losses)
        save_path: Path to save the plot
        title: Plot title
        show: Whether to display the plot
    """
    plt.figure(figsize=(12, 8))
    
    for name, metrics in training_curves.items():
        if 'train_losses' in metrics:
            plt.plot(
                metrics['train_losses'],
                label=f'{name} (train)',
                linestyle='-'
            )
        
        if 'val_losses' in metrics and metrics['val_losses']:
            plt.plot(
                metrics['val_losses'],
                label=f'{name} (val)',
                linestyle='--'
            )
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


def plot_accuracy_matrix(
    task_metrics: Dict,
    save_path: str = None,
    title: str = 'Accuracy Matrix',
    show: bool = False
):
    """
    Plot accuracy matrix for continual learning evaluation.
    The matrix shows the accuracy of the model on task j (columns)
    after learning task i (rows).
    
    Args:
        task_metrics: Dictionary with metrics for all tasks across learning sequence
        save_path: Path to save the plot
        title: Plot title
        show: Whether to display the plot
    """
    # Extract task sequence
    task_ids = []
    accuracy_matrix = []
    
    for key in sorted(task_metrics.keys()):
        if key.startswith('after_task_'):
            task_id = int(key.split('_')[-1])
            task_ids.append(task_id)
            
            # Extract accuracies for all tasks seen so far
            row = []
            for j in range(task_id + 1):
                if str(j) in task_metrics[key]:
                    row.append(task_metrics[key][str(j)]['accuracy'] * 100)  # Convert to percentage
                else:
                    row.append(0.0)  # Placeholder for missing data
            
            # Pad with zeros for future tasks
            row.extend([0.0] * (len(task_ids) - task_id))
            accuracy_matrix.append(row)
    
    # Convert to numpy array
    accuracy_matrix = np.array(accuracy_matrix)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    cax = plt.matshow(
        accuracy_matrix,
        cmap='viridis',
        fignum=1
    )
    plt.colorbar(cax, label='Accuracy (%)')
    
    # Set axis labels
    plt.xlabel('Task Index')
    plt.ylabel('After Learning Task')
    plt.title(title)
    
    # Set tick labels
    plt.xticks(range(len(task_ids)), task_ids)
    plt.yticks(range(len(task_ids)), task_ids)
    
    # Add accuracy values to cells
    for i in range(accuracy_matrix.shape[0]):
        for j in range(accuracy_matrix.shape[1]):
            if j <= i:  # Only for tasks seen so far
                plt.text(
                    j, i, f'{accuracy_matrix[i, j]:.1f}',
                    ha='center', va='center',
                    color='white' if accuracy_matrix[i, j] < 70 else 'black'
                )
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


def plot_forgetting_comparison(
    all_methods_metrics: Dict[str, Dict],
    save_path: str = None,
    title: str = 'Forgetting Metrics Comparison',
    show: bool = False
):
    """
    Plot comparison of forgetting metrics across different methods.
    
    Args:
        all_methods_metrics: Dictionary mapping method names to their metrics
        save_path: Path to save the plot
        title: Plot title
        show: Whether to display the plot
    """
    metrics_to_plot = ['average_accuracy', 'backward_transfer']
    
    # Prepare data for plotting
    methods = []
    data = {metric: [] for metric in metrics_to_plot}
    
    for method_name, metrics in all_methods_metrics.items():
        methods.append(method_name)
        
        # Extract forgetting metrics
        if 'forgetting_metrics' in metrics:
            forgetting_metrics = metrics['forgetting_metrics']
        elif 'task_metrics' in metrics and 'forgetting_metrics' in metrics['task_metrics']:
            forgetting_metrics = metrics['task_metrics']['forgetting_metrics']
        else:
            # Skip if metrics not available
            continue
        
        # Extract values for plotting
        for metric in metrics_to_plot:
            if metric in forgetting_metrics:
                # Convert accuracy to percentage
                if metric == 'average_accuracy':
                    data[metric].append(forgetting_metrics[metric] * 100)
                else:
                    data[metric].append(forgetting_metrics[metric] * 100)
            else:
                data[metric].append(0.0)  # Placeholder for missing data
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Number of metrics and methods
    n_metrics = len(metrics_to_plot)
    n_methods = len(methods)
    
    # Width of each bar group
    bar_width = 0.8 / n_metrics
    
    # X-positions for each method
    x = np.arange(n_methods)
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(
            x + i * bar_width - (n_metrics - 1) * bar_width / 2,
            data[metric],
            width=bar_width,
            label=metric.replace('_', ' ').title()
        )
    
    # Add values on top of bars
    for i, metric in enumerate(metrics_to_plot):
        for j, value in enumerate(data[metric]):
            plt.text(
                x[j] + i * bar_width - (n_metrics - 1) * bar_width / 2,
                value + 1,  # Small offset for visibility
                f'{value:.1f}',
                ha='center', va='bottom',
                fontsize=9
            )
    
    # Set axis labels and title
    plt.xlabel('Method')
    plt.ylabel('Value (%)')
    plt.title(title)
    
    # Set x-tick labels to method names
    plt.xticks(x, methods)
    
    # Add legend and grid
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


def plot_adaptation_speed(
    adaptation_data: Dict[str, List[float]],
    steps: List[int],
    save_path: str = None,
    title: str = 'Adaptation Speed Comparison',
    show: bool = False
):
    """
    Plot comparison of adaptation speed across different methods.
    
    Args:
        adaptation_data: Dictionary mapping method names to their performance at each step
        steps: List of step/epoch numbers
        save_path: Path to save the plot
        title: Plot title
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    
    for method, values in adaptation_data.items():
        plt.plot(steps, values, marker='o', label=method)
    
    plt.xlabel('Gradient Updates')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


def plot_parameter_efficiency(
    methods: List[str],
    trainable_params: List[int],
    accuracies: List[float],
    save_path: str = None,
    title: str = 'Parameter Efficiency vs Performance',
    show: bool = False
):
    """
    Plot parameter efficiency against performance.
    
    Args:
        methods: List of method names
        trainable_params: List of trainable parameters for each method
        accuracies: List of accuracy values for each method
        save_path: Path to save the plot
        title: Plot title
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot with method names as labels
    plt.scatter(trainable_params, accuracies, s=100)
    
    # Add method names as annotations
    for i, method in enumerate(methods):
        plt.annotate(
            method,
            (trainable_params[i], accuracies[i]),
            xytext=(10, 5),
            textcoords='offset points',
            fontsize=10
        )
    
    plt.xscale('log')  # Log scale for parameters
    plt.xlabel('Trainable Parameters (log scale)')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


def create_metrics_table(
    metrics_by_method: Dict[str, Dict],
    metrics_to_include: List[str] = None,
    precision: int = 2
):
    """
    Create a formatted table of metrics for different methods.
    
    Args:
        metrics_by_method: Dictionary mapping method names to their metrics
        metrics_to_include: List of metric names to include (default: all)
        precision: Number of decimal places for rounding
        
    Returns:
        DataFrame with formatted metrics table
    """
    if metrics_to_include is None:
        # Try to infer metrics from the first method's data
        first_method = next(iter(metrics_by_method.values()))
        if 'forgetting_metrics' in first_method:
            metrics_to_include = list(first_method['forgetting_metrics'].keys())
        elif 'task_metrics' in first_method and 'forgetting_metrics' in first_method['task_metrics']:
            metrics_to_include = list(first_method['task_metrics']['forgetting_metrics'].keys())
        else:
            # Default metrics
            metrics_to_include = ['average_accuracy', 'backward_transfer']
    
    # Prepare data for table
    table_data = []
    
    for method_name, metrics in metrics_by_method.items():
        row = {'Method': method_name}
        
        # Extract forgetting metrics
        if 'forgetting_metrics' in metrics:
            forgetting_metrics = metrics['forgetting_metrics']
        elif 'task_metrics' in metrics and 'forgetting_metrics' in metrics['task_metrics']:
            forgetting_metrics = metrics['task_metrics']['forgetting_metrics']
        else:
            # Skip if metrics not available
            continue
        
        # Extract values for table
        for metric in metrics_to_include:
            if metric in forgetting_metrics:
                # Format the value
                value = forgetting_metrics[metric]
                if metric == 'average_accuracy':
                    # Convert to percentage
                    value = value * 100
                
                row[metric] = round(value, precision)
            else:
                row[metric] = 'N/A'
        
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Rename columns to more readable format
    column_mapping = {
        'average_accuracy': 'Average Accuracy (%)',
        'backward_transfer': 'BWT (%)',
        'forward_transfer': 'FWT (%)'
    }
    
    df.rename(columns={k: column_mapping.get(k, k) for k in df.columns}, inplace=True)
    
    return df


def measure_adaptation_speed(
    model,
    adapter_name,
    dataloader,
    device,
    max_steps=100,
    step_interval=10,
    target_accuracy=None
):
    """
    Measure the adaptation speed of a model on a given dataset.
    
    Args:
        model: Model to adapt
        adapter_name: Name of the adapter to use
        dataloader: DataLoader with training data
        device: Device to run training on
        max_steps: Maximum number of gradient updates to perform
        step_interval: Interval for measuring performance
        target_accuracy: Stop when this accuracy is reached (optional)
        
    Returns:
        Dictionary with step counts and corresponding accuracies
    """
    model.train()
    
    # Setup optimizer for adapter parameters
    adapter_params = model.model.get_adapter_parameters(adapter_name)
    optimizer = torch.optim.Adam(adapter_params, lr=0.001)
    
    # Initialize tracking
    steps = []
    accuracies = []
    step_count = 0
    
    # Create validation dataloader
    if hasattr(dataloader.dataset, 'dataset'):
        # If dataset is a Subset, get the original dataset
        validation_dataset = dataloader.dataset.dataset
    else:
        validation_dataset = dataloader.dataset
    
    # Get a subset of the dataset for quick validation
    val_size = min(100, len(validation_dataset))
    val_indices = np.random.choice(len(validation_dataset), val_size, replace=False)
    validation_subset = torch.utils.data.Subset(validation_dataset, val_indices)
    validation_dataloader = DataLoader(validation_subset, batch_size=16, shuffle=False)
    
    # Training loop
    for epoch in range((max_steps // len(dataloader)) + 1):
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            optimizer.zero_grad()
            inputs = {k: v for k, v in batch.items() if k != 'labels'}
            outputs = model(inputs, adapter_name=adapter_name)
            loss = F.cross_entropy(outputs.logits, batch['labels'])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Increment step count
            step_count += 1
            
            # Check if we should measure performance
            if step_count % step_interval == 0 or step_count == max_steps:
                # Measure accuracy
                accuracy = compute_accuracy(model, validation_dataloader, adapter_name, device)
                
                # Record progress
                steps.append(step_count)
                accuracies.append(accuracy)
                
                # Check if target accuracy is reached
                if target_accuracy is not None and accuracy >= target_accuracy:
                    return {
                        'steps': steps,
                        'accuracies': accuracies,
                        'steps_to_target': step_count
                    }
            
            # Check if max steps reached
            if step_count >= max_steps:
                break
        
        # Break outer loop if max steps reached
        if step_count >= max_steps:
            break
    
    # Return results
    return {
        'steps': steps,
        'accuracies': accuracies,
        'steps_to_target': None  # Target not reached
    }
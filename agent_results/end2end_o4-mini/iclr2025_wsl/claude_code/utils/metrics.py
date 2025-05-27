"""
Evaluation metrics and visualization tools for permutation-equivariant weight embeddings.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.manifold import TSNE
from scipy.stats import spearmanr
import pandas as pd
import json


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot (if None, the plot is shown)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    
    if 'val_loss' in history and len(history['val_loss']) > 0:
        plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_embedding_visualization(embeddings, labels, label_type='architecture', 
                                title=None, save_path=None, perplexity=30):
    """
    Visualize embeddings with t-SNE.
    
    Args:
        embeddings: Embedding tensors [N, embedding_dim]
        labels: Labels for color coding [N]
        label_type: Type of labels (e.g., 'architecture', 'task', 'accuracy')
        title: Plot title
        save_path: Path to save the plot (if None, the plot is shown)
        perplexity: t-SNE perplexity parameter
    """
    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Different plotting based on label type
    if label_type == 'accuracy':
        # Continuous colormap for accuracy
        sc = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, 
                        cmap='viridis', alpha=0.8, s=50)
        plt.colorbar(sc, label='Accuracy')
    else:
        # Categorical labels
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                      c=[colors[i]], label=f'{label_type} {label}',
                      alpha=0.8, s=50)
        
        plt.legend(title=label_type.capitalize())
    
    if title is None:
        title = f'Embeddings Visualization by {label_type.capitalize()}'
    
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_retrieval_performance(retrieval_results, task_types=None, save_path=None):
    """
    Plot retrieval performance (Recall@k).
    
    Args:
        retrieval_results: Dictionary with retrieval metrics
        task_types: List of task types to include (e.g., ['architecture', 'task'])
        save_path: Path to save the plot (if None, the plot is shown)
    """
    if task_types is None:
        task_types = ['architecture', 'task']
    
    # Extract Recall@k values
    ks = []
    recalls = {task: [] for task in task_types}
    
    for key, value in retrieval_results.items():
        if key.startswith('recall@'):
            parts = key.split('_')
            k = int(parts[0].split('@')[1])
            task = parts[1]
            
            if task in task_types:
                if k not in ks:
                    ks.append(k)
                recalls[task].append((k, value))
    
    # Sort by k
    ks.sort()
    for task in task_types:
        recalls[task].sort(key=lambda x: x[0])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    for task in task_types:
        if recalls[task]:
            k_values, recall_values = zip(*recalls[task])
            plt.plot(k_values, recall_values, marker='o', label=f'{task.capitalize()}')
    
    plt.xlabel('k')
    plt.ylabel('Recall@k')
    plt.title('Retrieval Performance')
    plt.legend()
    plt.grid(True)
    
    # Set x-axis ticks to integers
    plt.xticks(ks)
    
    # Set y-axis limits
    plt.ylim(0, 1.05)
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_accuracy_prediction(true_values, predictions, save_path=None):
    """
    Plot actual vs. predicted accuracies.
    
    Args:
        true_values: True accuracy values
        predictions: Predicted accuracy values
        save_path: Path to save the plot (if None, the plot is shown)
    """
    # Convert to numpy
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.cpu().numpy()
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    # Ensure 1D arrays
    true_values = true_values.flatten()
    predictions = predictions.flatten()
    
    # Compute metrics
    mse = mean_squared_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    corr, _ = spearmanr(true_values, predictions)
    
    # Create plot
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(true_values, predictions, alpha=0.5, s=50)
    
    # Perfect prediction line
    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Add metrics to plot
    plt.text(0.05, 0.95, f'MSE: {mse:.4f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.90, f'R²: {r2:.4f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.85, f'Spearman ρ: {corr:.4f}', transform=plt.gca().transAxes, fontsize=12)
    
    plt.xlabel('True Accuracy')
    plt.ylabel('Predicted Accuracy')
    plt.title('Accuracy Prediction Performance')
    plt.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    plt.axis('equal')
    
    # Set axis limits
    plt.xlim(min_val - 0.05, max_val + 0.05)
    plt.ylim(min_val - 0.05, max_val + 0.05)
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_embedding_similarity_matrix(embeddings, labels, label_type='architecture', 
                                    save_path=None):
    """
    Plot embedding similarity matrix.
    
    Args:
        embeddings: Embedding tensors [N, embedding_dim]
        labels: Labels for grouping [N]
        label_type: Type of labels (e.g., 'architecture', 'task')
        save_path: Path to save the plot (if None, the plot is shown)
    """
    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_normalized = embeddings / norms
    
    # Compute similarity matrix
    similarity = np.matmul(embeddings_normalized, embeddings_normalized.T)
    
    # Sort by labels for grouped visualization
    sort_indices = np.argsort(labels)
    labels_sorted = labels[sort_indices]
    similarity_sorted = similarity[sort_indices][:, sort_indices]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot similarity matrix
    sns.heatmap(similarity_sorted, cmap='viridis', vmin=-1, vmax=1, 
              xticklabels=False, yticklabels=False)
    
    # Add label boundaries
    unique_labels = np.unique(labels_sorted)
    boundaries = [0]
    
    for label in unique_labels:
        boundaries.append(boundaries[-1] + np.sum(labels_sorted == label))
    
    # Draw boundaries
    for b in boundaries[1:-1]:
        plt.axhline(y=b, color='r', linestyle='-', linewidth=1)
        plt.axvline(x=b, color='r', linestyle='-', linewidth=1)
    
    plt.title(f'Embedding Similarity Matrix (Grouped by {label_type.capitalize()})')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_interpolation_path(model_A_id, model_B_id, alphas, accuracies, save_path=None):
    """
    Plot the performance of interpolated models.
    
    Args:
        model_A_id: ID of the first model
        model_B_id: ID of the second model
        alphas: Interpolation coefficients (0.0 to 1.0)
        accuracies: Corresponding accuracy values
        save_path: Path to save the plot (if None, the plot is shown)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot interpolation curve
    plt.plot(alphas, accuracies, marker='o', linestyle='-', linewidth=2)
    
    # Mark endpoints
    plt.scatter([0, 1], [accuracies[0], accuracies[-1]], color='red', s=100, 
              label='Original Models', zorder=10)
    
    # Add labels
    plt.text(0, accuracies[0], f'  Model A: {model_A_id}', verticalalignment='bottom')
    plt.text(1, accuracies[-1], f'  Model B: {model_B_id}', verticalalignment='bottom')
    
    # Find best interpolation point
    best_idx = np.argmax(accuracies)
    best_alpha = alphas[best_idx]
    best_accuracy = accuracies[best_idx]
    
    # Mark best point
    plt.scatter([best_alpha], [best_accuracy], color='green', s=100, 
              label=f'Best (α={best_alpha:.2f})', zorder=10)
    
    plt.xlabel('Interpolation Coefficient (α)')
    plt.ylabel('Accuracy')
    plt.title('Model Interpolation Performance')
    plt.grid(True)
    plt.legend()
    
    # Set x-axis limits
    plt.xlim(-0.05, 1.05)
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def create_summary_table(results, model_names, metrics, formats=None, save_path=None):
    """
    Create a formatted summary table of results.
    
    Args:
        results: Dictionary mapping model names to dictionaries of metrics
        model_names: List of model names to include
        metrics: List of metrics to include
        formats: Dictionary mapping metrics to format strings
        save_path: Path to save the table (if None, the table is returned as string)
        
    Returns:
        Formatted table as string (if save_path is None)
    """
    if formats is None:
        formats = {}
    
    # Default format is 4 decimal places
    default_format = '.4f'
    
    # Create DataFrame
    data = []
    
    for model in model_names:
        row = {'Model': model}
        
        if model in results:
            for metric in metrics:
                if metric in results[model]:
                    # Apply format if specified
                    value = results[model][metric]
                    
                    if isinstance(value, (int, float)):
                        format_str = formats.get(metric, default_format)
                        row[metric] = f'{value:{format_str}}'
                    else:
                        row[metric] = str(value)
                else:
                    row[metric] = 'N/A'
        else:
            row.update({metric: 'N/A' for metric in metrics})
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Format as table
    table = df.to_markdown(index=False)
    
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(table)
    
    return table


def aggregate_results(experiment_dir, model_names, save_path=None):
    """
    Aggregate results from multiple experiments.
    
    Args:
        experiment_dir: Directory containing experiment results
        model_names: List of model names
        save_path: Path to save the aggregated results (if None, results are returned)
        
    Returns:
        Dictionary with aggregated results (if save_path is None)
    """
    results = {}
    
    for model in model_names:
        model_dir = os.path.join(experiment_dir, model)
        
        if not os.path.isdir(model_dir):
            continue
        
        results[model] = {}
        
        # Retrieve retrieval results
        retrieval_path = os.path.join(model_dir, 'retrieval_results.json')
        if os.path.exists(retrieval_path):
            with open(retrieval_path, 'r') as f:
                retrieval_results = json.load(f)
                results[model].update(retrieval_results)
        
        # Retrieve accuracy prediction results
        accuracy_path = os.path.join(model_dir, 'accuracy_prediction_results.json')
        if os.path.exists(accuracy_path):
            with open(accuracy_path, 'r') as f:
                accuracy_results = json.load(f)
                results[model].update(accuracy_results)
        
        # Retrieve training history
        history_path = os.path.join(model_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
                
                # Extract final loss values
                if 'train_loss' in history and history['train_loss']:
                    results[model]['final_train_loss'] = history['train_loss'][-1]
                
                if 'val_loss' in history and history['val_loss']:
                    results[model]['final_val_loss'] = history['val_loss'][-1]
    
    # Save aggregated results
    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def plot_comparative_metrics(results, model_names, metrics, labels=None, save_path=None):
    """
    Create comparative bar plots for metrics across models.
    
    Args:
        results: Dictionary mapping model names to dictionaries of metrics
        model_names: List of model names to include
        metrics: List of metrics to include
        labels: Dictionary mapping metrics to display labels
        save_path: Path to save the plot (if None, the plot is shown)
    """
    if labels is None:
        labels = {metric: metric for metric in metrics}
    
    # Set up the figure
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 5, 6))
    
    # Handle case of a single metric
    if n_metrics == 1:
        axes = [axes]
    
    # Create bar plots for each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract values
        values = []
        for model in model_names:
            if model in results and metric in results[model]:
                values.append(results[model][metric])
            else:
                values.append(0)  # Or some placeholder value
        
        # Create bar plot
        bars = ax.bar(model_names, values, width=0.6)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        # Set labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel(labels.get(metric, metric))
        ax.set_title(labels.get(metric, metric))
        
        # Rotate x-tick labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
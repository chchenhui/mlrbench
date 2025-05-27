"""
Visualization utilities for the AEB project.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import transforms
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

def set_plot_style():
    """Set the plotting style."""
    plt.style.use('seaborn-whitegrid')
    sns.set_theme(style="whitegrid")
    
    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

def save_figure(fig, filename, output_dir='./results', dpi=300):
    """Save a figure to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    logger.info(f"Figure saved to {filepath}")
    return filepath

def plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir='./results'):
    """
    Plot training and validation losses and accuracies.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        output_dir: Directory to save the plots
    
    Returns:
        Tuple of filepaths to saved figures
    """
    set_plot_style()
    
    # Plot losses
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax_loss.plot(epochs, train_losses, 'b-', marker='o', label='Training Loss')
    ax_loss.plot(epochs, val_losses, 'r-', marker='s', label='Validation Loss')
    ax_loss.set_title('Training and Validation Loss')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True)
    
    # Add data points to the plot
    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        ax_loss.annotate(f'{train_loss:.3f}', (epochs[i], train_losses[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
        ax_loss.annotate(f'{val_loss:.3f}', (epochs[i], val_losses[i]), 
                    textcoords="offset points", xytext=(0,-15), ha='center')
    
    loss_path = save_figure(fig_loss, 'loss_curves.png', output_dir)
    plt.close(fig_loss)
    
    # Plot accuracies
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    ax_acc.plot(epochs, train_accs, 'b-', marker='o', label='Training Accuracy')
    ax_acc.plot(epochs, val_accs, 'r-', marker='s', label='Validation Accuracy')
    ax_acc.set_title('Training and Validation Accuracy')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.legend()
    ax_acc.grid(True)
    
    # Add data points to the plot
    for i, (train_acc, val_acc) in enumerate(zip(train_accs, val_accs)):
        ax_acc.annotate(f'{train_acc:.1f}%', (epochs[i], train_accs[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
        ax_acc.annotate(f'{val_acc:.1f}%', (epochs[i], val_accs[i]), 
                    textcoords="offset points", xytext=(0,-15), ha='center')
    
    acc_path = save_figure(fig_acc, 'accuracy_curves.png', output_dir)
    plt.close(fig_acc)
    
    return loss_path, acc_path

def plot_performance_comparison(model_names, metrics, metric_name, title, output_dir='./results'):
    """
    Plot comparison of model performances.
    
    Args:
        model_names: List of model names
        metrics: List of metric values for each model
        metric_name: Name of the metric
        title: Plot title
        output_dir: Directory to save the plot
    
    Returns:
        Path to saved figure
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar chart
    bars = ax.bar(model_names, metrics, color=sns.color_palette("muted"))
    
    # Add data labels on bars
    for bar, metric in zip(bars, metrics):
        ax.text(bar.get_x() + bar.get_width()/2., 
                bar.get_height() + 0.01*max(metrics), 
                f'{metric:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Add labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add horizontal line for standard performance
    if 'Standard' in model_names:
        std_idx = model_names.index('Standard')
        std_value = metrics[std_idx]
        ax.axhline(y=std_value, color='r', linestyle='--', 
                   label=f'Standard Model Performance ({std_value:.2f})')
        ax.legend()
    
    # Save figure
    filename = f'{metric_name.lower().replace(" ", "_")}_comparison.png'
    fig_path = save_figure(fig, filename, output_dir)
    plt.close(fig)
    
    return fig_path

def visualize_transformed_images(original_images, transformed_images, labels, class_names, output_dir='./results'):
    """
    Visualize original images and their transformed versions.
    
    Args:
        original_images: Tensor of original images (B, C, H, W)
        transformed_images: Tensor of transformed images (B, C, H, W)
        labels: Tensor of labels (B)
        class_names: List of class names
        output_dir: Directory to save the visualizations
    
    Returns:
        Path to saved figure
    """
    set_plot_style()
    
    num_samples = min(5, original_images.shape[0])
    fig, axs = plt.subplots(2, num_samples, figsize=(num_samples*3, 6))
    
    # Convert tensors to numpy for visualization
    if isinstance(original_images, torch.Tensor):
        original_images = original_images.detach().cpu().numpy()
    if isinstance(transformed_images, torch.Tensor):
        transformed_images = transformed_images.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Display images
    for i in range(num_samples):
        # Original image
        orig_img = np.transpose(original_images[i], (1, 2, 0))
        # If normalized, denormalize
        orig_img = np.clip(orig_img, 0, 1)
        axs[0, i].imshow(orig_img)
        axs[0, i].set_title(f"Original: {class_names[labels[i]]}")
        axs[0, i].axis('off')
        
        # Transformed image
        trans_img = np.transpose(transformed_images[i], (1, 2, 0))
        # If normalized, denormalize
        trans_img = np.clip(trans_img, 0, 1)
        axs[1, i].imshow(trans_img)
        axs[1, i].set_title(f"Transformed: {class_names[labels[i]]}")
        axs[1, i].axis('off')
    
    plt.tight_layout()
    fig_path = save_figure(fig, 'transformed_images.png', output_dir)
    plt.close(fig)
    
    return fig_path

def plot_evolution_progress(generations, best_fitnesses, avg_fitnesses, output_dir='./results'):
    """
    Plot the progress of the evolutionary algorithm.
    
    Args:
        generations: List of generation numbers
        best_fitnesses: List of best fitness values for each generation
        avg_fitnesses: List of average fitness values for each generation
        output_dir: Directory to save the plot
    
    Returns:
        Path to saved figure
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(generations, best_fitnesses, 'b-', marker='o', label='Best Fitness')
    ax.plot(generations, avg_fitnesses, 'g-', marker='s', label='Average Fitness')
    
    # Add data points to the plot
    for i, (best_fit, avg_fit) in enumerate(zip(best_fitnesses, avg_fitnesses)):
        if i % max(1, len(generations) // 10) == 0:  # Add labels for every 10% of points
            ax.annotate(f'{best_fit:.3f}', (generations[i], best_fitnesses[i]), 
                        textcoords="offset points", xytext=(0,10), ha='center')
            ax.annotate(f'{avg_fit:.3f}', (generations[i], avg_fitnesses[i]), 
                        textcoords="offset points", xytext=(0,-15), ha='center')
    
    ax.set_title('Evolutionary Progress')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.legend()
    ax.grid(True)
    
    fig_path = save_figure(fig, 'evolution_progress.png', output_dir)
    plt.close(fig)
    
    return fig_path

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', output_dir='./results'):
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix (n_classes, n_classes)
        class_names: List of class names
        title: Plot title
        output_dir: Directory to save the plot
    
    Returns:
        Path to saved figure
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    
    fig_path = save_figure(fig, 'confusion_matrix.png', output_dir)
    plt.close(fig)
    
    return fig_path
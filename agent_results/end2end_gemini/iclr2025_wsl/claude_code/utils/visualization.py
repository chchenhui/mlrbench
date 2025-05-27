import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import logging

logger = logging.getLogger(__name__)

def set_plotting_style():
    """Set matplotlib style for consistent visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.title_fontsize'] = 12

def plot_training_history(
    history_path: str, 
    save_dir: str,
    model_name: str = "model", 
    include_test: bool = True
) -> List[str]:
    """
    Plot training history.
    
    Args:
        history_path: Path to the training history JSON file
        save_dir: Directory to save plots
        model_name: Name of the model
        include_test: Whether to include test metrics
        
    Returns:
        List of paths to saved plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Get property names from history
    property_names = list(history.get('property_maes', {}).keys())
    
    saved_plots = []
    
    # Set style
    set_plotting_style()
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    if include_test and 'test_loss' in history:
        plt.plot(history['test_loss'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    loss_plot_path = os.path.join(save_dir, f'{model_name}_loss.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append(loss_plot_path)
    
    # Plot overall MAE
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_mae'], label='Train')
    plt.plot(history['val_mae'], label='Validation')
    if include_test and 'test_mae' in history:
        plt.plot(history['test_mae'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title(f'{model_name} - Overall Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    
    mae_plot_path = os.path.join(save_dir, f'{model_name}_overall_mae.png')
    plt.savefig(mae_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append(mae_plot_path)
    
    # Plot learning rate
    if 'learning_rate' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'{model_name} - Learning Rate Schedule')
        plt.grid(True)
        plt.yscale('log')
        
        lr_plot_path = os.path.join(save_dir, f'{model_name}_learning_rate.png')
        plt.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(lr_plot_path)
    
    # Plot per-property MAE
    if 'property_maes' in history:
        for prop in property_names:
            plt.figure(figsize=(10, 6))
            
            if prop in history['property_maes']:
                plt.plot(history['property_maes'][prop]['train'], label='Train')
                plt.plot(history['property_maes'][prop]['val'], label='Validation')
                if include_test and 'test' in history['property_maes'][prop]:
                    plt.plot(history['property_maes'][prop]['test'], label='Test')
                
                plt.xlabel('Epoch')
                plt.ylabel('MAE')
                plt.title(f'{model_name} - {prop} Mean Absolute Error')
                plt.legend()
                plt.grid(True)
                
                prop_mae_plot_path = os.path.join(save_dir, f'{model_name}_{prop}_mae.png')
                plt.savefig(prop_mae_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                saved_plots.append(prop_mae_plot_path)
    
    # Plot per-property R2
    if 'property_r2s' in history:
        for prop in property_names:
            plt.figure(figsize=(10, 6))
            
            if prop in history['property_r2s']:
                plt.plot(history['property_r2s'][prop]['train'], label='Train')
                plt.plot(history['property_r2s'][prop]['val'], label='Validation')
                if include_test and 'test' in history['property_r2s'][prop]:
                    plt.plot(history['property_r2s'][prop]['test'], label='Test')
                
                plt.xlabel('Epoch')
                plt.ylabel('R²')
                plt.title(f'{model_name} - {prop} R² Score')
                plt.legend()
                plt.grid(True)
                
                prop_r2_plot_path = os.path.join(save_dir, f'{model_name}_{prop}_r2.png')
                plt.savefig(prop_r2_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                saved_plots.append(prop_r2_plot_path)
    
    # Create a combined metrics plot using subplots
    num_properties = len(property_names)
    fig, axes = plt.subplots(num_properties + 1, 2, figsize=(18, 4 * (num_properties + 1)))
    
    # Plot overall metrics
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    if include_test and 'test_loss' in history:
        axes[0, 0].plot(history['test_loss'], label='Test')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Overall Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['train_mae'], label='Train')
    axes[0, 1].plot(history['val_mae'], label='Validation')
    if include_test and 'test_mae' in history:
        axes[0, 1].plot(history['test_mae'], label='Test')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Overall MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot per-property metrics
    for i, prop in enumerate(property_names):
        row = i + 1
        
        # Plot property MAE
        if 'property_maes' in history and prop in history['property_maes']:
            axes[row, 0].plot(history['property_maes'][prop]['train'], label='Train')
            axes[row, 0].plot(history['property_maes'][prop]['val'], label='Validation')
            if include_test and 'test' in history['property_maes'][prop]:
                axes[row, 0].plot(history['property_maes'][prop]['test'], label='Test')
            
            axes[row, 0].set_xlabel('Epoch')
            axes[row, 0].set_ylabel('MAE')
            axes[row, 0].set_title(f'{prop} MAE')
            axes[row, 0].legend()
            axes[row, 0].grid(True)
        
        # Plot property R2
        if 'property_r2s' in history and prop in history['property_r2s']:
            axes[row, 1].plot(history['property_r2s'][prop]['train'], label='Train')
            axes[row, 1].plot(history['property_r2s'][prop]['val'], label='Validation')
            if include_test and 'test' in history['property_r2s'][prop]:
                axes[row, 1].plot(history['property_r2s'][prop]['test'], label='Test')
            
            axes[row, 1].set_xlabel('Epoch')
            axes[row, 1].set_ylabel('R²')
            axes[row, 1].set_title(f'{prop} R² Score')
            axes[row, 1].legend()
            axes[row, 1].grid(True)
    
    plt.tight_layout()
    combined_plot_path = os.path.join(save_dir, f'{model_name}_combined_metrics.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append(combined_plot_path)
    
    return saved_plots

def plot_predictions_vs_targets(
    predictions: np.ndarray,
    targets: np.ndarray,
    property_names: List[str],
    save_dir: str,
    model_name: str = "model",
) -> List[str]:
    """
    Plot predictions vs targets for each property.
    
    Args:
        predictions: Model predictions [n_samples, n_properties]
        targets: Ground truth targets [n_samples, n_properties]
        property_names: Names of properties
        save_dir: Directory to save plots
        model_name: Name of the model
        
    Returns:
        List of paths to saved plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    set_plotting_style()
    
    plot_paths = []
    
    for i, prop in enumerate(property_names):
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(targets[:, i], predictions[:, i])
        rmse = np.sqrt(mean_squared_error(targets[:, i], predictions[:, i]))
        r2 = r2_score(targets[:, i], predictions[:, i])
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        
        # Plot scatter points
        scatter = plt.scatter(targets[:, i], predictions[:, i], alpha=0.6, 
                             edgecolor='k', linewidth=0.5)
        
        # Plot identity line
        min_val = min(np.min(targets[:, i]), np.min(predictions[:, i]))
        max_val = max(np.max(targets[:, i]), np.max(predictions[:, i]))
        padding = (max_val - min_val) * 0.05
        plt.plot([min_val - padding, max_val + padding], 
                [min_val - padding, max_val + padding], 
                'r--', label='y = x')
        
        # Add metrics text box
        plt.text(
            0.05, 0.95, 
            f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.xlabel(f'True {prop}')
        plt.ylabel(f'Predicted {prop}')
        plt.title(f'{model_name} - {prop} Predictions vs Targets')
        plt.grid(True)
        plt.xlim(min_val - padding, max_val + padding)
        plt.ylim(min_val - padding, max_val + padding)
        
        # Save plot
        plot_path = os.path.join(save_dir, f'{model_name}_{prop}_preds_vs_targets.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
        
        # Create density plot
        plt.figure(figsize=(10, 8))
        
        # Calculate point density
        from scipy.stats import gaussian_kde
        xy = np.vstack([targets[:, i], predictions[:, i]])
        density = gaussian_kde(xy)(xy)
        
        # Sort points by density for better visualization
        idx = density.argsort()
        x, y, z = targets[:, i][idx], predictions[:, i][idx], density[idx]
        
        plt.scatter(x, y, c=z, s=50, edgecolor='', alpha=0.8, cmap='viridis')
        plt.colorbar(label='Density')
        
        # Plot identity line
        plt.plot([min_val - padding, max_val + padding], 
                [min_val - padding, max_val + padding], 
                'r--', label='y = x')
        
        # Add metrics text box
        plt.text(
            0.05, 0.95, 
            f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.xlabel(f'True {prop}')
        plt.ylabel(f'Predicted {prop}')
        plt.title(f'{model_name} - {prop} Predictions vs Targets (Density)')
        plt.grid(True)
        plt.xlim(min_val - padding, max_val + padding)
        plt.ylim(min_val - padding, max_val + padding)
        
        # Save plot
        density_plot_path = os.path.join(save_dir, f'{model_name}_{prop}_preds_vs_targets_density.png')
        plt.savefig(density_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(density_plot_path)
        
        # Create hexbin plot
        plt.figure(figsize=(10, 8))
        
        plt.hexbin(targets[:, i], predictions[:, i], gridsize=30, cmap='Blues', mincnt=1)
        plt.colorbar(label='Count')
        
        # Plot identity line
        plt.plot([min_val - padding, max_val + padding], 
                [min_val - padding, max_val + padding], 
                'r--', label='y = x')
        
        # Add metrics text box
        plt.text(
            0.05, 0.95, 
            f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.xlabel(f'True {prop}')
        plt.ylabel(f'Predicted {prop}')
        plt.title(f'{model_name} - {prop} Predictions vs Targets (Hexbin)')
        plt.grid(True)
        plt.xlim(min_val - padding, max_val + padding)
        plt.ylim(min_val - padding, max_val + padding)
        
        # Save plot
        hexbin_plot_path = os.path.join(save_dir, f'{model_name}_{prop}_preds_vs_targets_hexbin.png')
        plt.savefig(hexbin_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(hexbin_plot_path)
    
    return plot_paths

def plot_error_distributions(
    predictions: np.ndarray,
    targets: np.ndarray,
    property_names: List[str],
    save_dir: str,
    model_name: str = "model",
) -> List[str]:
    """
    Plot error distributions for each property.
    
    Args:
        predictions: Model predictions [n_samples, n_properties]
        targets: Ground truth targets [n_samples, n_properties]
        property_names: Names of properties
        save_dir: Directory to save plots
        model_name: Name of the model
        
    Returns:
        List of paths to saved plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    set_plotting_style()
    
    plot_paths = []
    
    for i, prop in enumerate(property_names):
        # Calculate errors
        errors = predictions[:, i] - targets[:, i]
        abs_errors = np.abs(errors)
        
        # Create histogram plot
        plt.figure(figsize=(10, 6))
        
        plt.hist(errors, bins=30, alpha=0.7, color='royalblue', edgecolor='black', linewidth=1.2)
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        
        plt.xlabel(f'{prop} Error (Predicted - True)')
        plt.ylabel('Frequency')
        plt.title(f'{model_name} - {prop} Error Distribution')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(save_dir, f'{model_name}_{prop}_error_hist.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
        
        # Create KDE plot
        plt.figure(figsize=(10, 6))
        
        sns.kdeplot(errors, fill=True, color='royalblue', alpha=0.5, linewidth=2)
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        
        # Add statistics
        plt.text(
            0.05, 0.95, 
            f"Mean: {np.mean(errors):.4f}\nStd: {np.std(errors):.4f}\n"
            f"Median: {np.median(errors):.4f}\nMAE: {np.mean(abs_errors):.4f}",
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.xlabel(f'{prop} Error (Predicted - True)')
        plt.ylabel('Density')
        plt.title(f'{model_name} - {prop} Error Distribution (KDE)')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        kde_plot_path = os.path.join(save_dir, f'{model_name}_{prop}_error_kde.png')
        plt.savefig(kde_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(kde_plot_path)
        
        # Create box plot
        plt.figure(figsize=(8, 6))
        
        plt.boxplot(errors, vert=False, patch_artist=True, 
                  boxprops=dict(facecolor='lightblue', color='blue'),
                  whiskerprops=dict(color='blue'),
                  medianprops=dict(color='red'))
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        
        plt.xlabel(f'{prop} Error (Predicted - True)')
        plt.title(f'{model_name} - {prop} Error Distribution (Box Plot)')
        plt.grid(True)
        
        # Save plot
        box_plot_path = os.path.join(save_dir, f'{model_name}_{prop}_error_boxplot.png')
        plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(box_plot_path)
    
    return plot_paths

def plot_model_comparison(
    model_results: Dict[str, Dict[str, float]],
    property_names: List[str],
    metric_name: str = "mae",
    save_dir: str = "figures",
) -> str:
    """
    Plot comparison of multiple models.
    
    Args:
        model_results: Dictionary of model results {model_name: {property_name: metric_value}}
        property_names: Names of properties
        metric_name: Name of the metric to compare (e.g., 'mae', 'rmse', 'r2')
        save_dir: Directory to save plots
        
    Returns:
        Path to saved plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    set_plotting_style()
    
    # Prepare data for plotting
    models = list(model_results.keys())
    data = []
    
    for model in models:
        for prop in property_names:
            data.append({
                'Model': model,
                'Property': prop,
                'Value': model_results[model].get(prop, {}).get(metric_name, np.nan)
            })
    
    df = pd.DataFrame(data)
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    
    ax = sns.barplot(x='Property', y='Value', hue='Model', data=df)
    
    plt.xlabel('Property')
    plt.ylabel(metric_name.upper())
    plt.title(f'Model Comparison - {metric_name.upper()} by Property')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.grid(True, axis='y')
    
    # Adjust layout to make room for rotated x-labels
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f'model_comparison_{metric_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Pivot data for heatmap
    heatmap_data = df.pivot(index='Model', columns='Property', values='Value')
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5)
    
    plt.title(f'Model Comparison - {metric_name.upper()} by Property')
    plt.tight_layout()
    
    # Save plot
    heatmap_path = os.path.join(save_dir, f'model_comparison_{metric_name}_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_comparative_predictions_plot(
    predictions_dict: Dict[str, np.ndarray],
    targets: np.ndarray,
    property_names: List[str],
    save_dir: str,
) -> List[str]:
    """
    Create comparative prediction plots for multiple models.
    
    Args:
        predictions_dict: Dictionary of model predictions {model_name: predictions_array}
        targets: Ground truth targets [n_samples, n_properties]
        property_names: Names of properties
        save_dir: Directory to save plots
        
    Returns:
        List of paths to saved plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    set_plotting_style()
    
    plot_paths = []
    
    for i, prop in enumerate(property_names):
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, r2_score
        
        # Create plot with subplots for each model
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)
        
        # Calculate grid dimensions (aim for a roughly square grid)
        grid_size = int(np.ceil(np.sqrt(n_models)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(5*grid_size, 5*grid_size), 
                               sharex=True, sharey=True)
        
        # Flatten axes array for easier indexing
        axes = axes.flatten()
        
        # Calculate overall min and max for consistent axis limits
        min_val = min(np.min(targets[:, i]), min([np.min(preds[:, i]) for preds in predictions_dict.values()]))
        max_val = max(np.max(targets[:, i]), max([np.max(preds[:, i]) for preds in predictions_dict.values()]))
        padding = (max_val - min_val) * 0.05
        
        for j, model_name in enumerate(model_names):
            if j < len(axes):  # Check if we have enough subplots
                ax = axes[j]
                predictions = predictions_dict[model_name]
                
                # Calculate metrics
                mae = mean_absolute_error(targets[:, i], predictions[:, i])
                r2 = r2_score(targets[:, i], predictions[:, i])
                
                # Scatter plot
                ax.scatter(targets[:, i], predictions[:, i], alpha=0.6, s=30)
                
                # Identity line
                ax.plot([min_val - padding, max_val + padding], 
                       [min_val - padding, max_val + padding], 
                       'r--', linewidth=2)
                
                # Set labels and title
                ax.set_xlabel(f'True {prop}')
                ax.set_ylabel(f'Predicted {prop}')
                ax.set_title(f'{model_name}\nMAE: {mae:.4f}, R²: {r2:.4f}')
                
                # Set axis limits
                ax.set_xlim(min_val - padding, max_val + padding)
                ax.set_ylim(min_val - padding, max_val + padding)
                
                # Add grid
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(len(model_names), len(axes)):
            axes[j].set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, f'comparative_{prop}_predictions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
    
    return plot_paths

def visualize_attention_weights(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    layer_indices: torch.Tensor,
    save_dir: str,
    model_name: str = "model",
) -> List[str]:
    """
    Visualize attention weights for a transformer model.
    
    Args:
        model: Transformer model with attention mechanisms
        sample_input: Sample input tensor [batch_size, seq_length, feature_dim]
        layer_indices: Layer indices tensor [batch_size, seq_length]
        save_dir: Directory to save plots
        model_name: Name of the model
        
    Returns:
        List of paths to saved plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    set_plotting_style()
    
    plot_paths = []
    
    # Set model to eval mode
    model.eval()
    
    # Hook to capture attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        # Assuming the output is the attention weights
        attention_weights.append(output.detach().cpu())
    
    # Register hooks for attention modules
    hooks = []
    
    # Find attention modules in the model
    for name, module in model.named_modules():
        if 'attention' in name.lower() and 'weights' not in name.lower():
            # Customize this based on the specific model architecture
            if hasattr(module, 'forward'):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
    
    # Forward pass to capture attention weights
    with torch.no_grad():
        _ = model(sample_input.to(next(model.parameters()).device),
                 layer_indices.to(next(model.parameters()).device))
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Process and visualize attention weights
    for i, attn_weights in enumerate(attention_weights):
        # For this example, assume attention_weights shape is [batch_size, num_heads, seq_length, seq_length]
        if len(attn_weights.shape) != 4:
            continue
        
        batch_size, num_heads, seq_length, _ = attn_weights.shape
        
        # Visualize attention for each head in the first example
        for head in range(min(4, num_heads)):  # Limit to first 4 heads to avoid too many plots
            plt.figure(figsize=(10, 8))
            
            # Attention heatmap
            sns.heatmap(attn_weights[0, head].numpy(), cmap='viridis', square=True,
                      vmin=0, vmax=1)
            
            plt.title(f'{model_name} - Attention Weights - Layer {i} - Head {head}')
            plt.xlabel('Token Position (Target)')
            plt.ylabel('Token Position (Source)')
            
            # Save plot
            plot_path = os.path.join(save_dir, f'{model_name}_attn_weights_layer{i}_head{head}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
    
    return plot_paths

def plot_property_correlations(
    properties: np.ndarray,
    property_names: List[str],
    save_dir: str,
    title: str = "Property Correlations",
) -> str:
    """
    Plot correlations between properties.
    
    Args:
        properties: Property values [n_samples, n_properties]
        property_names: Names of properties
        save_dir: Directory to save plots
        title: Plot title
        
    Returns:
        Path to saved plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    set_plotting_style()
    
    # Create DataFrame
    df = pd.DataFrame(properties, columns=property_names)
    
    # Compute correlation matrix
    corr = df.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', square=True,
              linewidths=.5, vmin=-1, vmax=1)
    
    plt.title(title)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'property_correlations.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_radar_chart(
    model_results: Dict[str, Dict[str, float]],
    property_names: List[str],
    metric_name: str = "r2",
    save_dir: str = "figures",
) -> str:
    """
    Create a radar chart comparing models.
    
    Args:
        model_results: Dictionary of model results {model_name: {property_name: metric_value}}
        property_names: Names of properties
        metric_name: Name of the metric to compare (e.g., 'r2', 'mae')
        save_dir: Directory to save plots
        
    Returns:
        Path to saved plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    plt.figure(figsize=(10, 10))
    
    # Number of properties
    num_properties = len(property_names)
    
    # Compute angle for each property
    angles = np.linspace(0, 2*np.pi, num_properties, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add the first property at the end to close the polygon
    properties = property_names + [property_names[0]]
    
    # Set up the subplot with polar projection
    ax = plt.subplot(111, polar=True)
    
    # Draw property labels
    plt.xticks(angles[:-1], property_names, fontsize=12)
    
    # Draw y-labels (metrics)
    # For r2, use a 0-1 scale; for error metrics, need to determine appropriate scale
    if metric_name.lower() == 'r2':
        plt.ylim(0, 1)
        yticks = np.linspace(0, 1, 5)
        plt.yticks(yticks, [f"{y:.1f}" for y in yticks], fontsize=10)
    else:
        # For error metrics, normalize values for fair comparison
        all_values = [model_results[model].get(prop, {}).get(metric_name, 0) 
                     for model in model_results for prop in property_names]
        max_value = max(all_values) * 1.1  # Add 10% margin
        plt.ylim(0, max_value)
        yticks = np.linspace(0, max_value, 5)
        plt.yticks(yticks, [f"{y:.2f}" for y in yticks], fontsize=10)
    
    # Draw the radar chart for each model
    for i, model in enumerate(model_results.keys()):
        # Get metric values for all properties
        values = [model_results[model].get(prop, {}).get(metric_name, 0) for prop in property_names]
        
        # Close the loop by appending the first value
        values += values[:1]
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Set chart title
    plt.title(f'Model Comparison - {metric_name.upper()} by Property', size=15, y=1.1)
    
    # Save plot
    plot_path = os.path.join(save_dir, f'radar_chart_{metric_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_interactive_plots(
    predictions: np.ndarray,
    targets: np.ndarray,
    property_names: List[str],
    metadata: Optional[Dict[str, List[Any]]] = None,
    save_dir: str = "figures",
    model_name: str = "model",
) -> List[str]:
    """
    Create interactive plots using Plotly.
    
    Args:
        predictions: Model predictions [n_samples, n_properties]
        targets: Ground truth targets [n_samples, n_properties]
        property_names: Names of properties
        metadata: Optional metadata for samples {metadata_name: [values]}
        save_dir: Directory to save plots
        model_name: Name of the model
        
    Returns:
        List of paths to saved plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plot_paths = []
    
    for i, prop in enumerate(property_names):
        # Calculate errors
        errors = predictions[:, i] - targets[:, i]
        
        # Create DataFrame for easy plotting
        df = pd.DataFrame({
            f'True_{prop}': targets[:, i],
            f'Predicted_{prop}': predictions[:, i],
            'Error': errors,
            'Abs_Error': np.abs(errors)
        })
        
        # Add metadata if provided
        if metadata is not None:
            for meta_name, meta_values in metadata.items():
                df[meta_name] = meta_values
        
        # 3D scatter plot if metadata is available for color
        if metadata is not None and len(metadata) > 0:
            # Use the first metadata column for coloring
            color_column = list(metadata.keys())[0]
            
            # Create 3D scatter plot
            fig = px.scatter_3d(
                df, 
                x=f'True_{prop}', 
                y=f'Predicted_{prop}', 
                z='Abs_Error',
                color=color_column,
                opacity=0.7
            )
            
            fig.update_layout(
                title=f'{model_name} - {prop} Predictions (3D)',
                scene=dict(
                    xaxis_title=f'True {prop}',
                    yaxis_title=f'Predicted {prop}',
                    zaxis_title='Absolute Error'
                )
            )
            
            # Save plot
            plot_path = os.path.join(save_dir, f'{model_name}_{prop}_3d_scatter.html')
            fig.write_html(plot_path)
            plot_paths.append(plot_path)
        
        # 2D scatter plot with hover info
        fig = px.scatter(
            df, 
            x=f'True_{prop}', 
            y=f'Predicted_{prop}',
            color='Abs_Error',
            opacity=0.7,
            color_continuous_scale='Viridis'
        )
        
        # Add identity line
        min_val = min(df[f'True_{prop}'].min(), df[f'Predicted_{prop}'].min())
        max_val = max(df[f'True_{prop}'].max(), df[f'Predicted_{prop}'].max())
        padding = (max_val - min_val) * 0.05
        
        fig.add_trace(
            go.Scatter(
                x=[min_val - padding, max_val + padding],
                y=[min_val - padding, max_val + padding],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='y = x'
            )
        )
        
        fig.update_layout(
            title=f'{model_name} - {prop} Predictions vs Targets',
            xaxis_title=f'True {prop}',
            yaxis_title=f'Predicted {prop}',
            coloraxis_colorbar=dict(title='Absolute Error')
        )
        
        # Save plot
        plot_path = os.path.join(save_dir, f'{model_name}_{prop}_interactive_scatter.html')
        fig.write_html(plot_path)
        plot_paths.append(plot_path)
        
        # Error distribution
        fig = px.histogram(
            df, 
            x='Error',
            marginal='box',
            color_discrete_sequence=['royalblue'],
            opacity=0.7
        )
        
        fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title=f'{model_name} - {prop} Error Distribution',
            xaxis_title=f'{prop} Error (Predicted - True)',
            yaxis_title='Count'
        )
        
        # Save plot
        plot_path = os.path.join(save_dir, f'{model_name}_{prop}_error_dist_interactive.html')
        fig.write_html(plot_path)
        plot_paths.append(plot_path)
    
    return plot_paths

if __name__ == "__main__":
    # Test the visualization functions
    set_plotting_style()
    
    # Create some sample data
    n_samples = 100
    n_properties = 3
    property_names = ['accuracy', 'robustness', 'generalization_gap']
    
    # Generate random targets
    targets = np.random.rand(n_samples, n_properties)
    targets[:, 0] = 0.7 + 0.25 * targets[:, 0]  # accuracy between 0.7 and 0.95
    targets[:, 1] = 0.5 + 0.4 * targets[:, 1]   # robustness between 0.5 and 0.9
    targets[:, 2] = 0.02 + 0.13 * targets[:, 2] # generalization gap between 0.02 and 0.15
    
    # Generate predictions with some error
    predictions = targets + 0.05 * np.random.randn(n_samples, n_properties)
    predictions[:, 0] = np.clip(predictions[:, 0], 0.7, 0.95)
    predictions[:, 1] = np.clip(predictions[:, 1], 0.5, 0.9)
    predictions[:, 2] = np.clip(predictions[:, 2], 0.02, 0.15)
    
    # Create test save directory
    os.makedirs("test_visualizations", exist_ok=True)
    
    # Test prediction vs target plots
    plot_paths = plot_predictions_vs_targets(
        predictions=predictions,
        targets=targets,
        property_names=property_names,
        save_dir="test_visualizations",
        model_name="test_model"
    )
    
    print(f"Generated {len(plot_paths)} prediction vs target plots")
    
    # Test error distribution plots
    error_plots = plot_error_distributions(
        predictions=predictions,
        targets=targets,
        property_names=property_names,
        save_dir="test_visualizations",
        model_name="test_model"
    )
    
    print(f"Generated {len(error_plots)} error distribution plots")
    
    # Test model comparison
    model_results = {
        "Model1": {
            "accuracy": {"mae": 0.02, "rmse": 0.03, "r2": 0.95},
            "robustness": {"mae": 0.03, "rmse": 0.04, "r2": 0.92},
            "generalization_gap": {"mae": 0.01, "rmse": 0.015, "r2": 0.97},
        },
        "Model2": {
            "accuracy": {"mae": 0.03, "rmse": 0.04, "r2": 0.90},
            "robustness": {"mae": 0.025, "rmse": 0.035, "r2": 0.94},
            "generalization_gap": {"mae": 0.015, "rmse": 0.02, "r2": 0.95},
        },
    }
    
    comparison_plot = plot_model_comparison(
        model_results=model_results,
        property_names=property_names,
        metric_name="mae",
        save_dir="test_visualizations"
    )
    
    print(f"Generated model comparison plot: {comparison_plot}")
    
    # Test property correlations
    correlation_plot = plot_property_correlations(
        properties=targets,
        property_names=property_names,
        save_dir="test_visualizations",
        title="Test Property Correlations"
    )
    
    print(f"Generated property correlation plot: {correlation_plot}")
    
    # Test radar chart
    radar_plot = create_radar_chart(
        model_results=model_results,
        property_names=property_names,
        metric_name="r2",
        save_dir="test_visualizations"
    )
    
    print(f"Generated radar chart: {radar_plot}")
    
    # Test interactive plots
    metadata = {
        "architecture": ["resnet18"] * (n_samples // 2) + ["vgg16"] * (n_samples // 2),
        "parameters": np.random.randint(1000000, 50000000, n_samples)
    }
    
    interactive_plots = create_interactive_plots(
        predictions=predictions,
        targets=targets,
        property_names=property_names,
        metadata=metadata,
        save_dir="test_visualizations",
        model_name="test_model"
    )
    
    print(f"Generated {len(interactive_plots)} interactive plots")
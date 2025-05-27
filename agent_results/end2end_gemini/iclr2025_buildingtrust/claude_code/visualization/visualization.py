"""
Visualization Module

This module provides utilities for visualizing results from concept graph experiments.
"""

import os
import math
import logging
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

def visualize_attention_weights(
    attention_weights: List[np.ndarray],
    layer_indices: List[int],
    head_indices: List[int],
    tokens: List[str],
    save_path: Optional[str] = None,
    max_heads_per_plot: int = 4
) -> List[plt.Figure]:
    """
    Visualize attention weights as heatmaps.
    
    Args:
        attention_weights: List of attention weights matrices
        layer_indices: Layer indices corresponding to each matrix
        head_indices: Head indices corresponding to each matrix
        tokens: List of tokens for axis labels
        save_path: Path to save the visualization
        max_heads_per_plot: Maximum number of attention heads to include in a single plot
        
    Returns:
        List of matplotlib figures
    """
    logger.info(f"Visualizing attention weights for {len(attention_weights)} matrices")
    
    figures = []
    
    # Group attention matrices by layer
    layer_to_heads = {}
    for i, (attn_matrix, layer_idx, head_idx) in enumerate(zip(attention_weights, layer_indices, head_indices)):
        if layer_idx not in layer_to_heads:
            layer_to_heads[layer_idx] = []
        
        layer_to_heads[layer_idx].append((head_idx, attn_matrix))
    
    # Sort layers
    sorted_layers = sorted(layer_to_heads.keys())
    
    # For each layer, create plots with max_heads_per_plot heads
    for layer_idx in sorted_layers:
        heads = layer_to_heads[layer_idx]
        heads.sort(key=lambda x: x[0])  # Sort by head index
        
        # Split heads into groups
        head_groups = [heads[i:i+max_heads_per_plot] for i in range(0, len(heads), max_heads_per_plot)]
        
        # Create a plot for each group
        for group_idx, head_group in enumerate(head_groups):
            num_heads = len(head_group)
            fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 5, 5))
            
            if num_heads == 1:
                axes = [axes]
            
            for i, (head_idx, attn_matrix) in enumerate(head_group):
                ax = axes[i]
                
                # Truncate if the matrix is too large
                max_tokens = min(len(tokens), 30)  # Limit to 30 tokens for visibility
                
                # Ensure attn_matrix has the right dimensions
                if attn_matrix.shape[0] > max_tokens or attn_matrix.shape[1] > max_tokens:
                    attn_matrix = attn_matrix[:max_tokens, :max_tokens]
                
                # Update tokens to match matrix size
                display_tokens = tokens[:max_tokens]
                
                # Create heatmap
                sns.heatmap(
                    attn_matrix,
                    ax=ax,
                    cmap='viridis',
                    xticklabels=display_tokens,
                    yticklabels=display_tokens,
                    vmin=0,
                    vmax=attn_matrix.max()
                )
                
                ax.set_title(f"Layer {layer_idx}, Head {head_idx}")
                
                # Rotate x labels for better visibility
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            plt.tight_layout()
            plt.suptitle(f"Attention Weights - Layer {layer_idx}", fontsize=16)
            plt.subplots_adjust(top=0.9)
            
            figures.append(fig)
            
            # Save figure if save_path is provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                file_path = save_path.replace(".png", f"_layer{layer_idx}_group{group_idx}.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved attention visualization to {file_path}")
    
    return figures

def visualize_hidden_states_pca(
    hidden_states: Dict[int, List[np.ndarray]],
    tokens: List[str],
    num_components: int = 2,
    save_path: Optional[str] = None,
    layer_selection: Optional[List[int]] = None
) -> plt.Figure:
    """
    Visualize hidden states using PCA.
    
    Args:
        hidden_states: Dictionary of hidden states per layer
        tokens: List of tokens for labeling
        num_components: Number of PCA components (2 or 3)
        save_path: Path to save the visualization
        layer_selection: Optional list of layers to include
        
    Returns:
        Matplotlib figure
    """
    from sklearn.decomposition import PCA
    
    if num_components not in [2, 3]:
        logger.warning(f"Invalid num_components: {num_components}, using 2")
        num_components = 2
    
    logger.info(f"Visualizing hidden states with {num_components}D PCA")
    
    # Select layers if specified
    if layer_selection is not None:
        selected_states = {layer: hidden_states[layer] for layer in layer_selection if layer in hidden_states}
    else:
        selected_states = hidden_states
    
    # Flatten hidden states
    flat_states = []
    state_labels = []
    token_indices = []
    
    for layer_idx, layer_states in selected_states.items():
        for tok_idx, state in enumerate(layer_states):
            if isinstance(state, np.ndarray):
                flat_states.append(state)
            else:  # Handle PyTorch tensors
                flat_states.append(state.detach().cpu().numpy())
            
            state_labels.append(f"L{layer_idx}")
            token_indices.append(tok_idx)
    
    # Convert to numpy array
    X = np.array(flat_states)
    
    # Apply PCA
    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X)
    
    # Create the figure
    if num_components == 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create scatter plot
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=token_indices,
            cmap='viridis',
            alpha=0.7,
            s=100
        )
        
        # Add legend for layer colors
        unique_layers = sorted(set(state_labels))
        layer_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_layers)))
        
        patches = []
        for i, layer in enumerate(unique_layers):
            patch = mpatches.Patch(color=layer_colors[i], label=layer)
            patches.append(patch)
        
        plt.legend(handles=patches, title="Layers", loc="upper right")
        
        # Add colorbar for token positions
        cbar = plt.colorbar(scatter)
        cbar.set_label("Token Position")
        
        # Add labels
        plt.title("Hidden States PCA Visualization", fontsize=16)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=14)
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=14)
        
    else:  # 3D
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            X_pca[:, 2],
            c=token_indices,
            cmap='viridis',
            alpha=0.7,
            s=100
        )
        
        # Add legend for layer colors
        unique_layers = sorted(set(state_labels))
        layer_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_layers)))
        
        patches = []
        for i, layer in enumerate(unique_layers):
            patch = mpatches.Patch(color=layer_colors[i], label=layer)
            patches.append(patch)
        
        plt.legend(handles=patches, title="Layers", loc="upper right")
        
        # Add colorbar for token positions
        cbar = plt.colorbar(scatter)
        cbar.set_label("Token Position")
        
        # Add labels
        plt.title("Hidden States PCA Visualization (3D)", fontsize=16)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=14)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=14)
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)", fontsize=14)
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved hidden states PCA visualization to {save_path}")
    
    return fig

def visualize_token_importance(
    tokens: List[str],
    importance_scores: List[float],
    highlight_top_n: int = 5,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize token importance scores.
    
    Args:
        tokens: List of tokens
        importance_scores: Importance score for each token
        highlight_top_n: Number of top tokens to highlight
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Visualizing token importance for {len(tokens)} tokens")
    
    # Create a dataframe for easier manipulation
    df = pd.DataFrame({
        'token': tokens,
        'importance': importance_scores
    })
    
    # Sort by importance
    df = df.sort_values('importance', ascending=False)
    
    # Truncate if there are too many tokens
    max_tokens = min(len(df), 50)  # Limit to 50 tokens for visibility
    df = df.head(max_tokens)
    
    # Identify top N tokens
    top_n = min(highlight_top_n, len(df))
    df['is_top'] = [True if i < top_n else False for i in range(len(df))]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar plot
    bars = sns.barplot(
        x='token',
        y='importance',
        data=df,
        hue='is_top',
        palette={True: 'darkorange', False: 'lightblue'},
        ax=ax
    )
    
    # Customize plot
    plt.title(f"Token Importance Scores (Top {top_n} Highlighted)", fontsize=16)
    plt.xlabel("Token", fontsize=14)
    plt.ylabel("Importance Score", fontsize=14)
    
    # Rotate x labels for better visibility
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Hide legend
    ax.get_legend().remove()
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars.patches):
        if i < top_n:
            value = df.iloc[i]['importance']
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha='center',
                va='bottom',
                fontweight='bold'
            )
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved token importance visualization to {save_path}")
    
    return fig

def visualize_metrics_comparison(
    method_metrics: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str],
    higher_is_better: Dict[str, bool],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Visualize performance metrics comparison between different methods.
    
    Args:
        method_metrics: Dictionary mapping method names to their metrics
        metrics_to_plot: List of metric names to include in the visualization
        higher_is_better: Dictionary specifying whether higher is better for each metric
        save_path: Path to save the visualization
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Visualizing metrics comparison for {len(method_metrics)} methods")
    
    # Calculate the number of rows and columns for subplots
    num_metrics = len(metrics_to_plot)
    cols = min(3, num_metrics)
    rows = math.ceil(num_metrics / cols)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten axes if there are multiple rows and columns
    if rows > 1 or cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        if i < len(axes):
            ax = axes[i]
            
            # Extract metric values for each method
            methods = []
            values = []
            
            for method, metrics in method_metrics.items():
                if metric in metrics:
                    methods.append(method)
                    values.append(metrics[metric])
            
            # Create bar plot
            bars = ax.bar(methods, values, color='skyblue')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01 * max(values),
                    f"{height:.3f}",
                    ha='center',
                    va='bottom'
                )
            
            # Set title and labels
            ax.set_title(metric, fontsize=12)
            ax.set_ylabel("Value")
            
            # Rotate x labels if there are many methods
            if len(methods) > 3:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add indicator for better values
            if metric in higher_is_better:
                is_higher_better = higher_is_better[metric]
                best_idx = np.argmax(values) if is_higher_better else np.argmin(values)
                
                bars[best_idx].set_color('green')
                ax.text(
                    best_idx,
                    values[best_idx] + 0.05 * max(values),
                    "Best",
                    ha='center',
                    va='bottom',
                    color='green',
                    fontweight='bold'
                )
    
    # Hide unused subplots
    for i in range(num_metrics, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Performance Metrics Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    # Save figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metrics comparison visualization to {save_path}")
    
    return fig

def visualize_concept_examples(
    concept_data: Dict[str, Dict[str, Any]],
    text_segments: Dict[str, str],
    max_concepts: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize examples of identified concepts with corresponding text segments.
    
    Args:
        concept_data: Dictionary of concept information
        text_segments: Dictionary mapping concept names to text segments
        max_concepts: Maximum number of concepts to include
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Visualizing examples for {len(concept_data)} concepts")
    
    # Limit to max_concepts
    if len(concept_data) > max_concepts:
        # Sort concepts by size and take the largest ones
        sorted_concepts = sorted(
            concept_data.items(),
            key=lambda x: x[1].get('size', 0),
            reverse=True
        )
        selected_concepts = dict(sorted_concepts[:max_concepts])
    else:
        selected_concepts = concept_data
    
    # Calculate figure size based on number of concepts
    num_concepts = len(selected_concepts)
    fig_height = 2 + num_concepts * 0.8
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Hide axes
    ax.axis('off')
    
    # Create a table
    cell_text = []
    cell_colors = []
    
    # Prepare concept colors
    concept_colors = plt.cm.tab20(np.linspace(0, 1, num_concepts))
    
    for i, (concept_name, concept_info) in enumerate(selected_concepts.items()):
        # Get text segment if available
        text = text_segments.get(concept_name, "No text segment available")
        
        # Truncate text if too long
        if len(text) > 100:
            text = text[:97] + "..."
        
        # Add row to table
        cell_text.append([concept_name, text])
        
        # Set colors
        color = tuple(list(concept_colors[i]) + [0.2])  # Add alpha
        cell_colors.append([color, 'white'])
    
    # Create the table
    table = ax.table(
        cellText=cell_text,
        colLabels=["Concept", "Example Text Segment"],
        loc='center',
        cellLoc='left',
        cellColours=cell_colors,
        colWidths=[0.3, 0.7]
    )
    
    # Customize table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Set title
    plt.title("Concept Examples", fontsize=16, pad=20)
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved concept examples visualization to {save_path}")
    
    return fig

def visualize_training_curves(
    metrics: Dict[str, List[float]],
    epochs: List[int],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize training curves for concept probe training.
    
    Args:
        metrics: Dictionary mapping metric names to lists of values
        epochs: List of epoch numbers
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Visualizing training curves for {len(metrics)} metrics")
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Define colors for different metrics
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        if metric_name.lower() == 'loss' or 'loss' in metric_name.lower():
            # Plot loss on the left y-axis
            ax1.plot(
                epochs,
                values,
                color=colors[i],
                marker='o',
                linestyle='-',
                label=metric_name
            )
        else:
            # For other metrics (like accuracy), create a second y-axis
            ax2 = ax1.twinx()
            ax2.plot(
                epochs,
                values,
                color=colors[i],
                marker='s',
                linestyle='--',
                label=metric_name
            )
            ax2.set_ylabel("Accuracy / Other Metrics", color=colors[i])
    
    # Set labels and title
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14)
    plt.title("Training Curves", fontsize=16)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    try:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    except:
        ax1.legend(loc='upper right')
    
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves visualization to {save_path}")
    
    return fig
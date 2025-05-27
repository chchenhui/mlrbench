import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import json
from matplotlib.ticker import FormatStrFormatter
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

logger = logging.getLogger("influence_space")

def set_plot_style() -> None:
    """Set consistent style for all plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

def visualize_embeddings(
    embeddings: np.ndarray,
    labels: List[int],
    method: str = 'tsne',
    perplexity: int = 30,
    n_components: int = 2,
    title: str = "Embedding Visualization",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize high-dimensional embeddings in 2D.
    
    Args:
        embeddings: Numpy array of embeddings
        labels: List of cluster labels or categories
        method: Dimensionality reduction method ('tsne' or 'pca')
        perplexity: Perplexity parameter for t-SNE
        n_components: Number of components for dimensionality reduction
        title: Title for the plot
        save_path: Path to save the figure
    """
    set_plot_style()
    
    logger.info(f"Visualizing embeddings using {method}...")
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    else:  # PCA
        reducer = PCA(n_components=n_components, random_state=42)
    
    # Convert to numpy for safety
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    
    # Check if embeddings are valid
    if not isinstance(embeddings, np.ndarray) or embeddings.size == 0:
        logger.error("Invalid embeddings for visualization")
        return
    
    # Apply dimensionality reduction
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Convert labels to integers if they aren't already
    unique_labels = sorted(set(labels))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    label_indices = [label_to_idx[label] for label in labels]
    
    # Create a colormap with enough colors
    n_colors = len(unique_labels)
    cmap = plt.cm.get_cmap('tab20' if n_colors <= 20 else 'viridis', n_colors)
    
    # Plot points with colors based on labels
    scatter = plt.scatter(
        reduced_embeddings[:, 0], 
        reduced_embeddings[:, 1], 
        c=label_indices, 
        cmap=cmap, 
        alpha=0.7, 
        s=30
    )
    
    # Add colorbar for reference
    if len(unique_labels) <= 20:
        legend1 = plt.legend(
            *scatter.legend_elements(),
            title="Clusters",
            loc="upper right",
            bbox_to_anchor=(1.15, 1)
        )
        plt.gca().add_artist(legend1)
    else:
        plt.colorbar(scatter, label="Cluster")
    
    plt.title(title)
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_histogram(
    data: List[float],
    title: str = "Distribution",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    bins: int = 30,
    save_path: Optional[str] = None
) -> None:
    """
    Plot histogram of data.
    
    Args:
        data: List of values to plot
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        bins: Number of bins for histogram
        save_path: Path to save the figure
    """
    set_plot_style()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_violin_comparison(
    data: Dict[str, List[float]],
    title: str = "Method Comparison",
    xlabel: str = "Method",
    ylabel: str = "Value",
    save_path: Optional[str] = None
) -> None:
    """
    Create violin plot comparing distributions across methods.
    
    Args:
        data: Dictionary mapping method names to lists of values
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        save_path: Path to save the figure
    """
    set_plot_style()
    
    # Convert to dataframe for seaborn
    df_list = []
    for method, values in data.items():
        method_df = pd.DataFrame({
            'Method': [method] * len(values),
            'Value': values
        })
        df_list.append(method_df)
    
    df = pd.concat(df_list, ignore_index=True)
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x="Method", y="Value")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_heatmap(
    matrix: np.ndarray,
    title: str = "Heatmap",
    xlabel: str = "X",
    ylabel: str = "Y",
    cmap: str = "viridis",
    annot: bool = True,
    fmt: str = ".2f",
    save_path: Optional[str] = None
) -> None:
    """
    Create heatmap from matrix data.
    
    Args:
        matrix: 2D numpy array or nested list
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        cmap: Colormap for heatmap
        annot: Whether to annotate cells with values
        fmt: Format string for annotations
        save_path: Path to save the figure
    """
    set_plot_style()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=annot, fmt=fmt, cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_cluster_influence_distribution(
    clusters: List[List[int]],
    influence_scores: Dict[int, float],
    title: str = "Cluster Influence vs Size",
    save_path: Optional[str] = None
) -> None:
    """
    Plot cluster influence scores against cluster sizes.
    
    Args:
        clusters: List of indices for each cluster
        influence_scores: Dictionary mapping cluster indices to influence scores
        title: Title for the plot
        save_path: Path to save the figure
    """
    set_plot_style()
    
    # Prepare data
    cluster_sizes = [len(cluster) for cluster in clusters]
    cluster_influences = [influence_scores.get(i, 0) for i in range(len(clusters))]
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        cluster_sizes, 
        cluster_influences, 
        alpha=0.7,
        s=50,
        c=cluster_influences,
        cmap='coolwarm'
    )
    
    plt.colorbar(scatter, label="Influence Score")
    plt.title(title)
    plt.xlabel("Cluster Size")
    plt.ylabel("Influence Score")
    plt.xscale('log')
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_accuracy_vs_retention(
    retention_ratios: List[float],
    accuracies: Dict[str, List[float]],
    title: str = "Performance vs Data Retention",
    xlabel: str = "Data Retention Ratio",
    ylabel: str = "Accuracy (%)",
    save_path: Optional[str] = None
) -> None:
    """
    Plot performance metric against data retention ratio for different methods.
    
    Args:
        retention_ratios: List of data retention ratios (e.g., [0.1, 0.2, ..., 1.0])
        accuracies: Dictionary mapping method names to lists of accuracy values
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        save_path: Path to save the figure
    """
    set_plot_style()
    
    plt.figure(figsize=(10, 6))
    
    for method, acc_values in accuracies.items():
        plt.plot(retention_ratios, acc_values, marker='o', linewidth=2, label=method)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    
    # Set x-axis to show percentages
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.xticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_ablation_study(
    param_values: Dict[str, List[Any]],
    metrics: Dict[str, Dict[str, List[float]]],
    title: str = "Ablation Study",
    xlabel: str = "Parameter Value",
    ylabel: str = "Metric Value",
    save_path: Optional[str] = None
) -> None:
    """
    Plot ablation study results for different parameters.
    
    Args:
        param_values: Dictionary mapping parameter names to lists of values
        metrics: Dictionary mapping parameter names to dictionaries mapping metric names to lists of values
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        save_path: Path to save the figure
    """
    set_plot_style()
    
    # Create a subplot for each parameter
    n_params = len(param_values)
    fig, axes = plt.subplots(1, n_params, figsize=(15, 5), sharey=True)
    
    # If only one parameter, axes won't be an array
    if n_params == 1:
        axes = [axes]
    
    for i, (param_name, values) in enumerate(param_values.items()):
        ax = axes[i]
        
        # Plot each metric for this parameter
        for metric_name, metric_values in metrics[param_name].items():
            ax.plot(values, metric_values, marker='o', linewidth=2, label=metric_name)
        
        ax.set_title(f"{param_name}")
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.grid(True)
        
        # Add legend to the first subplot
        if i == 0:
            ax.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_demographic_performance(
    method_results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "avg_recall@1",
    title: str = "Performance Across Demographics",
    save_path: Optional[str] = None
) -> None:
    """
    Plot performance metrics across demographic groups for different methods.
    
    Args:
        method_results: Dictionary mapping method names to dictionaries mapping demographic groups to metrics
        metric: Name of the metric to plot
        title: Title for the plot
        save_path: Path to save the figure
    """
    set_plot_style()
    
    # Prepare data for plotting
    methods = list(method_results.keys())
    
    # Get all demographic groups
    all_demographics = set()
    for method_metrics in method_results.values():
        all_demographics.update(method_metrics.keys())
    
    all_demographics = sorted(all_demographics)
    
    # Create dataframe for plotting
    data = []
    for method in methods:
        for demo in all_demographics:
            if demo in method_results[method]:
                value = method_results[method][demo].get(metric, 0)
                data.append({
                    'Method': method,
                    'Demographic': demo,
                    'Value': value
                })
    
    df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(14, 6))
    
    # Use barplot for easier comparison
    ax = sns.barplot(data=df, x='Demographic', y='Value', hue='Method')
    
    plt.title(title)
    plt.xlabel("Demographic Group")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.legend(title="Method")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_results_summary(
    method_metrics: Dict[str, Dict[str, float]],
    efficiency_metrics: Dict[str, Dict[str, float]],
    demographic_gaps: Dict[str, Dict[str, float]],
    output_dir: str = "./"
) -> None:
    """
    Create a comprehensive summary of experimental results.
    
    Args:
        method_metrics: Dictionary mapping method names to performance metrics
        efficiency_metrics: Dictionary mapping method names to efficiency metrics
        demographic_gaps: Dictionary mapping method names to demographic performance gaps
        output_dir: Directory to save the summary
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary dataframe
    summary_data = []
    
    for method in method_metrics.keys():
        row = {
            'Method': method,
            'Recall@1': method_metrics[method].get('avg_recall@1', 0),
            'Recall@5': method_metrics[method].get('avg_recall@5', 0),
            'Recall@10': method_metrics[method].get('avg_recall@10', 0),
            'Data Reduction (%)': efficiency_metrics[method].get('data_reduction_ratio', 0) * 100,
            'Relative Training Time': efficiency_metrics[method].get('normalized_training_time', 0),
        }
        
        # Add demographic gaps
        if method in demographic_gaps:
            for demo, gap in demographic_gaps[method].items():
                row[f'{demo.capitalize()} Gap'] = gap
        
        summary_data.append(row)
    
    # Create dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    summary_df.to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)
    
    # Generate a markdown table
    markdown_table = summary_df.to_markdown(index=False, floatfmt=".2f")
    
    # Save as markdown
    with open(os.path.join(output_dir, "results_summary.md"), "w") as f:
        f.write("# Results Summary\n\n")
        f.write(markdown_table)
        f.write("\n")

def generate_final_figures(
    method_metrics: Dict[str, Dict[str, float]],
    method_histories: Dict[str, Dict[str, List[float]]],
    efficiency_metrics: Dict[str, Dict[str, float]],
    demographic_data: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    output_dir: str = "./"
) -> None:
    """
    Generate final figures for the experimental results.
    
    Args:
        method_metrics: Dictionary mapping method names to performance metrics
        method_histories: Dictionary mapping method names to training histories
        efficiency_metrics: Dictionary mapping method names to efficiency metrics
        demographic_data: Optional dictionary with demographic performance data
        output_dir: Directory to save the figures
    """
    set_plot_style()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot main performance metrics comparison
    metrics = ['avg_recall@1', 'avg_recall@5', 'avg_recall@10']
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(method_metrics))
    width = 0.25
    
    methods = list(method_metrics.keys())
    
    for i, metric in enumerate(metrics):
        values = [method_metrics[method].get(metric, 0) for method in methods]
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Method')
    plt.ylabel('Recall (%)')
    plt.title('Performance Comparison Across Methods')
    plt.xticks(x + width, methods, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "performance_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot efficiency metrics
    reduction_ratios = [efficiency_metrics[method].get('data_reduction_ratio', 0) * 100 for method in methods]
    training_times = [efficiency_metrics[method].get('normalized_training_time', 0) for method in methods]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Data Reduction (%)', color=color)
    ax1.bar(methods, reduction_ratios, color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticklabels(methods, rotation=45)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Normalized Training Time', color=color)
    ax2.plot(methods, training_times, 'o-', color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Efficiency Metrics Across Methods')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "efficiency_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot training curves for all methods
    plt.figure(figsize=(12, 6))
    
    for method, history in method_histories.items():
        if 'val_recall@1' in history:
            plt.plot(history['val_recall@1'], label=f"{method}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Recall@1 (%)')
    plt.title('Training Progress Across Methods')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "training_progress.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Plot demographic gaps if available
    if demographic_data:
        # Extract demographic gaps
        gaps = {}
        for method in methods:
            if method in demographic_data:
                gaps[method] = {}
                demos = demographic_data[method]
                
                # Calculate gaps for each demographic attribute
                attributes = set()
                for demo in demos:
                    attr = demo.split('_')[0]  # e.g., "gender_male" -> "gender"
                    attributes.add(attr)
                
                # Compute gaps for each attribute
                for attr in attributes:
                    attr_demos = {k: v for k, v in demos.items() if k.startswith(f"{attr}_")}
                    if attr_demos:
                        values = [d.get('avg_recall@1', 0) for d in attr_demos.values()]
                        gaps[method][attr] = max(values) - min(values)
        
        # Plot gaps
        fig, axes = plt.subplots(1, len(attributes), figsize=(15, 6), sharey=True)
        
        # If only one attribute, axes won't be an array
        if len(attributes) == 1:
            axes = [axes]
        
        for i, attr in enumerate(sorted(attributes)):
            ax = axes[i]
            
            values = [gaps[method].get(attr, 0) for method in methods]
            ax.bar(methods, values)
            
            ax.set_title(f"{attr.capitalize()} Gap")
            ax.set_xlabel('Method')
            if i == 0:
                ax.set_ylabel('Performance Gap (%)')
            ax.tick_params(axis='x', rotation=45)
            
            # Add values on top of bars
            for j, v in enumerate(values):
                ax.text(j, v + 0.5, f"{v:.1f}", ha='center')
        
        plt.suptitle('Demographic Performance Gaps', fontsize=16)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "demographic_gaps.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Plot performance vs efficiency trade-off
    plt.figure(figsize=(10, 8))
    
    x = [efficiency_metrics[method].get('data_reduction_ratio', 0) * 100 for method in methods]
    y = [method_metrics[method].get('avg_recall@1', 0) for method in methods]
    
    plt.scatter(x, y, s=100)
    
    # Add method labels to points
    for i, method in enumerate(methods):
        plt.annotate(method, (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Data Reduction (%)')
    plt.ylabel('Recall@1 (%)')
    plt.title('Performance vs. Data Efficiency')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "performance_efficiency_tradeoff.png"), dpi=300, bbox_inches='tight')
    plt.close()
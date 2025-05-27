"""
Visualization utilities for the Cluster-Driven Certified Unlearning experiment.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE


def set_plotting_style():
    """
    Set matplotlib and seaborn styling for consistent visualizations.
    """
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # Set matplotlib parameters
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18


def plot_perplexity_comparison(results, save_path=None):
    """
    Plot perplexity comparison between the original model and unlearned models.
    
    Args:
        results (dict): Dictionary of results
        save_path (str, optional): Path to save the plot
        
    Returns:
        fig: Matplotlib figure
    """
    set_plotting_style()
    
    # Extract perplexity data
    methods = list(results.keys())
    perplexities = [results[method]['perplexity'] for method in methods]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(methods, perplexities, width=0.6)
    
    # Add labels and title
    ax.set_xlabel('Unlearning Method')
    ax.set_ylabel('Perplexity (lower is better)')
    ax.set_title('Perplexity Comparison Across Unlearning Methods')
    
    # Add data labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Customize appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Highlight our method if present
    if 'Cluster-Driven' in methods:
        idx = methods.index('Cluster-Driven')
        bars[idx].set_color('darkred')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_metrics_radar(results, metrics=['KFR', 'KRR', 'Perplexity_Norm', 'Compute_Cost_Norm'], save_path=None):
    """
    Create a radar plot comparing multiple metrics across methods.
    
    Args:
        results (dict): Dictionary of results, with normalized metrics
        metrics (list): List of metrics to include
        save_path (str, optional): Path to save the plot
        
    Returns:
        fig: Matplotlib figure
    """
    set_plotting_style()
    
    # Extract methods and metrics
    methods = list(results.keys())
    
    # Create the radar plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # Number of variables
    N = len(metrics)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], metrics, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot each method
    cmap = plt.get_cmap('tab10')
    for i, method in enumerate(methods):
        values = [results[method].get(metric, 0) for metric in metrics]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=method, color=cmap(i))
        ax.fill(angles, values, color=cmap(i), alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("Comparison of Unlearning Methods", size=20, pad=20)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_knowledge_retention_vs_forgetting(results, save_path=None):
    """
    Create a scatter plot comparing Knowledge Retention Rate (KRR) vs Knowledge Forgetting Rate (KFR).
    
    Args:
        results (dict): Dictionary of results
        save_path (str, optional): Path to save the plot
        
    Returns:
        fig: Matplotlib figure
    """
    set_plotting_style()
    
    # Extract data
    methods = list(results.keys())
    krr_values = [results[method]['KRR'] for method in methods]
    kfr_values = [results[method]['KFR'] for method in methods]
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points
    cmap = plt.get_cmap('tab10')
    for i, method in enumerate(methods):
        ax.scatter(kfr_values[i], krr_values[i], s=150, label=method, color=cmap(i))
        
        # Annotate points
        ax.annotate(method, (kfr_values[i], krr_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=12)
    
    # Add reference lines
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95% KRR (Target)')
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, label='90% KRR')
    
    # Add labels and title
    ax.set_xlabel('Knowledge Forgetting Rate (KFR) - higher is better')
    ax.set_ylabel('Knowledge Retention Rate (KRR) - higher is better')
    ax.set_title('Trade-off Between Knowledge Retention and Forgetting')
    
    # Set axis limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.75, 1.05)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Pareto front: highlight non-dominated methods
    pareto_optimal = []
    for i, method in enumerate(methods):
        dominated = False
        for j, other_method in enumerate(methods):
            if i != j:
                # Check if other_method dominates method
                if kfr_values[j] >= kfr_values[i] and krr_values[j] >= krr_values[i] and \
                   (kfr_values[j] > kfr_values[i] or krr_values[j] > krr_values[i]):
                    dominated = True
                    break
        if not dominated:
            pareto_optimal.append(i)
    
    # Highlight Pareto optimal points
    for i in pareto_optimal:
        ax.scatter(kfr_values[i], krr_values[i], s=250, facecolors='none', 
                   edgecolors='red', linewidth=2, label='_nolegend_')
    
    # Add legend
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_computational_efficiency(results, save_path=None):
    """
    Create a plot comparing computational efficiency across methods.
    
    Args:
        results (dict): Dictionary of results
        save_path (str, optional): Path to save the plot
        
    Returns:
        fig: Matplotlib figure
    """
    set_plotting_style()
    
    # Extract data
    methods = list(results.keys())
    compute_times = [results[method]['compute_time'] / 60 for method in methods]  # Convert to minutes
    kfr_values = [results[method]['KFR'] for method in methods]
    
    # Sort by compute time
    sorted_indices = np.argsort(compute_times)
    methods = [methods[i] for i in sorted_indices]
    compute_times = [compute_times[i] for i in sorted_indices]
    kfr_values = [kfr_values[i] for i in sorted_indices]
    
    # Create bar plot with compute times
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Bar plot for compute times
    bars = ax1.bar(methods, compute_times, width=0.6, color='skyblue', label='Compute Time (minutes)')
    
    # Add labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Set y-axis label for compute times
    ax1.set_xlabel('Unlearning Method')
    ax1.set_ylabel('Compute Time (minutes)')
    
    # Create second y-axis for KFR
    ax2 = ax1.twinx()
    
    # Line plot for KFR
    ax2.plot(methods, kfr_values, 'ro-', linewidth=2, label='KFR')
    
    # Add KFR values as text
    for i, kfr in enumerate(kfr_values):
        ax2.text(i, kfr + 0.02, f'{kfr:.2f}', ha='center', va='bottom', color='darkred')
    
    # Set y-axis label for KFR
    ax2.set_ylabel('Knowledge Forgetting Rate (KFR)')
    ax2.set_ylim(0, 1.1)
    
    # Add title
    plt.title('Computational Efficiency vs. Knowledge Forgetting Rate')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Customize appearance
    ax1.spines['top'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Highlight our method if present
    if 'Cluster-Driven' in methods:
        idx = methods.index('Cluster-Driven')
        bars[idx].set_color('darkblue')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cluster_visualization(activations, clusters, save_path=None):
    """
    Create a t-SNE visualization of the clusters.
    
    Args:
        activations (np.ndarray): Hidden-layer activations
        clusters (np.ndarray): Cluster assignments
        save_path (str, optional): Path to save the plot
        
    Returns:
        fig: Matplotlib figure
    """
    set_plotting_style()
    
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    activations_2d = tsne.fit_transform(activations)
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot points colored by cluster
    cmap = plt.get_cmap('tab10')
    cluster_ids = np.unique(clusters)
    
    for i, cluster_id in enumerate(cluster_ids):
        mask = clusters == cluster_id
        ax.scatter(activations_2d[mask, 0], activations_2d[mask, 1], 
                   s=50, color=cmap(i % 10), label=f'Cluster {cluster_id}')
    
    # Add labels and title
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE Visualization of Model Representation Clusters')
    
    # Add legend
    if len(cluster_ids) <= 20:  # Only show legend if not too many clusters
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_sequential_unlearning_performance(sequential_results, metrics=['KFR', 'KRR'], save_path=None):
    """
    Plot performance metrics over sequential unlearning requests.
    
    Args:
        sequential_results (list): List of results for each unlearning request
        metrics (list): List of metrics to plot
        save_path (str, optional): Path to save the plot
        
    Returns:
        fig: Matplotlib figure
    """
    set_plotting_style()
    
    # Extract data
    num_requests = len(sequential_results)
    request_ids = list(range(1, num_requests + 1))
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot each metric
    cmap = plt.get_cmap('tab10')
    for i, metric in enumerate(metrics):
        metric_values = [result[metric] for result in sequential_results]
        
        # Plot line
        ax.plot(request_ids, metric_values, 'o-', linewidth=2, 
                label=metric, color=cmap(i))
        
        # Add data points
        for j, value in enumerate(metric_values):
            ax.text(request_ids[j], value + 0.02, f'{value:.2f}', 
                    ha='center', va='bottom', color=cmap(i))
    
    # Add labels and title
    ax.set_xlabel('Sequential Unlearning Request')
    ax.set_ylabel('Metric Value')
    ax.set_title('Performance Over Sequential Unlearning Requests')
    
    # Set x-axis ticks
    ax.set_xticks(request_ids)
    
    # Customize appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='both', linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_deletion_set_size_impact(size_results, methods, metrics=['KFR', 'KRR', 'compute_time'], save_path=None):
    """
    Plot the impact of deletion set size on various metrics.
    
    Args:
        size_results (dict): Dictionary mapping deletion set sizes to results
        methods (list): List of unlearning methods to include
        metrics (list): List of metrics to plot
        save_path (str, optional): Path to save the plot
        
    Returns:
        figs: List of Matplotlib figures
    """
    set_plotting_style()
    
    # Extract set sizes
    set_sizes = sorted(list(size_results.keys()))
    
    # Create a figure for each metric
    figs = []
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot each method
        cmap = plt.get_cmap('tab10')
        for i, method in enumerate(methods):
            metric_values = [size_results[size][method][metric] for size in set_sizes]
            
            # For compute time, convert to appropriate units
            if metric == 'compute_time':
                metric_values = [v / 60 for v in metric_values]  # Convert to minutes
                
            # Plot line
            ax.plot(set_sizes, metric_values, 'o-', linewidth=2, 
                    label=method, color=cmap(i))
        
        # Add labels and title
        ax.set_xlabel('Deletion Set Size')
        
        # Set appropriate y-label based on metric
        if metric == 'compute_time':
            ax.set_ylabel('Compute Time (minutes)')
        elif metric == 'KFR':
            ax.set_ylabel('Knowledge Forgetting Rate')
        elif metric == 'KRR':
            ax.set_ylabel('Knowledge Retention Rate')
        elif metric == 'perplexity':
            ax.set_ylabel('Perplexity')
        else:
            ax.set_ylabel(metric)
            
        ax.set_title(f'Impact of Deletion Set Size on {metric}')
        
        # Customize appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='both', linestyle='--', alpha=0.7)
        
        # Set x-axis to log scale if range is large
        if max(set_sizes) / min(set_sizes) > 10:
            ax.set_xscale('log')
            
        # Add legend
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            metric_save_path = save_path.replace('.png', f'_{metric}.png')
            plt.savefig(metric_save_path, dpi=300, bbox_inches='tight')
        
        figs.append(fig)
    
    return figs


def create_summary_dashboard(results, save_dir):
    """
    Create a comprehensive dashboard of visualizations for the unlearning experiments.
    
    Args:
        results (dict): Dictionary containing all experimental results
        save_dir (str): Directory to save visualizations
        
    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Perplexity comparison
    plot_perplexity_comparison(
        results['method_comparison'], 
        save_path=os.path.join(save_dir, 'perplexity_comparison.png')
    )
    
    # 2. Knowledge retention vs forgetting
    plot_knowledge_retention_vs_forgetting(
        results['method_comparison'],
        save_path=os.path.join(save_dir, 'knowledge_retention_vs_forgetting.png')
    )
    
    # 3. Computational efficiency
    plot_computational_efficiency(
        results['method_comparison'],
        save_path=os.path.join(save_dir, 'computational_efficiency.png')
    )
    
    # 4. Radar plot of normalized metrics
    plot_metrics_radar(
        results['normalized_metrics'],
        save_path=os.path.join(save_dir, 'metrics_radar.png')
    )
    
    # 5. Cluster visualization if available
    if 'cluster_activations' in results and 'cluster_assignments' in results:
        plot_cluster_visualization(
            results['cluster_activations'],
            results['cluster_assignments'],
            save_path=os.path.join(save_dir, 'cluster_visualization.png')
        )
    
    # 6. Sequential unlearning performance if available
    if 'sequential_results' in results:
        plot_sequential_unlearning_performance(
            results['sequential_results'],
            save_path=os.path.join(save_dir, 'sequential_unlearning.png')
        )
    
    # 7. Deletion set size impact if available
    if 'deletion_size_impact' in results:
        plot_deletion_set_size_impact(
            results['deletion_size_impact'],
            results['methods'],
            save_path=os.path.join(save_dir, 'deletion_size_impact.png')
        )
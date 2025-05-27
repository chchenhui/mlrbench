"""
Utility functions for the Reasoning Uncertainty Networks (RUNs) experiment.
"""
import os
import json
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import random
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logger.info(f"Random seed set to {seed}")

def setup_visualization_style() -> None:
    """Set up matplotlib visualization style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

def create_custom_colormap() -> LinearSegmentedColormap:
    """
    Create a custom colormap for uncertainty visualization.
    
    Returns:
        Custom colormap
    """
    # Define colors for low, medium, and high uncertainty
    colors = [(0.0, 'green'), (0.5, 'yellow'), (1.0, 'red')]
    
    # Create colormap
    cmap_name = 'uncertainty_cmap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors)
    
    return cm

def visualize_reasoning_graph(graph: nx.DiGraph, 
                              output_path: Optional[str] = None,
                              show_uncertainty: bool = True,
                              title: str = "Reasoning Graph") -> None:
    """
    Visualize a reasoning graph with uncertainty values.
    
    Args:
        graph: NetworkX directed graph
        output_path: Optional path to save the visualization
        show_uncertainty: Whether to color nodes by uncertainty
        title: Plot title
    """
    plt.figure(figsize=(14, 10))
    
    # Create position layout
    pos = nx.spring_layout(graph, seed=42, k=0.5)
    
    # Prepare node colors and sizes based on uncertainty
    if show_uncertainty and all('mean' in graph.nodes[n] for n in graph.nodes()):
        # Color by confidence (mean of Beta distribution)
        node_colors = [graph.nodes[n]['mean'] for n in graph.nodes()]
        
        # Size by variance
        node_sizes = [1000 * (0.5 + graph.nodes[n].get('variance', 0.1)) for n in graph.nodes()]
        
        # Use custom colormap
        cmap = create_custom_colormap()
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            graph, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=cmap,
            vmin=0, vmax=1
        )
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, label="Confidence")
    else:
        # Default visualization without uncertainty
        nx.draw_networkx_nodes(
            graph, pos,
            node_size=800,
            node_color="lightblue"
        )
    
    # Draw edges
    nx.draw_networkx_edges(
        graph, pos,
        width=1.5,
        arrowsize=20,
        edge_color="gray",
        alpha=0.7
    )
    
    # Create node labels
    labels = {}
    for node in graph.nodes():
        label = f"Step {graph.nodes[node].get('step_num', node+1)}"
        if 'is_hallucination' in graph.nodes[node] and graph.nodes[node]['is_hallucination']:
            label += " (H)"
        labels[node] = label
    
    # Draw labels
    nx.draw_networkx_labels(
        graph, pos,
        labels=labels,
        font_size=10,
        font_weight="bold"
    )
    
    plt.title(title)
    plt.axis("off")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved reasoning graph visualization to {output_path}")
    
    plt.close()

def plot_beta_distribution(alpha: float, beta: float, 
                           ax: Optional[plt.Axes] = None,
                           color: str = "blue",
                           label: Optional[str] = None) -> plt.Axes:
    """
    Plot a Beta distribution.
    
    Args:
        alpha: Alpha parameter for Beta distribution
        beta: Beta parameter for Beta distribution
        ax: Optional matplotlib axes
        color: Line color
        label: Optional line label
        
    Returns:
        Matplotlib axes
    """
    from scipy import stats
    
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    # Create x values
    x = np.linspace(0, 1, 1000)
    
    # Compute Beta PDF
    y = stats.beta.pdf(x, alpha, beta)
    
    # Plot
    ax.plot(x, y, color=color, label=label)
    
    # Add mean line
    mean = alpha / (alpha + beta)
    y_max = max(y)
    ax.axvline(mean, color=color, linestyle="--", alpha=0.5)
    ax.text(mean + 0.01, y_max * 0.9, f"{mean:.2f}", color=color)
    
    return ax

def visualize_uncertainty_distributions(nodes_data: List[Dict],
                                         output_path: Optional[str] = None,
                                         title: str = "Uncertainty Distributions") -> None:
    """
    Visualize uncertainty distributions for nodes in a reasoning graph.
    
    Args:
        nodes_data: List of node data dictionaries
        output_path: Optional path to save the visualization
        title: Plot title
    """
    num_nodes = len(nodes_data)
    if num_nodes == 0:
        return
    
    # Setup figure
    fig, axes = plt.subplots(nrows=min(num_nodes, 5), ncols=1, figsize=(10, 2*min(num_nodes, 5)))
    
    # Handle single node case
    if num_nodes == 1:
        axes = [axes]
    
    # Plot distributions for each node (limit to 5 for readability)
    for i, node in enumerate(nodes_data[:5]):
        ax = axes[i]
        
        # Plot initial distribution
        alpha_0 = node.get("alpha_0", node.get("alpha", 1))
        beta_0 = node.get("beta_0", node.get("beta", 1))
        plot_beta_distribution(alpha_0, beta_0, ax, color="blue", label="Initial")
        
        # Plot updated distribution (if available)
        if "alpha" in node and "beta" in node and ("alpha_0" in node or "beta_0" in node):
            alpha = node["alpha"]
            beta = node["beta"]
            plot_beta_distribution(alpha, beta, ax, color="red", label="After propagation")
        
        # Add node info
        ax.set_title(f"Step {node.get('step_num', i+1)}: {node.get('assertion', '')[:50]}...")
        
        # Add legend and labels
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Density")
        ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved uncertainty distributions to {output_path}")
    
    plt.close()

def create_radar_chart(metrics_dict: Dict[str, Dict[str, float]], 
                        metrics_to_plot: List[str],
                        output_path: Optional[str] = None,
                        title: str = "Model Comparison") -> None:
    """
    Create a radar chart comparing multiple models across metrics.
    
    Args:
        metrics_dict: Dictionary of model metrics
        metrics_to_plot: List of metrics to include in the radar chart
        output_path: Optional path to save the visualization
        title: Plot title
    """
    # Number of metrics and models
    num_metrics = len(metrics_to_plot)
    if num_metrics < 3:
        logger.warning("Need at least 3 metrics for a radar chart")
        return
    
    models = list(metrics_dict.keys())
    num_models = len(models)
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Define colors for models
    colors = plt.cm.tab10(np.linspace(0, 1, num_models))
    
    # Plot each model
    for i, model in enumerate(models):
        # Get metrics for this model
        model_metrics = metrics_dict[model]
        
        # Extract values for the specified metrics
        values = [model_metrics.get(metric, 0) for metric in metrics_to_plot]
        values += values[:1]  # Close the loop
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=model)
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    
    # Set y limits
    ax.set_ylim(0, 1)
    
    # Add title and legend
    plt.title(title, size=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved radar chart to {output_path}")
    
    plt.close()

def create_hallucination_types_chart(hallucination_data: Dict[str, Dict[str, int]],
                                     output_path: Optional[str] = None,
                                     title: str = "Hallucination Types Detected") -> None:
    """
    Create a chart showing types of hallucinations detected by each model.
    
    Args:
        hallucination_data: Dictionary mapping models to hallucination type counts
        output_path: Optional path to save the visualization
        title: Plot title
    """
    # Set up figure
    plt.figure(figsize=(12, 8))
    
    # Convert data to DataFrame for easier plotting
    models = []
    factual_counts = []
    logical_counts = []
    numerical_counts = []
    
    for model, counts in hallucination_data.items():
        models.append(model)
        factual_counts.append(counts.get("factual", 0))
        logical_counts.append(counts.get("logical", 0))
        numerical_counts.append(counts.get("numerical", 0))
    
    # Create DataFrame
    df = pd.DataFrame({
        "Factual": factual_counts,
        "Logical": logical_counts,
        "Numerical": numerical_counts
    }, index=models)
    
    # Create stacked bar chart
    ax = df.plot(kind="bar", stacked=True, figsize=(12, 8), colormap="viridis")
    
    # Add numbers on bars
    for container in ax.containers:
        ax.bar_label(container, label_type='center')
    
    plt.title(title, size=15)
    plt.xlabel("Model")
    plt.ylabel("Count")
    plt.legend(title="Hallucination Type")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved hallucination types chart to {output_path}")
    
    plt.close()

def create_confidence_distribution_plot(confidence_values: Dict[str, List[float]],
                                        output_path: Optional[str] = None,
                                        title: str = "Confidence Score Distribution") -> None:
    """
    Create a violin plot showing the distribution of confidence scores across models.
    
    Args:
        confidence_values: Dictionary mapping models to lists of confidence scores
        output_path: Optional path to save the visualization
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Prepare data for violin plot
    data = []
    labels = []
    
    for model, values in confidence_values.items():
        data.append(values)
        labels.append(model)
    
    # Create violin plot
    parts = plt.violinplot(data, showmeans=True, showmedians=True)
    
    # Set colors for each violin
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    
    # Set labels
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel("Confidence Score")
    plt.title(title)
    
    # Add grid
    plt.grid(True, axis="y", alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confidence distribution plot to {output_path}")
    
    plt.close()

def create_runtime_comparison_plot(runtime_data: Dict[str, float],
                                   output_path: Optional[str] = None,
                                   title: str = "Runtime Comparison") -> None:
    """
    Create a bar chart comparing runtime across models.
    
    Args:
        runtime_data: Dictionary mapping models to runtime in seconds
        output_path: Optional path to save the visualization
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Sort models by runtime (ascending)
    sorted_models = sorted(runtime_data.items(), key=lambda x: x[1])
    models, runtimes = zip(*sorted_models)
    
    # Create bar chart
    bars = plt.bar(models, runtimes)
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01 * max(runtimes),
            f'{height:.2f}s',
            ha='center', va='bottom',
            fontsize=10
        )
    
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved runtime comparison plot to {output_path}")
    
    plt.close()

def generate_synthetic_response(question: str, context: str = "", 
                               hallucination_type: Optional[str] = None) -> str:
    """
    Generate a synthetic response for testing purposes.
    
    Args:
        question: Question to answer
        context: Context information
        hallucination_type: Optional type of hallucination to include
            ("factual", "logical", "numerical", or None for no hallucination)
            
    Returns:
        Generated response
    """
    # Basic response templates
    templates = [
        "Based on the provided context, {answer}",
        "The answer is {answer}",
        "According to the information, {answer}",
        "From what we know, {answer}"
    ]
    
    # Generate a basic answer from the context
    words = context.split()
    if len(words) < 10:
        answer = context
    else:
        # Extract a random segment from the context
        start = random.randint(0, max(0, len(words) - 10))
        end = min(start + random.randint(5, 10), len(words))
        answer = " ".join(words[start:end])
    
    # Add hallucination if specified
    if hallucination_type == "factual":
        factual_hallucinations = [
            "studies at Harvard University have conclusively shown that",
            "according to a comprehensive meta-analysis published in Nature,",
            "recent scientific discoveries confirm that",
            "experts unanimously agree that",
            "historical evidence conclusively proves that"
        ]
        
        incorrect_facts = [
            "water boils at 50 degrees Celsius at standard pressure.",
            "humans can survive without oxygen for up to 30 minutes.",
            "the Earth completes a rotation around the Sun every 30 days.",
            "the human heart has five chambers.",
            "consuming vitamin C prevents all viral infections."
        ]
        
        hallucination = f" {random.choice(factual_hallucinations)} {random.choice(incorrect_facts)}"
        answer += hallucination
    
    elif hallucination_type == "logical":
        logical_contradictions = [
            "this clearly indicates X, but X is definitely not possible in this context.",
            "while all evidence points to Y, we can conclusively state that Y is not the case.",
            "the data suggests both A and not-A simultaneously.",
            "if we assume P, then Q follows, but we know Q and not-P are both true.",
            "the symptoms suggest the condition but definitely rule out the condition at the same time."
        ]
        
        answer += f" However, {random.choice(logical_contradictions)}"
    
    elif hallucination_type == "numerical":
        numerical_hallucinations = [
            "the probability is exactly 150%",
            "a 200% reduction in symptoms was observed",
            "adding 25 and 30 gives us 65",
            "when multiplying 7 by 8, we get 48",
            "dividing 100 by 4 results in 20"
        ]
        
        answer += f" Notably, {random.choice(numerical_hallucinations)}."
    
    # Format the response using a template
    template = random.choice(templates)
    response = template.format(answer=answer)
    
    return response

# Example usage
if __name__ == "__main__":
    # Set up visualization style
    setup_visualization_style()
    
    # Create a test graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_node(0, assertion="The Earth orbits the Sun", step_num=1, 
              alpha=5, beta=1, mean=0.83, variance=0.02)
    G.add_node(1, assertion="The Earth completes one orbit every 365.25 days", step_num=2,
              alpha=4, beta=2, mean=0.67, variance=0.03)
    G.add_node(2, assertion="The Earth's orbit is elliptical", step_num=3,
              alpha=3, beta=3, mean=0.5, variance=0.04)
    G.add_node(3, assertion="The Earth is closest to the Sun in January", step_num=4,
              alpha=2, beta=4, mean=0.33, variance=0.04, is_hallucination=True)
    
    # Add edges
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 3)
    
    # Test visualization functions
    visualize_reasoning_graph(G, "test_graph.png", show_uncertainty=True)
    
    # Test uncertainty distribution visualization
    nodes_data = [dict(G.nodes[i]) for i in range(4)]
    visualize_uncertainty_distributions(nodes_data, "test_distributions.png")
    
    # Test radar chart
    metrics = {
        "RUNs": {"precision": 0.85, "recall": 0.78, "f1": 0.81, "auroc": 0.92, "auprc": 0.88},
        "SelfCheckGPT": {"precision": 0.75, "recall": 0.82, "f1": 0.78, "auroc": 0.85, "auprc": 0.80},
        "HuDEx": {"precision": 0.82, "recall": 0.70, "f1": 0.76, "auroc": 0.88, "auprc": 0.82}
    }
    create_radar_chart(metrics, ["precision", "recall", "f1", "auroc", "auprc"], "test_radar.png")
    
    # Test hallucination types chart
    hallucination_data = {
        "RUNs": {"factual": 25, "logical": 15, "numerical": 10},
        "SelfCheckGPT": {"factual": 20, "logical": 10, "numerical": 5},
        "HuDEx": {"factual": 22, "logical": 12, "numerical": 8}
    }
    create_hallucination_types_chart(hallucination_data, "test_hallucination_types.png")
    
    # Test synthetic response generation
    context = "The Earth is the third planet from the Sun and orbits at an average distance of 149.6 million km."
    print(generate_synthetic_response("What is Earth?", context))
    print(generate_synthetic_response("What is Earth?", context, "factual"))
    print(generate_synthetic_response("What is Earth?", context, "logical"))
    print(generate_synthetic_response("What is Earth?", context, "numerical"))
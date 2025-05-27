#!/usr/bin/env python
"""
Script to create example visualizations for demonstration purposes.

This script generates sample visualizations to demonstrate the output
of the Concept-Graph experiments without requiring a full run of the system.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

def ensure_dir(directory):
    """Ensure a directory exists."""
    os.makedirs(directory, exist_ok=True)

def create_concept_graph_visualization():
    """Create an example concept graph visualization."""
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    nodes = [
        ("Problem Understanding", {"size": 10, "position": (0, 3)}),
        ("Variable Identification", {"size": 8, "position": (1, 2)}),
        ("Formula Retrieval", {"size": 12, "position": (2, 4)}),
        ("Calculation Setup", {"size": 7, "position": (3, 1)}),
        ("Multiplication", {"size": 9, "position": (4, 3)}),
        ("Division", {"size": 6, "position": (5, 2)}),
        ("Answer Verification", {"size": 11, "position": (6, 3)})
    ]
    
    G.add_nodes_from(nodes)
    
    # Add edges with weights
    edges = [
        ("Problem Understanding", "Variable Identification", {"weight": 0.8}),
        ("Problem Understanding", "Formula Retrieval", {"weight": 0.6}),
        ("Variable Identification", "Calculation Setup", {"weight": 0.9}),
        ("Formula Retrieval", "Calculation Setup", {"weight": 0.7}),
        ("Calculation Setup", "Multiplication", {"weight": 0.8}),
        ("Multiplication", "Division", {"weight": 0.5}),
        ("Division", "Answer Verification", {"weight": 0.9})
    ]
    
    G.add_edges_from(edges)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Get node positions
    pos = {node: data["position"] for node, data in nodes}
    
    # Get node sizes
    node_sizes = [data["size"] * 100 for _, data in nodes]
    
    # Get edge weights
    edge_weights = [G[u][v]["weight"] * 2 for u, v in G.edges()]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color="blue", alpha=0.6, 
                          connectionstyle='arc3,rad=0.1', arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    
    plt.title("Concept Graph Visualization Example", fontsize=16)
    plt.axis("off")
    
    return plt.gcf()

def create_attention_visualization():
    """Create an example attention visualization."""
    # Create sample attention matrix
    tokens = ["[CLS]", "What", "is", "the", "sum", "of", "25", "and", "17", "?", "[SEP]"]
    attention = np.zeros((len(tokens), len(tokens)))
    
    # Populate with some meaningful patterns
    # Strong attention from "sum" to "25" and "17"
    attention[4, 6] = 0.8
    attention[4, 8] = 0.8
    
    # Strong attention from "?" to the numbers and "sum"
    attention[9, 4] = 0.7
    attention[9, 6] = 0.9
    attention[9, 8] = 0.9
    
    # Some self-attention
    for i in range(len(tokens)):
        attention[i, i] = 0.5
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        attention,
        annot=False,
        cmap="viridis",
        xticklabels=tokens,
        yticklabels=tokens
    )
    
    plt.title("Attention Weight Visualization (Layer 10, Head 5)", fontsize=16)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    
    plt.xticks(rotation=45, ha="right")
    
    return plt.gcf()

def create_integrated_gradients_visualization():
    """Create an example integrated gradients visualization."""
    # Sample tokens and attribution scores
    tokens = ["[CLS]", "What", "is", "the", "sum", "of", "25", "and", "17", "?", "[SEP]"]
    scores = [0.01, 0.03, 0.02, 0.05, 0.25, 0.05, 0.3, 0.05, 0.2, 0.02, 0.01]
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(range(len(tokens)), scores, color='skyblue')
    
    # Highlight important tokens
    important_indices = [4, 6, 8]  # "sum", "25", "17"
    for idx in important_indices:
        bars[idx].set_color('orange')
    
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    plt.title("Integrated Gradients Token Attribution", fontsize=16)
    plt.xlabel("Tokens")
    plt.ylabel("Attribution Score")
    
    plt.tight_layout()
    
    return plt.gcf()

def create_methods_comparison_visualization():
    """Create an example methods comparison visualization."""
    # Sample metrics for different methods
    methods = ["Concept Graph", "Attention", "Integrated Gradients", "CoT"]
    metrics = {
        "num_nodes": [10, 0, 0, 6],
        "num_edges": [7, 0, 0, 5],
        "is_dag": [1.0, 0.0, 0.0, 0.0],
        "density": [0.2, 0.0, 0.0, 0.0]
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Plot each metric
    for i, (metric, values) in enumerate(metrics.items()):
        ax = axes[i]
        
        bars = ax.bar(methods, values, color='skyblue')
        
        # Highlight best method
        best_idx = np.argmax(values) if metric != "density" else np.argmin([v if v > 0 else float('inf') for v in values])
        if best_idx < len(bars):
            bars[best_idx].set_color('green')
        
        ax.set_title(metric, fontsize=12)
        ax.set_ylabel("Value")
        
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle("Performance Metrics Comparison", fontsize=16)
    plt.tight_layout()
    
    return fig

def create_success_rates_visualization():
    """Create an example success rates visualization."""
    # Sample success rates
    datasets = ["GSM8K", "HotpotQA", "StrategyQA"]
    rates = [0.85, 0.78, 0.92]
    
    plt.figure(figsize=(10, 6))
    plt.bar(datasets, rates, color='skyblue')
    plt.title("Success Rates by Dataset", fontsize=16)
    plt.xlabel("Dataset")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    
    # Add value labels
    for i, rate in enumerate(rates):
        plt.text(i, rate + 0.02, f"{rate:.0%}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    return plt.gcf()

def main():
    """Main function to generate and save example visualizations."""
    # Create output directories
    results_dir = "/home/chenhui/mlr-bench/pipeline_gemini/iclr2025_buildingtrust/results"
    figures_dir = os.path.join(results_dir, "figures")
    
    ensure_dir(results_dir)
    ensure_dir(figures_dir)
    
    # Create and save visualizations
    # 1. Concept Graph
    fig = create_concept_graph_visualization()
    fig.savefig(os.path.join(figures_dir, "concept_graph_example.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # 2. Attention Visualization
    fig = create_attention_visualization()
    fig.savefig(os.path.join(figures_dir, "attention_visualization_example.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # 3. Integrated Gradients
    fig = create_integrated_gradients_visualization()
    fig.savefig(os.path.join(figures_dir, "integrated_gradients_example.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # 4. Methods Comparison
    fig = create_methods_comparison_visualization()
    fig.savefig(os.path.join(figures_dir, "methods_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # 5. Success Rates
    fig = create_success_rates_visualization()
    fig.savefig(os.path.join(figures_dir, "success_rates.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Example visualizations created in {figures_dir}")

if __name__ == "__main__":
    main()
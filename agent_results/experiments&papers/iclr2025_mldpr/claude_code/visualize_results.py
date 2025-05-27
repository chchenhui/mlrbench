#!/usr/bin/env python3
"""
Visualize results from Benchmark Cards experiments.
This script creates additional visualizations for analysis.
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_results(results_dir, dataset_name):
    """Load results for a dataset."""
    logger.info(f"Loading results for {dataset_name}")
    
    # Paths to result files
    benchmark_card_path = os.path.join(results_dir, dataset_name, f"{dataset_name}_benchmark_card.json")
    model_results_path = os.path.join(results_dir, dataset_name, f"{dataset_name}_model_results.json")
    simulation_results_path = os.path.join(results_dir, dataset_name, f"{dataset_name}_simulation_results.json")
    
    results = {}
    
    # Load benchmark card
    if os.path.exists(benchmark_card_path):
        with open(benchmark_card_path, 'r') as f:
            results['benchmark_card'] = json.load(f)
    else:
        logger.warning(f"Benchmark card not found for {dataset_name}")
    
    # Load model results
    if os.path.exists(model_results_path):
        with open(model_results_path, 'r') as f:
            results['model_results'] = json.load(f)
    else:
        logger.warning(f"Model results not found for {dataset_name}")
    
    # Load simulation results
    if os.path.exists(simulation_results_path):
        with open(simulation_results_path, 'r') as f:
            results['simulation_results'] = json.load(f)
    else:
        logger.warning(f"Simulation results not found for {dataset_name}")
    
    return results


def create_radar_chart(model_results, output_path):
    """Create radar chart comparing models across metrics."""
    logger.info("Creating radar chart")
    
    # Select metrics for radar chart
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score"]
    
    # Get model names
    model_names = list(model_results.keys())
    
    # Check if metrics exist
    valid_metrics = []
    for metric in metrics:
        if all(metric in model_results[model] for model in model_names):
            valid_metrics.append(metric)
    
    if not valid_metrics:
        logger.warning("No valid metrics found for radar chart")
        return
    
    # Set up the figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of metrics
    N = len(valid_metrics)
    
    # Compute angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in valid_metrics])
    
    # Scale data to [0, 1] for better visualization
    min_values = {metric: min(model_results[model][metric] for model in model_names) for metric in valid_metrics}
    max_values = {metric: max(model_results[model][metric] for model in model_names) for metric in valid_metrics}
    
    # Add data for each model
    for i, model in enumerate(model_names):
        values = []
        for metric in valid_metrics:
            # Normalize to [0, 1]
            value = model_results[model][metric]
            min_val = min_values[metric]
            max_val = max_values[metric]
            
            if max_val > min_val:
                normalized = (value - min_val) / (max_val - min_val)
            else:
                normalized = 0.5  # Default if all values are the same
                
            values.append(normalized)
        
        values += values[:1]  # Close the polygon
        
        # Plot values
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_weight_heatmap(benchmark_card, output_path):
    """Create heatmap visualizing weights for different use cases."""
    logger.info("Creating weight heatmap")
    
    use_case_weights = benchmark_card.get("use_case_weights", {})
    
    if not use_case_weights:
        logger.warning("No use case weights found")
        return
    
    # Collect metrics and use cases
    all_metrics = set()
    for weights in use_case_weights.values():
        all_metrics.update(weights.keys())
    
    metrics = sorted(all_metrics)
    use_cases = sorted(use_case_weights.keys())
    
    # Create a matrix of weights
    weight_matrix = np.zeros((len(use_cases), len(metrics)))
    
    for i, use_case in enumerate(use_cases):
        for j, metric in enumerate(metrics):
            weight_matrix[i, j] = use_case_weights[use_case].get(metric, 0)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(weight_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=[m.replace('_', ' ').title() for m in metrics],
                yticklabels=[uc.replace('_', ' ').title() for uc in use_cases],
                cbar_kws={'label': 'Weight'})
    
    plt.title("Metric Weights by Use Case")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_selection_sankey(simulation_results, output_path):
    """Create Sankey diagram showing how model selections change with benchmark cards."""
    logger.info("Creating selection Sankey diagram")
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("Plotly not installed, skipping Sankey diagram")
        return
    
    default_selections = simulation_results.get("default_selections", {})
    card_selections = simulation_results.get("card_selections", {})
    
    if not default_selections or not card_selections:
        logger.warning("Selection results not found")
        return
    
    # Get unique models and use cases
    models = set(list(default_selections.values()) + list(card_selections.values()))
    use_cases = list(default_selections.keys())
    
    # Create node labels
    labels = []
    
    # Use case nodes
    use_case_nodes = [f"{uc} (Default)" for uc in use_cases]
    labels.extend(use_case_nodes)
    
    # Model nodes
    model_nodes = list(models)
    labels.extend(model_nodes)
    
    # Use case nodes (card)
    use_case_card_nodes = [f"{uc} (Card)" for uc in use_cases]
    labels.extend(use_case_card_nodes)
    
    # Create source, target, and value lists for the Sankey diagram
    source = []
    target = []
    value = []
    
    # Default selections
    for i, use_case in enumerate(use_cases):
        model = default_selections[use_case]
        model_idx = len(use_case_nodes) + model_nodes.index(model)
        
        source.append(i)
        target.append(model_idx)
        value.append(1)
    
    # Card selections
    for i, use_case in enumerate(use_cases):
        model = card_selections[use_case]
        model_idx = len(use_case_nodes) + model_nodes.index(model)
        use_case_card_idx = len(use_case_nodes) + len(model_nodes) + i
        
        source.append(model_idx)
        target.append(use_case_card_idx)
        value.append(1)
    
    # Create the figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=["blue"] * len(use_case_nodes) + ["green"] * len(model_nodes) + ["red"] * len(use_case_card_nodes)
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])
    
    fig.update_layout(title_text="Model Selection Flow", font_size=10)
    fig.write_image(output_path)


def create_metric_comparison(model_results, simulation_results, output_path):
    """Create bar chart comparing metrics for default vs card selections."""
    logger.info("Creating metric comparison chart")
    
    default_selections = simulation_results.get("default_selections", {})
    card_selections = simulation_results.get("card_selections", {})
    
    if not default_selections or not card_selections:
        logger.warning("Selection results not found")
        return
    
    # Get use cases with different selections
    different_selections = {}
    for use_case in default_selections:
        if default_selections[use_case] != card_selections.get(use_case):
            different_selections[use_case] = {
                "default": default_selections[use_case],
                "card": card_selections[use_case]
            }
    
    if not different_selections:
        logger.warning("No different selections found")
        return
    
    # Metrics to compare
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score"]
    
    # Create plots for each use case with different selections
    for use_case, selections in different_selections.items():
        default_model = selections["default"]
        card_model = selections["card"]
        
        # Check if models exist in results
        if default_model not in model_results or card_model not in model_results:
            continue
            
        # Get metrics for both models
        default_metrics = {m: model_results[default_model].get(m, 0) for m in metrics}
        card_metrics = {m: model_results[card_model].get(m, 0) for m in metrics}
        
        # Create a DataFrame for the comparison
        df = pd.DataFrame({
            "Metric": list(metrics),
            "Default Selection": [default_metrics[m] for m in metrics],
            "Card Selection": [card_metrics[m] for m in metrics]
        })
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Set up the positions
        x = np.arange(len(metrics))
        width = 0.35
        
        # Create the bars
        plt.bar(x - width/2, df["Default Selection"], width, label=f"Default ({default_model})")
        plt.bar(x + width/2, df["Card Selection"], width, label=f"Card ({card_model})")
        
        # Add labels and title
        plt.xlabel("Metrics")
        plt.ylabel("Value")
        plt.title(f"Metric Comparison for {use_case.replace('_', ' ').title()} Use Case")
        plt.xticks(x, [m.replace('_', ' ').title() for m in metrics])
        plt.legend()
        
        # Add value labels on top of bars
        for i, v in enumerate(df["Default Selection"]):
            plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center')
        
        for i, v in enumerate(df["Card Selection"]):
            plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(f"{output_path.replace('.png', f'_{use_case}.png')}", dpi=300)
        plt.close()


def main():
    """Main function to visualize results."""
    parser = argparse.ArgumentParser(description="Visualize Benchmark Cards results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing results")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset to visualize results for")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save visualizations (defaults to dataset results directory)")
    args = parser.parse_args()
    
    # Get full paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, args.results_dir)
    
    if args.output_dir:
        output_dir = os.path.join(script_dir, args.output_dir)
    else:
        output_dir = os.path.join(results_dir, args.dataset)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = load_results(results_dir, args.dataset)
    
    if not results:
        logger.error(f"No results found for {args.dataset}")
        return
    
    # Create visualizations
    if 'model_results' in results:
        create_radar_chart(
            results['model_results'],
            os.path.join(output_dir, f"{args.dataset}_radar_chart.png")
        )
    
    if 'benchmark_card' in results:
        create_weight_heatmap(
            results['benchmark_card'],
            os.path.join(output_dir, f"{args.dataset}_weight_heatmap.png")
        )
    
    if 'simulation_results' in results and 'model_results' in results:
        create_selection_sankey(
            results['simulation_results'],
            os.path.join(output_dir, f"{args.dataset}_selection_sankey.png")
        )
        
        create_metric_comparison(
            results['model_results'],
            results['simulation_results'],
            os.path.join(output_dir, f"{args.dataset}_metric_comparison.png")
        )
    
    logger.info(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
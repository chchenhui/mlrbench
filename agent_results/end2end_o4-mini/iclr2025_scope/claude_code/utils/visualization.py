import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import logging

logger = logging.getLogger(__name__)

def set_plotting_style():
    """Set consistent plotting style for all figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)
    
    # Set consistent font styles
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })

def plot_loss_curves(train_losses: List[float], 
                    val_losses: List[float], 
                    save_path: str):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the figure
    """
    set_plotting_style()
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o', linestyle='-', markersize=4)
    plt.plot(val_losses, label='Validation Loss', marker='s', linestyle='-', markersize=4)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Loss curves saved to {save_path}")

def plot_perplexity_comparison(model_perplexities: Dict[str, float], save_path: str):
    """
    Plot perplexity comparison across models.
    
    Args:
        model_perplexities: Dictionary mapping model names to perplexity values
        save_path: Path to save the figure
    """
    set_plotting_style()
    
    # Sort by perplexity
    sorted_items = sorted(model_perplexities.items(), key=lambda x: x[1])
    models, perplexities = zip(*sorted_items)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(models, perplexities, color=sns.color_palette("muted", len(models)))
    
    # Add value labels on top of bars
    for bar, ppl in zip(bars, perplexities):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{ppl:.2f}', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Models')
    plt.ylabel('Perplexity (lower is better)')
    plt.title('Perplexity Comparison Across Models')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Perplexity comparison saved to {save_path}")

def plot_throughput_comparison(model_throughputs: Dict[str, float], save_path: str):
    """
    Plot throughput comparison across models.
    
    Args:
        model_throughputs: Dictionary mapping model names to throughput values (tokens/sec)
        save_path: Path to save the figure
    """
    set_plotting_style()
    
    # Sort by throughput (descending)
    sorted_items = sorted(model_throughputs.items(), key=lambda x: x[1], reverse=True)
    models, throughputs = zip(*sorted_items)
    
    plt.figure(figsize=(12, 8))
    
    # Create bars with gradient color based on throughput
    palette = sns.color_palette("viridis", len(models))
    bars = plt.bar(models, throughputs, color=palette)
    
    # Add value labels on top of bars
    for bar, tput in zip(bars, throughputs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{tput:.2f}', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Models')
    plt.ylabel('Throughput (tokens/sec)')
    plt.title('Throughput Comparison Across Models')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Throughput comparison saved to {save_path}")

def plot_memory_usage_comparison(model_memory_usage: Dict[str, float], save_path: str):
    """
    Plot memory usage comparison across models.
    
    Args:
        model_memory_usage: Dictionary mapping model names to memory usage values (GB)
        save_path: Path to save the figure
    """
    set_plotting_style()
    
    # Sort by memory usage (ascending)
    sorted_items = sorted(model_memory_usage.items(), key=lambda x: x[1])
    models, memory_usage = zip(*sorted_items)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(models, memory_usage, color=sns.color_palette("cool", len(models)))
    
    # Add value labels on top of bars
    for bar, mem in zip(bars, memory_usage):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{mem:.2f} GB', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Models')
    plt.ylabel('Peak Memory Usage (GB)')
    plt.title('Memory Usage Comparison Across Models')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Memory usage comparison saved to {save_path}")

def plot_compression_ratio_comparison(model_compression_ratios: Dict[str, float], save_path: str):
    """
    Plot compression ratio comparison across models.
    
    Args:
        model_compression_ratios: Dictionary mapping model names to compression ratios
        save_path: Path to save the figure
    """
    set_plotting_style()
    
    # Sort by compression ratio (descending)
    sorted_items = sorted(model_compression_ratios.items(), key=lambda x: x[1], reverse=True)
    models, ratios = zip(*sorted_items)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(models, ratios, color=sns.color_palette("magma", len(models)))
    
    # Add value labels on top of bars
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{ratio:.2f}x', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Models')
    plt.ylabel('Compression Ratio (higher is better)')
    plt.title('KV Cache Compression Ratio Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Compression ratio comparison saved to {save_path}")

def plot_latency_vs_sequence_length(
    model_results: Dict[str, Dict[str, List[float]]],
    save_path: str
):
    """
    Plot latency vs sequence length for different models.
    
    Args:
        model_results: Dictionary mapping model names to dictionaries of sequence lengths and latencies
                     {model_name: {"seq_lengths": [...], "latencies_ms": [...]}}
        save_path: Path to save the figure
    """
    set_plotting_style()
    
    plt.figure(figsize=(12, 8))
    
    # Create line plots for each model
    for i, (model_name, data) in enumerate(model_results.items()):
        seq_lengths = data["seq_lengths"]
        latencies = data["latencies_ms"]
        
        # Use a colorful palette
        color = sns.color_palette("tab10")[i % 10]
        
        # Plot with markers and lines
        plt.plot(seq_lengths, latencies, 
                marker='o', linestyle='-', linewidth=2, markersize=8,
                label=model_name, color=color)
    
    plt.xlabel('Sequence Length (tokens)')
    plt.ylabel('Latency (ms/token)')
    plt.title('Latency vs Sequence Length')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add log scale for better visualization of quadratic vs linear behavior
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    
    # Add reference lines for linear and quadratic scaling
    x_ref = np.array([seq_lengths[0], seq_lengths[-1]])
    y_linear = x_ref * (model_results[list(model_results.keys())[0]]["latencies_ms"][0] / model_results[list(model_results.keys())[0]]["seq_lengths"][0])
    y_quadratic = x_ref**2 * (model_results[list(model_results.keys())[0]]["latencies_ms"][0] / model_results[list(model_results.keys())[0]]["seq_lengths"][0]**2)
    
    plt.plot(x_ref, y_linear, 'k--', alpha=0.5, linewidth=1, label='Linear Scaling')
    plt.plot(x_ref, y_quadratic, 'k:', alpha=0.5, linewidth=1, label='Quadratic Scaling')
    
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Latency vs sequence length plot saved to {save_path}")

def plot_tradeoff_bubble_chart(
    model_results: Dict[str, Dict[str, float]], 
    save_path: str
):
    """
    Plot bubble chart showing tradeoff between perplexity, throughput, and memory usage.
    
    Args:
        model_results: Dictionary mapping model names to dictionaries of metrics
                     {model_name: {"perplexity": float, "throughput": float, "memory_gb": float}}
        save_path: Path to save the figure
    """
    set_plotting_style()
    
    # Extract data
    models = list(model_results.keys())
    perplexities = [model_results[model]["perplexity"] for model in models]
    throughputs = [model_results[model]["throughput"] for model in models]
    memories = [model_results[model]["memory_gb"] for model in models]
    
    # Create bubble chart
    plt.figure(figsize=(12, 10))
    
    # Use a colorful palette
    colors = sns.color_palette("husl", len(models))
    
    # Plot bubbles
    for i, (model, perplexity, throughput, memory) in enumerate(zip(models, perplexities, throughputs, memories)):
        plt.scatter(throughput, perplexity, s=memory*100, alpha=0.7, 
                   c=[colors[i]], label=model, edgecolors='white', linewidth=1.5)
        
        # Add model name as annotation
        plt.annotate(model, (throughput, perplexity), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10)
    
    plt.xlabel('Throughput (tokens/sec) - higher is better')
    plt.ylabel('Perplexity - lower is better')
    plt.title('Tradeoff: Perplexity vs Throughput vs Memory Usage')
    
    # Add custom legend for bubble size
    memory_sizes = [min(memories), (min(memories) + max(memories))/2, max(memories)]
    for memory in memory_sizes:
        plt.scatter([], [], s=memory*100, alpha=0.5, c='gray', 
                  label=f'{memory:.1f} GB')
    
    plt.legend(title="Memory Usage", loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Invert y-axis so lower perplexity (better) is at the top
    plt.gca().invert_yaxis()
    
    # Add a light reference line for Pareto frontier
    # (not a true Pareto calculation, just a visual guide)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Tradeoff bubble chart saved to {save_path}")

def plot_ablation_results(
    ablation_results: Dict[str, Dict[str, float]],
    metric_name: str,
    save_path: str
):
    """
    Plot ablation study results.
    
    Args:
        ablation_results: Dictionary mapping ablation configuration names to dictionaries of metrics
                         {config_name: {"perplexity": float, "throughput": float, ...}}
        metric_name: Name of the metric to plot
        save_path: Path to save the figure
    """
    set_plotting_style()
    
    # Extract data
    configs = list(ablation_results.keys())
    metric_values = [ablation_results[config][metric_name] for config in configs]
    
    # Sort by metric value (appropriate direction depends on metric)
    should_reverse = metric_name in ["throughput", "compression_ratio"]  # Higher is better
    sorted_pairs = sorted(zip(configs, metric_values), key=lambda x: x[1], reverse=should_reverse)
    configs, metric_values = zip(*sorted_pairs)
    
    plt.figure(figsize=(12, 8))
    
    # Create bars with gradient color
    bars = plt.bar(configs, metric_values, color=sns.color_palette("viridis", len(configs)))
    
    # Add value labels on top of bars
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        offset = 0.05 * max(metric_values) if height > 0 else -0.1 * max(metric_values)
        plt.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, rotation=0)
    
    plt.xlabel('Configuration')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'Ablation Study: Effect on {metric_name.replace("_", " ").title()}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Ablation results for {metric_name} saved to {save_path}")

def plot_rouge_scores(
    model_rouge_scores: Dict[str, Dict[str, Dict[str, float]]],
    save_path: str
):
    """
    Plot ROUGE scores for different models.
    
    Args:
        model_rouge_scores: Dictionary mapping model names to ROUGE score dictionaries
                           {model_name: {"rouge-1": {"f": float, ...}, ...}}
        save_path: Path to save the figure
    """
    set_plotting_style()
    
    # Extract F1 scores for different ROUGE metrics
    models = list(model_rouge_scores.keys())
    rouge1_f1 = [model_rouge_scores[model]["rouge-1"]["f"] for model in models]
    rouge2_f1 = [model_rouge_scores[model]["rouge-2"]["f"] for model in models]
    rougeL_f1 = [model_rouge_scores[model]["rouge-l"]["f"] for model in models]
    
    # Create dataframe for grouped bar chart
    data = {
        "Model": models * 3,
        "ROUGE Type": ["ROUGE-1"] * len(models) + ["ROUGE-2"] * len(models) + ["ROUGE-L"] * len(models),
        "F1 Score": rouge1_f1 + rouge2_f1 + rougeL_f1
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(14, 8))
    
    # Create grouped bar chart
    sns.barplot(x="Model", y="F1 Score", hue="ROUGE Type", data=df, palette="Set2")
    
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.title('ROUGE F1 Scores Across Models')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="ROUGE Metric")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROUGE scores comparison saved to {save_path}")

def create_summary_dashboard(
    results_dir: str,
    model_results: Dict[str, Dict[str, Any]]
):
    """
    Create a comprehensive summary dashboard with all plots.
    
    Args:
        results_dir: Directory to save results
        model_results: Dictionary of all model results
    """
    set_plotting_style()
    
    # Ensure directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract metrics for each plot
    perplexities = {model: data.get("perplexity", 0) for model, data in model_results.items()}
    throughputs = {model: data.get("throughput", 0) for model, data in model_results.items()}
    memory_usage = {model: data.get("memory_gb", 0) for model, data in model_results.items()}
    compression_ratios = {model: data.get("compression_ratio", 1.0) for model, data in model_results.items()}
    
    # Generate individual plots
    plot_perplexity_comparison(perplexities, os.path.join(results_dir, "perplexity_comparison.png"))
    plot_throughput_comparison(throughputs, os.path.join(results_dir, "throughput_comparison.png"))
    plot_memory_usage_comparison(memory_usage, os.path.join(results_dir, "memory_usage_comparison.png"))
    plot_compression_ratio_comparison(compression_ratios, os.path.join(results_dir, "compression_ratio_comparison.png"))
    
    # Extract sequence length vs latency data if available
    seq_latency_data = {}
    for model, data in model_results.items():
        if "seq_lengths" in data and "latencies_ms" in data:
            seq_latency_data[model] = {
                "seq_lengths": data["seq_lengths"],
                "latencies_ms": data["latencies_ms"]
            }
    
    if seq_latency_data:
        plot_latency_vs_sequence_length(seq_latency_data, os.path.join(results_dir, "latency_vs_sequence_length.png"))
    
    # Create tradeoff bubble chart
    tradeoff_data = {}
    for model, data in model_results.items():
        if "perplexity" in data and "throughput" in data and "memory_gb" in data:
            tradeoff_data[model] = {
                "perplexity": data["perplexity"],
                "throughput": data["throughput"],
                "memory_gb": data["memory_gb"]
            }
    
    if tradeoff_data:
        plot_tradeoff_bubble_chart(tradeoff_data, os.path.join(results_dir, "tradeoff_analysis.png"))
    
    # Extract ROUGE scores if available
    rouge_data = {}
    for model, data in model_results.items():
        if "rouge-1" in data and "rouge-2" in data and "rouge-l" in data:
            rouge_data[model] = {
                "rouge-1": data["rouge-1"],
                "rouge-2": data["rouge-2"],
                "rouge-l": data["rouge-l"]
            }
    
    if rouge_data:
        plot_rouge_scores(rouge_data, os.path.join(results_dir, "rouge_scores_comparison.png"))
    
    logger.info(f"All summary visualizations saved to {results_dir}")
    
    return {
        "perplexity_plot": os.path.join(results_dir, "perplexity_comparison.png"),
        "throughput_plot": os.path.join(results_dir, "throughput_comparison.png"),
        "memory_plot": os.path.join(results_dir, "memory_usage_comparison.png"),
        "compression_plot": os.path.join(results_dir, "compression_ratio_comparison.png"),
        "latency_plot": os.path.join(results_dir, "latency_vs_sequence_length.png") if seq_latency_data else None,
        "tradeoff_plot": os.path.join(results_dir, "tradeoff_analysis.png") if tradeoff_data else None,
        "rouge_plot": os.path.join(results_dir, "rouge_scores_comparison.png") if rouge_data else None
    }
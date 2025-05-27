"""
Utility functions for the ATSKV (Adaptive Token-Relevance Sparse KV-Cache) implementation.
"""
import os
import time
import json
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get the appropriate device (GPU or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def save_json(data: Dict, filename: str):
    """Save data as JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filename: str) -> Dict:
    """Load data from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def memory_usage_stats(model: PreTrainedModel) -> Dict[str, float]:
    """Calculate memory usage statistics for a model."""
    if not torch.cuda.is_available():
        return {"cuda_memory_allocated_MB": 0, "cuda_memory_reserved_MB": 0}
    
    # Get memory usage statistics
    cuda_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
    cuda_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
    
    return {
        "cuda_memory_allocated_MB": cuda_memory_allocated,
        "cuda_memory_reserved_MB": cuda_memory_reserved,
    }

def calculate_kv_cache_size(
    model: PreTrainedModel,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    max_seq_length: int
) -> Dict[str, float]:
    """
    Calculate the size of KV cache for a given model and sequence length.
    
    Args:
        model: The pre-trained model
        num_layers: Number of transformer layers
        hidden_size: Hidden size dimension
        num_attention_heads: Number of attention heads
        max_seq_length: Maximum sequence length
        
    Returns:
        Dictionary containing KV cache size in different units
    """
    # Calculate size per layer
    head_dim = hidden_size // num_attention_heads
    
    # For K and V, we need to store tensors of shape (batch_size, num_heads, seq_len, head_dim)
    # Assuming batch_size = 1 for inference
    bytes_per_element = torch.tensor([], dtype=model.dtype).element_size()
    
    # Total size for K and V
    kv_cache_size_bytes = 2 * num_layers * num_attention_heads * max_seq_length * head_dim * bytes_per_element
    
    return {
        "kv_cache_size_bytes": kv_cache_size_bytes,
        "kv_cache_size_MB": kv_cache_size_bytes / (1024 * 1024),
        "kv_cache_size_GB": kv_cache_size_bytes / (1024 * 1024 * 1024),
    }

def compute_attention_statistics(
    attention_scores: torch.Tensor,
    layer_idx: int
) -> Dict[str, torch.Tensor]:
    """
    Compute statistics from attention scores.
    
    Args:
        attention_scores: Attention scores tensor with shape [batch_size, num_heads, seq_len, seq_len]
        layer_idx: Index of the current layer
        
    Returns:
        Dictionary containing attention statistics
    """
    # Attention scores have shape [batch_size, num_heads, seq_len, seq_len]
    # Mean attention received by each token across all heads
    mean_attention_received = attention_scores.mean(dim=(0, 1))  # [seq_len, seq_len] -> [seq_len]
    
    # Mean attention given by each token
    mean_attention_given = attention_scores.mean(dim=(0, 1)).mean(dim=1)  # [seq_len]
    
    # Entropy of attention distribution for each token
    attention_probs = torch.softmax(attention_scores, dim=-1)
    entropy = -(attention_probs * torch.log(attention_probs + 1e-10)).sum(dim=-1)  # [batch_size, num_heads, seq_len]
    mean_entropy = entropy.mean(dim=(0, 1))  # [seq_len]
    
    return {
        f"layer_{layer_idx}_mean_attention_received": mean_attention_received,
        f"layer_{layer_idx}_mean_attention_given": mean_attention_given,
        f"layer_{layer_idx}_attention_entropy": mean_entropy,
    }

def visualize_attention_patterns(
    attention_weights: Dict[int, torch.Tensor],
    save_path: str,
    max_tokens: int = 100
):
    """
    Visualize attention patterns across layers.
    
    Args:
        attention_weights: Dictionary mapping layer indices to attention tensors
        save_path: Path to save the visualization
        max_tokens: Maximum number of tokens to visualize
    """
    num_layers = len(attention_weights)
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, num_layers * 3))
    
    if num_layers == 1:
        axes = [axes]
        
    for layer_idx, attention in attention_weights.items():
        # Get attention map for first head, first batch
        attn = attention[0, 0, :max_tokens, :max_tokens].cpu().numpy()
        
        # Plot heatmap
        sns.heatmap(attn, ax=axes[layer_idx], cmap="viridis")
        axes[layer_idx].set_title(f"Layer {layer_idx} Attention")
        axes[layer_idx].set_xlabel("Key tokens")
        axes[layer_idx].set_ylabel("Query tokens")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_token_relevance_scores(
    relevance_scores: Dict[int, torch.Tensor],
    save_path: str,
    max_tokens: int = 100
):
    """
    Plot token relevance scores across layers.
    
    Args:
        relevance_scores: Dictionary mapping layer indices to relevance score tensors
        save_path: Path to save the plot
        max_tokens: Maximum number of tokens to visualize
    """
    num_layers = len(relevance_scores)
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, num_layers * 2))
    
    if num_layers == 1:
        axes = [axes]
    
    for i, (layer_idx, scores) in enumerate(relevance_scores.items()):
        # Get scores for first batch
        scores_np = scores[0, :max_tokens].cpu().numpy()
        
        # Plot bars
        axes[i].bar(range(len(scores_np)), scores_np)
        axes[i].set_title(f"Layer {layer_idx} Token Relevance Scores")
        axes[i].set_xlabel("Token Position")
        axes[i].set_ylabel("Relevance Score")
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_comparison_metrics(
    metrics: Dict[str, Dict[str, List[float]]],
    metric_name: str,
    save_path: str
):
    """
    Plot comparison metrics across different methods.
    
    Args:
        metrics: Dictionary mapping method names to metric dictionaries
        metric_name: Name of the metric to plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for method, method_metrics in metrics.items():
        if metric_name in method_metrics:
            plt.plot(method_metrics[metric_name], label=method)
    
    plt.title(f"{metric_name} Comparison")
    plt.xlabel("Sequence Length")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_memory_usage_comparison(
    results: Dict[str, Dict[str, List[float]]],
    seq_lengths: List[int],
    save_path: str
):
    """
    Plot memory usage comparison across different methods.
    
    Args:
        results: Dictionary mapping method names to results
        seq_lengths: List of sequence lengths
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for method, method_results in results.items():
        if "memory_usage" in method_results:
            plt.plot(seq_lengths, method_results["memory_usage"], label=method, marker='o')
    
    plt.title("Memory Usage Comparison")
    plt.xlabel("Sequence Length")
    plt.ylabel("Memory Usage (MB)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def measure_inference_time(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_samples: int = 5,
    generate_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Measure inference time statistics.
    
    Args:
        model: The pre-trained model
        input_ids: Input token IDs
        attention_mask: Attention mask
        num_samples: Number of samples for measurement
        generate_kwargs: Additional arguments for model.generate()
        
    Returns:
        Dictionary containing timing statistics
    """
    if generate_kwargs is None:
        generate_kwargs = {}
    
    # Warm-up
    _ = model.generate(input_ids, attention_mask=attention_mask, **generate_kwargs)
    
    # Measure time-to-first-token
    start_times = []
    end_times = []
    throughputs = []
    
    for _ in range(num_samples):
        # Time to first token
        torch.cuda.synchronize()
        start_time = time.time()
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=1,
            **generate_kwargs
        )
        torch.cuda.synchronize()
        first_token_time = time.time() - start_time
        start_times.append(first_token_time)
        
        # Full generation time
        torch.cuda.synchronize()
        start_time = time.time()
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            **generate_kwargs
        )
        torch.cuda.synchronize()
        end_time = time.time() - start_time
        end_times.append(end_time)
        
        # Calculate throughput
        num_new_tokens = outputs.shape[1] - input_ids.shape[1]
        throughput = num_new_tokens / end_time
        throughputs.append(throughput)
    
    return {
        "time_to_first_token": np.mean(start_times),
        "full_generation_time": np.mean(end_times),
        "tokens_per_second": np.mean(throughputs),
        "time_to_first_token_std": np.std(start_times),
        "full_generation_time_std": np.std(end_times),
        "tokens_per_second_std": np.std(throughputs),
    }

def format_results_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format results as a markdown table.
    
    Args:
        results: Dictionary mapping method names to result metrics
        
    Returns:
        Markdown-formatted table string
    """
    # Get all unique metrics across all methods
    all_metrics = set()
    for method_results in results.values():
        all_metrics.update(method_results.keys())
    
    # Create table header
    header = "| Method | " + " | ".join(all_metrics) + " |"
    separator = "| --- | " + " | ".join(["---"] * len(all_metrics)) + " |"
    
    # Create table rows
    rows = []
    for method, method_results in results.items():
        row = f"| {method} |"
        for metric in all_metrics:
            if metric in method_results:
                value = method_results[metric]
                # Format based on type
                if isinstance(value, float):
                    row += f" {value:.4f} |"
                else:
                    row += f" {value} |"
            else:
                row += " - |"
        rows.append(row)
    
    return "\n".join([header, separator] + rows)

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_sparsity_impact(
    sparsity_levels: List[float],
    performance_metrics: Dict[str, List[float]],
    save_path: str
):
    """
    Plot the impact of different sparsity levels on model performance.
    
    Args:
        sparsity_levels: List of sparsity levels
        performance_metrics: Dictionary mapping metric names to lists of values
        save_path: Path to save the plot
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # First y-axis for accuracy metrics
    if "accuracy" in performance_metrics:
        ax1.plot(sparsity_levels, performance_metrics["accuracy"], 'b-', marker='o', label="Accuracy")
    if "f1" in performance_metrics:
        ax1.plot(sparsity_levels, performance_metrics["f1"], 'g-', marker='s', label="F1 Score")
    
    ax1.set_xlabel("Sparsity Level (%)")
    ax1.set_ylabel("Performance Metric")
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Second y-axis for memory and time efficiency
    ax2 = ax1.twinx()
    
    if "memory_reduction" in performance_metrics:
        ax2.plot(sparsity_levels, performance_metrics["memory_reduction"], 'r-', marker='^', label="Memory Reduction (%)")
    if "speedup" in performance_metrics:
        ax2.plot(sparsity_levels, performance_metrics["speedup"], 'm-', marker='*', label="Speedup Factor")
    
    ax2.set_ylabel("Efficiency Metric")
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    
    plt.title("Impact of Sparsity on Performance and Efficiency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
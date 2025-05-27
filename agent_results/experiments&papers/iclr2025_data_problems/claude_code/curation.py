import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import DataLoader, Dataset, Subset
import logging
from utils import Timer, plot_method_comparison
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import random
from scipy.optimize import minimize
from tqdm import tqdm

logger = logging.getLogger("influence_space")

def categorize_clusters(
    influence_scores: Dict[int, float],
    harmful_threshold: float = -0.001,
    beneficial_threshold: float = 0.01
) -> Tuple[List[int], List[int], List[int]]:
    """
    Categorize clusters based on their influence scores.
    
    Args:
        influence_scores: Dictionary mapping cluster indices to influence scores
        harmful_threshold: Threshold for identifying harmful clusters
        beneficial_threshold: Threshold for identifying beneficial clusters
        
    Returns:
        Tuple of lists for (harmful, neutral, beneficial) cluster indices
    """
    harmful = []
    neutral = []
    beneficial = []
    
    for cluster_idx, score in influence_scores.items():
        if score < harmful_threshold:
            harmful.append(cluster_idx)
        elif score > beneficial_threshold:
            beneficial.append(cluster_idx)
        else:
            neutral.append(cluster_idx)
    
    logger.info(f"Categorized clusters: {len(harmful)} harmful, {len(neutral)} neutral, {len(beneficial)} beneficial")
    return harmful, neutral, beneficial

def compute_cluster_sizes(clusters: List[List[int]]) -> Dict[int, int]:
    """
    Compute size of each cluster.
    
    Args:
        clusters: List of indices for each cluster
        
    Returns:
        Dictionary mapping cluster indices to their sizes
    """
    return {i: len(cluster) for i, cluster in enumerate(clusters)}

def compute_cluster_demographics(
    clusters: List[List[int]],
    dataset: Dataset
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Compute demographic statistics for each cluster.
    
    Args:
        clusters: List of indices for each cluster
        dataset: Dataset containing demographic information
        
    Returns:
        Dictionary mapping demographic attributes to cluster statistics
    """
    # Initialize demographic statistics
    demographics = ["gender", "ethnicity"]
    stats = {demo: {i: defaultdict(int) for i in range(len(clusters))} for demo in demographics}
    
    # Count demographics for each cluster
    for cluster_idx, cluster in enumerate(clusters):
        for idx in cluster:
            try:
                sample = dataset[idx]
                for demo in demographics:
                    if demo in sample:
                        value = sample[demo]
                        stats[demo][cluster_idx][value] += 1
            except:
                # Skip samples that can't be accessed
                continue
    
    # Convert counts to proportions
    for demo in demographics:
        for cluster_idx, counts in stats[demo].items():
            total = sum(counts.values())
            if total > 0:
                for value in counts:
                    counts[value] /= total
    
    return stats

def compute_representativeness(
    cluster_demographics: Dict[str, Dict[int, Dict[str, float]]],
    dataset_demographics: Dict[str, Dict[str, float]]
) -> Dict[int, float]:
    """
    Compute representativeness score for each cluster.
    
    Args:
        cluster_demographics: Demographic statistics for each cluster
        dataset_demographics: Overall demographic statistics for the dataset
        
    Returns:
        Dictionary mapping cluster indices to representativeness scores
    """
    representativeness = {}
    
    # Get all cluster indices
    cluster_indices = set()
    for demo in cluster_demographics:
        cluster_indices.update(cluster_demographics[demo].keys())
    
    # Compute representativeness for each cluster
    for cluster_idx in cluster_indices:
        # Compute average absolute difference for each demographic attribute
        diffs = []
        
        for demo in cluster_demographics:
            if cluster_idx in cluster_demographics[demo]:
                cluster_stats = cluster_demographics[demo][cluster_idx]
                dataset_stats = dataset_demographics[demo]
                
                # Compute absolute differences for each value
                values = set(cluster_stats.keys()) | set(dataset_stats.keys())
                abs_diffs = []
                
                for value in values:
                    cluster_prop = cluster_stats.get(value, 0)
                    dataset_prop = dataset_stats.get(value, 0)
                    abs_diffs.append(abs(cluster_prop - dataset_prop))
                
                # Average difference for this demographic
                if abs_diffs:
                    diffs.append(sum(abs_diffs) / len(abs_diffs))
        
        # Overall representativeness for this cluster
        # Lower difference means higher representativeness
        if diffs:
            representativeness[cluster_idx] = 1.0 - sum(diffs) / len(diffs)
        else:
            representativeness[cluster_idx] = 0.0
    
    return representativeness

def optimize_weights(
    influence_scores: Dict[int, float],
    cluster_sizes: Dict[int, int],
    representativeness: Dict[int, float],
    target_size: int,
    max_weight: float = 5.0,
    alpha: float = 0.7,
    beta: float = 0.3
) -> Dict[int, float]:
    """
    Optimize cluster weights to maximize utility while respecting constraints.
    
    Args:
        influence_scores: Dictionary mapping cluster indices to influence scores
        cluster_sizes: Dictionary mapping cluster indices to their sizes
        representativeness: Dictionary mapping cluster indices to representativeness scores
        target_size: Target number of samples after curation
        max_weight: Maximum weight for a cluster
        alpha: Weight for influence score in the utility function
        beta: Weight for representativeness in the utility function
        
    Returns:
        Dictionary mapping cluster indices to their optimal weights
    """
    logger.info("Optimizing cluster weights...")
    
    # Get common cluster indices
    cluster_indices = sorted(set(influence_scores.keys()) & set(cluster_sizes.keys()) & set(representativeness.keys()))
    
    # Convert to numpy arrays for optimization
    influence_arr = np.array([influence_scores.get(idx, 0) for idx in cluster_indices])
    sizes_arr = np.array([cluster_sizes.get(idx, 0) for idx in cluster_indices])
    repr_arr = np.array([representativeness.get(idx, 0) for idx in cluster_indices])
    
    # Normalize influence scores and representativeness to [0, 1]
    if len(influence_arr) > 0:
        influence_min = min(0, influence_arr.min())  # Ensure non-positive values get mapped to 0
        influence_max = max(0, influence_arr.max())  # Ensure non-negative values get mapped to 1
        influence_range = influence_max - influence_min
        if influence_range > 0:
            influence_norm = (influence_arr - influence_min) / influence_range
        else:
            influence_norm = np.zeros_like(influence_arr)
    else:
        influence_norm = np.array([])
    
    # Define the objective function to maximize
    def objective(weights):
        # Combined utility of influence and representativeness
        utility = alpha * np.sum(weights * influence_norm) + beta * np.sum(weights * repr_arr * sizes_arr) / np.sum(weights * sizes_arr)
        # Negative for minimization
        return -utility
    
    # Define constraint: total weighted size <= target_size
    def size_constraint(weights):
        return target_size - np.sum(weights * sizes_arr)
    
    # Initial weights: distribute equally
    if len(cluster_indices) > 0:
        initial_weights = np.ones(len(cluster_indices))
        
        # Set up constraints
        constraints = [
            {'type': 'ineq', 'fun': size_constraint}
        ]
        
        # Set up bounds: weights between 0 and max_weight
        bounds = [(0, max_weight) for _ in range(len(cluster_indices))]
        
        # Run optimization
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False, 'maxiter': 1000}
        )
        
        # Extract optimal weights
        optimal_weights = result.x
        
        # Convert back to dictionary
        weights = {idx: float(w) for idx, w in zip(cluster_indices, optimal_weights)}
    else:
        weights = {}
    
    logger.info(f"Optimized weights for {len(weights)} clusters")
    return weights

def apply_curation(
    dataset: Dataset,
    clusters: List[List[int]],
    cluster_weights: Dict[int, float],
    pruned_clusters: List[int],
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, List[int], List[float]]:
    """
    Apply curation strategy to create a curated dataloader.
    
    Args:
        dataset: Original dataset
        clusters: List of indices for each cluster
        cluster_weights: Dictionary mapping cluster indices to their weights
        pruned_clusters: List of cluster indices to prune
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (curated_dataloader, curated_indices, sample_weights)
    """
    logger.info("Applying curation strategy...")
    
    # Collect all curated indices and their weights
    curated_indices = []
    sample_weights = []
    
    for cluster_idx, cluster in enumerate(clusters):
        # Skip empty or pruned clusters
        if not cluster or cluster_idx in pruned_clusters:
            continue
        
        # Get weight for this cluster (default to 1.0)
        weight = cluster_weights.get(cluster_idx, 1.0)
        
        # Add indices and weights
        curated_indices.extend(cluster)
        sample_weights.extend([weight] * len(cluster))
    
    # Create curated dataset
    from data_loader import get_dataloader_from_indices
    curated_dataloader = get_dataloader_from_indices(
        dataset,
        curated_indices,
        weights=sample_weights,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    logger.info(f"Created curated dataset with {len(curated_indices)} samples from {len(clusters) - len(pruned_clusters)} clusters")
    return curated_dataloader, curated_indices, sample_weights

def run_curation(
    influence_scores: Dict[int, float],
    clusters: List[List[int]],
    dataset: Dataset,
    target_size: Optional[int] = None,
    harmful_threshold: float = -0.001,
    beneficial_threshold: float = 0.01,
    max_weight: float = 5.0,
    batch_size: int = 32,
    num_workers: int = 4,
    output_dir: str = "./",
    visualize: bool = True
) -> Tuple[DataLoader, List[int], List[float]]:
    """
    Run the full curation pipeline.
    
    Args:
        influence_scores: Dictionary mapping cluster indices to influence scores
        clusters: List of indices for each cluster
        dataset: Original dataset
        target_size: Target number of samples after curation (if None, use 80% of original)
        harmful_threshold: Threshold for identifying harmful clusters
        beneficial_threshold: Threshold for identifying beneficial clusters
        max_weight: Maximum weight for a cluster
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for data loading
        output_dir: Directory to save results
        visualize: Whether to create visualizations
        
    Returns:
        Tuple of (curated_dataloader, curated_indices, sample_weights)
    """
    # Set default target size if not specified
    if target_size is None:
        total_samples = sum(len(cluster) for cluster in clusters)
        target_size = int(0.8 * total_samples)
    
    # Categorize clusters
    harmful, neutral, beneficial = categorize_clusters(
        influence_scores, 
        harmful_threshold=harmful_threshold,
        beneficial_threshold=beneficial_threshold
    )
    
    # Compute cluster sizes
    cluster_sizes = compute_cluster_sizes(clusters)
    
    # Compute cluster demographics
    cluster_demographics = compute_cluster_demographics(clusters, dataset)
    
    # Compute overall dataset demographics
    dataset_demographics = {}
    for demo in cluster_demographics:
        # Combine all cluster demographics
        dataset_demographics[demo] = defaultdict(float)
        total_samples = 0
        
        for cluster_idx, stats in cluster_demographics[demo].items():
            size = cluster_sizes.get(cluster_idx, 0)
            for value, prop in stats.items():
                dataset_demographics[demo][value] += prop * size
            total_samples += size
        
        # Normalize to proportions
        if total_samples > 0:
            for value in dataset_demographics[demo]:
                dataset_demographics[demo][value] /= total_samples
    
    # Compute representativeness scores
    representativeness = compute_representativeness(cluster_demographics, dataset_demographics)
    
    # Optimize weights (only for neutral and beneficial clusters)
    valid_clusters = neutral + beneficial
    valid_influence = {idx: influence_scores[idx] for idx in valid_clusters if idx in influence_scores}
    valid_sizes = {idx: cluster_sizes[idx] for idx in valid_clusters if idx in cluster_sizes}
    valid_repr = {idx: representativeness[idx] for idx in valid_clusters if idx in representativeness}
    
    # Optimize weights
    cluster_weights = optimize_weights(
        valid_influence,
        valid_sizes,
        valid_repr,
        target_size,
        max_weight=max_weight
    )
    
    # Apply curation strategy
    curated_dataloader, curated_indices, sample_weights = apply_curation(
        dataset,
        clusters,
        cluster_weights,
        pruned_clusters=harmful,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save cluster categorization
    categorization = {
        "harmful": harmful,
        "neutral": neutral,
        "beneficial": beneficial
    }
    with open(os.path.join(output_dir, "cluster_categorization.json"), "w") as f:
        json.dump({k: [int(i) for i in v] for k, v in categorization.items()}, f)
    
    # Save cluster weights
    with open(os.path.join(output_dir, "cluster_weights.json"), "w") as f:
        json.dump({str(k): float(v) for k, v in cluster_weights.items()}, f)
    
    # Save curated indices
    np.save(os.path.join(output_dir, "curated_indices.npy"), np.array(curated_indices))
    
    # Save sample weights
    np.save(os.path.join(output_dir, "sample_weights.npy"), np.array(sample_weights))
    
    # Visualize curation results
    if visualize:
        # Plot cluster categorization
        categories = ["Harmful", "Neutral", "Beneficial"]
        counts = [len(harmful), len(neutral), len(beneficial)]
        
        plt.figure(figsize=(10, 6))
        plt.bar(categories, counts)
        plt.title("Cluster Categorization")
        plt.xlabel("Category")
        plt.ylabel("Number of Clusters")
        plt.grid(True, axis='y')
        
        # Add values on top of bars
        for i, count in enumerate(counts):
            plt.text(i, count + 0.1, str(count), ha='center')
        
        # Save plot
        plt.savefig(os.path.join(output_dir, "cluster_categorization.png"))
        plt.close()
        
        # Plot weight distribution
        weights = list(cluster_weights.values())
        
        plt.figure(figsize=(10, 6))
        plt.hist(weights, bins=20)
        plt.title("Cluster Weight Distribution")
        plt.xlabel("Weight")
        plt.ylabel("Frequency")
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(output_dir, "weight_distribution.png"))
        plt.close()
    
    return curated_dataloader, curated_indices, sample_weights

def random_sampling_baseline(
    dataset: Dataset,
    target_size: int,
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, List[int]]:
    """
    Create a baseline using random sampling.
    
    Args:
        dataset: Original dataset
        target_size: Target number of samples
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for data loading
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (dataloader, sampled_indices)
    """
    logger.info(f"Creating random sampling baseline with {target_size} samples...")
    
    # Set random seed
    random.seed(random_seed)
    
    # Sample random indices
    all_indices = list(range(len(dataset)))
    sampled_indices = random.sample(all_indices, min(target_size, len(all_indices)))
    
    # Create dataloader
    from data_loader import get_dataloader_from_indices
    dataloader = get_dataloader_from_indices(
        dataset,
        sampled_indices,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return dataloader, sampled_indices

def clip_score_filtering_baseline(
    dataset: Dataset,
    clip_scores: np.ndarray,
    target_size: int,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, List[int]]:
    """
    Create a baseline using CLIP score filtering.
    
    Args:
        dataset: Original dataset
        clip_scores: Array of CLIP scores for each sample
        target_size: Target number of samples
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (dataloader, sampled_indices)
    """
    logger.info(f"Creating CLIP score filtering baseline with {target_size} samples...")
    
    # Sort indices by CLIP scores
    sorted_indices = np.argsort(-clip_scores)  # Negative for descending order
    
    # Take top indices to reach target size
    sampled_indices = sorted_indices[:min(target_size, len(sorted_indices))].tolist()
    
    # Create dataloader
    from data_loader import get_dataloader_from_indices
    dataloader = get_dataloader_from_indices(
        dataset,
        sampled_indices,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return dataloader, sampled_indices

def individual_influence_baseline(
    dataset: Dataset,
    sample_influence_scores: Dict[int, float],
    target_size: int,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, List[int]]:
    """
    Create a baseline using individual sample influence scores.
    
    Args:
        dataset: Original dataset
        sample_influence_scores: Dictionary mapping sample indices to influence scores
        target_size: Target number of samples
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (dataloader, sampled_indices)
    """
    logger.info(f"Creating individual influence baseline with {target_size} samples...")
    
    # Sort indices by influence scores
    sorted_indices = sorted(sample_influence_scores.keys(), 
                           key=lambda idx: sample_influence_scores.get(idx, -float('inf')),
                           reverse=True)
    
    # Take top indices to reach target size
    sampled_indices = sorted_indices[:min(target_size, len(sorted_indices))]
    
    # Create dataloader
    from data_loader import get_dataloader_from_indices
    dataloader = get_dataloader_from_indices(
        dataset,
        sampled_indices,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return dataloader, sampled_indices
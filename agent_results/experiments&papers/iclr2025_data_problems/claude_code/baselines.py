import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import DataLoader, Dataset, Subset
import logging
from utils import Timer
import random
from collections import defaultdict
import json
from data_loader import get_dataloader_from_indices

logger = logging.getLogger("influence_space")

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
    dataloader = get_dataloader_from_indices(
        dataset,
        sampled_indices,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    logger.info(f"Created random sampling baseline with {len(sampled_indices)} samples")
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
    dataloader = get_dataloader_from_indices(
        dataset,
        sampled_indices,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    logger.info(f"Created CLIP score filtering baseline with {len(sampled_indices)} samples")
    return dataloader, sampled_indices

def individual_influence_baseline(
    dataset: Dataset,
    sample_influence_scores: Dict[int, float],
    target_size: int,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, List[int]]:
    """
    Create a baseline using individual sample influence scores (DataInf-style).
    
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
    dataloader = get_dataloader_from_indices(
        dataset,
        sampled_indices,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    logger.info(f"Created individual influence baseline with {len(sampled_indices)} samples")
    return dataloader, sampled_indices

def compute_individual_influence_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    val_gradient: torch.Tensor,
    device: torch.device,
    max_samples: int = 1000
) -> Dict[int, float]:
    """
    Compute influence scores for individual samples (DataInf-style).
    
    Args:
        model: Trained model
        dataloader: DataLoader containing samples
        val_gradient: Validation gradient vector
        device: Device to run computation on
        max_samples: Maximum number of samples to compute influence for
        
    Returns:
        Dictionary mapping sample indices to influence scores
    """
    from influence_estimation import compute_sample_gradients, contrastive_loss
    
    logger.info(f"Computing individual influence scores for up to {max_samples} samples...")
    
    # Compute gradients for individual samples
    sample_gradients = compute_sample_gradients(
        model=model,
        dataloader=dataloader,
        device=device,
        batch_size=1,
        max_samples=max_samples
    )
    
    # Compute influence scores using dot product
    influence_scores = {}
    for idx, gradient in sample_gradients.items():
        # Simple dot product for efficiency (approximating DataInf)
        score = -torch.dot(val_gradient, gradient).item()
        influence_scores[idx] = score
    
    logger.info(f"Computed influence scores for {len(influence_scores)} samples")
    return influence_scores

def diversity_sampling_baseline(
    dataset: Dataset,
    embeddings: np.ndarray,
    target_size: int,
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, List[int]]:
    """
    Create a baseline using diversity-based sampling (greedy farthest-point sampling).
    
    Args:
        dataset: Original dataset
        embeddings: Matrix of embeddings for each sample
        target_size: Target number of samples
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for data loading
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (dataloader, sampled_indices)
    """
    logger.info(f"Creating diversity sampling baseline with {target_size} samples...")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.maximum(norms, 1e-8)
    
    # Initialize with a random index
    all_indices = list(range(len(dataset)))
    sampled_indices = [np.random.choice(all_indices)]
    remaining_indices = list(set(all_indices) - set(sampled_indices))
    
    # Iteratively add samples that are farthest from the currently selected set
    pbar = tqdm(total=min(target_size, len(all_indices)) - 1, desc="Sampling diverse points")
    while len(sampled_indices) < min(target_size, len(all_indices)):
        # Compute distances to already sampled points
        sampled_embeddings = normalized_embeddings[sampled_indices]
        remaining_embeddings = normalized_embeddings[remaining_indices]
        
        # Compute cosine similarity (dot product of normalized vectors)
        similarities = np.dot(remaining_embeddings, sampled_embeddings.T)
        
        # For each remaining point, get its maximum similarity to any sampled point
        max_similarities = np.max(similarities, axis=1)
        
        # Choose the point with the lowest maximum similarity (most dissimilar)
        idx = np.argmin(max_similarities)
        next_index = remaining_indices[idx]
        
        # Update sampled and remaining indices
        sampled_indices.append(next_index)
        remaining_indices.remove(next_index)
        
        pbar.update(1)
    
    pbar.close()
    
    # Create dataloader
    dataloader = get_dataloader_from_indices(
        dataset,
        sampled_indices,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    logger.info(f"Created diversity sampling baseline with {len(sampled_indices)} samples")
    return dataloader, sampled_indices

def coreset_selection_baseline(
    dataset: Dataset,
    embeddings: np.ndarray,
    target_size: int,
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, List[int]]:
    """
    Create a baseline using coreset selection (K-means++ initialization).
    
    Args:
        dataset: Original dataset
        embeddings: Matrix of embeddings for each sample
        target_size: Target number of samples
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for data loading
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (dataloader, sampled_indices)
    """
    logger.info(f"Creating coreset selection baseline with {target_size} samples...")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.maximum(norms, 1e-8)
    
    # Initialize with a random index
    all_indices = list(range(len(dataset)))
    sampled_indices = [np.random.choice(all_indices)]
    remaining_indices = list(set(all_indices) - set(sampled_indices))
    
    # Iteratively add samples using K-means++ initialization strategy
    pbar = tqdm(total=min(target_size, len(all_indices)) - 1, desc="Selecting coreset points")
    while len(sampled_indices) < min(target_size, len(all_indices)):
        # Compute distances to already sampled points
        sampled_embeddings = normalized_embeddings[sampled_indices]
        remaining_embeddings = normalized_embeddings[remaining_indices]
        
        # Compute squared distances (2 - 2*cosine_similarity for normalized vectors)
        similarities = np.dot(remaining_embeddings, sampled_embeddings.T)
        max_similarities = np.max(similarities, axis=1)
        squared_distances = 2.0 - 2.0 * max_similarities
        
        # Choose the next point with probability proportional to squared distance
        probs = squared_distances / np.sum(squared_distances)
        idx = np.random.choice(len(remaining_indices), p=probs)
        next_index = remaining_indices[idx]
        
        # Update sampled and remaining indices
        sampled_indices.append(next_index)
        remaining_indices.remove(next_index)
        
        pbar.update(1)
    
    pbar.close()
    
    # Create dataloader
    dataloader = get_dataloader_from_indices(
        dataset,
        sampled_indices,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    logger.info(f"Created coreset selection baseline with {len(sampled_indices)} samples")
    return dataloader, sampled_indices

def run_all_baselines(
    dataset: Dataset,
    embeddings: np.ndarray, 
    clip_scores: np.ndarray,
    target_size: int,
    model: Optional[torch.nn.Module] = None,
    val_gradient: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: int = 42,
    output_dir: str = "./baselines"
) -> Dict[str, Tuple[DataLoader, List[int]]]:
    """
    Run all baseline methods for comparison.
    
    Args:
        dataset: Original dataset
        embeddings: Matrix of embeddings for each sample
        clip_scores: Array of CLIP scores for each sample
        target_size: Target number of samples
        model: Optional trained model for influence-based baselines
        val_gradient: Optional validation gradient for influence-based baselines
        device: Device for computation
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        random_seed: Random seed for reproducibility
        output_dir: Directory to save baseline results
        
    Returns:
        Dictionary mapping method names to tuples of (dataloader, sampled_indices)
    """
    logger.info("Running all baseline methods...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store baseline results
    baseline_results = {}
    
    # 1. Random sampling
    random_dataloader, random_indices = random_sampling_baseline(
        dataset=dataset,
        target_size=target_size,
        batch_size=batch_size,
        num_workers=num_workers,
        random_seed=random_seed
    )
    
    baseline_results["Random Sampling"] = (random_dataloader, random_indices)
    
    # Save random sampling results
    np.save(os.path.join(output_dir, "random_indices.npy"), np.array(random_indices))
    
    # 2. CLIP score filtering
    clip_dataloader, clip_indices = clip_score_filtering_baseline(
        dataset=dataset,
        clip_scores=clip_scores,
        target_size=target_size,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    baseline_results["CLIP Score Filtering"] = (clip_dataloader, clip_indices)
    
    # Save CLIP score filtering results
    np.save(os.path.join(output_dir, "clip_score_filtering_indices.npy"), np.array(clip_indices))
    
    # 3. Diversity sampling
    diversity_dataloader, diversity_indices = diversity_sampling_baseline(
        dataset=dataset,
        embeddings=embeddings,
        target_size=target_size,
        batch_size=batch_size,
        num_workers=num_workers,
        random_seed=random_seed
    )
    
    baseline_results["Diversity Sampling"] = (diversity_dataloader, diversity_indices)
    
    # Save diversity sampling results
    np.save(os.path.join(output_dir, "diversity_sampling_indices.npy"), np.array(diversity_indices))
    
    # 4. Coreset selection
    coreset_dataloader, coreset_indices = coreset_selection_baseline(
        dataset=dataset,
        embeddings=embeddings,
        target_size=target_size,
        batch_size=batch_size,
        num_workers=num_workers,
        random_seed=random_seed
    )
    
    baseline_results["Coreset Selection"] = (coreset_dataloader, coreset_indices)
    
    # Save coreset selection results
    np.save(os.path.join(output_dir, "coreset_selection_indices.npy"), np.array(coreset_indices))
    
    # 5. Individual influence (if model and validation gradient provided)
    if model is not None and val_gradient is not None and device is not None:
        # Create a dataloader for the dataset
        full_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Compute individual influence scores
        individual_influence_scores = compute_individual_influence_scores(
            model=model,
            dataloader=full_dataloader,
            val_gradient=val_gradient,
            device=device,
            max_samples=min(5000, len(dataset))  # Limit for computational efficiency
        )
        
        # Save influence scores
        influence_dict = {str(k): float(v) for k, v in individual_influence_scores.items()}
        with open(os.path.join(output_dir, "individual_influence_scores.json"), "w") as f:
            json.dump(influence_dict, f)
        
        # Create baseline using these scores
        individual_dataloader, individual_indices = individual_influence_baseline(
            dataset=dataset,
            sample_influence_scores=individual_influence_scores,
            target_size=target_size,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        baseline_results["Individual Influence"] = (individual_dataloader, individual_indices)
        
        # Save individual influence results
        np.save(os.path.join(output_dir, "individual_influence_indices.npy"), np.array(individual_indices))
    
    logger.info(f"Completed running {len(baseline_results)} baseline methods")
    return baseline_results
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import logging

logger = logging.getLogger(__name__)

class TokenImportanceScorer:
    """Computes token importance scores based on attention weights."""
    
    def __init__(self, 
                 num_layers: int, 
                 num_heads: int, 
                 lookback_window: int = 256):
        """
        Initialize token importance scorer.
        
        Args:
            num_layers: Number of layers in the transformer model
            num_heads: Number of attention heads per layer
            lookback_window: Number of recent positions to consider (Δ in the paper)
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.lookback_window = lookback_window
        
    def compute_importance_scores(self, 
                                  attention_matrices: Dict[Tuple[int, int], torch.Tensor],
                                  seq_len: int) -> torch.Tensor:
        """
        Compute importance scores for each token based on attention weights.
        
        Args:
            attention_matrices: Dictionary mapping (layer, head) to attention matrices
                                of shape (batch_size, seq_len, seq_len)
            seq_len: Current sequence length
        
        Returns:
            importance_scores: Tensor of shape (batch_size, seq_len) with importance score for each token
        """
        batch_size = attention_matrices[(0, 0)].shape[0]
        importance_scores = torch.zeros((batch_size, seq_len), device=attention_matrices[(0, 0)].device)
        
        # Consider only the last lookback_window positions as queries
        lookback_start = max(0, seq_len - self.lookback_window)
        
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                if (layer, head) in attention_matrices:
                    # Get attention weights: shape (batch_size, seq_len, seq_len)
                    attn_weights = attention_matrices[(layer, head)]
                    
                    # Sum the attention each token receives from recent queries
                    # shape: (batch_size, seq_len)
                    token_importance = attn_weights[:, lookback_start:seq_len, :seq_len].sum(dim=1)
                    importance_scores += token_importance
        
        return importance_scores


class OnlineKMeansClustering:
    """Performs online k-means clustering for KV cache compression."""
    
    def __init__(self, 
                 num_clusters: int, 
                 dim: int, 
                 learning_rate: float = 0.01,
                 device: torch.device = None):
        """
        Initialize online k-means clustering.
        
        Args:
            num_clusters: Number of cluster centroids (K in the paper)
            dim: Dimension of vectors to cluster
            learning_rate: Learning rate for centroid updates (η in the paper)
            device: Device to use for computation
        """
        self.num_clusters = num_clusters
        self.dim = dim
        self.learning_rate = learning_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize centroids randomly
        self.initialized = False
        self.key_centroids = None
        self.value_centroids = None
        
        # Count vectors assigned to each cluster, for weighted averaging
        self.cluster_counts = torch.zeros(num_clusters, device=self.device)
    
    def initialize(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Initialize centroids using the first batch of data.
        
        Args:
            keys: Key vectors of shape (..., dim)
            values: Value vectors of shape (..., dim)
        """
        if self.initialized:
            return
        
        # Flatten to 2D
        keys_flat = keys.reshape(-1, self.dim)
        values_flat = values.reshape(-1, self.dim)
        
        # If we have fewer vectors than clusters, duplicate them
        if keys_flat.shape[0] < self.num_clusters:
            repeats = (self.num_clusters + keys_flat.shape[0] - 1) // keys_flat.shape[0]
            keys_flat = keys_flat.repeat(repeats, 1)
            values_flat = values_flat.repeat(repeats, 1)
        
        # Select random vectors as initial centroids
        idx = torch.randperm(keys_flat.shape[0], device=self.device)[:self.num_clusters]
        self.key_centroids = keys_flat[idx].clone()
        self.value_centroids = values_flat[idx].clone()
        
        self.cluster_counts = torch.ones(self.num_clusters, device=self.device)
        self.initialized = True
    
    def cluster(self, 
                keys: torch.Tensor, 
                values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assign keys and values to clusters and update centroids.
        
        Args:
            keys: Key vectors of shape (..., dim)
            values: Value vectors of shape (..., dim)
            
        Returns:
            cluster_idx: Cluster assignments for each vector
            key_centroids: Updated key centroids
            value_centroids: Updated value centroids
        """
        # Initialize centroids if not done yet
        self.initialize(keys, values)
        
        # Flatten input tensors to 2D
        original_shape = keys.shape[:-1]
        keys_flat = keys.reshape(-1, self.dim)
        values_flat = values.reshape(-1, self.dim)
        
        # Compute distances to centroids
        # Expand for broadcasting: (num_vectors, 1, dim) - (1, num_clusters, dim)
        distances = torch.cdist(keys_flat, self.key_centroids)
        
        # Assign each vector to nearest centroid
        cluster_idx = torch.argmin(distances, dim=1)
        
        # Update centroids using online learning
        for k in range(self.num_clusters):
            mask = (cluster_idx == k)
            if mask.sum() > 0:
                # Get vectors assigned to this cluster
                cluster_keys = keys_flat[mask]
                cluster_values = values_flat[mask]
                
                # Compute mean of assigned vectors
                mean_key = cluster_keys.mean(dim=0)
                mean_value = cluster_values.mean(dim=0)
                
                # Update centroid with learning rate
                self.key_centroids[k] = (1 - self.learning_rate) * self.key_centroids[k] + self.learning_rate * mean_key
                self.value_centroids[k] = (1 - self.learning_rate) * self.value_centroids[k] + self.learning_rate * mean_value
                
                # Update count (for weighted updates)
                self.cluster_counts[k] += mask.sum()
        
        # Reshape cluster_idx back to original shape
        cluster_idx = cluster_idx.reshape(original_shape)
        
        return cluster_idx, self.key_centroids, self.value_centroids


class KVCacheCompressor(nn.Module):
    """
    KV Cache compression module that implements pruning and clustering strategies.
    """
    
    def __init__(self, 
                 num_layers: int, 
                 num_heads: int, 
                 head_dim: int,
                 max_cache_size: int,
                 num_clusters: int,
                 pruning_interval: int = 512,
                 lookback_window: int = 256,
                 kmeans_learning_rate: float = 0.01,
                 device: torch.device = None):
        """
        Initialize KV cache compressor.
        
        Args:
            num_layers: Number of layers in the transformer model
            num_heads: Number of attention heads per layer
            head_dim: Dimension of each attention head
            max_cache_size: Maximum number of KV pairs to retain after pruning (B in the paper)
            num_clusters: Number of cluster centroids for low-rank summarization (K in the paper)
            pruning_interval: Interval (in tokens) between pruning operations (P in the paper)
            lookback_window: Number of recent positions to consider for importance (Δ in the paper)
            kmeans_learning_rate: Learning rate for online k-means updates (η in the paper)
            device: Device to use for computation
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_cache_size = max_cache_size
        self.num_clusters = num_clusters
        self.pruning_interval = pruning_interval
        self.lookback_window = lookback_window
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize token importance scorer
        self.importance_scorer = TokenImportanceScorer(
            num_layers=num_layers,
            num_heads=num_heads,
            lookback_window=lookback_window
        )
        
        # Initialize online k-means clustering
        self.online_kmeans = OnlineKMeansClustering(
            num_clusters=num_clusters,
            dim=head_dim,
            learning_rate=kmeans_learning_rate,
            device=self.device
        )
        
        # Maintain token positions and ids for mapping between original and compressed indices
        self.token_positions = None
        self.current_seq_len = 0
        self.steps_since_last_pruning = 0
        
        # Statistics
        self.stats = {
            'pruning_count': 0,
            'avg_pruning_time': 0,
            'avg_tokens_removed': 0,
            'compression_ratio': 0,
        }
    
    def forward(self, 
                key_cache: Dict[int, torch.Tensor],
                value_cache: Dict[int, torch.Tensor],
                attention_matrices: Dict[Tuple[int, int], torch.Tensor] = None,
                do_pruning: bool = False) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[str, Any]]:
        """
        Compress KV cache by pruning and clustering.
        
        Args:
            key_cache: Dictionary mapping layer to key cache tensors of shape (batch_size, seq_len, num_heads, head_dim)
            value_cache: Dictionary mapping layer to value cache tensors of shape (batch_size, seq_len, num_heads, head_dim)
            attention_matrices: Dictionary mapping (layer, head) to attention matrices 
                               of shape (batch_size, seq_len, seq_len)
            do_pruning: Whether to perform pruning, regardless of interval
            
        Returns:
            compressed_key_cache: Compressed key cache
            compressed_value_cache: Compressed value cache
            stats: Dictionary of compression statistics
        """
        # Update sequence length and steps count
        batch_size = next(iter(key_cache.values())).shape[0]
        seq_len = next(iter(key_cache.values())).shape[1]
        self.current_seq_len = seq_len
        self.steps_since_last_pruning += 1
        
        # Initialize token positions for tracking original positions
        if self.token_positions is None:
            self.token_positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        elif self.token_positions.shape[1] < seq_len:
            # Append new positions for new tokens
            new_positions = torch.arange(
                self.token_positions.shape[1], seq_len, device=self.device
            ).unsqueeze(0).expand(batch_size, -1)
            self.token_positions = torch.cat([self.token_positions, new_positions], dim=1)
        
        # Check if pruning should be performed
        should_prune = do_pruning or (
            self.steps_since_last_pruning >= self.pruning_interval and 
            seq_len > self.max_cache_size
        )
        
        # Skip if no pruning needed
        if not should_prune:
            return key_cache, value_cache, self.stats
        
        # Reset counter
        self.steps_since_last_pruning = 0
        
        # Measure time for statistics
        start_time = time.time()
        
        # Compute token importance scores
        if attention_matrices is None:
            # If attention matrices not provided, we can't compute importance scores
            logger.warning("Attention matrices not provided, skipping importance-based pruning")
            return key_cache, value_cache, self.stats
        
        importance_scores = self.importance_scorer.compute_importance_scores(
            attention_matrices=attention_matrices,
            seq_len=seq_len
        )
        
        # Sort tokens by importance scores and select top max_cache_size tokens
        _, sorted_indices = torch.sort(importance_scores, dim=1, descending=True)
        keep_indices = sorted_indices[:, :self.max_cache_size]
        
        # Sort indices for efficient gather operation
        keep_indices, _ = torch.sort(keep_indices, dim=1)
        
        # Track token positions
        self.token_positions = torch.gather(self.token_positions, 1, keep_indices)
        
        # Prune KV cache
        compressed_key_cache = {}
        compressed_value_cache = {}
        
        for layer in key_cache.keys():
            layer_key_cache = key_cache[layer]  # (batch_size, seq_len, num_heads, head_dim)
            layer_value_cache = value_cache[layer]  # (batch_size, seq_len, num_heads, head_dim)
            
            # Gather only important tokens
            # Expand keep_indices to gather from (seq_len, num_heads, head_dim) dimension
            expanded_indices = keep_indices.unsqueeze(-1).unsqueeze(-1).expand(
                batch_size, self.max_cache_size, self.num_heads, self.head_dim
            )
            
            compressed_keys = torch.gather(layer_key_cache, 1, expanded_indices)
            compressed_values = torch.gather(layer_value_cache, 1, expanded_indices)
            
            # Apply online k-means clustering if num_clusters < max_cache_size
            if self.num_clusters < self.max_cache_size:
                # Reshape for clustering: (batch_size, seq_len, num_heads, head_dim) 
                # -> (batch_size * seq_len * num_heads, head_dim)
                batch_seq_heads = batch_size * self.max_cache_size * self.num_heads
                keys_flat = compressed_keys.reshape(batch_seq_heads, self.head_dim)
                values_flat = compressed_values.reshape(batch_seq_heads, self.head_dim)
                
                # Perform clustering
                _, key_centroids, value_centroids = self.online_kmeans.cluster(
                    keys=keys_flat, 
                    values=values_flat
                )
                
                # Use centroids as the compressed cache
                # Expand centroids for broadcasting: (num_clusters, head_dim) 
                # -> (batch_size, num_clusters, num_heads, head_dim)
                expanded_key_centroids = key_centroids.unsqueeze(0).unsqueeze(2).expand(
                    batch_size, self.num_clusters, self.num_heads, self.head_dim
                )
                expanded_value_centroids = value_centroids.unsqueeze(0).unsqueeze(2).expand(
                    batch_size, self.num_clusters, self.num_heads, self.head_dim
                )
                
                compressed_key_cache[layer] = expanded_key_centroids
                compressed_value_cache[layer] = expanded_value_centroids
            else:
                # If not using clustering, just keep the pruned cache
                compressed_key_cache[layer] = compressed_keys
                compressed_value_cache[layer] = compressed_values
        
        # Update statistics
        elapsed_time = time.time() - start_time
        tokens_removed = seq_len - min(self.max_cache_size, seq_len)
        compression_ratio = seq_len / min(self.max_cache_size, seq_len)
        
        self.stats['pruning_count'] += 1
        self.stats['avg_pruning_time'] = (
            (self.stats['avg_pruning_time'] * (self.stats['pruning_count'] - 1) + elapsed_time) / 
            self.stats['pruning_count'] 
        )
        self.stats['avg_tokens_removed'] = (
            (self.stats['avg_tokens_removed'] * (self.stats['pruning_count'] - 1) + tokens_removed) / 
            self.stats['pruning_count']
        )
        self.stats['compression_ratio'] = compression_ratio
        
        # Log statistics
        logger.info(f"KV Cache compression: removed {tokens_removed} tokens, "
                    f"compression ratio: {compression_ratio:.2f}, "
                    f"time: {elapsed_time:.4f}s")
        
        return compressed_key_cache, compressed_value_cache, self.stats


class DistillationLoss(nn.Module):
    """
    Implements distillation loss between teacher (full KV cache) 
    and student (compressed KV cache) models.
    """
    
    def __init__(self, temperature: float = 2.0):
        """
        Initialize distillation loss.
        
        Args:
            temperature: Temperature for softmax (T in the paper)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, 
                teacher_logits: torch.Tensor, 
                student_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between teacher and student distributions.
        
        Args:
            teacher_logits: Logits from teacher model (full KV cache)
            student_logits: Logits from student model (compressed KV cache)
            
        Returns:
            loss: KL divergence loss
        """
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL divergence
        kl_div = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='batchmean',
            log_target=False
        )
        
        # Scale by temperature squared (as in original Knowledge Distillation)
        return kl_div * (self.temperature ** 2)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time

class MatrixEntropy:
    """Utility to compute matrix entropy for uncertainty estimation."""
    
    @staticmethod
    def compute_entropy(matrix: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute entropy of a probability matrix along specified dimension.
        
        Args:
            matrix: Tensor of probabilities
            dim: Dimension along which to compute entropy
            
        Returns:
            entropy: Tensor of entropy values
        """
        # Ensure matrix contains valid probabilities
        eps = 1e-10
        matrix = torch.clamp(matrix, min=eps, max=1.0)
        
        # Normalize if not already probabilities summing to 1
        if not torch.allclose(matrix.sum(dim=dim, keepdim=True), 
                              torch.ones_like(matrix.sum(dim=dim, keepdim=True)),
                              rtol=1e-3):
            matrix = F.normalize(matrix, p=1, dim=dim)
        
        # Compute entropy: -∑(p * log(p))
        entropy = -torch.sum(matrix * torch.log2(matrix), dim=dim)
        
        return entropy
    
    @staticmethod
    def compute_attention_entropy(attention_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of attention matrix.
        
        Args:
            attention_matrix: Attention weights tensor (batch_size, seq_len, seq_len)
            
        Returns:
            entropy: Tensor of entropy values (batch_size, seq_len)
        """
        # Compute entropy along the key dimension
        return MatrixEntropy.compute_entropy(attention_matrix, dim=-1)


class UNCompCompressor(nn.Module):
    """
    UNComp: Uncertainty-Aware Long-Context Compressor for Efficient LLM Inference
    
    Implements uncertainty-aware compression by estimating model uncertainty 
    across layers and heads at the token level.
    """
    
    def __init__(self, 
                 num_layers: int, 
                 num_heads: int, 
                 head_dim: int,
                 uncertainty_threshold: float = 0.5,
                 target_compression_ratio: float = 0.3,
                 update_interval: int = 256,
                 device: torch.device = None):
        """
        Initialize UNComp compressor.
        
        Args:
            num_layers: Number of layers in the transformer model
            num_heads: Number of attention heads per layer
            head_dim: Dimension of each attention head
            uncertainty_threshold: Threshold for considering a token as uncertain
            target_compression_ratio: Target ratio for compression (lower = more compression)
            update_interval: Interval between KV cache updates
            device: Device to use for computation
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.uncertainty_threshold = uncertainty_threshold
        self.target_compression_ratio = target_compression_ratio
        self.update_interval = update_interval
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tracking
        self.token_positions = None
        self.steps_since_last_update = 0
        
        # Layer-specific compression settings
        self.layer_compression_ratios = self._compute_layer_compression_ratios()
        
        # Statistics
        self.stats = {
            'update_count': 0,
            'avg_update_time': 0,
            'avg_tokens_removed': 0,
            'compression_ratio': 0,
        }
    
    def _compute_layer_compression_ratios(self) -> Dict[int, float]:
        """
        Compute compression ratios for each layer.
        Early layers generally need less compression than later layers.
        
        Returns:
            layer_compression_ratios: Dictionary mapping layer index to compression ratio
        """
        # Simple linear scaling: later layers get more compression
        ratios = {}
        for layer in range(self.num_layers):
            # Scale from 0.5*target to 1.5*target from first to last layer
            scale = 0.5 + layer / (self.num_layers - 1) if self.num_layers > 1 else 1.0
            ratios[layer] = min(0.8, max(0.1, self.target_compression_ratio * scale))
        
        return ratios
    
    def compute_uncertainty_scores(self, 
                                  attention_matrices: Dict[Tuple[int, int], torch.Tensor],
                                  layer: int) -> torch.Tensor:
        """
        Compute uncertainty scores for tokens in a specific layer.
        
        Args:
            attention_matrices: Dictionary mapping (layer, head) to attention matrices
            layer: Layer index
            
        Returns:
            uncertainty_scores: Tensor of shape (batch_size, seq_len) with uncertainty score for each token
        """
        batch_size = next(iter(attention_matrices.values())).shape[0]
        seq_len = next(iter(attention_matrices.values())).shape[1]
        
        # Initialize uncertainty scores
        uncertainty_scores = torch.zeros((batch_size, seq_len), device=self.device)
        
        # Count valid heads for normalization
        valid_head_count = 0
        
        # Compute entropy for each head in this layer
        for head in range(self.num_heads):
            if (layer, head) in attention_matrices:
                valid_head_count += 1
                
                # Get attention weights: (batch_size, seq_len, seq_len)
                attn_weights = attention_matrices[(layer, head)]
                
                # Compute entropy along key dimension: (batch_size, seq_len)
                entropy = MatrixEntropy.compute_attention_entropy(attn_weights)
                
                # Add to uncertainty scores
                uncertainty_scores += entropy
        
        # Normalize by valid head count
        if valid_head_count > 0:
            uncertainty_scores /= valid_head_count
        
        # Normalize to [0, 1]
        max_entropy = torch.log2(torch.tensor(seq_len, dtype=torch.float, device=self.device))
        uncertainty_scores /= max_entropy
        
        return uncertainty_scores
    
    def compress_layer_kv_cache(self,
                               key_cache: torch.Tensor,
                               value_cache: torch.Tensor,
                               uncertainty_scores: torch.Tensor,
                               layer_compression_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress KV cache for a specific layer based on uncertainty scores and target ratio.
        
        Args:
            key_cache: Key cache tensor of shape (batch_size, seq_len, num_heads, head_dim)
            value_cache: Value cache tensor of shape (batch_size, seq_len, num_heads, head_dim)
            uncertainty_scores: Uncertainty scores of shape (batch_size, seq_len)
            layer_compression_ratio: Target ratio for this layer
            
        Returns:
            compressed_key_cache: Compressed key cache
            compressed_value_cache: Compressed value cache
            keep_indices: Indices of tokens that were kept
        """
        batch_size, seq_len = uncertainty_scores.shape
        
        # Target number of tokens to keep
        target_tokens = max(int(seq_len * layer_compression_ratio), 256)
        
        # Split tokens into certain and uncertain based on threshold
        uncertain_mask = uncertainty_scores > self.uncertainty_threshold
        
        # Always keep uncertain tokens
        uncertain_indices = torch.where(uncertain_mask)
        
        # Create tensor of indices to keep
        keep_indices_list = []
        for b in range(batch_size):
            # Get uncertain tokens for this batch
            batch_uncertain = uncertain_indices[0] == b
            uncertain_idx = uncertain_indices[1][batch_uncertain]
            
            # Count uncertain tokens
            num_uncertain = uncertain_idx.size(0)
            
            # Determine how many certain tokens to keep
            num_certain_to_keep = max(0, target_tokens - num_uncertain)
            
            # If we need to keep certain tokens
            if num_certain_to_keep > 0:
                # Get indices of certain tokens
                certain_mask = ~uncertain_mask[b]
                certain_idx = torch.where(certain_mask)[0]
                
                # If we have more certain tokens than needed, sample them
                if certain_idx.size(0) > num_certain_to_keep:
                    # Sort by uncertainty (higher gets kept)
                    certain_scores = uncertainty_scores[b, certain_idx]
                    _, sorted_idx = torch.sort(certain_scores, descending=True)
                    certain_idx = certain_idx[sorted_idx[:num_certain_to_keep]]
                
                # Combine uncertain and certain indices
                batch_keep_indices = torch.cat([uncertain_idx, certain_idx])
            else:
                # Only keep uncertain tokens (if too many, prioritize highest uncertainty)
                if num_uncertain > target_tokens:
                    scores = uncertainty_scores[b, uncertain_idx]
                    _, sorted_idx = torch.sort(scores, descending=True)
                    uncertain_idx = uncertain_idx[sorted_idx[:target_tokens]]
                batch_keep_indices = uncertain_idx
            
            # Sort indices for efficient gathering
            batch_keep_indices, _ = torch.sort(batch_keep_indices)
            
            # Store batch indices
            keep_indices_list.append(batch_keep_indices)
        
        # Pad indices to same length
        max_indices = max(indices.size(0) for indices in keep_indices_list)
        padded_indices = torch.zeros((batch_size, max_indices), dtype=torch.long, device=self.device)
        
        for b, indices in enumerate(keep_indices_list):
            padded_indices[b, :indices.size(0)] = indices
        
        # Create mask of valid indices (non-padding)
        valid_mask = torch.zeros_like(padded_indices, dtype=torch.bool)
        for b, indices in enumerate(keep_indices_list):
            valid_mask[b, :indices.size(0)] = True
        
        # Gather important tokens from key and value caches
        # Expand indices for gathering from the right dimensions
        expanded_indices = padded_indices.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, max_indices, self.num_heads, self.head_dim
        )
        
        # Initialize compressed caches
        compressed_keys = torch.zeros(
            (batch_size, max_indices, self.num_heads, self.head_dim),
            device=self.device
        )
        compressed_values = torch.zeros(
            (batch_size, max_indices, self.num_heads, self.head_dim),
            device=self.device
        )
        
        # Gather for each batch item individually using valid indices
        for b in range(batch_size):
            valid_indices = padded_indices[b, valid_mask[b]]
            if valid_indices.size(0) > 0:
                # Expand for gathering
                expanded_valid = valid_indices.unsqueeze(-1).unsqueeze(-1).expand(
                    valid_indices.size(0), self.num_heads, self.head_dim
                )
                
                # Gather using valid indices
                compressed_keys[b, :valid_indices.size(0)] = torch.gather(
                    key_cache[b], 0, expanded_valid
                )
                compressed_values[b, :valid_indices.size(0)] = torch.gather(
                    value_cache[b], 0, expanded_valid
                )
        
        return compressed_keys, compressed_values, padded_indices
    
    def forward(self, 
                key_cache: Dict[int, torch.Tensor],
                value_cache: Dict[int, torch.Tensor],
                attention_matrices: Dict[Tuple[int, int], torch.Tensor] = None,
                do_update: bool = False) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[str, Any]]:
        """
        Compress KV cache using uncertainty-aware compression.
        
        Args:
            key_cache: Dictionary mapping layer to key cache tensors
            value_cache: Dictionary mapping layer to value cache tensors
            attention_matrices: Dictionary mapping (layer, head) to attention matrices
            do_update: Whether to force an update, regardless of interval
            
        Returns:
            compressed_key_cache: Compressed key cache
            compressed_value_cache: Compressed value cache
            stats: Dictionary of compression statistics
        """
        # Skip if attention matrices not provided
        if attention_matrices is None:
            return key_cache, value_cache, self.stats
        
        # Get sequence length
        seq_len = next(iter(key_cache.values())).shape[1]
        batch_size = next(iter(key_cache.values())).shape[0]
        
        # Update step counter
        self.steps_since_last_update += 1
        
        # Check if update should be performed
        should_update = do_update or (
            self.steps_since_last_update >= self.update_interval and seq_len > 512
        )
        
        # Skip if no update needed
        if not should_update:
            return key_cache, value_cache, self.stats
        
        # Reset counter
        self.steps_since_last_update = 0
        
        # Measure time for stats
        start_time = time.time()
        
        # Process each layer
        compressed_key_cache = {}
        compressed_value_cache = {}
        total_tokens_before = 0
        total_tokens_after = 0
        
        for layer in range(self.num_layers):
            if layer not in key_cache:
                continue
                
            layer_key_cache = key_cache[layer]
            layer_value_cache = value_cache[layer]
            
            # Compute uncertainty scores
            uncertainty_scores = self.compute_uncertainty_scores(
                attention_matrices=attention_matrices,
                layer=layer
            )
            
            # Compress layer KV cache
            compressed_keys, compressed_values, _ = self.compress_layer_kv_cache(
                key_cache=layer_key_cache,
                value_cache=layer_value_cache,
                uncertainty_scores=uncertainty_scores,
                layer_compression_ratio=self.layer_compression_ratios[layer]
            )
            
            # Store compressed caches
            compressed_key_cache[layer] = compressed_keys
            compressed_value_cache[layer] = compressed_values
            
            # Update token counts for stats
            total_tokens_before += layer_key_cache.shape[1]
            total_tokens_after += compressed_keys.shape[1]
        
        # Update statistics
        elapsed_time = time.time() - start_time
        avg_tokens_removed = total_tokens_before - total_tokens_after
        compression_ratio = total_tokens_before / (total_tokens_after + 1e-10)
        
        self.stats['update_count'] += 1
        self.stats['avg_update_time'] = (
            (self.stats['avg_update_time'] * (self.stats['update_count'] - 1) + elapsed_time) / 
            self.stats['update_count']
        )
        self.stats['avg_tokens_removed'] = (
            (self.stats['avg_tokens_removed'] * (self.stats['update_count'] - 1) + avg_tokens_removed) / 
            self.stats['update_count']
        )
        self.stats['compression_ratio'] = compression_ratio
        
        return compressed_key_cache, compressed_value_cache, self.stats
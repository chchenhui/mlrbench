import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time

class DynamicKVCompressor(nn.Module):
    """
    DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs
    
    Implements task-aware adaptive compression by adjusting the number of tokens
    retained at each layer.
    """
    
    def __init__(self, 
                 num_layers: int, 
                 num_heads: int, 
                 head_dim: int,
                 global_budget: int = 2048,
                 layer_budget_decay: float = 0.9,
                 token_scoring_window: int = 128,
                 update_interval: int = 256,
                 device: torch.device = None):
        """
        Initialize DynamicKV compressor.
        
        Args:
            num_layers: Number of layers in the transformer model
            num_heads: Number of attention heads per layer
            head_dim: Dimension of each attention head
            global_budget: Maximum number of tokens to retain globally
            layer_budget_decay: Decay factor for layer-specific budgets (deeper layers get smaller budgets)
            token_scoring_window: Window size for token scoring
            update_interval: Interval between KV cache updates
            device: Device to use for computation
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.global_budget = global_budget
        self.layer_budget_decay = layer_budget_decay
        self.token_scoring_window = token_scoring_window
        self.update_interval = update_interval
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Compute per-layer budgets with decay
        self.layer_budgets = self._compute_layer_budgets()
        
        # Token positions tracking
        self.token_positions = None
        self.steps_since_last_update = 0
        
        # Statistics
        self.stats = {
            'update_count': 0,
            'avg_update_time': 0,
            'avg_tokens_removed': 0,
            'compression_ratio': 0,
        }
    
    def _compute_layer_budgets(self) -> Dict[int, int]:
        """
        Compute token budgets for each layer based on decay factor.
        Deeper layers get smaller budgets.
        
        Returns:
            layer_budgets: Dictionary mapping layer index to token budget
        """
        # Compute raw budgets with exponential decay
        raw_budgets = [
            self.global_budget * (self.layer_budget_decay ** layer)
            for layer in range(self.num_layers)
        ]
        
        # Normalize to ensure sum equals global budget
        total_raw = sum(raw_budgets)
        normalized_budgets = [
            max(128, int(raw * self.global_budget / total_raw))
            for raw in raw_budgets
        ]
        
        # Ensure the sum doesn't exceed global budget
        while sum(normalized_budgets) > self.global_budget:
            # Find layer with largest budget
            max_idx = normalized_budgets.index(max(normalized_budgets))
            # Reduce it by 1
            normalized_budgets[max_idx] -= 1
        
        # Convert to dictionary
        return {layer: budget for layer, budget in enumerate(normalized_budgets)}
    
    def compute_token_scores(self, attention_matrices: Dict[Tuple[int, int], torch.Tensor], layer: int) -> torch.Tensor:
        """
        Compute importance scores for each token in a specific layer.
        
        Args:
            attention_matrices: Dictionary mapping (layer, head) to attention matrices
                               of shape (batch_size, seq_len, seq_len)
            layer: Layer index to compute scores for
            
        Returns:
            token_scores: Tensor of shape (batch_size, seq_len) with score for each token
        """
        seq_len = next(iter(attention_matrices.values())).shape[1]
        batch_size = next(iter(attention_matrices.values())).shape[0]
        scores = torch.zeros((batch_size, seq_len), device=self.device)
        
        # Consider a window of recent tokens as queries
        lookback_start = max(0, seq_len - self.token_scoring_window)
        
        # Sum attention weights across heads for this layer
        for head in range(self.num_heads):
            if (layer, head) in attention_matrices:
                attn_weights = attention_matrices[(layer, head)]
                # Sum attention each token receives from recent queries
                token_importance = attn_weights[:, lookback_start:seq_len, :seq_len].sum(dim=1)
                scores += token_importance
        
        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def compress_layer_kv_cache(self,
                               key_cache: torch.Tensor,
                               value_cache: torch.Tensor,
                               token_scores: torch.Tensor,
                               layer_budget: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress KV cache for a specific layer based on token scores.
        
        Args:
            key_cache: Key cache tensor of shape (batch_size, seq_len, num_heads, head_dim)
            value_cache: Value cache tensor of shape (batch_size, seq_len, num_heads, head_dim)
            token_scores: Importance scores of shape (batch_size, seq_len)
            layer_budget: Maximum number of tokens to retain for this layer
            
        Returns:
            compressed_key_cache: Compressed key cache
            compressed_value_cache: Compressed value cache
            keep_indices: Indices of tokens that were kept
        """
        batch_size, seq_len = token_scores.shape
        
        # Sort tokens by importance scores
        _, sorted_indices = torch.sort(token_scores, dim=1, descending=True)
        
        # Keep top tokens according to layer budget
        keep_top_k = min(layer_budget, seq_len)
        keep_indices = sorted_indices[:, :keep_top_k]
        
        # Sort indices for efficient gather operation
        keep_indices, _ = torch.sort(keep_indices, dim=1)
        
        # Gather important tokens from key and value caches
        # Expand indices for gathering from the right dimensions
        expanded_indices = keep_indices.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, keep_top_k, self.num_heads, self.head_dim
        )
        
        # Gather
        compressed_keys = torch.gather(key_cache, 1, expanded_indices)
        compressed_values = torch.gather(value_cache, 1, expanded_indices)
        
        return compressed_keys, compressed_values, keep_indices
    
    def forward(self, 
                key_cache: Dict[int, torch.Tensor],
                value_cache: Dict[int, torch.Tensor],
                attention_matrices: Dict[Tuple[int, int], torch.Tensor] = None,
                do_update: bool = False) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[str, Any]]:
        """
        Compress KV cache using task-aware adaptive token retention.
        
        Args:
            key_cache: Dictionary mapping layer to key cache tensors of shape (batch_size, seq_len, num_heads, head_dim)
            value_cache: Dictionary mapping layer to value cache tensors of shape (batch_size, seq_len, num_heads, head_dim)
            attention_matrices: Dictionary mapping (layer, head) to attention matrices 
                               of shape (batch_size, seq_len, seq_len)
            do_update: Whether to force an update, regardless of interval
            
        Returns:
            compressed_key_cache: Compressed key cache
            compressed_value_cache: Compressed value cache
            stats: Dictionary of compression statistics
        """
        # Get sequence length
        seq_len = next(iter(key_cache.values())).shape[1]
        batch_size = next(iter(key_cache.values())).shape[0]
        
        # Update step counter
        self.steps_since_last_update += 1
        
        # Initialize token positions for tracking
        if self.token_positions is None:
            self.token_positions = {
                layer: torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
                for layer in range(self.num_layers)
            }
        
        # Check if update should be performed
        should_update = do_update or (
            self.steps_since_last_update >= self.update_interval
        )
        
        # Skip if no update needed or attention matrices not provided
        if not should_update or attention_matrices is None:
            return key_cache, value_cache, self.stats
        
        # Reset counter
        self.steps_since_last_update = 0
        
        # Measure time for stats
        start_time = time.time()
        total_tokens_removed = 0
        
        # Compress each layer
        compressed_key_cache = {}
        compressed_value_cache = {}
        
        for layer in range(self.num_layers):
            if layer in key_cache:
                # Compute token scores for this layer
                token_scores = self.compute_token_scores(
                    attention_matrices=attention_matrices, 
                    layer=layer
                )
                
                # Compress layer KV cache
                compressed_keys, compressed_values, keep_indices = self.compress_layer_kv_cache(
                    key_cache=key_cache[layer],
                    value_cache=value_cache[layer],
                    token_scores=token_scores,
                    layer_budget=self.layer_budgets[layer]
                )
                
                # Update token positions for this layer
                self.token_positions[layer] = torch.gather(
                    self.token_positions[layer], 1, keep_indices
                )
                
                # Store compressed caches
                compressed_key_cache[layer] = compressed_keys
                compressed_value_cache[layer] = compressed_values
                
                # Track tokens removed
                tokens_removed = seq_len - compressed_keys.shape[1]
                total_tokens_removed += tokens_removed
        
        # Update statistics
        elapsed_time = time.time() - start_time
        avg_tokens_removed = total_tokens_removed / len(key_cache)
        compression_ratio = seq_len / (seq_len - (avg_tokens_removed / len(key_cache)))
        
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
    
    def update_layer_budgets(self, task_id: int = None, scale_factor: float = None):
        """
        Update layer budgets based on task or scale factor.
        
        Args:
            task_id: Optional task identifier to load predefined budgets
            scale_factor: Optional scale factor to adjust global budget
        """
        if scale_factor is not None:
            self.global_budget = max(128, int(self.global_budget * scale_factor))
        
        # Recompute layer budgets
        self.layer_budgets = self._compute_layer_budgets()
        
        # If task-specific adjustments are provided, apply them
        if task_id is not None:
            # Example task-specific adjustments (would be learned or predefined)
            if task_id == 0:  # e.g., summarization: deeper layers more important
                for layer in range(self.num_layers // 2, self.num_layers):
                    self.layer_budgets[layer] = int(self.layer_budgets[layer] * 1.2)
            elif task_id == 1:  # e.g., QA: middle layers more important
                mid_layer = self.num_layers // 2
                for layer in range(mid_layer - 2, mid_layer + 2):
                    if 0 <= layer < self.num_layers:
                        self.layer_budgets[layer] = int(self.layer_budgets[layer] * 1.2)
            
            # Normalize to ensure we don't exceed global budget
            current_sum = sum(self.layer_budgets.values())
            if current_sum > self.global_budget:
                scale = self.global_budget / current_sum
                for layer in self.layer_budgets:
                    self.layer_budgets[layer] = max(128, int(self.layer_budgets[layer] * scale))
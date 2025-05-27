"""
Baseline methods for KV cache management to compare against ATSKV.
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class FullKVCache:
    """
    Standard full KV cache without compression (baseline).
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device = None
    ):
        """
        Initialize the full KV cache.
        
        Args:
            hidden_size: Hidden size of the transformer model
            num_attention_heads: Number of attention heads in the transformer model
            num_hidden_layers: Number of hidden layers in the transformer model
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            device: Device to run the model on
        """
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Storage for KV cache
        self.kv_cache = {}
        
        # Metrics tracking
        self.metrics = {
            "memory_usage": [],
        }
    
    def reset_cache(self):
        """Reset the KV cache."""
        self.kv_cache = {}
    
    def update_kv_cache(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor
    ):
        """
        Update the KV cache.
        
        Args:
            layer_idx: Index of the current layer
            key_states: Key states [batch_size, num_heads, seq_len, head_dim]
            value_states: Value states [batch_size, num_heads, seq_len, head_dim]
        """
        layer_key = f"layer_{layer_idx}"
        
        if layer_key not in self.kv_cache:
            self.kv_cache[layer_key] = {}
        
        # Store in cache
        self.kv_cache[layer_key]['k'] = key_states
        self.kv_cache[layer_key]['v'] = value_states
    
    def get_cached_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the cached key-value tensors for a layer.
        
        Args:
            layer_idx: Index of the current layer
            
        Returns:
            Tuple of (key_states, value_states)
        """
        layer_key = f"layer_{layer_idx}"
        
        if layer_key in self.kv_cache:
            return (
                self.kv_cache[layer_key]['k'],
                self.kv_cache[layer_key]['v']
            )
        
        return None, None
    
    def compute_memory_usage(self) -> Dict[str, float]:
        """
        Compute the current memory usage of the KV cache.
        
        Returns:
            Dictionary containing memory usage statistics
        """
        total_elements = 0
        
        for layer_key, layer_cache in self.kv_cache.items():
            if 'k' in layer_cache and 'v' in layer_cache:
                # Count elements in key and value tensors
                k_elements = layer_cache['k'].numel()
                v_elements = layer_cache['v'].numel()
                
                total_elements += k_elements + v_elements
        
        # Compute memory in bytes (assuming float32)
        bytes_per_element = 4  # float32
        total_memory = total_elements * bytes_per_element
        
        # Convert to MB
        total_memory_mb = total_memory / (1024 * 1024)
        
        # Store metrics
        self.metrics["memory_usage"].append(total_memory_mb)
        
        return {
            "total_memory_mb": total_memory_mb,
        }


class SlidingWindowKVCache:
    """
    KV cache with a sliding window approach that keeps only the most recent tokens.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        head_dim: int,
        max_seq_len: int,
        window_size: int = 1024,
        device: torch.device = None
    ):
        """
        Initialize the sliding window KV cache.
        
        Args:
            hidden_size: Hidden size of the transformer model
            num_attention_heads: Number of attention heads in the transformer model
            num_hidden_layers: Number of hidden layers in the transformer model
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            window_size: Size of the sliding window
            device: Device to run the model on
        """
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Storage for KV cache
        self.kv_cache = {}
        
        # Metrics tracking
        self.metrics = {
            "memory_usage": [],
        }
    
    def reset_cache(self):
        """Reset the KV cache."""
        self.kv_cache = {}
    
    def update_kv_cache(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor
    ):
        """
        Update the KV cache, keeping only the most recent window_size tokens.
        
        Args:
            layer_idx: Index of the current layer
            key_states: Key states [batch_size, num_heads, seq_len, head_dim]
            value_states: Value states [batch_size, num_heads, seq_len, head_dim]
        """
        layer_key = f"layer_{layer_idx}"
        
        if layer_key not in self.kv_cache:
            self.kv_cache[layer_key] = {}
        
        # Apply sliding window
        seq_len = key_states.size(2)
        
        if seq_len > self.window_size:
            # Keep only the most recent window_size tokens
            key_states = key_states[:, :, -self.window_size:, :]
            value_states = value_states[:, :, -self.window_size:, :]
        
        # Store in cache
        self.kv_cache[layer_key]['k'] = key_states
        self.kv_cache[layer_key]['v'] = value_states
    
    def get_cached_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the cached key-value tensors for a layer.
        
        Args:
            layer_idx: Index of the current layer
            
        Returns:
            Tuple of (key_states, value_states)
        """
        layer_key = f"layer_{layer_idx}"
        
        if layer_key in self.kv_cache:
            return (
                self.kv_cache[layer_key]['k'],
                self.kv_cache[layer_key]['v']
            )
        
        return None, None
    
    def compute_memory_usage(self) -> Dict[str, float]:
        """
        Compute the current memory usage of the KV cache.
        
        Returns:
            Dictionary containing memory usage statistics
        """
        total_elements = 0
        
        for layer_key, layer_cache in self.kv_cache.items():
            if 'k' in layer_cache and 'v' in layer_cache:
                # Count elements in key and value tensors
                k_elements = layer_cache['k'].numel()
                v_elements = layer_cache['v'].numel()
                
                total_elements += k_elements + v_elements
        
        # Compute memory in bytes (assuming float32)
        bytes_per_element = 4  # float32
        total_memory = total_elements * bytes_per_element
        
        # Convert to MB
        total_memory_mb = total_memory / (1024 * 1024)
        
        # Store metrics
        self.metrics["memory_usage"].append(total_memory_mb)
        
        return {
            "total_memory_mb": total_memory_mb,
        }


class DynamicKVCache:
    """
    Implementation of DynamicKV approach for KV cache management.
    Dynamically adjusts the number of tokens retained at each layer.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        head_dim: int,
        max_seq_len: int,
        initial_retention_ratio: float = 0.5,
        min_retention_ratio: float = 0.1,
        max_retention_ratio: float = 0.9,
        device: torch.device = None
    ):
        """
        Initialize the DynamicKV cache.
        
        Args:
            hidden_size: Hidden size of the transformer model
            num_attention_heads: Number of attention heads in the transformer model
            num_hidden_layers: Number of hidden layers in the transformer model
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            initial_retention_ratio: Initial ratio of tokens to retain
            min_retention_ratio: Minimum ratio of tokens to retain
            max_retention_ratio: Maximum ratio of tokens to retain
            device: Device to run the model on
        """
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.initial_retention_ratio = initial_retention_ratio
        self.min_retention_ratio = min_retention_ratio
        self.max_retention_ratio = max_retention_ratio
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Layer-specific retention ratios (different layers may keep different amounts)
        self.layer_retention_ratios = {
            i: initial_retention_ratio for i in range(num_hidden_layers)
        }
        
        # Storage for KV cache
        self.kv_cache = {}
        
        # Metrics tracking
        self.metrics = {
            "memory_usage": [],
            "retention_ratios": {i: [] for i in range(num_hidden_layers)},
        }
    
    def reset_cache(self):
        """Reset the KV cache."""
        self.kv_cache = {}
    
    def compute_retention_mask(
        self,
        layer_idx: int,
        seq_len: int,
        attention_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the retention mask based on attention scores.
        
        Args:
            layer_idx: Index of the current layer
            seq_len: Sequence length
            attention_scores: Attention scores [batch_size, num_heads, seq_len, seq_len]
            
        Returns:
            Binary mask [batch_size, seq_len]
        """
        batch_size = attention_scores.shape[0]
        
        # Compute token importance based on average attention received
        token_importance = attention_scores.mean(dim=(0, 1)).mean(dim=1)  # [seq_len]
        
        # Determine how many tokens to keep
        retention_ratio = self.layer_retention_ratios[layer_idx]
        num_tokens_to_keep = max(1, int(seq_len * retention_ratio))
        
        # Find the top-k tokens by importance
        if seq_len <= num_tokens_to_keep:
            # Keep all tokens if seq_len is small
            mask = torch.ones((batch_size, seq_len), device=self.device)
        else:
            # Find indices of top-k tokens
            _, top_indices = torch.topk(token_importance, num_tokens_to_keep)
            
            # Create mask
            mask = torch.zeros((batch_size, seq_len), device=self.device)
            mask[:, top_indices] = 1.0
        
        # Store metrics
        self.metrics["retention_ratios"][layer_idx].append(retention_ratio)
        
        return mask
    
    def update_kv_cache(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_scores: torch.Tensor
    ):
        """
        Update the KV cache based on dynamic retention policy.
        
        Args:
            layer_idx: Index of the current layer
            key_states: Key states [batch_size, num_heads, seq_len, head_dim]
            value_states: Value states [batch_size, num_heads, seq_len, head_dim]
            attention_scores: Attention scores [batch_size, num_heads, seq_len, seq_len]
        """
        layer_key = f"layer_{layer_idx}"
        
        if layer_key not in self.kv_cache:
            self.kv_cache[layer_key] = {}
        
        # Compute retention mask
        seq_len = key_states.size(2)
        retention_mask = self.compute_retention_mask(layer_idx, seq_len, attention_scores)
        
        # Expand mask for broadcasting
        mask_expanded = retention_mask.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, seq_len, 1]
        
        # Apply mask to key and value states
        masked_keys = key_states * mask_expanded
        masked_values = value_states * mask_expanded
        
        # Store in cache
        self.kv_cache[layer_key]['k'] = masked_keys
        self.kv_cache[layer_key]['v'] = masked_values
        self.kv_cache[layer_key]['mask'] = retention_mask
    
    def get_cached_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the cached key-value tensors for a layer.
        
        Args:
            layer_idx: Index of the current layer
            
        Returns:
            Tuple of (key_states, value_states, mask)
        """
        layer_key = f"layer_{layer_idx}"
        
        if layer_key in self.kv_cache:
            return (
                self.kv_cache[layer_key]['k'],
                self.kv_cache[layer_key]['v'],
                self.kv_cache[layer_key]['mask']
            )
        
        return None, None, None
    
    def compute_memory_usage(self) -> Dict[str, float]:
        """
        Compute the current memory usage of the KV cache.
        
        Returns:
            Dictionary containing memory usage statistics
        """
        total_elements = 0
        total_active_elements = 0
        
        for layer_key, layer_cache in self.kv_cache.items():
            if 'k' in layer_cache and 'v' in layer_cache and 'mask' in layer_cache:
                # Count elements in key and value tensors
                k_elements = layer_cache['k'].numel()
                v_elements = layer_cache['v'].numel()
                
                # Count active elements (non-zero)
                k_active = torch.count_nonzero(layer_cache['k']).item()
                v_active = torch.count_nonzero(layer_cache['v']).item()
                
                total_elements += k_elements + v_elements
                total_active_elements += k_active + v_active
        
        # Compute memory in bytes (assuming float32)
        bytes_per_element = 4  # float32
        total_memory = total_elements * bytes_per_element
        active_memory = total_active_elements * bytes_per_element
        
        # Convert to MB
        total_memory_mb = total_memory / (1024 * 1024)
        active_memory_mb = active_memory / (1024 * 1024)
        
        # Compute sparsity
        sparsity = 1.0 - (active_memory / total_memory) if total_memory > 0 else 0.0
        
        # Store metrics
        self.metrics["memory_usage"].append(active_memory_mb)
        
        return {
            "total_memory_mb": total_memory_mb,
            "active_memory_mb": active_memory_mb,
            "sparsity": sparsity
        }
    
    def adjust_layer_retention_ratios(self, performance_metric: float, memory_usage: float, target_memory: float):
        """
        Adjust the layer-specific retention ratios based on performance and memory usage.
        
        Args:
            performance_metric: Current performance metric (higher is better)
            memory_usage: Current memory usage
            target_memory: Target memory usage
        """
        # Simple heuristic to adjust retention ratios
        memory_ratio = memory_usage / target_memory if target_memory > 0 else 1.0
        
        for layer_idx in range(self.num_hidden_layers):
            if memory_ratio > 1.1:  # Using too much memory
                # Reduce retention ratio
                new_ratio = max(
                    self.min_retention_ratio,
                    self.layer_retention_ratios[layer_idx] * 0.9
                )
            elif memory_ratio < 0.9:  # Using too little memory
                # Increase retention ratio if performance is not good
                if performance_metric < 0.95:  # Assuming normalized performance
                    new_ratio = min(
                        self.max_retention_ratio,
                        self.layer_retention_ratios[layer_idx] * 1.1
                    )
                else:
                    new_ratio = self.layer_retention_ratios[layer_idx]
            else:
                # Keep current ratio
                new_ratio = self.layer_retention_ratios[layer_idx]
            
            self.layer_retention_ratios[layer_idx] = new_ratio


class RocketKVCache:
    """
    Implementation of RocketKV approach for KV cache management.
    Uses a two-stage KV cache compression strategy.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        head_dim: int,
        max_seq_len: int,
        coarse_eviction_rate: float = 0.5,
        fine_sparsity_rate: float = 0.7,
        device: torch.device = None
    ):
        """
        Initialize the RocketKV cache.
        
        Args:
            hidden_size: Hidden size of the transformer model
            num_attention_heads: Number of attention heads in the transformer model
            num_hidden_layers: Number of hidden layers in the transformer model
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            coarse_eviction_rate: Rate of tokens to evict in coarse-grain stage
            fine_sparsity_rate: Sparsity rate for fine-grain stage
            device: Device to run the model on
        """
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.coarse_eviction_rate = coarse_eviction_rate
        self.fine_sparsity_rate = fine_sparsity_rate
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Storage for KV cache
        self.kv_cache = {}
        
        # Metrics tracking
        self.metrics = {
            "memory_usage": [],
            "coarse_eviction_rates": [],
            "fine_sparsity_rates": [],
        }
    
    def reset_cache(self):
        """Reset the KV cache."""
        self.kv_cache = {}
    
    def coarse_grain_eviction(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply coarse-grain KV cache eviction.
        
        Args:
            key_states: Key states [batch_size, num_heads, seq_len, head_dim]
            value_states: Value states [batch_size, num_heads, seq_len, head_dim]
            attention_scores: Attention scores [batch_size, num_heads, seq_len, seq_len]
            
        Returns:
            Tuple of (evicted_keys, evicted_values, eviction_mask)
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Compute token importance as the sum of attention received
        # We sum across all query tokens and heads
        token_importance = attention_scores.sum(dim=1).sum(dim=1)  # [batch_size, seq_len]
        
        # Determine number of tokens to keep
        num_tokens_to_keep = max(1, int(seq_len * (1 - self.coarse_eviction_rate)))
        
        # Get top-k token indices for each batch
        _, top_indices = torch.topk(token_importance, num_tokens_to_keep, dim=1)
        
        # Create mask (1 for tokens to keep, 0 for tokens to evict)
        eviction_mask = torch.zeros((batch_size, seq_len), device=self.device)
        
        # Set mask values for tokens to keep
        for b in range(batch_size):
            eviction_mask[b, top_indices[b]] = 1.0
        
        # Expand mask for broadcasting
        mask_expanded = eviction_mask.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, seq_len, 1]
        
        # Apply mask
        evicted_keys = key_states * mask_expanded
        evicted_values = value_states * mask_expanded
        
        return evicted_keys, evicted_values, eviction_mask
    
    def fine_grain_sparsification(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply fine-grain sparsification to non-evicted tokens.
        
        Args:
            key_states: Key states after coarse eviction [batch_size, num_heads, seq_len, head_dim]
            value_states: Value states after coarse eviction [batch_size, num_heads, seq_len, head_dim]
            attention_scores: Attention scores [batch_size, num_heads, seq_len, seq_len]
            
        Returns:
            Tuple of (sparse_keys, sparse_values)
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Create a mask that keeps only the top (1 - fine_sparsity_rate) elements
        # in each head and token dimension
        sparse_mask = torch.zeros_like(key_states)
        
        for b in range(batch_size):
            for h in range(num_heads):
                # Get attention scores for this batch and head
                scores = attention_scores[b, h]  # [seq_len, seq_len]
                
                # Compute importance of each dimension in the key/value vectors
                # by looking at how much they're attended to
                dim_importance = torch.abs(key_states[b, h]).mean(dim=0)  # [head_dim]
                
                # Determine number of dimensions to keep
                num_dims_to_keep = max(1, int(head_dim * (1 - self.fine_sparsity_rate)))
                
                # Get top dimensions
                _, top_dims = torch.topk(dim_importance, num_dims_to_keep)
                
                # Set mask for these dimensions
                sparse_mask[b, h, :, top_dims] = 1.0
        
        # Apply sparsity mask
        sparse_keys = key_states * sparse_mask
        sparse_values = value_states * sparse_mask
        
        return sparse_keys, sparse_values
    
    def update_kv_cache(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_scores: torch.Tensor
    ):
        """
        Update the KV cache using the two-stage compression strategy.
        
        Args:
            layer_idx: Index of the current layer
            key_states: Key states [batch_size, num_heads, seq_len, head_dim]
            value_states: Value states [batch_size, num_heads, seq_len, head_dim]
            attention_scores: Attention scores [batch_size, num_heads, seq_len, seq_len]
        """
        layer_key = f"layer_{layer_idx}"
        
        if layer_key not in self.kv_cache:
            self.kv_cache[layer_key] = {}
        
        # Stage 1: Coarse-grain eviction
        evicted_keys, evicted_values, eviction_mask = self.coarse_grain_eviction(
            key_states, value_states, attention_scores
        )
        
        # Stage 2: Fine-grain sparsification
        sparse_keys, sparse_values = self.fine_grain_sparsification(
            evicted_keys, evicted_values, attention_scores
        )
        
        # Store in cache
        self.kv_cache[layer_key]['k'] = sparse_keys
        self.kv_cache[layer_key]['v'] = sparse_values
        self.kv_cache[layer_key]['mask'] = eviction_mask
        
        # Store metrics
        self.metrics["coarse_eviction_rates"].append(self.coarse_eviction_rate)
        self.metrics["fine_sparsity_rates"].append(self.fine_sparsity_rate)
    
    def get_cached_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the cached key-value tensors for a layer.
        
        Args:
            layer_idx: Index of the current layer
            
        Returns:
            Tuple of (key_states, value_states, mask)
        """
        layer_key = f"layer_{layer_idx}"
        
        if layer_key in self.kv_cache:
            return (
                self.kv_cache[layer_key]['k'],
                self.kv_cache[layer_key]['v'],
                self.kv_cache[layer_key]['mask']
            )
        
        return None, None, None
    
    def compute_memory_usage(self) -> Dict[str, float]:
        """
        Compute the current memory usage of the KV cache.
        
        Returns:
            Dictionary containing memory usage statistics
        """
        total_elements = 0
        total_active_elements = 0
        
        for layer_key, layer_cache in self.kv_cache.items():
            if 'k' in layer_cache and 'v' in layer_cache:
                # Count elements in key and value tensors
                k_elements = layer_cache['k'].numel()
                v_elements = layer_cache['v'].numel()
                
                # Count active elements (non-zero)
                k_active = torch.count_nonzero(layer_cache['k']).item()
                v_active = torch.count_nonzero(layer_cache['v']).item()
                
                total_elements += k_elements + v_elements
                total_active_elements += k_active + v_active
        
        # Compute memory in bytes (assuming float32)
        bytes_per_element = 4  # float32
        total_memory = total_elements * bytes_per_element
        active_memory = total_active_elements * bytes_per_element
        
        # Convert to MB
        total_memory_mb = total_memory / (1024 * 1024)
        active_memory_mb = active_memory / (1024 * 1024)
        
        # Compute sparsity
        sparsity = 1.0 - (active_memory / total_memory) if total_memory > 0 else 0.0
        
        # Store metrics
        self.metrics["memory_usage"].append(active_memory_mb)
        
        return {
            "total_memory_mb": total_memory_mb,
            "active_memory_mb": active_memory_mb,
            "sparsity": sparsity
        }


class KVCacheFactory:
    """
    Factory class for creating different KV cache implementations.
    """
    @staticmethod
    def create(
        method: str,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        head_dim: int,
        max_seq_len: int,
        **kwargs
    ):
        """
        Create a KV cache implementation based on the specified method.
        
        Args:
            method: Name of the KV cache method
            hidden_size: Hidden size of the transformer model
            num_attention_heads: Number of attention heads in the transformer model
            num_hidden_layers: Number of hidden layers in the transformer model
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            **kwargs: Additional arguments specific to each method
            
        Returns:
            KV cache implementation
        """
        if method == "full":
            return FullKVCache(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                **kwargs
            )
        elif method == "sliding_window":
            window_size = kwargs.get("window_size", 1024)
            return SlidingWindowKVCache(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                window_size=window_size,
                **kwargs
            )
        elif method == "dynamic_kv":
            initial_retention_ratio = kwargs.get("initial_retention_ratio", 0.5)
            min_retention_ratio = kwargs.get("min_retention_ratio", 0.1)
            max_retention_ratio = kwargs.get("max_retention_ratio", 0.9)
            return DynamicKVCache(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                initial_retention_ratio=initial_retention_ratio,
                min_retention_ratio=min_retention_ratio,
                max_retention_ratio=max_retention_ratio,
                **kwargs
            )
        elif method == "rocket_kv":
            coarse_eviction_rate = kwargs.get("coarse_eviction_rate", 0.5)
            fine_sparsity_rate = kwargs.get("fine_sparsity_rate", 0.7)
            return RocketKVCache(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                coarse_eviction_rate=coarse_eviction_rate,
                fine_sparsity_rate=fine_sparsity_rate,
                **kwargs
            )
        elif method == "atskv":
            from sparse_kv_cache import AdaptiveSparseKVCache
            feature_dim = kwargs.get("feature_dim", 64)
            lambda_momentum = kwargs.get("lambda_momentum", 0.8)
            initial_sparsity = kwargs.get("initial_sparsity", 0.7)
            return AdaptiveSparseKVCache(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                feature_dim=feature_dim,
                lambda_momentum=lambda_momentum,
                initial_sparsity=initial_sparsity,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown KV cache method: {method}")
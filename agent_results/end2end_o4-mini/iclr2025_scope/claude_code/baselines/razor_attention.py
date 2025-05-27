import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time

class RazorAttentionCompressor(nn.Module):
    """
    RazorAttention: Efficient KV Cache Compression Through Retrieval Heads
    
    Implements a compression technique that maintains a full cache for 
    crucial retrieval heads and discards remote tokens in non-retrieval heads.
    """
    
    def __init__(self, 
                 num_layers: int, 
                 num_heads: int, 
                 head_dim: int,
                 retrieval_head_ratio: float = 0.2,
                 non_retrieval_token_ratio: float = 0.3,
                 compensation_tokens: int = 8,
                 update_interval: int = 256,
                 device: torch.device = None):
        """
        Initialize RazorAttention compressor.
        
        Args:
            num_layers: Number of layers in the transformer model
            num_heads: Number of attention heads per layer
            head_dim: Dimension of each attention head
            retrieval_head_ratio: Ratio of heads to treat as retrieval heads
            non_retrieval_token_ratio: Ratio of tokens to keep for non-retrieval heads
            compensation_tokens: Number of compensation tokens to add
            update_interval: Interval between KV cache updates
            device: Device to use for computation
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.retrieval_head_ratio = retrieval_head_ratio
        self.non_retrieval_token_ratio = non_retrieval_token_ratio
        self.compensation_tokens = compensation_tokens
        self.update_interval = update_interval
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine retrieval heads (fixed for simplicity, but could be dynamic)
        num_retrieval_heads = max(1, int(num_heads * retrieval_head_ratio))
        self.retrieval_heads = {
            layer: set(range(num_retrieval_heads)) 
            for layer in range(num_layers)
        }
        
        # Tracking
        self.token_positions = None
        self.steps_since_last_update = 0
        
        # Compensation token embeddings
        self.has_compensation_tokens = compensation_tokens > 0
        if self.has_compensation_tokens:
            self.compensation_key_embeddings = nn.Parameter(
                torch.randn(num_layers, compensation_tokens, num_heads, head_dim, device=self.device) / np.sqrt(head_dim)
            )
            self.compensation_value_embeddings = nn.Parameter(
                torch.randn(num_layers, compensation_tokens, num_heads, head_dim, device=self.device) / np.sqrt(head_dim)
            )
        
        # Statistics
        self.stats = {
            'update_count': 0,
            'avg_update_time': 0,
            'compression_ratio': 0,
        }
    
    def identify_retrieval_heads(self, attention_matrices: Dict[Tuple[int, int], torch.Tensor]) -> Dict[int, set]:
        """
        Identify retrieval heads based on attention patterns.
        
        In a real implementation, this would analyze attention matrices to identify 
        heads that look far back in the context (retrieval heads).
        
        Args:
            attention_matrices: Dictionary mapping (layer, head) to attention matrices
            
        Returns:
            retrieval_heads: Dictionary mapping layer to set of retrieval head indices
        """
        # For simplicity, we use a fixed set of retrieval heads
        # A real implementation would analyze attention patterns
        
        # However, for demonstration, we'll use a heuristic based on average attention distance
        retrieval_heads = {layer: set() for layer in range(self.num_layers)}
        
        for layer in range(self.num_layers):
            head_distances = {}
            for head in range(self.num_heads):
                if (layer, head) in attention_matrices:
                    attn_matrix = attention_matrices[(layer, head)]
                    seq_len = attn_matrix.shape[1]
                    
                    # Create position indices
                    q_indices = torch.arange(seq_len, device=self.device).unsqueeze(1)  # [seq_len, 1]
                    k_indices = torch.arange(seq_len, device=self.device).unsqueeze(0)  # [1, seq_len]
                    
                    # Compute absolute distance between query and key positions
                    distances = torch.abs(q_indices - k_indices).float()  # [seq_len, seq_len]
                    
                    # Weight distances by attention weights
                    weighted_distances = distances.unsqueeze(0) * attn_matrix  # [batch, seq_len, seq_len]
                    
                    # Compute average attention distance
                    avg_distance = weighted_distances.sum() / (attn_matrix.sum() + 1e-10)
                    head_distances[head] = avg_distance.item()
            
            # Sort heads by average attention distance (higher = more retrieval-like)
            sorted_heads = sorted(head_distances.keys(), key=lambda h: head_distances[h], reverse=True)
            num_retrieval_heads = max(1, int(self.num_heads * self.retrieval_head_ratio))
            
            # Select top heads as retrieval heads
            retrieval_heads[layer] = set(sorted_heads[:num_retrieval_heads])
        
        return retrieval_heads
    
    def compute_token_scores(self, attention_matrices: Dict[Tuple[int, int], torch.Tensor], layer: int) -> torch.Tensor:
        """
        Compute importance scores for tokens in non-retrieval heads.
        
        Args:
            attention_matrices: Dictionary mapping (layer, head) to attention matrices
            layer: Layer index
            
        Returns:
            token_scores: Tensor of shape (batch_size, seq_len) with score for each token
        """
        seq_len = next(iter(attention_matrices.values())).shape[1]
        batch_size = next(iter(attention_matrices.values())).shape[0]
        scores = torch.zeros((batch_size, seq_len), device=self.device)
        
        # Get non-retrieval heads for this layer
        non_retrieval_heads = set(range(self.num_heads)) - self.retrieval_heads[layer]
        
        # Consider recent positions for scoring (assuming recent tokens more important)
        recent_window = min(128, seq_len)
        recent_start = max(0, seq_len - recent_window)
        
        # Sum attention weights across non-retrieval heads
        for head in non_retrieval_heads:
            if (layer, head) in attention_matrices:
                attn_weights = attention_matrices[(layer, head)]
                
                # Sum attention each token receives from recent queries
                token_importance = attn_weights[:, recent_start:seq_len, :seq_len].sum(dim=1)
                scores += token_importance
        
        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def compute_compensation_embeddings(self, 
                                        key_cache: torch.Tensor, 
                                        value_cache: torch.Tensor,
                                        drop_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute compensation embeddings from dropped tokens.
        
        Args:
            key_cache: Key cache tensor of shape (batch_size, seq_len, num_heads, head_dim)
            value_cache: Value cache tensor of shape (batch_size, seq_len, num_heads, head_dim)
            drop_indices: Indices of tokens to drop
            
        Returns:
            compensation_keys: Compensation key embeddings
            compensation_values: Compensation value embeddings
        """
        if not self.has_compensation_tokens or drop_indices.numel() == 0:
            return None, None
        
        batch_size = key_cache.shape[0]
        
        # Gather dropped tokens
        expanded_drop_indices = drop_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, self.num_heads, self.head_dim
        )
        dropped_keys = torch.gather(key_cache, 1, expanded_drop_indices)
        dropped_values = torch.gather(value_cache, 1, expanded_drop_indices)
        
        # Compute mean of dropped tokens to initialize compensation tokens
        # If no tokens dropped, use learnable parameters
        if dropped_keys.shape[1] > 0:
            # Average dropped tokens along sequence dimension
            dropped_keys_mean = dropped_keys.mean(dim=1, keepdim=True)
            dropped_values_mean = dropped_values.mean(dim=1, keepdim=True)
            
            # Expand to compensation_tokens
            compensation_keys = dropped_keys_mean.expand(-1, self.compensation_tokens, -1, -1)
            compensation_values = dropped_values_mean.expand(-1, self.compensation_tokens, -1, -1)
        else:
            # Use learnable parameters (repeat for batch size)
            compensation_keys = self.compensation_key_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            compensation_values = self.compensation_value_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        
        return compensation_keys, compensation_values
    
    def forward(self, 
                key_cache: Dict[int, torch.Tensor],
                value_cache: Dict[int, torch.Tensor],
                attention_matrices: Dict[Tuple[int, int], torch.Tensor] = None,
                do_update: bool = False) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[str, Any]]:
        """
        Compress KV cache using RazorAttention strategy.
        
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
        
        # Update retrieval heads based on attention patterns
        if False:  # Disabled for simplicity
            self.retrieval_heads = self.identify_retrieval_heads(attention_matrices)
        
        # Process each layer
        compressed_key_cache = {}
        compressed_value_cache = {}
        
        for layer in range(self.num_layers):
            if layer not in key_cache:
                continue
                
            layer_key_cache = key_cache[layer]
            layer_value_cache = value_cache[layer]
            
            # Compute token scores for non-retrieval heads
            token_scores = self.compute_token_scores(attention_matrices, layer)
            
            # Determine tokens to keep for non-retrieval heads
            num_tokens_to_keep = max(int(seq_len * self.non_retrieval_token_ratio), 512)
            _, sorted_indices = torch.sort(token_scores, dim=1, descending=True)
            keep_indices = sorted_indices[:, :num_tokens_to_keep]
            
            # Determine tokens to drop
            all_indices = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            drop_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=self.device)
            
            # Mark kept indices as False in drop_mask
            for b in range(batch_size):
                drop_mask[b, keep_indices[b]] = False
            
            drop_indices = all_indices[drop_mask].reshape(batch_size, -1)
            
            # Compute compensation embeddings from dropped tokens
            if self.has_compensation_tokens:
                comp_keys, comp_values = self.compute_compensation_embeddings(
                    key_cache=layer_key_cache,
                    value_cache=layer_value_cache,
                    drop_indices=drop_indices
                )
            
            # Create compressed KV cache by head type
            new_keys = []
            new_values = []
            
            for head in range(self.num_heads):
                head_keys = layer_key_cache[:, :, head, :].unsqueeze(2)  # [batch, seq, 1, dim]
                head_values = layer_value_cache[:, :, head, :].unsqueeze(2)  # [batch, seq, 1, dim]
                
                if head in self.retrieval_heads[layer]:
                    # Retrieval head: keep all tokens
                    new_head_keys = head_keys
                    new_head_values = head_values
                else:
                    # Non-retrieval head: keep only important tokens
                    expanded_keep_indices = keep_indices.unsqueeze(-1).unsqueeze(-1).expand(
                        batch_size, num_tokens_to_keep, 1, self.head_dim
                    )
                    kept_keys = torch.gather(head_keys, 1, expanded_keep_indices)
                    kept_values = torch.gather(head_values, 1, expanded_keep_indices)
                    
                    # Add compensation tokens if enabled
                    if self.has_compensation_tokens and comp_keys is not None:
                        head_comp_keys = comp_keys[:, layer, :, head, :].unsqueeze(2)  # [batch, comp, 1, dim]
                        head_comp_values = comp_values[:, layer, :, head, :].unsqueeze(2)  # [batch, comp, 1, dim]
                        new_head_keys = torch.cat([kept_keys, head_comp_keys], dim=1)
                        new_head_values = torch.cat([kept_values, head_comp_values], dim=1)
                    else:
                        new_head_keys = kept_keys
                        new_head_values = kept_values
                
                new_keys.append(new_head_keys)
                new_values.append(new_head_values)
            
            # Concat along head dimension (may have different sequence lengths)
            # For simplicity, we'll use the maximum length and pad
            max_seq_len = max(keys.shape[1] for keys in new_keys)
            padded_keys = []
            padded_values = []
            
            for head in range(self.num_heads):
                head_keys = new_keys[head]
                head_values = new_values[head]
                
                # Pad to max_seq_len if necessary
                if head_keys.shape[1] < max_seq_len:
                    padding = torch.zeros(
                        (batch_size, max_seq_len - head_keys.shape[1], 1, self.head_dim),
                        device=self.device
                    )
                    head_keys = torch.cat([head_keys, padding], dim=1)
                    head_values = torch.cat([head_values, padding], dim=1)
                
                padded_keys.append(head_keys)
                padded_values.append(head_values)
            
            # Concatenate along head dimension and reshape
            compressed_keys = torch.cat(padded_keys, dim=2)
            compressed_values = torch.cat(padded_values, dim=2)
            
            compressed_key_cache[layer] = compressed_keys
            compressed_value_cache[layer] = compressed_values
        
        # Update statistics
        elapsed_time = time.time() - start_time
        
        # Estimate compression ratio
        original_size = sum(cache.numel() for cache in key_cache.values()) + sum(cache.numel() for cache in value_cache.values())
        compressed_size = sum(cache.numel() for cache in compressed_key_cache.values()) + sum(cache.numel() for cache in compressed_value_cache.values())
        compression_ratio = original_size / (compressed_size + 1e-10)
        
        self.stats['update_count'] += 1
        self.stats['avg_update_time'] = (
            (self.stats['avg_update_time'] * (self.stats['update_count'] - 1) + elapsed_time) / 
            self.stats['update_count']
        )
        self.stats['compression_ratio'] = compression_ratio
        
        return compressed_key_cache, compressed_value_cache, self.stats
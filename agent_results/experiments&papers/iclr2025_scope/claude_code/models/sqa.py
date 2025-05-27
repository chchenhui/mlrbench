#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sub-Quadratic Sparse Attention (SQA) module.

This module implements the Sub-Quadratic Sparse Attention component of the proposed
architecture. The SQA processes only the tokens selected by the DSR, dramatically
reducing the computational complexity compared to standard attention mechanisms.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ClusteringLayer(nn.Module):
    """
    Lightweight clustering layer for key-value pairs.
    """
    def __init__(self, key_dim: int, num_clusters: int):
        super().__init__()
        self.key_dim = key_dim
        self.num_clusters = num_clusters
        
        # Initialize cluster centroids
        self.register_parameter(
            "centroids",
            nn.Parameter(torch.randn(num_clusters, key_dim))
        )
        
        # Initialize cluster assignments as a buffer (not parameters)
        self.register_buffer(
            "assignments",
            torch.zeros(0, dtype=torch.long)
        )
    
    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cluster the key-value pairs.
        
        Args:
            keys: Tensor of shape [batch_size, seq_length, key_dim]
            values: Tensor of shape [batch_size, seq_length, value_dim]
            mask: Optional tensor of shape [batch_size, seq_length] indicating valid tokens
        
        Returns:
            cluster_centroids: Tensor of shape [batch_size, num_clusters, key_dim]
            cluster_values: Tensor of shape [batch_size, num_clusters, value_dim]
            assignments: Tensor of shape [batch_size, seq_length] indicating cluster assignments
        """
        batch_size, seq_length, _ = keys.shape
        
        # Apply mask if provided
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=keys.device)
        
        # Normalize centroids and keys for cosine similarity
        centroids_norm = F.normalize(self.centroids, p=2, dim=-1)
        keys_norm = F.normalize(keys, p=2, dim=-1)
        
        # Compute distances/similarities to cluster centroids
        # [batch_size, seq_length, num_clusters]
        similarities = torch.bmm(
            keys_norm,
            centroids_norm.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)
        )
        
        # Apply mask to similarities (set masked tokens to -inf)
        if mask is not None:
            similarities = similarities.masked_fill(~mask.unsqueeze(-1).bool(), float('-inf'))
        
        # Assign tokens to nearest cluster
        # [batch_size, seq_length]
        assignments = torch.argmax(similarities, dim=-1)
        self.assignments = assignments.detach()  # Save for visualization/analysis
        
        # Compute cluster centroids and aggregated values
        cluster_centroids = torch.zeros(
            batch_size, self.num_clusters, self.key_dim, device=keys.device
        )
        cluster_values = torch.zeros(
            batch_size, self.num_clusters, values.shape[-1], device=values.device
        )
        
        # Aggregate keys and values per cluster
        for b in range(batch_size):
            for c in range(self.num_clusters):
                # Get mask for tokens assigned to this cluster
                cluster_mask = (assignments[b] == c) & mask[b].bool()
                
                if torch.any(cluster_mask):
                    # Average keys and values for this cluster
                    cluster_centroids[b, c] = torch.mean(keys[b, cluster_mask], dim=0)
                    cluster_values[b, c] = torch.mean(values[b, cluster_mask], dim=0)
                else:
                    # If no tokens assigned to this cluster, use the centroid
                    cluster_centroids[b, c] = self.centroids[c]
                    # For values, we can use zeros or a learned default
                    cluster_values[b, c] = 0.0
        
        return cluster_centroids, cluster_values, assignments
    
    def update_centroids(self, keys: torch.Tensor, assignments: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Update cluster centroids based on assignments (only used during training).
        
        Args:
            keys: Tensor of shape [batch_size, seq_length, key_dim]
            assignments: Tensor of shape [batch_size, seq_length] indicating cluster assignments
            mask: Optional tensor of shape [batch_size, seq_length] indicating valid tokens
        """
        if not self.training:
            return
        
        batch_size, seq_length, _ = keys.shape
        
        # Apply mask if provided
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=keys.device)
        
        # Compute new centroids as the average of assigned keys
        new_centroids = []
        
        for c in range(self.num_clusters):
            cluster_keys = []
            
            for b in range(batch_size):
                # Get mask for tokens assigned to this cluster
                cluster_mask = (assignments[b] == c) & mask[b].bool()
                
                if torch.any(cluster_mask):
                    cluster_keys.append(keys[b, cluster_mask])
            
            if cluster_keys:
                # Concatenate and average keys from all batches
                all_keys = torch.cat(cluster_keys, dim=0)
                new_centroid = torch.mean(all_keys, dim=0)
                new_centroids.append(new_centroid)
            else:
                # No assignments to this cluster, keep the old centroid
                new_centroids.append(self.centroids[c])
        
        # Update centroids with moving average
        alpha = 0.1  # Learning rate for centroid updates
        updated_centroids = torch.stack(new_centroids)
        self.centroids.data = (1 - alpha) * self.centroids.data + alpha * updated_centroids


class SubQuadraticAttention(nn.Module):
    """
    Sub-Quadratic Sparse Attention module that operates efficiently on clustered tokens.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_clusters: int = 32,
        top_k_clusters: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize the Sub-Quadratic Sparse Attention module.
        
        Args:
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
            num_clusters: Number of clusters for sparsification
            top_k_clusters: Top-k clusters to consider for each query
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.top_k_clusters = top_k_clusters
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Clustering layer (one per attention head)
        self.clustering_layers = nn.ModuleList([
            ClusteringLayer(self.head_dim, num_clusters)
            for _ in range(num_heads)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Track active clusters for compute loss
        self.active_clusters = 0
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        selection_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sub-quadratic sparse attention.
        
        Args:
            query: Tensor of shape [batch_size, query_length, hidden_dim]
            key: Tensor of shape [batch_size, key_length, hidden_dim]
            value: Tensor of shape [batch_size, key_length, hidden_dim]
            selection_mask: Binary tensor of shape [batch_size, key_length] indicating tokens
                           selected by the DSR
            attention_mask: Optional tensor of shape [batch_size, query_length, key_length]
                           for masking attention weights
        
        Returns:
            output: Tensor of shape [batch_size, query_length, hidden_dim]
            attention_weights: Tensor of shape [batch_size, num_heads, query_length, key_length]
                              containing sparse attention weights
        """
        batch_size, query_length, _ = query.shape
        _, key_length, _ = key.shape
        
        # If selection_mask is provided, only use selected tokens for key and value
        if selection_mask is not None:
            # Expand selection_mask for broadcasting
            expanded_mask = selection_mask.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            
            # Apply mask - replace non-selected tokens with zeros
            key = key * expanded_mask
            value = value * expanded_mask
        
        # Project query, key, value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        # [batch_size, seq_length, num_heads, head_dim]
        q = q.view(batch_size, query_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, key_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, key_length, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_length, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Cluster-based attention sparsification
        outputs = []
        attention_weights = []
        total_active_clusters = 0
        
        for h in range(self.num_heads):
            # Cluster the key-value pairs for this head
            cluster_mask = selection_mask if selection_mask is not None else None
            cluster_centroids, cluster_values, assignments = self.clustering_layers[h](
                k[:, h], v[:, h], mask=cluster_mask
            )
            
            # Compute attention scores with cluster centroids
            # [batch_size, query_length, num_clusters]
            cluster_scores = torch.bmm(
                q[:, h],  # [batch_size, query_length, head_dim]
                cluster_centroids.transpose(1, 2)  # [batch_size, head_dim, num_clusters]
            ) * self.scale
            
            # Select top-k clusters for each query token
            # [batch_size, query_length, top_k_clusters]
            topk_scores, topk_indices = torch.topk(
                cluster_scores, k=min(self.top_k_clusters, self.num_clusters), dim=-1
            )
            
            # Count active clusters for compute loss
            active_clusters_count = torch.sum(topk_indices.ne(-1)).item()
            total_active_clusters += active_clusters_count
            
            # Compute sparse attention for each query token
            head_outputs = []
            head_weights = []
            
            for b in range(batch_size):
                for q_idx in range(query_length):
                    # Get indices of tokens in the selected clusters for this query
                    selected_clusters = topk_indices[b, q_idx]
                    token_mask = torch.zeros(key_length, device=query.device)
                    
                    for c_idx in selected_clusters:
                        # Get tokens assigned to this cluster
                        if c_idx == -1:  # Handle padding
                            continue
                        
                        cluster_tokens = (assignments[b] == c_idx)
                        token_mask = token_mask | cluster_tokens
                    
                    # Apply selection mask if provided
                    if selection_mask is not None:
                        token_mask = token_mask & selection_mask[b].bool()
                    
                    # If no tokens selected, use all tokens (fallback)
                    if not torch.any(token_mask):
                        if selection_mask is not None:
                            token_mask = selection_mask[b].bool()
                        else:
                            token_mask = torch.ones(key_length, device=query.device, dtype=torch.bool)
                    
                    # Get selected tokens
                    selected_k = k[b, h, token_mask]  # [selected_length, head_dim]
                    selected_v = v[b, h, token_mask]  # [selected_length, head_dim]
                    
                    # Compute attention with selected tokens
                    q_token = q[b, h, q_idx].unsqueeze(0)  # [1, head_dim]
                    
                    # [1, selected_length]
                    attn_weights = torch.mm(q_token, selected_k.transpose(0, 1)) * self.scale
                    
                    # Apply attention mask if provided
                    if attention_mask is not None:
                        selected_attn_mask = attention_mask[b, q_idx, token_mask]
                        attn_weights = attn_weights + selected_attn_mask
                    
                    # Softmax and dropout
                    attn_weights = F.softmax(attn_weights, dim=-1)
                    attn_weights = self.dropout(attn_weights)
                    
                    # Weighted sum of values
                    # [1, head_dim]
                    attn_output = torch.mm(attn_weights, selected_v)
                    
                    # Store outputs and weights
                    head_outputs.append(attn_output)
                    
                    # Create sparse attention weights matrix
                    full_weights = torch.zeros(key_length, device=query.device)
                    full_weights[token_mask] = attn_weights.squeeze(0)
                    head_weights.append(full_weights)
            
            # Stack outputs and weights
            # [batch_size, query_length, head_dim]
            head_outputs = torch.stack([
                torch.stack(head_outputs[b * query_length:(b + 1) * query_length])
                for b in range(batch_size)
            ])
            
            # [batch_size, query_length, key_length]
            head_weights = torch.stack([
                torch.stack(head_weights[b * query_length:(b + 1) * query_length])
                for b in range(batch_size)
            ])
            
            outputs.append(head_outputs)
            attention_weights.append(head_weights)
        
        # Track active clusters for compute loss
        self.active_clusters = total_active_clusters / (batch_size * self.num_heads)
        
        # Concatenate outputs from all heads
        # [batch_size, query_length, hidden_dim]
        output = torch.cat([
            outputs[h] for h in range(self.num_heads)
        ], dim=-1)
        
        # Apply output projection
        output = self.out_proj(output.view(batch_size, query_length, self.hidden_dim))
        
        # Stack attention weights from all heads
        # [batch_size, num_heads, query_length, key_length]
        stacked_weights = torch.stack([
            attention_weights[h] for h in range(self.num_heads)
        ], dim=1)
        
        # Update cluster centroids during training
        if self.training:
            for h in range(self.num_heads):
                self.clustering_layers[h].update_centroids(
                    k[:, h], self.clustering_layers[h].assignments, mask=selection_mask
                )
        
        return output, stacked_weights
    
    def get_active_clusters(self) -> int:
        """Get the average number of active clusters per head."""
        return self.active_clusters
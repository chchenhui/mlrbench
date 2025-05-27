#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rotating Compressive KV Cache (RCKV) module.

This module implements the Rotating Compressive KV Cache component of the proposed
architecture. The RCKV maintains a fixed-size representation of historical context
using low-rank projections and a rotating buffer mechanism.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class LowRankProjection(nn.Module):
    """
    Low-rank projection for compressing KV cache.
    """
    def __init__(self, original_dim: int, compressed_dim: int):
        super().__init__()
        self.original_dim = original_dim
        self.compressed_dim = compressed_dim
        
        # Forward projection matrix
        self.projection = nn.Linear(original_dim, compressed_dim, bias=False)
        
        # Reconstruction matrix (transpose of projection for orthogonal initialization)
        self.reconstruction = nn.Linear(compressed_dim, original_dim, bias=False)
        
        # Initialize with orthogonal matrices for minimal reconstruction error
        self._init_orthogonal()
    
    def _init_orthogonal(self):
        """Initialize projection matrices orthogonally."""
        nn.init.orthogonal_(self.projection.weight)
        self.reconstruction.weight.data = self.projection.weight.t()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress the input using low-rank projection.
        
        Args:
            x: Tensor of shape [..., original_dim]
        
        Returns:
            compressed: Tensor of shape [..., compressed_dim]
        """
        return self.projection(x)
    
    def reconstruct(self, compressed: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the original representation from compressed form.
        
        Args:
            compressed: Tensor of shape [..., compressed_dim]
        
        Returns:
            reconstructed: Tensor of shape [..., original_dim]
        """
        return self.reconstruction(compressed)
    
    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the reconstruction error for the compression.
        
        Args:
            x: Tensor of shape [..., original_dim]
        
        Returns:
            error: Reconstruction error as Frobenius norm
        """
        compressed = self.forward(x)
        reconstructed = self.reconstruct(compressed)
        error = torch.norm(x - reconstructed, p='fro')
        return error


class ImportanceEstimator(nn.Module):
    """
    Estimates the information value of KV pairs for buffer rotation.
    """
    def __init__(self, key_dim: int, value_dim: int):
        super().__init__()
        input_dim = key_dim + value_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.activation = nn.ReLU()
    
    def forward(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Estimate importance scores for KV pairs.
        
        Args:
            keys: Tensor of shape [batch_size, buffer_size, key_dim]
            values: Tensor of shape [batch_size, buffer_size, value_dim]
        
        Returns:
            importance: Tensor of shape [batch_size, buffer_size] with importance scores
        """
        # Concatenate keys and values
        batch_size, buffer_size, _ = keys.shape
        kv_pairs = torch.cat([keys, values], dim=-1)
        
        # Flatten batch and buffer dimensions
        flat_input = kv_pairs.view(-1, kv_pairs.size(-1))
        
        # Feed through MLP
        hidden1 = self.activation(self.fc1(flat_input))
        hidden2 = self.activation(self.fc2(hidden1))
        importance = torch.sigmoid(self.fc3(hidden2))
        
        # Reshape back to [batch_size, buffer_size]
        importance = importance.view(batch_size, buffer_size)
        
        return importance


class RotatingCompressiveKVCache(nn.Module):
    """
    Rotating Compressive KV Cache module that maintains a fixed-size
    representation of historical context.
    """
    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        compressed_dim: int,
        buffer_size: int = 1024
    ):
        """
        Initialize the Rotating Compressive KV Cache.
        
        Args:
            key_dim: Dimension of key vectors
            value_dim: Dimension of value vectors
            compressed_dim: Dimension of compressed representations
            buffer_size: Size of the rotating buffer
        """
        super().__init__()
        
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.compressed_dim = compressed_dim
        self.buffer_size = buffer_size
        
        # Low-rank projections for keys and values
        self.key_projection = LowRankProjection(key_dim, compressed_dim)
        self.value_projection = LowRankProjection(value_dim, compressed_dim)
        
        # Importance estimator for buffer rotation
        self.importance_estimator = ImportanceEstimator(key_dim, value_dim)
        
        # Initialize empty buffers
        self.register_buffer(
            "key_buffer",
            torch.zeros(0, buffer_size, compressed_dim)
        )
        
        self.register_buffer(
            "value_buffer",
            torch.zeros(0, buffer_size, compressed_dim)
        )
        
        # Track buffer usage for metrics
        self.register_buffer(
            "buffer_usage",
            torch.zeros(0, buffer_size, dtype=torch.bool)
        )
        
        # Store the last reconstruction error for the loss function
        self.reconstruction_error = 0.0
        
        # Information retention metrics for evaluation
        self.retention_score = 0.0
    
    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process and store keys and values in the compressed cache.
        
        Args:
            keys: Tensor of shape [batch_size, seq_length, key_dim]
            values: Tensor of shape [batch_size, seq_length, value_dim]
            mask: Optional tensor of shape [batch_size, seq_length] indicating valid tokens
        
        Returns:
            Dictionary containing:
                - cached_keys: Tensor of shape [batch_size, buffer_size, key_dim]
                - cached_values: Tensor of shape [batch_size, buffer_size, value_dim]
                - usage_mask: Tensor of shape [batch_size, buffer_size] indicating used buffer slots
        """
        batch_size, seq_length, _ = keys.shape
        
        # Apply mask if provided
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=keys.device)
        
        # Initialize buffers if not already done
        if self.key_buffer.size(0) != batch_size:
            self.key_buffer = torch.zeros(
                batch_size, self.buffer_size, self.compressed_dim, device=keys.device
            )
            self.value_buffer = torch.zeros(
                batch_size, self.buffer_size, self.compressed_dim, device=keys.device
            )
            self.buffer_usage = torch.zeros(
                batch_size, self.buffer_size, dtype=torch.bool, device=keys.device
            )
        
        # Compress new keys and values
        compressed_keys = self.key_projection(keys)
        compressed_values = self.value_projection(values)
        
        # Track reconstruction error for loss function
        if self.training:
            self.reconstruction_error = (
                self.key_projection.compute_reconstruction_error(keys) +
                self.value_projection.compute_reconstruction_error(values)
            )
        
        # If buffer is not full, just add the new tokens
        for b in range(batch_size):
            valid_tokens = mask[b].bool()
            num_valid = valid_tokens.sum().item()
            
            if num_valid == 0:
                continue
            
            # Count unused buffer slots
            unused_slots = (~self.buffer_usage[b]).sum().item()
            
            if unused_slots >= num_valid:
                # Enough space to add all new tokens
                first_empty = 0
                while self.buffer_usage[b, first_empty]:
                    first_empty += 1
                
                # Add new tokens to empty slots
                for i, idx in enumerate(torch.where(valid_tokens)[0][:unused_slots]):
                    self.key_buffer[b, first_empty + i] = compressed_keys[b, idx]
                    self.value_buffer[b, first_empty + i] = compressed_values[b, idx]
                    self.buffer_usage[b, first_empty + i] = True
            else:
                # Not enough space, need to replace existing tokens
                
                # Get importance scores for existing buffer
                existing_keys = self.get_reconstructed_keys(b).unsqueeze(0)
                existing_values = self.get_reconstructed_values(b).unsqueeze(0)
                existing_importance = self.importance_estimator(existing_keys, existing_values).squeeze(0)
                
                # Get importance scores for new tokens
                valid_keys = keys[b, valid_tokens].unsqueeze(0)
                valid_values = values[b, valid_tokens].unsqueeze(0)
                new_importance = self.importance_estimator(valid_keys, valid_values).squeeze(0)
                
                # Create candidate pool with existing and new tokens
                candidate_keys = torch.cat([
                    compressed_keys[b, valid_tokens],
                    self.key_buffer[b, self.buffer_usage[b]]
                ])
                
                candidate_values = torch.cat([
                    compressed_values[b, valid_tokens],
                    self.value_buffer[b, self.buffer_usage[b]]
                ])
                
                candidate_importance = torch.cat([
                    new_importance,
                    existing_importance
                ])
                
                # Select top buffer_size tokens by importance
                _, top_indices = torch.topk(
                    candidate_importance, k=min(self.buffer_size, len(candidate_importance))
                )
                
                # Reset buffer
                self.key_buffer[b] = torch.zeros_like(self.key_buffer[b])
                self.value_buffer[b] = torch.zeros_like(self.value_buffer[b])
                self.buffer_usage[b] = torch.zeros_like(self.buffer_usage[b])
                
                # Fill buffer with selected tokens
                for i, idx in enumerate(top_indices):
                    self.key_buffer[b, i] = candidate_keys[idx]
                    self.value_buffer[b, i] = candidate_values[idx]
                    self.buffer_usage[b, i] = True
                
                # Calculate retention score for metrics
                # (proportion of old tokens retained)
                num_old_retained = sum(idx >= num_valid for idx in top_indices).item()
                self.retention_score = num_old_retained / min(self.buffer_size, len(candidate_importance) - num_valid)
        
        # Return the reconstructed cache
        return {
            'cached_keys': self.get_reconstructed_keys(),
            'cached_values': self.get_reconstructed_values(),
            'usage_mask': self.buffer_usage
        }
    
    def get_reconstructed_keys(self, batch_idx: Optional[int] = None) -> torch.Tensor:
        """
        Get reconstructed keys from the compressed cache.
        
        Args:
            batch_idx: Optional batch index to get keys for a specific batch item
        
        Returns:
            reconstructed_keys: Tensor of shape [batch_size, buffer_size, key_dim]
                               or [buffer_size, key_dim] if batch_idx is provided
        """
        if batch_idx is not None:
            # Reconstruct for specific batch item
            compressed_keys = self.key_buffer[batch_idx]
            used_mask = self.buffer_usage[batch_idx]
            
            # Only reconstruct used slots
            reconstructed = torch.zeros(
                self.buffer_size, self.key_dim, device=compressed_keys.device
            )
            
            if used_mask.any():
                reconstructed[used_mask] = self.key_projection.reconstruct(
                    compressed_keys[used_mask]
                )
            
            return reconstructed
        else:
            # Reconstruct for all batch items
            batch_size = self.key_buffer.size(0)
            reconstructed = torch.zeros(
                batch_size, self.buffer_size, self.key_dim, device=self.key_buffer.device
            )
            
            for b in range(batch_size):
                used_mask = self.buffer_usage[b]
                if used_mask.any():
                    reconstructed[b, used_mask] = self.key_projection.reconstruct(
                        self.key_buffer[b, used_mask]
                    )
            
            return reconstructed
    
    def get_reconstructed_values(self, batch_idx: Optional[int] = None) -> torch.Tensor:
        """
        Get reconstructed values from the compressed cache.
        
        Args:
            batch_idx: Optional batch index to get values for a specific batch item
        
        Returns:
            reconstructed_values: Tensor of shape [batch_size, buffer_size, value_dim]
                                 or [buffer_size, value_dim] if batch_idx is provided
        """
        if batch_idx is not None:
            # Reconstruct for specific batch item
            compressed_values = self.value_buffer[batch_idx]
            used_mask = self.buffer_usage[batch_idx]
            
            # Only reconstruct used slots
            reconstructed = torch.zeros(
                self.buffer_size, self.value_dim, device=compressed_values.device
            )
            
            if used_mask.any():
                reconstructed[used_mask] = self.value_projection.reconstruct(
                    compressed_values[used_mask]
                )
            
            return reconstructed
        else:
            # Reconstruct for all batch items
            batch_size = self.value_buffer.size(0)
            reconstructed = torch.zeros(
                batch_size, self.buffer_size, self.value_dim, device=self.value_buffer.device
            )
            
            for b in range(batch_size):
                used_mask = self.buffer_usage[b]
                if used_mask.any():
                    reconstructed[b, used_mask] = self.value_projection.reconstruct(
                        self.value_buffer[b, used_mask]
                    )
            
            return reconstructed
    
    def get_usage_ratio(self) -> float:
        """
        Get the average buffer usage ratio.
        
        Returns:
            usage_ratio: Ratio of used buffer slots
        """
        if self.buffer_usage.numel() == 0:
            return 0.0
        
        return self.buffer_usage.float().mean().item()
    
    def get_reconstruction_error(self) -> float:
        """
        Get the current reconstruction error.
        
        Returns:
            error: Current reconstruction error
        """
        return self.reconstruction_error
    
    def get_information_retention(self) -> float:
        """
        Get the information retention score.
        
        Returns:
            retention: Information retention score
        """
        return self.retention_score
    
    def clear_cache(self, batch_idx: Optional[int] = None):
        """
        Clear the KV cache.
        
        Args:
            batch_idx: Optional batch index to clear cache for a specific batch item
        """
        if batch_idx is not None:
            self.key_buffer[batch_idx] = torch.zeros_like(self.key_buffer[batch_idx])
            self.value_buffer[batch_idx] = torch.zeros_like(self.value_buffer[batch_idx])
            self.buffer_usage[batch_idx] = torch.zeros_like(self.buffer_usage[batch_idx])
        else:
            self.key_buffer = torch.zeros_like(self.key_buffer)
            self.value_buffer = torch.zeros_like(self.value_buffer)
            self.buffer_usage = torch.zeros_like(self.buffer_usage)
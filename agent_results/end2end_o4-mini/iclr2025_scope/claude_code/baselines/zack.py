import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time

class ZACKCompressor(nn.Module):
    """
    ZACK: Zero-Overhead LLM Inference Acceleration via Dimensionality Compression of the Key-Value Cache
    
    Implements dimensionality compression for KV caches based on the importance of attention heads.
    """
    
    def __init__(self, 
                 num_layers: int, 
                 num_heads: int, 
                 head_dim: int,
                 target_compression_ratio: float = 0.5,
                 device: torch.device = None):
        """
        Initialize ZACK compressor.
        
        Args:
            num_layers: Number of layers in the transformer model
            num_heads: Number of attention heads per layer
            head_dim: Dimension of each attention head
            target_compression_ratio: Target ratio of compressed to original dimensions
            device: Device to use for computation
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.target_compression_ratio = target_compression_ratio
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize dimensionality reduction matrices
        self.compression_matrices = nn.ParameterDict()
        self.decompression_matrices = nn.ParameterDict()
        
        # Calculate compressed dimension size
        self.compressed_dim = max(1, int(head_dim * target_compression_ratio))
        
        # Create compression/decompression matrices for each layer and head
        for layer in range(num_layers):
            for head in range(num_heads):
                # Initialize compression matrix: (head_dim, compressed_dim)
                compression_key = f"layer{layer}_head{head}_comp"
                self.compression_matrices[compression_key] = nn.Parameter(
                    torch.randn(head_dim, self.compressed_dim, device=self.device) / np.sqrt(head_dim)
                )
                
                # Initialize decompression matrix: (compressed_dim, head_dim)
                decompression_key = f"layer{layer}_head{head}_decomp"
                self.decompression_matrices[decompression_key] = nn.Parameter(
                    torch.randn(self.compressed_dim, head_dim, device=self.device) / np.sqrt(self.compressed_dim)
                )
        
        # Statistics
        self.stats = {
            'compression_ratio': head_dim / self.compressed_dim,
            'memory_savings': 1 - (self.compressed_dim / head_dim),
            'last_compression_time': 0,
        }
    
    def compress_kv_cache(self, 
                          key_cache: Dict[int, torch.Tensor],
                          value_cache: Dict[int, torch.Tensor]) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Compress KV cache by reducing dimensionality.
        
        Args:
            key_cache: Dictionary mapping layer to key cache tensors
                      of shape (batch_size, seq_len, num_heads, head_dim)
            value_cache: Dictionary mapping layer to value cache tensors
                         of shape (batch_size, seq_len, num_heads, head_dim)
            
        Returns:
            compressed_key_cache: Compressed key cache
            compressed_value_cache: Compressed value cache
        """
        start_time = time.time()
        
        compressed_key_cache = {}
        compressed_value_cache = {}
        
        for layer, layer_key_cache in key_cache.items():
            batch_size, seq_len, num_heads, _ = layer_key_cache.shape
            layer_value_cache = value_cache[layer]
            
            # Initialize compressed caches
            compressed_keys = torch.zeros(
                (batch_size, seq_len, num_heads, self.compressed_dim),
                device=self.device
            )
            compressed_values = torch.zeros(
                (batch_size, seq_len, num_heads, self.compressed_dim),
                device=self.device
            )
            
            # Compress each head
            for head in range(num_heads):
                # Get compression matrices
                comp_key = f"layer{layer}_head{head}_comp"
                compression_matrix = self.compression_matrices[comp_key]
                
                # Extract keys and values for this head
                head_keys = layer_key_cache[:, :, head, :]  # (batch_size, seq_len, head_dim)
                head_values = layer_value_cache[:, :, head, :]  # (batch_size, seq_len, head_dim)
                
                # Compress: (batch_size, seq_len, head_dim) @ (head_dim, compressed_dim)
                # -> (batch_size, seq_len, compressed_dim)
                compressed_head_keys = torch.matmul(head_keys, compression_matrix)
                compressed_head_values = torch.matmul(head_values, compression_matrix)
                
                # Store
                compressed_keys[:, :, head, :] = compressed_head_keys
                compressed_values[:, :, head, :] = compressed_head_values
            
            compressed_key_cache[layer] = compressed_keys
            compressed_value_cache[layer] = compressed_values
        
        # Update statistics
        self.stats['last_compression_time'] = time.time() - start_time
        
        return compressed_key_cache, compressed_value_cache
    
    def decompress_kv_cache(self,
                           compressed_key_cache: Dict[int, torch.Tensor],
                           compressed_value_cache: Dict[int, torch.Tensor]) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Decompress KV cache.
        
        Args:
            compressed_key_cache: Compressed key cache
            compressed_value_cache: Compressed value cache
            
        Returns:
            decompressed_key_cache: Decompressed key cache
            decompressed_value_cache: Decompressed value cache
        """
        decompressed_key_cache = {}
        decompressed_value_cache = {}
        
        for layer, layer_compressed_keys in compressed_key_cache.items():
            batch_size, seq_len, num_heads, _ = layer_compressed_keys.shape
            layer_compressed_values = compressed_value_cache[layer]
            
            # Initialize decompressed caches
            decompressed_keys = torch.zeros(
                (batch_size, seq_len, num_heads, self.head_dim),
                device=self.device
            )
            decompressed_values = torch.zeros(
                (batch_size, seq_len, num_heads, self.head_dim),
                device=self.device
            )
            
            # Decompress each head
            for head in range(num_heads):
                # Get decompression matrices
                decomp_key = f"layer{layer}_head{head}_decomp"
                decompression_matrix = self.decompression_matrices[decomp_key]
                
                # Extract compressed keys and values for this head
                compressed_head_keys = layer_compressed_keys[:, :, head, :]  # (batch_size, seq_len, compressed_dim)
                compressed_head_values = layer_compressed_values[:, :, head, :]  # (batch_size, seq_len, compressed_dim)
                
                # Decompress: (batch_size, seq_len, compressed_dim) @ (compressed_dim, head_dim)
                # -> (batch_size, seq_len, head_dim)
                decompressed_head_keys = torch.matmul(compressed_head_keys, decompression_matrix)
                decompressed_head_values = torch.matmul(compressed_head_values, decompression_matrix)
                
                # Store
                decompressed_keys[:, :, head, :] = decompressed_head_keys
                decompressed_values[:, :, head, :] = decompressed_head_values
            
            decompressed_key_cache[layer] = decompressed_keys
            decompressed_value_cache[layer] = decompressed_values
        
        return decompressed_key_cache, decompressed_value_cache
    
    def forward(self, 
                key_cache: Dict[int, torch.Tensor],
                value_cache: Dict[int, torch.Tensor]) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[str, Any]]:
        """
        Forward pass: Compress and then decompress KV cache.
        
        Args:
            key_cache: Dictionary mapping layer to key cache tensors
            value_cache: Dictionary mapping layer to value cache tensors
            
        Returns:
            decompressed_key_cache: Decompressed key cache
            decompressed_value_cache: Decompressed value cache
            stats: Dictionary of compression statistics
        """
        # Compress
        compressed_key_cache, compressed_value_cache = self.compress_kv_cache(
            key_cache=key_cache,
            value_cache=value_cache
        )
        
        # Decompress
        decompressed_key_cache, decompressed_value_cache = self.decompress_kv_cache(
            compressed_key_cache=compressed_key_cache,
            compressed_value_cache=compressed_value_cache
        )
        
        return decompressed_key_cache, decompressed_value_cache, self.stats
    
    def optimize_compression_rates(self, 
                                  key_cache: Dict[int, torch.Tensor],
                                  value_cache: Dict[int, torch.Tensor],
                                  num_steps: int = 1000):
        """
        Optimize compression matrices to minimize reconstruction error.
        
        Args:
            key_cache: Dictionary mapping layer to key cache tensors
            value_cache: Dictionary mapping layer to value cache tensors
            num_steps: Number of optimization steps
            
        Returns:
            stats: Dictionary of optimization statistics
        """
        # Freeze current parameters for training
        for param in self.parameters():
            param.requires_grad = True
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        # Optimization loop
        for step in range(num_steps):
            # Forward pass
            decompressed_key_cache, decompressed_value_cache, _ = self.forward(
                key_cache=key_cache,
                value_cache=value_cache
            )
            
            # Compute reconstruction loss
            key_loss = sum(
                torch.nn.functional.mse_loss(decompressed_keys, key_cache[layer])
                for layer, decompressed_keys in decompressed_key_cache.items()
            )
            
            value_loss = sum(
                torch.nn.functional.mse_loss(decompressed_values, value_cache[layer])
                for layer, decompressed_values in decompressed_value_cache.items()
            )
            
            total_loss = key_loss + value_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Log progress
            if (step + 1) % 100 == 0:
                print(f"Step {step+1}/{num_steps}, Reconstruction loss: {total_loss.item():.6f}")
        
        # Freeze parameters after training
        for param in self.parameters():
            param.requires_grad = False
        
        return {
            'final_loss': total_loss.item(),
            'key_loss': key_loss.item(),
            'value_loss': value_loss.item()
        }
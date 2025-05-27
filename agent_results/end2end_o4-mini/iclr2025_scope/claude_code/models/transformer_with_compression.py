import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import logging
import numpy as np
from .kv_cache_compressor import KVCacheCompressor, DistillationLoss

logger = logging.getLogger(__name__)


class TransformerWithCompression(nn.Module):
    """
    Wrapper around a transformer model that adds KV cache compression functionality.
    """
    
    def __init__(self, 
                 model_name_or_path: str,
                 max_cache_size: int = 1024,
                 num_clusters: int = 256,
                 pruning_interval: int = 512,
                 lookback_window: int = 256,
                 kmeans_learning_rate: float = 0.01,
                 temperature: float = 2.0,
                 distillation_weight: float = 0.5,
                 use_compression: bool = True,
                 device: torch.device = None):
        """
        Initialize transformer model with KV cache compression.
        
        Args:
            model_name_or_path: HuggingFace model name or path
            max_cache_size: Maximum number of KV pairs to retain after pruning (B in the paper)
            num_clusters: Number of cluster centroids for low-rank summarization (K in the paper)
            pruning_interval: Interval (in tokens) between pruning operations (P in the paper)
            lookback_window: Number of recent positions to consider for importance (Δ in the paper)
            kmeans_learning_rate: Learning rate for online k-means updates (η in the paper)
            temperature: Temperature for distillation loss
            distillation_weight: Weight of distillation loss (λ in the paper)
            use_compression: Whether to use compression or not (for baseline comparison)
            device: Device to use for computation
        """
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            config=self.config
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Extract model dimensions
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        
        # Initialize KV cache compressor
        self.compressor = KVCacheCompressor(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_cache_size=max_cache_size,
            num_clusters=num_clusters,
            pruning_interval=pruning_interval,
            lookback_window=lookback_window,
            kmeans_learning_rate=kmeans_learning_rate,
            device=self.device
        )
        
        # Initialize distillation loss
        self.distillation_loss = DistillationLoss(temperature=temperature)
        self.distillation_weight = distillation_weight
        
        # Set up compression flag
        self.use_compression = use_compression
        
        # Keep track of key/value caches and attention matrices
        self.key_cache = {}
        self.value_cache = {}
        self.attention_matrices = {}
        
        # Monkey patch attention modules to capture attention weights
        self._patch_attention_modules()
        
        # Statistics
        self.stats = {
            'inference_time': 0,
            'tokens_processed': 0,
            'tokens_per_second': 0,
            'peak_memory_usage': 0,
            'compression_stats': {}
        }
    
    def _patch_attention_modules(self):
        """
        Patch the attention modules to capture attention weights.
        """
        # This is model-specific, so we'll implement a simple approach for common HF models
        for layer_idx, layer in enumerate(self.model.transformer.h):
            # Store original attention forward method
            original_forward = layer.attn.forward
            
            # Define new forward method that captures attention weights
            def make_new_forward(original_function, layer_idx):
                def new_forward(*args, **kwargs):
                    # Call original function
                    outputs = original_function(*args, **kwargs)
                    
                    # Store attention weights if available
                    if len(outputs) > 1 and outputs[1] is not None:
                        attn_weights = outputs[1]  # Shape: (batch_size, num_heads, seq_len, seq_len)
                        batch_size, num_heads, seq_len, _ = attn_weights.shape
                        
                        # Store for each head
                        for head_idx in range(num_heads):
                            self.attention_matrices[(layer_idx, head_idx)] = attn_weights[:, head_idx]
                    
                    return outputs
                return new_forward
            
            # Replace original forward with new one
            layer.attn.forward = make_new_forward(original_forward, layer_idx)
    
    def _reset_caches(self):
        """Reset KV caches and attention matrices."""
        self.key_cache = {}
        self.value_cache = {}
        self.attention_matrices = {}
    
    def _extract_kv_caches(self):
        """
        Extract KV caches from the model.
        This is model-specific, so we'll need to adapt it for different model architectures.
        """
        # For demonstration, assuming caches are stored in model.transformer.h[layer_idx].attn.{k_cache, v_cache}
        for layer_idx in range(self.num_layers):
            layer = self.model.transformer.h[layer_idx]
            if hasattr(layer.attn, 'k_cache') and hasattr(layer.attn, 'v_cache'):
                self.key_cache[layer_idx] = layer.attn.k_cache
                self.value_cache[layer_idx] = layer.attn.v_cache
    
    def _apply_compressed_kv_caches(self, compressed_key_cache, compressed_value_cache):
        """
        Apply compressed KV caches to the model.
        This is model-specific, so we'll need to adapt it for different model architectures.
        """
        for layer_idx in range(self.num_layers):
            if layer_idx in compressed_key_cache and layer_idx in compressed_value_cache:
                layer = self.model.transformer.h[layer_idx]
                if hasattr(layer.attn, 'k_cache') and hasattr(layer.attn, 'v_cache'):
                    layer.attn.k_cache = compressed_key_cache[layer_idx]
                    layer.attn.v_cache = compressed_value_cache[layer_idx]
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                do_compression: bool = True,
                return_dict: bool = True) -> Dict[str, Any]:
        """
        Forward pass with optional KV cache compression.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels for language modeling
            do_compression: Whether to apply compression
            return_dict: Whether to return dict or tuple
            
        Returns:
            outputs: Model outputs with loss
        """
        # Record start time for inference speed measurement
        start_time = time.time()
        
        # Extract model's current KV caches
        self._extract_kv_caches()
        
        # Run teacher model (full KV cache) forward pass if training with distillation
        if self.training and self.use_compression and do_compression:
            with torch.no_grad():
                teacher_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,  # Don't compute loss yet
                    return_dict=True
                )
                teacher_logits = teacher_outputs.logits
        
        # Apply compression if enabled and not in training mode
        # (during training, compression is used for distillation only)
        if self.use_compression and do_compression and not self.training:
            # Get current KV caches
            # Collect attention matrices during the forward pass
            
            # Apply compression
            compressed_key_cache, compressed_value_cache, compression_stats = self.compressor(
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                attention_matrices=self.attention_matrices
            )
            
            # Update model with compressed caches
            self._apply_compressed_kv_caches(compressed_key_cache, compressed_value_cache)
            
            # Update compression statistics
            self.stats['compression_stats'] = compression_stats
        
        # Run model forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        # Apply distillation loss during training
        if self.training and self.use_compression and do_compression:
            # Combine MLE loss and distillation loss
            mle_loss = outputs.loss
            distill_loss = self.distillation_loss(
                teacher_logits=teacher_logits,
                student_logits=outputs.logits
            )
            
            # Combined loss with weighting
            outputs.loss = mle_loss + self.distillation_weight * distill_loss
        
        # Record statistics
        elapsed_time = time.time() - start_time
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        tokens_processed = batch_size * seq_len
        
        self.stats['inference_time'] = elapsed_time
        self.stats['tokens_processed'] = tokens_processed
        self.stats['tokens_per_second'] = tokens_processed / elapsed_time
        
        # Record peak memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
            self.stats['peak_memory_usage'] = peak_memory
        
        # Reset caches
        self._reset_caches()
        
        return outputs if return_dict else (outputs.loss, outputs.logits)
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 max_length: int = 100,
                 do_sample: bool = True,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 use_compression: Optional[bool] = None,
                 **kwargs) -> torch.Tensor:
        """
        Generate text with KV cache compression.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum generation length
            do_sample: Whether to sample from distribution
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            use_compression: Override default compression setting
            **kwargs: Additional arguments to pass to the model's generate method
            
        Returns:
            generated_ids: Generated token IDs
        """
        # Override compression setting if specified
        original_compression = self.use_compression
        if use_compression is not None:
            self.use_compression = use_compression
        
        # Record start time
        start_time = time.time()
        
        # Generate text
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )
        
        # Restore original compression setting
        self.use_compression = original_compression
        
        # Record statistics
        elapsed_time = time.time() - start_time
        tokens_generated = generated_ids.shape[0] * (generated_ids.shape[1] - input_ids.shape[1])
        
        self.stats['inference_time'] = elapsed_time
        self.stats['tokens_processed'] = tokens_generated
        self.stats['tokens_per_second'] = tokens_generated / elapsed_time
        
        # Record peak memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
            self.stats['peak_memory_usage'] = peak_memory
        
        return generated_ids
    
    def save_pretrained(self, save_dir: str):
        """
        Save model, tokenizer, and compressor configuration.
        
        Args:
            save_dir: Directory to save to
        """
        # Save underlying model and tokenizer
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save compressor configuration
        compressor_config = {
            'max_cache_size': self.compressor.max_cache_size,
            'num_clusters': self.compressor.num_clusters,
            'pruning_interval': self.compressor.pruning_interval,
            'lookback_window': self.compressor.lookback_window,
            'kmeans_learning_rate': self.compressor.online_kmeans.learning_rate,
            'temperature': self.distillation_loss.temperature,
            'distillation_weight': self.distillation_weight,
            'use_compression': self.use_compression
        }
        
        torch.save(compressor_config, f"{save_dir}/compressor_config.pt")
        
    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path: str,
                        **kwargs):
        """
        Load model from pretrained weights, with compressor configuration if available.
        
        Args:
            model_name_or_path: HuggingFace model name or path
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            model: Loaded model
        """
        # Check if compressor config exists
        try:
            compressor_config = torch.load(f"{model_name_or_path}/compressor_config.pt")
            # Update kwargs with saved config
            for k, v in compressor_config.items():
                if k not in kwargs:
                    kwargs[k] = v
        except:
            # No saved compressor config, use defaults or provided kwargs
            pass
        
        # Initialize model
        return cls(model_name_or_path=model_name_or_path, **kwargs)
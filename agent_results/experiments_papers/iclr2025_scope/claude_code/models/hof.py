#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid Optimization Framework (HOF) module.

This module implements the Hybrid Optimization Framework that integrates all components
of the proposed architecture (DSR, SQA, RCKV) and provides a unified end-to-end
training approach with a multi-objective loss function.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import namedtuple

from models.dsr import DynamicSparseRetriever
from models.sqa import SubQuadraticAttention
from models.rckv import RotatingCompressiveKVCache


class TransformerEncoder(nn.Module):
    """
    Simplified Transformer encoder with the proposed sub-quadratic attention mechanism.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        sqa: SubQuadraticAttention,
        rckv: RotatingCompressiveKVCache,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.sqa = sqa
        self.rckv = rckv
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        selection_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the transformer encoder.
        
        Args:
            x: Tensor of shape [batch_size, seq_length, hidden_dim]
            selection_mask: Binary tensor of shape [batch_size, seq_length] indicating
                           tokens selected by the DSR
            attention_mask: Optional tensor of shape [batch_size, seq_length, seq_length]
                           for masking attention weights
            use_cache: Whether to use and update the KV cache
        
        Returns:
            output: Tensor of shape [batch_size, seq_length, hidden_dim]
            attention_info: Dictionary containing attention weights and KV cache info
        """
        # Normalize input
        residual = x
        x = self.norm1(x)
        
        # Process keys and values through RCKV if using cache
        kv_cache_info = {}
        if use_cache:
            cache_output = self.rckv(x, x, mask=selection_mask)
            cached_keys = cache_output['cached_keys']
            cached_values = cache_output['cached_values']
            cache_mask = cache_output['usage_mask']
            
            # Combine current keys/values with cached keys/values
            batch_size, seq_length, _ = x.shape
            buffer_size = cached_keys.size(1)
            
            # Create combined keys and values
            combined_keys = torch.cat([cached_keys, x], dim=1)
            combined_values = torch.cat([cached_values, x], dim=1)
            
            # Create combined mask
            combined_mask = torch.cat([
                cache_mask,
                torch.ones(batch_size, seq_length, device=x.device)
                if selection_mask is None else selection_mask
            ], dim=1)
            
            # Store info for return
            kv_cache_info = {
                'cached_keys': cached_keys,
                'cached_values': cached_values,
                'cache_mask': cache_mask,
                'combined_keys': combined_keys,
                'combined_values': combined_values,
                'combined_mask': combined_mask
            }
            
            # Use combined keys/values for attention
            attn_output, attn_weights = self.sqa(
                query=x,
                key=combined_keys,
                value=combined_values,
                selection_mask=combined_mask,
                attention_mask=attention_mask
            )
        else:
            # Standard attention without cache
            attn_output, attn_weights = self.sqa(
                query=x,
                key=x,
                value=x,
                selection_mask=selection_mask,
                attention_mask=attention_mask
            )
        
        # Residual connection
        x = residual + self.dropout(attn_output)
        
        # Feed-forward network
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        
        return x, {'attention_weights': attn_weights, 'kv_cache': kv_cache_info}


class TransformerDecoderLayer(nn.Module):
    """
    Simplified Transformer decoder layer with the proposed sub-quadratic attention mechanism.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        sqa: SubQuadraticAttention,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.sqa = sqa
        
        # Self-attention
        self.self_attn = SubQuadraticAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_clusters=sqa.num_clusters,
            top_k_clusters=sqa.top_k_clusters,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        selection_mask: Optional[torch.Tensor] = None,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the transformer decoder layer.
        
        Args:
            x: Tensor of shape [batch_size, seq_length, hidden_dim]
            memory: Tensor of shape [batch_size, memory_length, hidden_dim]
            selection_mask: Binary tensor of shape [batch_size, memory_length] indicating
                           tokens selected by the DSR
            self_attention_mask: Optional tensor for masking self-attention
            cross_attention_mask: Optional tensor for masking cross-attention
        
        Returns:
            output: Tensor of shape [batch_size, seq_length, hidden_dim]
            attention_info: Dictionary containing attention weights
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        self_attn_output, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            attention_mask=self_attention_mask
        )
        
        x = residual + self.dropout(self_attn_output)
        
        # Cross-attention with memory and SQA
        residual = x
        x = self.norm2(x)
        
        cross_attn_output, cross_attn_weights = self.sqa(
            query=x,
            key=memory,
            value=memory,
            selection_mask=selection_mask,
            attention_mask=cross_attention_mask
        )
        
        x = residual + self.dropout(cross_attn_output)
        
        # Feed-forward network
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout(self.ffn(x))
        
        return x, {
            'self_attention_weights': self_attn_weights,
            'cross_attention_weights': cross_attn_weights
        }


class HybridOptimizationFramework(nn.Module):
    """
    Hybrid Optimization Framework that integrates all components and
    provides end-to-end training with a multi-objective loss function.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dsr: Optional[DynamicSparseRetriever] = None,
        sqa: Optional[SubQuadraticAttention] = None,
        rckv: Optional[RotatingCompressiveKVCache] = None,
        lambda_task: float = 1.0,
        lambda_retrieval: float = 0.5,
        lambda_compression: float = 0.3,
        lambda_compute: float = 0.2,
        ramp_up_period: int = 1000,
        pad_token_id: int = 0,
        dropout: float = 0.1
    ):
        """
        Initialize the Hybrid Optimization Framework.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of hidden states
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dsr: Dynamic Sparse Retriever module (created if None)
            sqa: Sub-Quadratic Sparse Attention module (created if None)
            rckv: Rotating Compressive KV Cache module (created if None)
            lambda_task: Weight for task loss
            lambda_retrieval: Weight for retrieval loss
            lambda_compression: Weight for compression loss
            lambda_compute: Weight for compute loss
            ramp_up_period: Ramp-up period for curriculum learning
            pad_token_id: ID of the padding token
            dropout: Dropout probability
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        
        # Loss weights
        self.lambda_task = lambda_task
        self.lambda_retrieval = lambda_retrieval
        self.lambda_compression = lambda_compression
        self.lambda_compute = lambda_compute
        self.ramp_up_period = ramp_up_period
        
        # Current training step (for curriculum learning)
        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(4096, embedding_dim)  # Fixed max length
        
        # Initialize or use provided components
        if dsr is None:
            self.dsr = DynamicSparseRetriever(
                embedding_dim=embedding_dim,
                reduced_dim=hidden_dim // 4,
                base_budget=512,
                alpha=0.5
            )
        else:
            self.dsr = dsr
        
        if sqa is None:
            self.sqa = SubQuadraticAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_clusters=32,
                top_k_clusters=8,
                dropout=dropout
            )
        else:
            self.sqa = sqa
        
        if rckv is None:
            head_dim = hidden_dim // num_heads
            self.rckv = RotatingCompressiveKVCache(
                key_dim=head_dim,
                value_dim=head_dim,
                compressed_dim=head_dim // 2,
                buffer_size=1024
            )
        else:
            self.rckv = rckv
        
        # Input projection
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                sqa=self.sqa,
                rckv=self.rckv,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Decoder layers (optional for generation tasks)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                sqa=self.sqa,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Create an output type for structured returns
        self.ModelOutput = namedtuple(
            "ModelOutput",
            ["logits", "loss", "attention_weights", "selection_mask", "cache_info"]
        )
    
    def _get_current_loss_weights(self) -> Tuple[float, float, float, float]:
        """
        Get the current loss weights based on curriculum learning schedule.
        
        Returns:
            Tuple of (lambda_task, lambda_retrieval, lambda_compression, lambda_compute)
        """
        # Scale based on current step
        progress = min(1.0, self.current_step.item() / self.ramp_up_period)
        
        return (
            self.lambda_task,
            self.lambda_retrieval * progress,
            self.lambda_compression * progress,
            self.lambda_compute * progress
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        return_dict: bool = True
    ) -> Union[Tuple, namedtuple]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_length] containing token ids
            attention_mask: Optional tensor of shape [batch_size, seq_length] indicating
                           valid tokens (1) and padding (0)
            labels: Optional tensor of shape [batch_size, seq_length] containing target ids
            use_cache: Whether to use the KV cache
            return_dict: Whether to return a namedtuple
        
        Returns:
            If return_dict=True, a namedtuple containing:
                - logits: Tensor of shape [batch_size, seq_length, vocab_size]
                - loss: Loss value if labels are provided, otherwise None
                - attention_weights: Dictionary of attention weights
                - selection_mask: Binary tensor indicating tokens selected by DSR
                - cache_info: Information about the KV cache
            
            If return_dict=False, a tuple of:
                (logits, loss, attention_weights, selection_mask, cache_info)
        """
        batch_size, seq_length = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Add positional embeddings
        positions = torch.arange(seq_length, device=input_ids.device).expand(batch_size, seq_length)
        pos_embeds = self.pos_embedding(positions)
        
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)
        
        # Dynamic sparse retrieval of relevant tokens
        retrieval_output = self.dsr(
            query_embeddings=embeddings,
            context_embeddings=embeddings,
            context_mask=attention_mask,
            return_scores=True
        )
        
        selection_mask = retrieval_output['selection_mask']
        relevance_scores = retrieval_output['relevance_scores']
        
        # Project embeddings to hidden dimension
        hidden_states = self.input_projection(embeddings)
        
        # Process through encoder layers
        all_attention_weights = []
        cache_info = {}
        
        for layer_idx, encoder_layer in enumerate(self.encoder_layers):
            hidden_states, layer_outputs = encoder_layer(
                hidden_states,
                selection_mask=selection_mask,
                attention_mask=attention_mask.unsqueeze(1).unsqueeze(2),  # [batch_size, 1, 1, seq_length]
                use_cache=use_cache
            )
            
            all_attention_weights.append(layer_outputs['attention_weights'])
            
            if layer_idx == 0:
                cache_info = layer_outputs['kv_cache']
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        # If we are computing the loss (training or validation)
        loss = None
        if labels is not None:
            # Increment step counter (for curriculum learning)
            if self.training:
                self.current_step += 1
            
            # Get loss weights
            lambda_task, lambda_retrieval, lambda_compression, lambda_compute = self._get_current_loss_weights()
            
            # Task loss (next token prediction)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            task_loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
            
            # Retrieval loss (if DSR is enabled)
            retrieval_loss = torch.tensor(0.0, device=logits.device)
            if lambda_retrieval > 0 and hasattr(self.dsr, 'compute_rl_loss'):
                # Pseudo-reward based on prediction accuracy
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    correct = (predictions == labels) & (labels != -100)
                    task_reward = correct.float().mean(dim=-1)
                
                retrieval_loss = self.dsr.compute_rl_loss(
                    task_reward=task_reward,
                    selection_mask=selection_mask,
                    token_count_penalty=0.01
                )
            
            # Compression loss (if RCKV is enabled)
            compression_loss = torch.tensor(0.0, device=logits.device)
            if lambda_compression > 0 and hasattr(self.rckv, 'get_reconstruction_error'):
                compression_loss = self.rckv.get_reconstruction_error()
            
            # Compute efficiency loss (if components report metrics)
            compute_loss = torch.tensor(0.0, device=logits.device)
            if lambda_compute > 0:
                # Penalize excessive token selection
                if hasattr(self.dsr, 'get_token_efficiency'):
                    token_efficiency = self.dsr.get_token_efficiency()
                    # Lower efficiency (higher ratio) means higher loss
                    efficiency_loss = token_efficiency
                    compute_loss += efficiency_loss
                
                # Penalize excessive cluster usage
                if hasattr(self.sqa, 'get_active_clusters'):
                    active_clusters = self.sqa.get_active_clusters()
                    # More active clusters means higher loss
                    cluster_loss = active_clusters / self.sqa.num_clusters
                    compute_loss += cluster_loss
            
            # Combine losses
            loss = (
                lambda_task * task_loss +
                lambda_retrieval * retrieval_loss +
                lambda_compression * compression_loss +
                lambda_compute * compute_loss
            )
        
        if return_dict:
            return self.ModelOutput(
                logits=logits,
                loss=loss,
                attention_weights=all_attention_weights,
                selection_mask=selection_mask,
                cache_info=cache_info
            )
        else:
            return logits, loss, all_attention_weights, selection_mask, cache_info
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        min_length: int = 0,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        early_stopping: bool = False
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_length] containing token ids
            attention_mask: Optional tensor of shape [batch_size, seq_length] indicating
                           valid tokens (1) and padding (0)
            max_length: Maximum length of generated sequences
            min_length: Minimum length of generated sequences
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling
            top_k: Top-k filtering parameter
            top_p: Top-p filtering parameter
            repetition_penalty: Repetition penalty parameter
            no_repeat_ngram_size: Size of n-grams to avoid repeating
            early_stopping: Whether to stop generation when all beams are finished
        
        Returns:
            generated_ids: Tensor of shape [batch_size, max_length] containing generated token ids
        """
        # This is a simplified generation method - in practice, you'd use more sophisticated
        # beam search or sampling algorithms
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Start with the input ids
        curr_ids = input_ids
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(curr_ids)
        
        # Generate tokens auto-regressively
        for _ in range(max_length - curr_ids.shape[1]):
            # Forward pass
            outputs = self.forward(
                input_ids=curr_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
            
            # Get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(curr_ids[i].tolist()):
                        if token_id != self.pad_token_id:
                            next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(-1, top_k_indices, top_k_values)
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Append next token to sequence
            next_token = next_token.unsqueeze(-1)
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)
            
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=device)
            ], dim=1)
        
        return curr_ids
    
    def evaluate_adaptation(self) -> Dict[str, float]:
        """
        Evaluate the model's adaptation capabilities.
        
        Returns:
            Dictionary containing adaptation metrics:
                - information_retention: Score for information retention over time
                - temporal_consistency: Score for consistency in streaming scenarios
                - adaptation_speed: Score for adaptation speed to new contexts
        """
        # Information retention from RCKV
        information_retention = self.rckv.get_information_retention() if hasattr(self.rckv, 'get_information_retention') else 0.0
        
        # Temporal consistency (placeholder - would be measured with task-specific metrics)
        temporal_consistency = 0.5
        
        # Adaptation speed (placeholder - would be measured with task-specific metrics)
        adaptation_speed = 0.5
        
        return {
            'information_retention': information_retention,
            'temporal_consistency': temporal_consistency,
            'adaptation_speed': adaptation_speed
        }
    
    def get_token_efficiency(self) -> float:
        """
        Get the token efficiency ratio.
        
        Returns:
            efficiency: Token efficiency ratio (selected tokens / total tokens)
        """
        if hasattr(self.dsr, 'get_token_efficiency'):
            return self.dsr.get_token_efficiency()
        return 1.0  # Default to 1.0 (no efficiency) if DSR not available
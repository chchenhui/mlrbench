#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dynamic Sparse Retriever (DSR) module.

This module implements the Dynamic Sparse Retriever component of the proposed
sub-quadratic architecture. The DSR selectively fetches context tokens most
relevant to the input query, minimizing redundant prefill.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class QueryAnalyzer(nn.Module):
    """
    A lightweight module that estimates query complexity.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()
        
    def forward(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Estimate the complexity of a query.
        
        Args:
            query_embeddings: Tensor of shape [batch_size, query_length, embedding_dim]
        
        Returns:
            complexity: Tensor of shape [batch_size] containing complexity scores
        """
        # Average pooling over token dimension
        pooled = torch.mean(query_embeddings, dim=1)  # [batch_size, embedding_dim]
        
        # Feed through MLP
        hidden = self.activation(self.fc1(pooled))
        complexity = torch.sigmoid(self.fc2(hidden)).squeeze(-1)  # [batch_size]
        
        return complexity


class DynamicSparseRetriever(nn.Module):
    """
    Dynamic Sparse Retriever module that selects the most relevant tokens 
    from the context window based on the current query.
    """
    def __init__(
        self,
        embedding_dim: int,
        reduced_dim: int,
        base_budget: int = 512,
        alpha: float = 0.5,
        temperature: float = 1.0
    ):
        """
        Initialize the Dynamic Sparse Retriever.
        
        Args:
            embedding_dim: Dimension of the input embeddings
            reduced_dim: Dimension of the reduced representations
            base_budget: Base number of tokens to select
            alpha: Adaptation sensitivity for query complexity
            temperature: Temperature for sampling during RL training
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.reduced_dim = reduced_dim
        self.base_budget = base_budget
        self.alpha = alpha
        self.temperature = temperature
        
        # Query and context encoders (implemented as dimension reduction projections)
        self.query_encoder = nn.Linear(embedding_dim, reduced_dim)
        self.context_encoder = nn.Linear(embedding_dim, reduced_dim)
        
        # Query complexity analyzer
        self.query_analyzer = QueryAnalyzer(embedding_dim)
        
        # Track metrics for RL training and evaluation
        self.selected_tokens = None
        self.selection_logits = None
        self.token_count = 0
        self.total_tokens = 0
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        context_embeddings: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        return_scores: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute relevance scores and select tokens from the context.
        
        Args:
            query_embeddings: Tensor of shape [batch_size, query_length, embedding_dim]
            context_embeddings: Tensor of shape [batch_size, context_length, embedding_dim]
            context_mask: Optional tensor of shape [batch_size, context_length] indicating valid context tokens
            return_scores: Whether to return the relevance scores
        
        Returns:
            Dictionary containing:
                - selected_tokens: Tensor of shape [batch_size, selected_length] containing indices of selected tokens
                - selection_mask: Binary tensor of shape [batch_size, context_length] indicating selected tokens
                - relevance_scores: Tensor of shape [batch_size, context_length] containing token relevance scores
                  (only if return_scores is True)
        """
        batch_size, query_length, _ = query_embeddings.shape
        _, context_length, _ = context_embeddings.shape
        
        # Apply mask if provided
        if context_mask is None:
            context_mask = torch.ones(batch_size, context_length, device=query_embeddings.device)
        
        # Project embeddings to reduced dimension
        query_reduced = self.query_encoder(query_embeddings)  # [batch_size, query_length, reduced_dim]
        context_reduced = self.context_encoder(context_embeddings)  # [batch_size, context_length, reduced_dim]
        
        # Normalize embeddings
        query_reduced = F.normalize(query_reduced, p=2, dim=-1)
        context_reduced = F.normalize(context_reduced, p=2, dim=-1)
        
        # Compute relevance scores between query and context tokens
        # First, average pool query tokens
        query_pooled = torch.mean(query_reduced, dim=1)  # [batch_size, reduced_dim]
        query_pooled = F.normalize(query_pooled, p=2, dim=-1)
        
        # Compute similarity scores
        # [batch_size, context_length]
        relevance_scores = torch.bmm(
            query_pooled.unsqueeze(1),
            context_reduced.transpose(1, 2)
        ).squeeze(1)
        
        # Apply mask to scores (set masked tokens to -inf)
        relevance_scores = relevance_scores.masked_fill(~context_mask.bool(), float('-inf'))
        
        # Estimate query complexity and adjust budget
        complexity = self.query_analyzer(query_embeddings)  # [batch_size]
        token_budget = torch.round(
            self.base_budget * (1 + self.alpha * complexity)
        ).long()  # [batch_size]
        
        # Ensure budget doesn't exceed context length
        token_budget = torch.minimum(token_budget, torch.sum(context_mask, dim=1))
        
        # Select top-k tokens per batch based on relevance scores
        # Get indices of top-k tokens
        # For RL training during exploration, we'll use these scores as logits for sampling
        self.selection_logits = relevance_scores / self.temperature
        
        if self.training:
            # During training, use Gumbel-softmax for differentiable sampling
            # This enables exploration while maintaining differentiability
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(relevance_scores) + 1e-10) + 1e-10)
            noisy_logits = relevance_scores + gumbel_noise
            
            # Get indices of top-k tokens per batch item
            _, selected_indices = [], []
            selection_mask = torch.zeros_like(context_mask)
            
            for i in range(batch_size):
                budget = token_budget[i].item()
                # Get top-k indices for this batch item
                _, indices = torch.topk(noisy_logits[i], k=budget)
                
                # Create selection mask
                batch_mask = torch.zeros_like(context_mask[i])
                batch_mask[indices] = 1
                selection_mask[i] = batch_mask
                
                selected_indices.append(indices)
            
            # Track selected token count for efficiency metrics
            self.token_count = token_budget.sum().item()
            self.total_tokens = context_mask.sum().item()
            
            # For RL training, we also need to track the actual indices
            self.selected_tokens = selected_indices
        else:
            # During inference, just take the top-k deterministically
            _, selected_indices = [], []
            selection_mask = torch.zeros_like(context_mask)
            
            for i in range(batch_size):
                budget = token_budget[i].item()
                # Get top-k indices for this batch item
                _, indices = torch.topk(relevance_scores[i], k=budget)
                
                # Create selection mask
                batch_mask = torch.zeros_like(context_mask[i])
                batch_mask[indices] = 1
                selection_mask[i] = batch_mask
                
                selected_indices.append(indices)
            
            # Track token efficiency
            self.token_count = token_budget.sum().item()
            self.total_tokens = context_mask.sum().item()
            
            self.selected_tokens = selected_indices
        
        result = {
            'selected_tokens': selected_indices,
            'selection_mask': selection_mask
        }
        
        if return_scores:
            result['relevance_scores'] = relevance_scores
        
        return result
    
    def get_token_efficiency(self) -> float:
        """
        Calculate token efficiency as the ratio of selected tokens to total tokens.
        
        Returns:
            efficiency: Token efficiency ratio
        """
        if self.total_tokens == 0:
            return 0.0
        return self.token_count / self.total_tokens
    
    def compute_rl_loss(
        self,
        task_reward: torch.Tensor,
        selection_mask: torch.Tensor,
        token_count_penalty: float = 0.01
    ) -> torch.Tensor:
        """
        Compute the reinforcement learning loss for optimizing token selection.
        
        Args:
            task_reward: Tensor of shape [batch_size] containing task rewards
            selection_mask: Binary tensor of shape [batch_size, context_length] indicating selected tokens
            token_count_penalty: Penalty factor for the number of selected tokens
        
        Returns:
            loss: RL loss value
        """
        batch_size = task_reward.shape[0]
        
        # Compute token count penalty
        token_counts = torch.sum(selection_mask, dim=1)  # [batch_size]
        token_penalty = token_counts * token_count_penalty
        
        # Combine rewards (higher task reward, lower token count is better)
        combined_reward = task_reward - token_penalty
        
        # Compute policy gradient loss
        log_probs = []
        
        for i in range(batch_size):
            # Get the log probabilities of the selected tokens
            selected_logits = self.selection_logits[i][self.selected_tokens[i]]
            selected_log_probs = F.log_softmax(selected_logits, dim=0)
            log_probs.append(selected_log_probs.mean())
        
        log_probs = torch.stack(log_probs)
        
        # Policy gradient loss
        loss = -torch.mean(log_probs * combined_reward)
        
        return loss
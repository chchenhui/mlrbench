"""
Token Relevance Prediction module for the ATSKV (Adaptive Token-Relevance Sparse KV-Cache).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class TokenRelevancePredictor(nn.Module):
    """
    Lightweight neural network to predict the relevance of each token for KV cache retention.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        feature_dim: int = 64,
        hidden_dim: int = 32
    ):
        """
        Initialize the token relevance predictor.
        
        Args:
            hidden_size: Hidden size of the transformer model
            num_attention_heads: Number of attention heads in the transformer model
            feature_dim: Dimension of the feature vector
            hidden_dim: Dimension of the hidden layer in the MLP
        """
        super().__init__()
        
        # Input features:
        # 1. Hidden state: [hidden_size]
        # 2. Attention features: [num_attention_heads]
        # 3. Handcrafted features: [feature_dim]
        input_dim = hidden_size + num_attention_heads + feature_dim
        
        # MLP for relevance prediction
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Feature extractors
        self.hidden_state_projector = nn.Linear(hidden_size, hidden_size)
        self.attention_projector = nn.Linear(num_attention_heads, num_attention_heads)
        self.feature_projector = nn.Linear(feature_dim, feature_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_patterns: torch.Tensor,
        handcrafted_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to compute relevance scores.
        
        Args:
            hidden_states: Hidden state representations [batch_size, seq_len, hidden_size]
            attention_patterns: Attention patterns [batch_size, seq_len, num_heads]
            handcrafted_features: Handcrafted features [batch_size, seq_len, feature_dim]
            
        Returns:
            Relevance scores [batch_size, seq_len]
        """
        # Project each feature type
        hidden_features = self.hidden_state_projector(hidden_states)
        attention_features = self.attention_projector(attention_patterns)
        custom_features = self.feature_projector(handcrafted_features)
        
        # Concatenate features
        combined_features = torch.cat([hidden_features, attention_features, custom_features], dim=-1)
        
        # Compute relevance scores
        relevance_scores = self.mlp(combined_features).squeeze(-1)
        
        return relevance_scores

class AttentionStatisticsExtractor:
    """
    Extract statistics from attention patterns to create features for the relevance predictor.
    """
    def __init__(self, num_heads: int, seq_len: int):
        """
        Initialize the attention statistics extractor.
        
        Args:
            num_heads: Number of attention heads
            seq_len: Maximum sequence length
        """
        self.num_heads = num_heads
        self.seq_len = seq_len
        
    def extract_statistics(
        self,
        attention_scores: torch.Tensor,
        layer_idx: int,
        token_types: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract attention statistics for relevance prediction.
        
        Args:
            attention_scores: Attention scores [batch_size, num_heads, seq_len, seq_len]
            layer_idx: Index of the current layer
            token_types: Optional tensor indicating token types [batch_size, seq_len]
            
        Returns:
            Attention features [batch_size, seq_len, num_heads]
        """
        batch_size = attention_scores.shape[0]
        device = attention_scores.device
        
        # Initialize features
        attention_features = torch.zeros(
            (batch_size, self.seq_len, self.num_heads),
            device=device
        )
        
        # Extract features for each head
        for head_idx in range(self.num_heads):
            # Get attention scores for this head
            head_scores = attention_scores[:, head_idx, :, :]  # [batch_size, seq_len, seq_len]
            
            # 1. Mean attention received by each token
            mean_attention_received = head_scores.mean(dim=1)  # [batch_size, seq_len]
            
            # Store the feature
            attention_features[:, :, head_idx] = mean_attention_received
        
        return attention_features

class HandcraftedFeatureExtractor(nn.Module):
    """
    Extract handcrafted features for token relevance prediction.
    """
    def __init__(self, feature_dim: int, vocab_size: int):
        """
        Initialize the handcrafted feature extractor.
        
        Args:
            feature_dim: Dimension of the feature vector
            vocab_size: Size of the vocabulary
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size
        
        # Embeddings for token types
        self.token_type_embedding = nn.Embedding(10, feature_dim // 4)  # Assuming up to 10 token types
        
    def extract_features(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        hidden_state_norms: Optional[torch.Tensor] = None,
        layer_idx: int = 0
    ) -> torch.Tensor:
        """
        Extract handcrafted features for relevance prediction.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            token_type_ids: Optional token type IDs [batch_size, seq_len]
            positions: Optional positions [batch_size, seq_len]
            hidden_state_norms: Optional norms of hidden states [batch_size, seq_len]
            layer_idx: Index of the current layer
            
        Returns:
            Handcrafted features [batch_size, seq_len, feature_dim]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize feature tensor
        features = torch.zeros(
            (batch_size, seq_len, self.feature_dim),
            device=device
        )
        
        # 1. Token type features (if provided)
        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embedding(token_type_ids)
            features[:, :, :token_type_embeddings.size(-1)] = token_type_embeddings
        
        # 2. Positional information (normalized)
        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        normalized_positions = positions.float() / seq_len
        features[:, :, self.feature_dim // 4] = normalized_positions
        
        # 3. Special token indicators (e.g., [CLS], [SEP], etc.)
        # This is a simplified implementation - in practice, you'd identify special tokens based on your tokenizer
        special_token_indicator = (input_ids < 5).float()  # Assuming first few token IDs are special tokens
        features[:, :, self.feature_dim // 4 + 1] = special_token_indicator
        
        # 4. Hidden state norms (if provided)
        if hidden_state_norms is not None:
            features[:, :, self.feature_dim // 4 + 2] = hidden_state_norms / hidden_state_norms.max()
        
        # 5. Layer-specific embedding
        layer_embedding = torch.ones((batch_size, seq_len), device=device) * (layer_idx / 100.0)
        features[:, :, self.feature_dim // 4 + 3] = layer_embedding
        
        # Fill remaining features with zeros (placeholder for more handcrafted features)
        
        return features
    
class RelevanceThresholdController:
    """
    Control the threshold for token retention based on relevance scores.
    """
    def __init__(
        self,
        num_layers: int,
        initial_quantile: float = 0.7,
        min_quantile: float = 0.5,
        max_quantile: float = 0.9,
        beta: float = 1.0,
        lambda_momentum: float = 0.8
    ):
        """
        Initialize the relevance threshold controller.
        
        Args:
            num_layers: Number of layers in the model
            initial_quantile: Initial quantile threshold
            min_quantile: Minimum quantile threshold
            max_quantile: Maximum quantile threshold
            beta: Scaling factor for thresholds
            lambda_momentum: Momentum factor for mask updates
        """
        self.num_layers = num_layers
        self.initial_quantile = initial_quantile
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile
        self.beta = beta
        self.lambda_momentum = lambda_momentum
        
        # Initialize layer-specific thresholds
        self.layer_quantiles = {l: initial_quantile for l in range(num_layers)}
        self.layer_betas = {l: beta for l in range(num_layers)}
        
        # Cache for previous masks
        self.previous_masks = {}
        
    def compute_threshold(
        self,
        relevance_scores: torch.Tensor,
        layer_idx: int,
        current_memory: float,
        target_memory: float
    ) -> Tuple[float, torch.Tensor]:
        """
        Compute the threshold for token retention.
        
        Args:
            relevance_scores: Relevance scores [batch_size, seq_len]
            layer_idx: Index of the current layer
            current_memory: Current memory usage
            target_memory: Target memory usage
            
        Returns:
            Tuple of (threshold_value, binary_mask)
        """
        # Get the current quantile
        q = self.layer_quantiles[layer_idx]
        
        # Adjust quantile based on memory usage
        memory_ratio = min(1.0, current_memory / target_memory) if target_memory > 0 else 1.0
        q_adjusted = self.min_quantile + (self.max_quantile - self.min_quantile) * memory_ratio
        
        # Update layer quantile with some smoothing
        self.layer_quantiles[layer_idx] = 0.9 * self.layer_quantiles[layer_idx] + 0.1 * q_adjusted
        
        # Compute the threshold value
        flat_scores = relevance_scores.reshape(-1)
        k = int(len(flat_scores) * self.layer_quantiles[layer_idx])
        if k < 1:
            k = 1
        elif k >= len(flat_scores):
            k = len(flat_scores) - 1
            
        # Get the k-th largest value as threshold
        threshold = torch.kthvalue(flat_scores, len(flat_scores) - k).values.item()
        threshold = threshold * self.layer_betas[layer_idx]
        
        # Create binary mask
        mask = (relevance_scores > threshold).float()
        
        # Apply momentum for stability if previous mask exists
        if layer_idx in self.previous_masks:
            mask = self.lambda_momentum * self.previous_masks[layer_idx] + (1 - self.lambda_momentum) * mask
            # Re-binarize
            mask = (mask > 0.5).float()
        
        # Store the mask for next iteration
        self.previous_masks[layer_idx] = mask.clone()
        
        return threshold, mask
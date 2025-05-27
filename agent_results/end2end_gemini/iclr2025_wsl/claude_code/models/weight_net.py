import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import math
import logging
from einops import rearrange, repeat

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        """
        Args:
            d_model: Hidden dimension of the embedding
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension [1, max_seq_length, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings [batch_size, seq_length, embedding_dim]
            
        Returns:
            Embeddings with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]

class SegmentEncoding(nn.Module):
    """Segment encoding for distinguishing different parts of the input."""
    
    def __init__(self, d_model: int, num_segments: int = 10):
        """
        Args:
            d_model: Hidden dimension of the embedding
            num_segments: Number of distinct segments
        """
        super().__init__()
        
        # Create learnable segment embeddings
        self.segment_embeddings = nn.Embedding(num_segments, d_model)
    
    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        """
        Add segment encoding to input embeddings.
        
        Args:
            x: Input embeddings [batch_size, seq_length, embedding_dim]
            segment_ids: Segment IDs for each position [batch_size, seq_length]
            
        Returns:
            Embeddings with segment encoding added
        """
        segment_embeddings = self.segment_embeddings(segment_ids)
        return x + segment_embeddings

class PermutationInvariantAttention(nn.Module):
    """Attention module that is invariant to permutations within specified segments."""
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Output projection
        self.output = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim).
        
        Args:
            x: [batch_size, seq_length, d_model]
            
        Returns:
            [batch_size, num_heads, seq_length, head_dim]
        """
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge the (num_heads, head_dim) into d_model.
        
        Args:
            x: [batch_size, num_heads, seq_length, head_dim]
            
        Returns:
            [batch_size, seq_length, d_model]
        """
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_indices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for permutation-invariant attention.

        Args:
            query: Query embeddings [batch_size, seq_length, d_model]
            key: Key embeddings [batch_size, seq_length, d_model]
            value: Value embeddings [batch_size, seq_length, d_model]
            layer_indices: Layer indices for tokens [batch_size, seq_length]
            attention_mask: Optional attention mask [batch_size, seq_length]

        Returns:
            Output embeddings [batch_size, seq_length, d_model]
        """
        batch_size = query.size(0)

        # Linear projections
        q = self.query(query)  # [batch_size, seq_length, d_model]
        k = self.key(key)  # [batch_size, seq_length, d_model]
        v = self.value(value)  # [batch_size, seq_length, d_model]

        # Split heads
        q = self._split_heads(q)  # [batch_size, num_heads, seq_length, head_dim]
        k = self._split_heads(k)  # [batch_size, num_heads, seq_length, head_dim]
        v = self._split_heads(v)  # [batch_size, num_heads, seq_length, head_dim]

        # Scale query
        q = q / math.sqrt(self.head_dim)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq_length, seq_length]

        # Apply permutation invariance within layers
        # Create a mask where tokens in the same layer can attend to each other
        layer_mask = (layer_indices.unsqueeze(2) == layer_indices.unsqueeze(1))  # [batch_size, seq_length, seq_length]
        # Handle case where layer_indices are all 0 by making sure masks have at least one attention path
        layer_mask = layer_mask | (torch.sum(layer_mask, dim=-1, keepdim=True) == 0)
        layer_mask = layer_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [batch_size, num_heads, seq_length, seq_length]

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention mask to match scores shape
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_length]
            attention_mask = (1.0 - attention_mask) * -10000.0  # Convert mask to additive
            scores = scores + attention_mask

        # Apply layer mask (set scores to -inf for tokens not in the same layer)
        scores = scores.masked_fill(~layer_mask, -10000.0)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_length, seq_length]
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)  # [batch_size, num_heads, seq_length, head_dim]

        # Merge heads
        context = self._merge_heads(context)  # [batch_size, seq_length, d_model]

        # Apply output projection
        output = self.output(context)  # [batch_size, seq_length, d_model]

        return output

class PermutationInvariantTransformerLayer(nn.Module):
    """Transformer layer with permutation-invariant attention."""
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int = 2048, 
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward layer
            dropout: Dropout probability
        """
        super().__init__()
        
        # Permutation-invariant multi-head attention
        self.attention = PermutationInvariantAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        layer_indices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for transformer layer.
        
        Args:
            x: Input embeddings [batch_size, seq_length, d_model]
            layer_indices: Layer indices for tokens [batch_size, seq_length]
            attention_mask: Optional attention mask [batch_size, seq_length]
            
        Returns:
            Output embeddings [batch_size, seq_length, d_model]
        """
        # Self-attention
        attn_output = self.attention(
            query=x,
            key=x,
            value=x,
            layer_indices=layer_indices,
            attention_mask=attention_mask,
        )
        
        # Add & Norm
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        
        # Add & Norm
        x = self.layer_norm2(x + ff_output)
        
        return x

class CrossLayerAttention(nn.Module):
    """Attention module for communication across different layers."""
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Output projection
        self.output = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim).
        
        Args:
            x: [batch_size, seq_length, d_model]
            
        Returns:
            [batch_size, num_heads, seq_length, head_dim]
        """
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge the (num_heads, head_dim) into d_model.
        
        Args:
            x: [batch_size, num_heads, seq_length, head_dim]
            
        Returns:
            [batch_size, seq_length, d_model]
        """
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for cross-layer attention.
        
        Args:
            query: Query embeddings [batch_size, seq_length, d_model]
            key: Key embeddings [batch_size, seq_length, d_model]
            value: Value embeddings [batch_size, seq_length, d_model]
            attention_mask: Optional attention mask [batch_size, seq_length]
            
        Returns:
            Output embeddings [batch_size, seq_length, d_model]
        """
        batch_size = query.size(0)
        
        # Linear projections
        q = self.query(query)  # [batch_size, seq_length, d_model]
        k = self.key(key)  # [batch_size, seq_length, d_model]
        v = self.value(value)  # [batch_size, seq_length, d_model]
        
        # Split heads
        q = self._split_heads(q)  # [batch_size, num_heads, seq_length, head_dim]
        k = self._split_heads(k)  # [batch_size, num_heads, seq_length, head_dim]
        v = self._split_heads(v)  # [batch_size, num_heads, seq_length, head_dim]
        
        # Scale query
        q = q / math.sqrt(self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq_length, seq_length]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention mask to match scores shape
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_length]
            attention_mask = (1.0 - attention_mask) * -10000.0  # Convert mask to additive
            scores = scores + attention_mask
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_length, seq_length]
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)  # [batch_size, num_heads, seq_length, head_dim]
        
        # Merge heads
        context = self._merge_heads(context)  # [batch_size, seq_length, d_model]
        
        # Apply output projection
        output = self.output(context)  # [batch_size, seq_length, d_model]
        
        return output

class CrossLayerTransformerLayer(nn.Module):
    """Transformer layer for communication across different layers."""
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int = 2048, 
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward layer
            dropout: Dropout probability
        """
        super().__init__()
        
        # Cross-layer multi-head attention
        self.attention = CrossLayerAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for transformer layer.
        
        Args:
            x: Input embeddings [batch_size, seq_length, d_model]
            attention_mask: Optional attention mask [batch_size, seq_length]
            
        Returns:
            Output embeddings [batch_size, seq_length, d_model]
        """
        # Self-attention
        attn_output = self.attention(
            query=x,
            key=x,
            value=x,
            attention_mask=attention_mask,
        )
        
        # Add & Norm
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        
        # Add & Norm
        x = self.layer_norm2(x + ff_output)
        
        return x

class WeightNetTransformer(nn.Module):
    """
    Permutation-invariant transformer model for weight-based property prediction.
    
    This model has a two-stage architecture:
    1. Permutation-invariant attention within each layer
    2. Cross-layer attention between different layers
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_intra_layer_heads: int = 4,
        num_cross_layer_heads: int = 8,
        num_intra_layer_blocks: int = 2,
        num_cross_layer_blocks: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 1024,
        num_segments: int = 100,
        num_properties: int = 3,
        token_dim: int = 64,
    ):
        """
        Args:
            d_model: Hidden dimension of the model
            num_intra_layer_heads: Number of attention heads for intra-layer attention
            num_cross_layer_heads: Number of attention heads for cross-layer attention
            num_intra_layer_blocks: Number of intra-layer transformer blocks
            num_cross_layer_blocks: Number of cross-layer transformer blocks
            d_ff: Dimension of feed-forward layer
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            num_segments: Number of distinct segments (layer types)
            num_properties: Number of properties to predict
            token_dim: Dimension of input tokens
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_intra_layer_heads = num_intra_layer_heads
        self.num_cross_layer_heads = num_cross_layer_heads
        self.num_intra_layer_blocks = num_intra_layer_blocks
        self.num_cross_layer_blocks = num_cross_layer_blocks
        
        # Input embedding
        self.embedding = nn.Linear(token_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Segment encoding
        self.segment_encoding = SegmentEncoding(d_model, num_segments)
        
        # Intra-layer transformer blocks
        self.intra_layer_blocks = nn.ModuleList([
            PermutationInvariantTransformerLayer(
                d_model=d_model,
                num_heads=num_intra_layer_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_intra_layer_blocks)
        ])
        
        # Cross-layer transformer blocks
        self.cross_layer_blocks = nn.ModuleList([
            CrossLayerTransformerLayer(
                d_model=d_model,
                num_heads=num_cross_layer_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_cross_layer_blocks)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pooling attention
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool_attn = nn.Linear(d_model, d_model)
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_properties),
        )
    
    def _attention_pooling(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Attention pooling to aggregate sequence into a single vector.
        
        Args:
            x: Input sequence [batch_size, seq_length, d_model]
            mask: Optional attention mask [batch_size, seq_length]
            
        Returns:
            Pooled vector [batch_size, d_model]
        """
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Expand pool query to batch size
        query = self.pool_query.expand(batch_size, 1, self.d_model)
        
        # Compute attention scores
        keys = self.pool_attn(x)  # [batch_size, seq_length, d_model]
        scores = torch.bmm(query, keys.transpose(1, 2))  # [batch_size, 1, seq_length]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_length]
            scores = scores.masked_fill(~mask, -10000.0)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, 1, seq_length]
        
        # Apply attention weights
        pooled = torch.bmm(attn_weights, x)  # [batch_size, 1, d_model]
        pooled = pooled.squeeze(1)  # [batch_size, d_model]
        
        return pooled
    
    def forward(
        self, 
        x: torch.Tensor, 
        layer_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for WeightNet.
        
        Args:
            x: Input tokens [batch_size, seq_length, token_dim]
            layer_indices: Layer indices for each token [batch_size, seq_length]
                           If not provided, extracted from x
            attention_mask: Optional attention mask [batch_size, seq_length]
            
        Returns:
            Predicted properties [batch_size, num_properties]
        """
        batch_size, seq_length, token_dim = x.size()
        
        # Extract layer indices from last dimension if not provided
        if layer_indices is None:
            # Assuming layer indices are stored at index -4 of the token
            layer_indices = x[:, :, -4].long()
        
        # Extract token features
        token_features = x[:, :, :token_dim - 4]  # Assuming last 4 dimensions are metadata
        
        # Embed tokens
        embedded = self.embedding(token_features)  # [batch_size, seq_length, d_model]
        
        # Add positional encoding
        pos_encoded = self.positional_encoding(embedded)  # [batch_size, seq_length, d_model]
        
        # Add segment encoding
        segment_encoded = self.segment_encoding(pos_encoded, layer_indices)  # [batch_size, seq_length, d_model]
        
        # Intra-layer transformer blocks
        x = segment_encoded
        for intra_layer_block in self.intra_layer_blocks:
            x = intra_layer_block(x, layer_indices, attention_mask)
        
        # Cross-layer transformer blocks
        for cross_layer_block in self.cross_layer_blocks:
            x = cross_layer_block(x, attention_mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Attention pooling
        pooled = self._attention_pooling(x, attention_mask)
        
        # Output MLP
        output = self.output_mlp(pooled)
        
        return output

class MLPBaseline(nn.Module):
    """Simple MLP baseline model for weight-based property prediction."""
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: List[int] = [1024, 512, 256],
        output_dim: int = 3,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Input dimension (token dimension)
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of properties to predict
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP baseline.
        
        Args:
            x: Input tokens [batch_size, seq_length, token_dim]
            
        Returns:
            Predicted properties [batch_size, output_dim]
        """
        # Average pooling over sequence dimension
        x_mean = torch.mean(x, dim=1)
        
        # Forward pass through MLP
        return self.model(x_mean)

class StatsBaseline(nn.Module):
    """
    Baseline model using aggregated statistics from weights.
    
    This model extracts statistical features from weights and uses an MLP to predict properties.
    """
    
    def __init__(
        self,
        token_dim: int = 64,
        num_features: int = 20,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 3,
        dropout: float = 0.2,
    ):
        """
        Args:
            token_dim: Input token dimension
            num_features: Number of statistical features to extract
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of properties to predict
            dropout: Dropout probability
        """
        super().__init__()
        
        self.token_dim = token_dim
        self.num_features = num_features
        
        # Feature extraction layers
        self.feature_extractors = nn.ModuleList([
            nn.Linear(token_dim, 1) for _ in range(num_features)
        ])
        
        # MLP for prediction
        layers = []
        in_dim = num_features
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for stats baseline.
        
        Args:
            x: Input tokens [batch_size, seq_length, token_dim]
            
        Returns:
            Predicted properties [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # Extract features
        features = []
        for extractor in self.feature_extractors:
            # Apply extractor to each token
            feature = extractor(x[:, :, :self.token_dim - 4])  # Exclude metadata dimensions
            
            # Compute statistics of the feature across tokens
            feature_mean = torch.mean(feature, dim=1)
            features.append(feature_mean)
        
        # Concatenate features
        features = torch.cat(features, dim=1)  # [batch_size, num_features]
        
        # Forward pass through MLP
        return self.mlp(features)

if __name__ == "__main__":
    # Test the models with random inputs
    batch_size = 2
    seq_length = 128
    token_dim = 68  # 64 + 4 metadata dimensions
    d_model = 256
    num_properties = 3
    
    # Create random inputs
    tokens = torch.randn(batch_size, seq_length, token_dim)
    layer_indices = torch.randint(0, 10, (batch_size, seq_length))
    
    # Test WeightNet
    weight_net = WeightNetTransformer(
        d_model=d_model,
        num_intra_layer_heads=4,
        num_cross_layer_heads=8,
        num_intra_layer_blocks=2,
        num_cross_layer_blocks=2,
        num_properties=num_properties,
        token_dim=token_dim - 4,  # Exclude metadata dimensions
    )
    
    weight_net_output = weight_net(tokens, layer_indices)
    print(f"WeightNet output shape: {weight_net_output.shape}")
    
    # Test MLP baseline
    mlp_baseline = MLPBaseline(
        input_dim=token_dim,
        hidden_dims=[1024, 512, 256],
        output_dim=num_properties,
    )
    
    mlp_output = mlp_baseline(tokens)
    print(f"MLP baseline output shape: {mlp_output.shape}")
    
    # Test stats baseline
    stats_baseline = StatsBaseline(
        token_dim=token_dim,
        num_features=20,
        hidden_dims=[256, 128, 64],
        output_dim=num_properties,
    )
    
    stats_output = stats_baseline(tokens)
    print(f"Stats baseline output shape: {stats_output.shape}")
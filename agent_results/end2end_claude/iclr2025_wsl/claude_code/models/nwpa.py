"""
Neural Weight Pattern Analyzer (NWPA) model for neural network weight analysis.

This module implements the NWPA model as proposed in the research, which uses
graph neural networks and attention mechanisms to analyze neural network weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer implementation.
    Applies self-attention over the nodes in a graph.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        dropout: float = 0.1, 
        alpha: float = 0.2, 
        concat: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # Attention mechanism
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        # Initialize with Glorot
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)
        
        # Leaky ReLU for attention
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GAT layer.
        
        Args:
            x: Node features tensor of shape [N, in_features]
            adj: Adjacency matrix of shape [N, N]
            
        Returns:
            Updated node features of shape [N, out_features]
        """
        # Apply linear transformation
        Wh = self.W(x)  # [N, out_features]
        
        # Self-attention on the nodes
        # Create all possible pairs of nodes
        a_input = self._prepare_attentional_mechanism_input(Wh)  # [N, N, 2*out_features]
        
        # Compute attention coefficients
        e = self.leakyrelu(self.a(a_input).squeeze(-1))  # [N, N]
        
        # Mask attention coefficients using adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        
        # Apply softmax to normalize attention coefficients
        attention = F.softmax(attention, dim=1)  # [N, N]
        attention = F.dropout(attention, self.dropout, training=self.training)  # [N, N]
        
        # Apply attention to node features
        h_prime = torch.matmul(attention, Wh)  # [N, out_features]
        
        # Apply final non-linearity if specified
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """
        Prepare the input for the attention mechanism.
        
        Args:
            Wh: Transformed node features of shape [N, out_features]
            
        Returns:
            Tensor of shape [N, N, 2*out_features] containing pairs of node features
        """
        N = Wh.size(0)
        
        # Repeat node features for all possible node pairs
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # [N*N, out_features]
        Wh_repeated_alternating = Wh.repeat(N, 1)  # [N*N, out_features]
        
        # Combine features for all node pairs
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)  # [N*N, 2*out_features]
        
        # Reshape to [N, N, 2*out_features]
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

class MessagePassingGNN(nn.Module):
    """
    Message Passing Graph Neural Network (MP-GNN) for weight pattern analysis.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        num_layers: int = 2, 
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.layers = nn.ModuleList()
        
        if use_attention:
            # Graph Attention layers
            for i in range(num_layers):
                in_dim = hidden_dim if i > 0 else hidden_dim
                out_dim = hidden_dim if i < num_layers - 1 else output_dim
                concat = i < num_layers - 1
                
                self.layers.append(
                    GraphAttentionLayer(
                        in_features=in_dim,
                        out_features=out_dim,
                        dropout=dropout,
                        concat=concat
                    )
                )
        else:
            # Simple message passing layers using linear projections and neighborhood aggregation
            for i in range(num_layers):
                in_dim = hidden_dim if i > 0 else hidden_dim
                out_dim = hidden_dim if i < num_layers - 1 else output_dim
                
                self.layers.append(nn.Linear(in_dim, out_dim))
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GNN.
        
        Args:
            x: Node features tensor of shape [N, input_dim]
            adj: Adjacency matrix of shape [N, N]
            
        Returns:
            Updated node features of shape [N, output_dim]
        """
        # Initial projection
        h = self.input_projection(x)
        h = F.relu(h)
        h = self.dropout_layer(h)
        
        # Apply GNN layers
        for i, layer in enumerate(self.layers):
            if self.use_attention:
                h = layer(h, adj)
            else:
                # Simple message passing using adjacency matrix
                h_neigh = torch.matmul(adj, h)
                h = layer(h_neigh)
                
                if i < self.num_layers - 1:
                    h = F.relu(h)
                    h = self.dropout_layer(h)
        
        return h

class HierarchicalPooling(nn.Module):
    """
    Hierarchical pooling layer for graph-level representation.
    """
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        pooling_ratio: float = 0.5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pooling_ratio = pooling_ratio
        self.dropout = dropout
        
        # Scoring network for node importance
        self.score_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Projection for readout
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the hierarchical pooling layer.
        
        Args:
            x: Node features tensor of shape [N, input_dim]
            adj: Adjacency matrix of shape [N, N]
            
        Returns:
            Tuple containing:
                - Pooled node features [N', hidden_dim]
                - Pooled adjacency matrix [N', N']
                - Graph-level representation [hidden_dim]
        """
        # Compute node scores
        scores = self.score_layer(x)  # [N, 1]
        scores = torch.sigmoid(scores)  # [N, 1]
        
        # Determine number of nodes to keep
        N = x.size(0)
        k = int(N * self.pooling_ratio)
        k = max(1, min(k, N))  # Ensure at least 1 node and at most N nodes
        
        # Select top k nodes
        _, idx = torch.topk(scores.view(-1), k)  # [k]
        
        # Get the masked features and adjacency
        x_pooled = x[idx]  # [k, input_dim]
        adj_pooled = adj[idx][:, idx]  # [k, k]
        
        # Project features
        x_projected = self.projection(x_pooled)  # [k, hidden_dim]
        x_projected = F.relu(x_projected)
        x_projected = self.dropout_layer(x_projected)
        
        # Readout for graph-level representation
        graph_rep = x_projected.mean(dim=0)  # [hidden_dim]
        
        return x_projected, adj_pooled, graph_rep

class NWPA(nn.Module):
    """
    Neural Weight Pattern Analyzer (NWPA) model.
    
    This model represents neural networks as graphs and analyzes weight patterns
    using graph neural networks and attention mechanisms.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 128, 
        num_layers: int = 3,
        num_classes: Optional[Dict[str, int]] = None,
        num_regression_targets: Optional[int] = None,
        dropout: float = 0.1,
        use_attention: bool = True,
        pooling_ratio: float = 0.5
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes or {}
        self.num_regression_targets = num_regression_targets or 0
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Feature extraction network for raw weight features
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # GNN for weight graph structure
        self.gnn = MessagePassingGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention,
        )
        
        # Hierarchical pooling for multi-resolution analysis
        self.pooling = HierarchicalPooling(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            pooling_ratio=pooling_ratio,
            dropout=dropout,
        )
        
        # Combination layer
        self.combination = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classification heads (one per class type)
        self.classification_heads = nn.ModuleDict()
        for class_name, num_class in self.num_classes.items():
            self.classification_heads[class_name] = nn.Linear(hidden_dim, num_class)
        
        # Regression head
        if self.num_regression_targets > 0:
            self.regression_head = nn.Linear(hidden_dim, self.num_regression_targets)
    
    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass through the model
        
        Args:
            batch: Dictionary containing:
                - 'features': Tensor of shape [batch_size, input_dim]
                
        Returns:
            Dictionary containing:
                - 'classification': Dict of {class_name: logits} for each class
                - 'regression': Regression outputs if regression targets exist
        """
        # Extract features
        features = batch['features']
        
        # Since we're working with a simplified version where we don't have actual graph structures,
        # we'll construct a simple fully-connected graph adjacency matrix for each sample
        batch_size = features.size(0)
        adj = torch.ones(batch_size, batch_size, device=features.device)  # Fully connected
        
        # Process with feature extractor
        h_features = self.feature_extractor(features)
        
        # Process with GNN
        h_gnn = self.gnn(features, adj)
        
        # Apply hierarchical pooling
        _, _, h_graph = self.pooling(h_gnn, adj)
        
        # Combine feature-based and graph-based representations
        h_combined = torch.cat([h_features.mean(dim=0), h_graph], dim=0)
        h = self.combination(h_combined.unsqueeze(0)).squeeze(0)
        
        outputs = {}
        
        # Classification outputs
        if self.num_classes:
            classification_outputs = {}
            for class_name, head in self.classification_heads.items():
                # For batch processing
                if len(features.shape) > 1 and features.shape[0] > 1:
                    h_batch = h.unsqueeze(0).expand(batch_size, -1)
                    classification_outputs[class_name] = head(h_batch)
                else:
                    classification_outputs[class_name] = head(h.unsqueeze(0))
            outputs['classification'] = classification_outputs
        
        # Regression outputs
        if self.num_regression_targets > 0:
            # For batch processing
            if len(features.shape) > 1 and features.shape[0] > 1:
                h_batch = h.unsqueeze(0).expand(batch_size, -1)
                outputs['regression'] = self.regression_head(h_batch)
            else:
                outputs['regression'] = self.regression_head(h.unsqueeze(0))
        
        return outputs
    
    def loss_function(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """
        Compute the loss for model outputs
        
        Args:
            outputs: Dictionary from the forward pass
            batch: Dictionary containing ground truth labels
            
        Returns:
            Combined loss value
        """
        loss = 0.0
        
        # Classification loss
        if 'classification' in outputs:
            for class_name, logits in outputs['classification'].items():
                target = batch[f'class_{class_name}']
                class_loss = F.cross_entropy(logits, target)
                loss += class_loss
        
        # Regression loss
        if 'regression' in outputs:
            regression_targets = batch['regression_targets']
            regression_loss = F.mse_loss(outputs['regression'], regression_targets)
            loss += regression_loss
        
        return loss
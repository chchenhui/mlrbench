"""
Implementation of the permutation-equivariant graph embedding models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch
import torch.optim as optim


class NodeInitializer(nn.Module):
    """Initialize node features from layer information."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),  # 4 features: weight norm, bias magnitude, input_dim, output_dim
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: Node features [batch_size, num_nodes, 4]
                where features are [weight_norm, bias_mag, input_dim, output_dim]
        Returns:
            Initial node embeddings [batch_size, num_nodes, hidden_dim]
        """
        return self.mlp(x)


class MessageFunction(nn.Module):
    """Message function for GNN message passing."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Edge feature is scalar weight
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),  # src + dst + edge features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, h_i, h_j, edge_attr):
        """
        Args:
            h_i: Destination node features [E, hidden_dim]
            h_j: Source node features [E, hidden_dim]
            edge_attr: Edge features [E, 1]
        Returns:
            Messages [E, hidden_dim]
        """
        edge_embedding = self.edge_encoder(edge_attr)
        return self.message_mlp(torch.cat([h_i, h_j, edge_embedding], dim=1))


class UpdateFunction(nn.Module):
    """Node update function for GNN message passing."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Current state + aggregated message
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, h_i, m_i):
        """
        Args:
            h_i: Current node features [N, hidden_dim]
            m_i: Aggregated messages [N, hidden_dim]
        Returns:
            Updated node features [N, hidden_dim]
        """
        return self.update_mlp(torch.cat([h_i, m_i], dim=1))


class EquivariantLayerGNN(MessagePassing):
    """Permutation-equivariant GNN for processing a single layer graph."""
    def __init__(self, hidden_dim=128):
        super().__init__(aggr='add')  # Use sum aggregation for permutation equivariance
        self.hidden_dim = hidden_dim
        self.message_function = MessageFunction(hidden_dim)
        self.update_function = UpdateFunction(hidden_dim)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [N, hidden_dim]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge features [E, 1]
        Returns:
            Updated node features [N, hidden_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        """Generate messages from source to destination nodes."""
        return self.message_function(x_i, x_j, edge_attr)
    
    def update(self, aggr_out, x):
        """Update node features with aggregated messages."""
        return self.update_function(x, aggr_out)


class LayerEmbedder(nn.Module):
    """Process a single layer graph to produce a layer embedding."""
    def __init__(self, hidden_dim=128, message_passing_steps=3):
        super().__init__()
        self.node_initializer = NodeInitializer(hidden_dim)
        self.gnns = nn.ModuleList([
            EquivariantLayerGNN(hidden_dim) for _ in range(message_passing_steps)
        ])
        self.pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, node_features, edge_index, edge_attr, batch=None):
        """
        Args:
            node_features: Node features [N, 4]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge features [E, 1]
            batch: Batch indices [N] for batched graphs
        Returns:
            Layer embedding [batch_size, hidden_dim]
        """
        # Initialize node embeddings
        x = self.node_initializer(node_features)
        
        # Apply GNN layers
        for gnn in self.gnns:
            x = gnn(x, edge_index, edge_attr)
        
        # Global pooling (mean pooling for permutation invariance)
        if batch is not None:
            x_dense, mask = to_dense_batch(x, batch)
            pooled = x_dense.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            pooled = x.mean(dim=0, keepdim=True)
        
        return self.pooling(pooled)


class TransformerAggregator(nn.Module):
    """Aggregate layer embeddings using a transformer."""
    def __init__(self, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, layer_embeddings, mask=None):
        """
        Args:
            layer_embeddings: Layer embeddings [batch_size, num_layers, hidden_dim]
            mask: Padding mask for variable-length networks [batch_size, num_layers]
        Returns:
            Global network embedding [batch_size, hidden_dim]
        """
        # Process with transformer
        if mask is not None:
            transformer_mask = mask.logical_not()
        else:
            transformer_mask = None
            
        transformed = self.transformer(layer_embeddings, src_key_padding_mask=transformer_mask)
        
        # Global pooling (mean over layers)
        if mask is not None:
            # Apply mask before pooling
            pooled = (transformed * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            pooled = transformed.mean(dim=1)
            
        return self.pooling(pooled)


class WeightGraphEmbedding(nn.Module):
    """
    Complete model for embedding neural network weights as permutation-invariant representations.
    """
    def __init__(
        self,
        hidden_dim=128,
        message_passing_steps=3,
        transformer_heads=4,
        transformer_layers=2,
        global_embedding_dim=256
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.global_embedding_dim = global_embedding_dim
        
        # Layer embedder (weight matrix to permutation-equivariant embedding)
        self.layer_embedder = LayerEmbedder(hidden_dim, message_passing_steps)
        
        # Transformer to aggregate layer embeddings
        self.transformer = TransformerAggregator(
            hidden_dim=hidden_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers
        )
        
        # Project to final embedding dimension
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, global_embedding_dim),
            nn.LayerNorm(global_embedding_dim)
        )
    
    def forward(self, batch_data):
        """
        Args:
            batch_data: Dictionary containing:
                - layer_node_features: List of node features for each layer
                - layer_edge_indices: List of edge indices for each layer
                - layer_edge_attrs: List of edge attributes for each layer
                - layer_batch_indices: List of batch indices for each layer
                - num_layers: Number of layers per network [batch_size]
                
        Returns:
            Network embedding [batch_size, global_embedding_dim]
        """
        batch_size = len(batch_data['num_layers'])
        max_layers = max(batch_data['num_layers']).item()
        device = batch_data['layer_node_features'][0].device
        
        # Process each layer to get layer embeddings
        all_layer_embeddings = []
        layer_start_idx = 0
        
        for layer_idx in range(max_layers):
            # Count networks that have this layer
            networks_with_layer = sum(1 for n in batch_data['num_layers'] if n > layer_idx)
            if networks_with_layer == 0:
                break
                
            # Get data for current layer across all networks that have it
            end_idx = layer_start_idx + networks_with_layer
            node_features = batch_data['layer_node_features'][layer_idx]
            edge_index = batch_data['layer_edge_indices'][layer_idx]
            edge_attr = batch_data['layer_edge_attrs'][layer_idx]
            batch_indices = batch_data['layer_batch_indices'][layer_idx]
            
            # Process layer
            layer_embeddings = self.layer_embedder(node_features, edge_index, edge_attr, batch_indices)
            
            # Store embeddings (with padding for networks without this layer)
            padded_embeddings = torch.zeros(batch_size, self.hidden_dim, device=device)
            present_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            # Fill in embeddings for networks that have this layer
            for batch_idx, num_layers in enumerate(batch_data['num_layers']):
                if layer_idx < num_layers:
                    network_layer_idx = sum(1 for n in batch_data['num_layers'][:batch_idx] if n > layer_idx)
                    padded_embeddings[batch_idx] = layer_embeddings[network_layer_idx]
                    present_mask[batch_idx] = True
            
            all_layer_embeddings.append(padded_embeddings)
            layer_start_idx = end_idx
        
        # Stack layer embeddings
        layer_embeddings_tensor = torch.stack(all_layer_embeddings, dim=1)  # [batch_size, max_layers, hidden_dim]
        
        # Create mask for transformer
        mask = torch.zeros(batch_size, max_layers, dtype=torch.bool, device=device)
        for i, num_layers in enumerate(batch_data['num_layers']):
            mask[i, :num_layers] = True
        
        # Process with transformer
        global_embedding = self.transformer(layer_embeddings_tensor, mask)
        
        # Project to final dimension
        return self.projector(global_embedding)


class MultiLayerPerceptron(nn.Module):
    """Simple MLP baseline for embedding flattened weights."""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Flattened weights [batch_size, input_dim]
        Returns:
            Embedding [batch_size, output_dim]
        """
        return self.model(x)


class PCAPlusMLPBaseline(nn.Module):
    """Baseline that applies PCA to flattened weights then an MLP."""
    def __init__(self, input_dim, pca_components, hidden_dims, output_dim):
        super().__init__()
        self.register_buffer('mean', torch.zeros(input_dim))
        self.register_buffer('components', torch.eye(input_dim, pca_components))
        
        self.mlp = MultiLayerPerceptron(pca_components, hidden_dims, output_dim)
        
    def fit_pca(self, data):
        """
        Fit PCA to the data.
        
        Args:
            data: Tensor of shape [num_samples, input_dim]
        """
        # Center data
        self.mean = data.mean(dim=0)
        X = data - self.mean
        
        # SVD decomposition
        U, S, V = torch.svd(X)
        
        # Store principal components
        self.components = V[:, :self.components.shape[1]]
        
    def forward(self, x):
        """
        Args:
            x: Flattened weights [batch_size, input_dim]
        Returns:
            Embedding [batch_size, output_dim]
        """
        # Center and project
        x_centered = x - self.mean
        x_pca = x_centered @ self.components
        
        # Pass through MLP
        return self.mlp(x_pca)


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, positives_mask):
        """
        Args:
            embeddings: Embeddings [batch_size, embedding_dim]
            positives_mask: Boolean mask indicating positive pairs [batch_size, batch_size]
                           (i,j) is 1 if embeddings[i] and embeddings[j] form a positive pair
        Returns:
            Loss value
        """
        # Normalize embeddings
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings_normalized, embeddings_normalized.t()) / self.temperature
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(embeddings.size(0), dtype=torch.bool, device=embeddings.device)
        sim_matrix.masked_fill_(mask, -float('inf'))
        
        # InfoNCE loss computation
        batch_size = embeddings.size(0)
        loss = 0.0
        
        for i in range(batch_size):
            positive_indices = positives_mask[i].nonzero(as_tuple=True)[0]
            if len(positive_indices) == 0:
                continue
                
            positive_logits = sim_matrix[i, positive_indices]
            negative_logits = sim_matrix[i, ~positives_mask[i]]
            negative_logits = negative_logits[~mask[i, ~positives_mask[i]]]  # Remove self
            
            logits = torch.cat([positive_logits, negative_logits])
            labels = torch.zeros(len(logits), device=embeddings.device, dtype=torch.long)
            labels[:len(positive_logits)] = 1
            
            loss += F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))
        
        return loss / batch_size


class AccuracyRegressor(nn.Module):
    """Regressor for zero-shot performance prediction from embeddings."""
    def __init__(self, embedding_dim, hidden_dims=[128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = embedding_dim
        
        # Hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
            
        # Output layer (single value for accuracy)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Accuracy is in [0, 1]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, embeddings):
        """
        Args:
            embeddings: Model embeddings [batch_size, embedding_dim]
        Returns:
            Predicted accuracy [batch_size, 1]
        """
        return self.model(embeddings)


class EmbeddingDecoder(nn.Module):
    """Decoder to convert embeddings back to weights for model merging."""
    def __init__(self, embedding_dim, layer_sizes):
        """
        Args:
            embedding_dim: Dimension of the global embedding
            layer_sizes: List of tuples (input_dim, output_dim) for each layer
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layer_sizes = layer_sizes
        
        # Create a decoder for each layer
        self.layer_decoders = nn.ModuleList()
        for in_dim, out_dim in layer_sizes:
            decoder = nn.Sequential(
                nn.Linear(embedding_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, in_dim * out_dim + out_dim)  # weights + biases
            )
            self.layer_decoders.append(decoder)
    
    def forward(self, embedding):
        """
        Args:
            embedding: Global embedding [batch_size, embedding_dim]
        Returns:
            List of (weight, bias) tuples for each layer
        """
        batch_size = embedding.size(0)
        decoded_layers = []
        
        for i, (in_dim, out_dim) in enumerate(self.layer_sizes):
            # Decode layer parameters
            params = self.layer_decoders[i](embedding)
            
            # Split into weights and biases
            weights_size = in_dim * out_dim
            weights = params[:, :weights_size].view(batch_size, out_dim, in_dim)
            biases = params[:, weights_size:].view(batch_size, out_dim)
            
            decoded_layers.append((weights, biases))
            
        return decoded_layers
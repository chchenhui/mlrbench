"""
Permutation-Equivariant GNN Encoder for Model Zoo Retrieval.
This module implements the permutation-equivariant GNN encoder for 
embedding neural network weights into a functional similarity space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import degree
import logging

# Local imports
from config import MODEL_CONFIG, LOG_CONFIG

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["log_level"]),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gnn_encoder")

class GeometricMessagePassing(MessagePassing):
    """
    Geometric message passing layer that respects permutation symmetry.
    This is inspired by Geom-GCN but with additional equivariance guarantees.
    """
    
    def __init__(self, in_channels, out_channels, edge_dim=None):
        super(GeometricMessagePassing, self).__init__(aggr='mean')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Linear transformation for node features
        self.linear = nn.Linear(in_channels, out_channels)
        
        # Edge feature processing
        self.edge_dim = edge_dim
        if edge_dim is not None:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
            
        # Transformation matrix generator
        self.transform_generator = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, out_channels * out_channels)
        )
        
        # Initialize the transform generator to produce identity-like transformations
        for layer in self.transform_generator:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x, edge_index, edge_attr=None):
        # Transform node features
        x = self.linear(x)
        
        # Start propagation
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr=None, edge_index_i=None, edge_index_j=None):
        # Generate edge-specific transformation based on invariants
        edge_invariants = torch.stack([
            degree(edge_index_i, dtype=torch.float),
            degree(edge_index_j, dtype=torch.float)
        ], dim=1)
        edge_invariants = edge_invariants / (edge_invariants.sum(dim=1, keepdim=True) + 1e-8)
        
        # Generate transformation matrices for each edge
        transform_params = self.transform_generator(edge_invariants)
        transform_matrices = transform_params.view(-1, self.out_channels, self.out_channels)
        
        # Apply transformation to source node features
        transformed_features = torch.bmm(
            x_j.unsqueeze(1),
            transform_matrices
        ).squeeze(1)
        
        # If edge attributes are provided, incorporate them
        if edge_attr is not None and self.edge_dim is not None:
            edge_features = self.edge_encoder(edge_attr)
            return transformed_features + edge_features
        
        return transformed_features
    
    def update(self, aggr_out):
        # Identity update function
        return aggr_out


class AttentionReadout(nn.Module):
    """
    Attention-based readout function for graph-level embeddings.
    Preserves permutation symmetry through attention pooling.
    """
    
    def __init__(self, in_channels, out_channels):
        super(AttentionReadout, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Output transformation
        self.transform = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, batch):
        # Compute attention scores
        attention_scores = self.attention(x)
        
        # Normalize attention scores (within each graph via softmax)
        attention_scores = global_add_pool(attention_scores.exp(), batch)
        attention_scores = attention_scores[batch].reciprocal()
        attention_scores = attention_scores * self.attention(x).exp()
        
        # Apply attention pooling
        pooled = global_add_pool(x * attention_scores, batch)
        
        # Transform to output dimension
        out = self.transform(pooled)
        
        return out


class EquivariantGNN(nn.Module):
    """
    Permutation-equivariant GNN for encoding neural network weights.
    """
    
    def __init__(self, 
                 node_dim=MODEL_CONFIG["gnn_encoder"]["node_dim"],
                 edge_dim=MODEL_CONFIG["gnn_encoder"]["edge_dim"],
                 hidden_dim=MODEL_CONFIG["gnn_encoder"]["hidden_dim"],
                 output_dim=MODEL_CONFIG["gnn_encoder"]["output_dim"],
                 num_layers=MODEL_CONFIG["gnn_encoder"]["num_layers"],
                 dropout=MODEL_CONFIG["gnn_encoder"]["dropout"],
                 readout=MODEL_CONFIG["gnn_encoder"]["readout"]):
        super(EquivariantGNN, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.readout_type = readout
        
        # Input transformation
        self.input_transform = nn.Linear(node_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim if i > 0 else hidden_dim
            self.convs.append(GeometricMessagePassing(in_channels, hidden_dim, edge_dim))
        
        # Readout function
        if readout == "attention":
            self.readout = AttentionReadout(hidden_dim, output_dim)
        else:  # Default to mean pooling
            self.readout = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim)
            )
        
        # Layer-level GRU for combining representations from different layers
        self.layer_gru = nn.GRUCell(output_dim, output_dim)
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
        logger.info(f"Initialized EquivariantGNN with {num_layers} layers")
    
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass for a single graph.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch assignment vector [num_nodes]
            
        Returns:
            Graph embedding vector [batch_size, output_dim]
        """
        # Input transformation
        x = self.input_transform(x)
        
        # Apply GNN layers with residual connections
        all_layer_reps = []
        for i in range(self.num_layers):
            x_res = x  # Save for residual connection
            x = self.convs[i](x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual connection
            
            # Create layer representation
            if self.readout_type == "attention":
                layer_rep = self.readout(x, batch)
            else:
                if self.readout_type == "mean":
                    pooled = global_mean_pool(x, batch)
                elif self.readout_type == "sum":
                    pooled = global_add_pool(x, batch)
                elif self.readout_type == "max":
                    pooled = global_max_pool(x, batch)
                else:  # Default to mean
                    pooled = global_mean_pool(x, batch)
                layer_rep = self.readout(pooled)
            
            all_layer_reps.append(layer_rep)
        
        # Combine layer representations using GRU
        h = all_layer_reps[0]
        for i in range(1, len(all_layer_reps)):
            h = self.layer_gru(all_layer_reps[i], h)
        
        # Apply layer normalization
        h = self.layer_norm(h)
        
        return h
    
    def encode_model(self, model_graphs):
        """
        Encode a model's weights (represented as a list of layer graphs).
        
        Args:
            model_graphs: A list of torch_geometric.data.Data objects, one for each layer.
            
        Returns:
            A model embedding vector [output_dim].
        """
        # Process each layer graph
        layer_embeddings = []
        
        for graph in model_graphs:
            # Extract graph data
            x = graph.x
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr
            
            # Create batch vector (all nodes belong to same graph)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
            # Get layer embedding
            with torch.no_grad():
                layer_embedding = self.forward(x, edge_index, edge_attr, batch)
            
            layer_embeddings.append(layer_embedding)
        
        # Combine layer embeddings
        if not layer_embeddings:
            logger.warning("No layer embeddings generated")
            return torch.zeros(self.output_dim)
        
        # Stack and reduce
        model_embedding = torch.cat(layer_embeddings, dim=0)
        model_embedding = torch.mean(model_embedding, dim=0)
        
        return model_embedding


class MultiheadAttention(nn.Module):
    """
    Multi-head attention layer for combining layer embeddings.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Apply multi-head attention over a set of embeddings.
        
        Args:
            x: Input embeddings [batch_size, seq_len, embed_dim]
            
        Returns:
            Attended embeddings [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project inputs
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class ModelEmbedder(nn.Module):
    """
    Full model encoder combining layer-level GNN with attention-based layer aggregation.
    """
    
    def __init__(self,
                 node_dim=MODEL_CONFIG["gnn_encoder"]["node_dim"],
                 edge_dim=MODEL_CONFIG["gnn_encoder"]["edge_dim"],
                 hidden_dim=MODEL_CONFIG["gnn_encoder"]["hidden_dim"],
                 output_dim=MODEL_CONFIG["gnn_encoder"]["output_dim"],
                 num_layers=MODEL_CONFIG["gnn_encoder"]["num_layers"],
                 dropout=MODEL_CONFIG["gnn_encoder"]["dropout"],
                 readout=MODEL_CONFIG["gnn_encoder"]["readout"]):
        super(ModelEmbedder, self).__init__()
        
        # GNN for processing each layer graph
        self.gnn = EquivariantGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            readout=readout
        )
        
        # Attention mechanism for combining layer embeddings
        self.attention = MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=dropout
        )
        
        # Final MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(output_dim)
        
        logger.info("Initialized ModelEmbedder")
    
    def forward(self, model_graphs):
        """
        Encode a model's weights into an embedding space.
        
        Args:
            model_graphs: List of graph representations for each layer.
            
        Returns:
            Model embedding vector.
        """
        # Process each layer
        layer_embeddings = []
        
        for graph in model_graphs:
            # Extract graph data
            x = graph.x
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr
            
            # Create batch vector (all nodes belong to same graph)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
            # Get layer embedding
            layer_embedding = self.gnn(x, edge_index, edge_attr, batch)
            layer_embeddings.append(layer_embedding)
        
        # Handle case with no layers
        if not layer_embeddings:
            return torch.zeros(MODEL_CONFIG["gnn_encoder"]["output_dim"])
        
        # Stack layer embeddings
        layer_embeddings = torch.stack(layer_embeddings, dim=1)  # [batch_size, num_layers, output_dim]
        
        # Apply attention to aggregate layer embeddings
        combined = self.attention(layer_embeddings)
        
        # Apply mean pooling over layers
        model_embedding = torch.mean(combined, dim=1)  # [batch_size, output_dim]
        
        # Apply final MLP
        model_embedding = self.final_mlp(model_embedding)
        
        # Normalize
        model_embedding = self.layer_norm(model_embedding)
        
        return model_embedding
    
    def encode_batch(self, batch_graphs):
        """
        Encode a batch of models.
        
        Args:
            batch_graphs: List of lists of layer graphs, one list per model.
            
        Returns:
            Batch of model embeddings.
        """
        embeddings = []
        
        for model_graphs in batch_graphs:
            embedding = self.forward(model_graphs)
            embeddings.append(embedding)
        
        return torch.stack(embeddings, dim=0)


# Test code
if __name__ == "__main__":
    # Import for testing
    import torch
    from torch_geometric.data import Data
    import random
    
    # Create some synthetic graph data for testing
    def create_test_graph(num_nodes=10, dim=MODEL_CONFIG["gnn_encoder"]["node_dim"], 
                         edge_dim=MODEL_CONFIG["gnn_encoder"]["edge_dim"]):
        # Create random node features
        x = torch.randn(num_nodes, dim)
        
        # Create random edges (ensuring all nodes are connected)
        edge_index = []
        for i in range(num_nodes):
            # Each node connects to at least one other node
            targets = random.sample(range(num_nodes), max(1, num_nodes // 3))
            for t in targets:
                if t != i:  # Avoid self-loops
                    edge_index.append([i, t])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create random edge features
        edge_attr = torch.randn(edge_index.size(1), edge_dim)
        
        # Create graph
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph.n_in = num_nodes // 2
        graph.n_out = num_nodes - graph.n_in
        graph.layer_name = "test_layer"
        graph.total_nodes = num_nodes
        
        return graph
    
    # Create test data
    num_models = 3
    num_layers_per_model = 4
    
    test_models = []
    for _ in range(num_models):
        model_graphs = []
        for _ in range(num_layers_per_model):
            graph = create_test_graph()
            model_graphs.append(graph)
        test_models.append(model_graphs)
    
    # Create model
    model = ModelEmbedder()
    
    # Test encoder
    print("Testing encoder...")
    embeddings = model.encode_batch(test_models)
    
    print(f"Batch embeddings shape: {embeddings.shape}")
    print("First model embedding:", embeddings[0][:5])  # Show first 5 dimensions
    
    # Check if encoder is equivariant
    print("\nTesting equivariance...")
    
    # Create a permutation of the first graph in the first model
    original_graph = test_models[0][0]
    
    # Create a permuted version by shuffling output nodes
    permuted_graph = Data(
        x=original_graph.x.clone(),
        edge_index=original_graph.edge_index.clone(),
        edge_attr=original_graph.edge_attr.clone(),
        n_in=original_graph.n_in,
        n_out=original_graph.n_out,
        layer_name=original_graph.layer_name,
        total_nodes=original_graph.total_nodes
    )
    
    # Apply node permutation to output nodes
    n_in = original_graph.n_in
    n_out = original_graph.n_out
    
    # Generate random permutation for output nodes
    perm = torch.randperm(n_out)
    
    # Permute node features for output nodes
    for i in range(n_out):
        old_idx = n_in + i
        new_idx = n_in + perm[i]
        permuted_graph.x[old_idx] = original_graph.x[new_idx]
    
    # Update edges to reflect the new node ordering
    for i in range(permuted_graph.edge_index.size(1)):
        if permuted_graph.edge_index[1, i] >= n_in:
            # Convert to output node index
            out_idx = permuted_graph.edge_index[1, i] - n_in
            # Apply permutation
            permuted_graph.edge_index[1, i] = n_in + perm[out_idx]
    
    # Create new model lists with original and permuted graphs
    original_model = test_models[0].copy()
    permuted_model = test_models[0].copy()
    permuted_model[0] = permuted_graph
    
    # Get embeddings
    with torch.no_grad():
        orig_embedding = model.forward(original_model)
        perm_embedding = model.forward(permuted_model)
    
    # Compute distance between embeddings
    dist = torch.norm(orig_embedding - perm_embedding)
    print(f"Distance between original and permuted embeddings: {dist.item()}")
    print("(A small distance indicates equivariance is preserved)")
    
    # Test model performance with multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"\nLet's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    # Move to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert test data to device
    test_models_device = []
    for model_graphs in test_models:
        model_graphs_device = []
        for graph in model_graphs:
            graph = graph.to(device)
            model_graphs_device.append(graph)
        test_models_device.append(model_graphs_device)
    
    # Test on device
    embeddings_device = model.encode_batch(test_models_device)
    print(f"Device embeddings shape: {embeddings_device.shape}")
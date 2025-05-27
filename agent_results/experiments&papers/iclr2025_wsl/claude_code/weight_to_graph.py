"""
Weight-to-Graph conversion module.
This module provides functions to convert neural network weight tensors into graph structures
that preserve the important connectivity patterns while enabling equivariant processing.
"""

import torch
import numpy as np
import networkx as nx
import torch_geometric
from torch_geometric.data import Data, Batch
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
logger = logging.getLogger("weight_to_graph")

class WeightToGraph:
    """Convert neural network weights to graph representations."""
    
    def __init__(self, edge_dim=MODEL_CONFIG["gnn_encoder"]["edge_dim"], 
                 node_dim=MODEL_CONFIG["gnn_encoder"]["node_dim"]):
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"WeightToGraph initialized with edge_dim={edge_dim}, node_dim={node_dim}, device={self.device}")
    
    def convert_model_to_graphs(self, model_weights):
        """
        Convert a model's weights to a list of graph objects.
        
        Args:
            model_weights: Dictionary of model weights.
            
        Returns:
            A list of torch_geometric.data.Data objects, one for each layer.
        """
        graphs = []
        
        # Process each weight tensor in the model
        for key, weight in model_weights.items():
            # Skip non-tensor weights
            if not isinstance(weight, torch.Tensor):
                continue
            
            # For now, we only consider 2D weight matrices (ignore biases, batchnorm, etc.)
            if len(weight.shape) != 2:
                continue
            
            # Create a graph for this layer
            graph = self._weight_matrix_to_graph(weight, layer_name=key)
            if graph is not None:
                graphs.append(graph)
        
        logger.info(f"Converted model weights to {len(graphs)} layer graphs")
        return graphs
    
    def _weight_matrix_to_graph(self, weight_matrix, layer_name=""):
        """
        Convert a single weight matrix to a graph representation.
        
        Args:
            weight_matrix: 2D tensor of shape (n_out, n_in)
            layer_name: Name of the layer (for reference)
            
        Returns:
            A torch_geometric.data.Data object representing the graph.
        """
        try:
            # Extract dimensions
            n_out, n_in = weight_matrix.shape
            
            # Create bipartite graph structure
            # Each node is either an input neuron or an output neuron
            # Edges go from input neurons to output neurons
            
            # Generate edge indices
            edge_indices = []
            edge_weights = []
            edge_features = []
            
            # Create edges for non-zero weights (or all weights if preferred)
            for i in range(n_out):
                for j in range(n_in):
                    # Skip if weight is zero (optional for sparse representation)
                    # if weight_matrix[i, j] == 0:
                    #     continue
                    
                    # Edge from input j to output i
                    # For input nodes, we use indices [0, n_in-1]
                    # For output nodes, we use indices [n_in, n_in+n_out-1]
                    # This ensures a proper bipartite structure
                    edge_indices.append([j, n_in + i])
                    
                    # Edge weight is the weight matrix value
                    w_val = weight_matrix[i, j].item()
                    edge_weights.append(w_val)
                    
                    # Edge feature (weight value and its magnitude)
                    edge_feat = self._create_edge_feature(w_val)
                    edge_features.append(edge_feat)
            
            # If no edges, return None (this should not happen for regular layers)
            if not edge_indices:
                logger.warning(f"No edges created for layer {layer_name}")
                return None
            
            # Convert to tensors
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.stack([torch.tensor(f, dtype=torch.float) for f in edge_features])
            
            # Create node features
            num_nodes = n_in + n_out
            
            # Generate initial node features (can be enhanced with layer-specific info)
            # Here we distinguish input nodes from output nodes with a one-hot feature
            node_features = []
            for i in range(num_nodes):
                if i < n_in:
                    # Input node
                    node_feat = self._create_node_feature(is_input=True, idx=i, layer_name=layer_name)
                else:
                    # Output node
                    node_feat = self._create_node_feature(is_input=False, idx=i-n_in, layer_name=layer_name)
                node_features.append(node_feat)
            
            # Convert to tensor
            x = torch.stack([torch.tensor(f, dtype=torch.float) for f in node_features])
            
            # Create additional graph-level features
            layer_info = {
                'layer_name': layer_name,
                'n_in': n_in,
                'n_out': n_out,
                'weight_stats': {
                    'mean': weight_matrix.mean().item(),
                    'std': weight_matrix.std().item(),
                    'min': weight_matrix.min().item(),
                    'max': weight_matrix.max().item(),
                }
            }
            
            # Create the graph with all data
            graph = Data(
                x=x,                     # Node features
                edge_index=edge_index,   # Edge connections
                edge_attr=edge_attr,     # Edge features
                n_in=n_in,               # Number of input nodes
                n_out=n_out,             # Number of output nodes
                layer_name=layer_name,   # Layer name
                total_nodes=num_nodes    # Total number of nodes
            )
            
            return graph
            
        except Exception as e:
            logger.error(f"Error converting weight matrix to graph: {e}")
            return None
    
    def _create_edge_feature(self, weight_value):
        """
        Create a feature vector for an edge.
        
        Args:
            weight_value: The weight value for this edge.
            
        Returns:
            A feature vector of size edge_dim.
        """
        # Start with basic features: the weight value and its magnitude
        features = [weight_value, abs(weight_value)]
        
        # Add sign indicator (positive/negative)
        features.append(1.0 if weight_value >= 0 else -1.0)
        
        # Add some transformations of weight to capture non-linear properties
        features.append(np.sign(weight_value) * np.log(abs(weight_value) + 1e-8))
        
        # Pad to edge_dim size with zeros
        if len(features) < self.edge_dim:
            features.extend([0.0] * (self.edge_dim - len(features)))
        
        # Truncate if longer (should not happen with current setup)
        return features[:self.edge_dim]
    
    def _create_node_feature(self, is_input, idx, layer_name=""):
        """
        Create a feature vector for a node.
        
        Args:
            is_input: Whether this is an input node.
            idx: The index of the node.
            layer_name: The name of the layer.
            
        Returns:
            A feature vector of size node_dim.
        """
        # Start with basic features
        features = []
        
        # Type indicator (input or output)
        features.append(1.0 if is_input else 0.0)
        features.append(0.0 if is_input else 1.0)
        
        # Normalized index (relative position in layer)
        features.append(idx / 1000.0)  # Scale down to avoid large values
        
        # Layer type encoding (can be enhanced with more layer-specific info)
        is_conv = 1.0 if 'conv' in layer_name.lower() else 0.0
        is_linear = 1.0 if 'linear' in layer_name.lower() or 'fc' in layer_name.lower() else 0.0
        is_bn = 1.0 if 'batch' in layer_name.lower() or 'bn' in layer_name.lower() else 0.0
        
        features.extend([is_conv, is_linear, is_bn])
        
        # Pad to node_dim size with zeros
        if len(features) < self.node_dim:
            features.extend([0.0] * (self.node_dim - len(features)))
        
        # Truncate if longer
        return features[:self.node_dim]
    
    def batch_graphs(self, graphs):
        """
        Batch multiple graphs into a single batched graph.
        
        Args:
            graphs: List of torch_geometric.data.Data objects.
            
        Returns:
            A batched graph or list if batching fails.
        """
        try:
            if not graphs:
                logger.warning("No graphs to batch")
                return None
            
            # Move graphs to device if needed
            for g in graphs:
                g.to(self.device)
            
            # Batch the graphs
            batched_graph = Batch.from_data_list(graphs)
            return batched_graph
        
        except Exception as e:
            logger.error(f"Error batching graphs: {e}")
            return graphs
    
    def apply_permutation_transform(self, graph, permutation_prob=0.15):
        """
        Apply a permutation transformation to a graph.
        
        Args:
            graph: torch_geometric.data.Data object
            permutation_prob: Probability of applying permutation
            
        Returns:
            Transformed graph
        """
        if random.random() > permutation_prob:
            return graph.clone()
        
        try:
            # Extract necessary attributes
            n_in = graph.n_in
            n_out = graph.n_out
            edge_index = graph.edge_index.clone()
            edge_attr = graph.edge_attr.clone()
            x = graph.x.clone()
            
            # Generate output permutation (only permute output nodes)
            perm = torch.randperm(n_out)
            
            # Update edge indices based on permutation
            for i in range(edge_index.size(1)):
                # Check if it's an output node (index >= n_in)
                if edge_index[1, i] >= n_in:
                    # Get the original output node index (relative to output space)
                    orig_idx = edge_index[1, i] - n_in
                    # Apply permutation and update
                    edge_index[1, i] = n_in + perm[orig_idx]
            
            # Update node features for output nodes
            for i in range(n_in, n_in + n_out):
                # Get the original output node index (relative to output space)
                orig_idx = i - n_in
                # Get the new index after permutation
                new_idx = n_in + perm[orig_idx]
                # Swap node features
                x[i], x[new_idx] = x[new_idx].clone(), x[i].clone()
            
            # Create new transformed graph
            transformed_graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                n_in=n_in,
                n_out=n_out,
                layer_name=graph.layer_name,
                total_nodes=graph.total_nodes
            )
            
            return transformed_graph
            
        except Exception as e:
            logger.error(f"Error applying permutation transform: {e}")
            return graph.clone()
    
    def apply_scaling_transform(self, graph, scale_range=(0.5, 2.0)):
        """
        Apply a scaling transformation to a graph.
        
        Args:
            graph: torch_geometric.data.Data object
            scale_range: Range for scaling factors
            
        Returns:
            Transformed graph
        """
        try:
            # Extract necessary attributes
            n_in = graph.n_in
            n_out = graph.n_out
            edge_index = graph.edge_index.clone()
            edge_attr = graph.edge_attr.clone()
            x = graph.x.clone()
            
            # Generate random scaling factors for output nodes
            min_scale, max_scale = scale_range
            scales = torch.rand(n_out) * (max_scale - min_scale) + min_scale
            
            # Apply scaling to edge attributes
            for i in range(edge_index.size(1)):
                # Check if it's an output node (index >= n_in)
                if edge_index[1, i] >= n_in:
                    # Get the output node index (relative to output space)
                    out_idx = edge_index[1, i] - n_in
                    # Get the scaling factor
                    scale = scales[out_idx]
                    # Update the edge attribute (weight value)
                    edge_attr[i, 0] *= scale  # Assuming weight value is at index 0
                    edge_attr[i, 1] = abs(edge_attr[i, 0])  # Update magnitude feature
            
            # Create new transformed graph
            transformed_graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                n_in=n_in,
                n_out=n_out,
                layer_name=graph.layer_name,
                total_nodes=graph.total_nodes
            )
            
            return transformed_graph
            
        except Exception as e:
            logger.error(f"Error applying scaling transform: {e}")
            return graph.clone()
    
    def apply_dropout_transform(self, graph, dropout_prob=0.05):
        """
        Apply a dropout transformation to a graph.
        
        Args:
            graph: torch_geometric.data.Data object
            dropout_prob: Probability of dropping edges
            
        Returns:
            Transformed graph
        """
        try:
            # Extract necessary attributes
            edge_index = graph.edge_index.clone()
            edge_attr = graph.edge_attr.clone()
            
            # Create dropout mask
            mask = torch.rand(edge_index.size(1)) > dropout_prob
            
            # Apply mask
            edge_index = edge_index[:, mask]
            edge_attr = edge_attr[mask]
            
            # Create new transformed graph
            transformed_graph = Data(
                x=graph.x.clone(),
                edge_index=edge_index,
                edge_attr=edge_attr,
                n_in=graph.n_in,
                n_out=graph.n_out,
                layer_name=graph.layer_name,
                total_nodes=graph.total_nodes
            )
            
            return transformed_graph
            
        except Exception as e:
            logger.error(f"Error applying dropout transform: {e}")
            return graph.clone()
    
    def visualize_graph(self, graph, filename=None):
        """
        Visualize a graph using NetworkX.
        
        Args:
            graph: torch_geometric.data.Data object
            filename: Optional filename to save the visualization
            
        Returns:
            NetworkX graph object
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes
            n_in = graph.n_in
            n_out = graph.n_out
            
            # Add input nodes
            for i in range(n_in):
                G.add_node(i, type='input', pos=(0, i))
                
            # Add output nodes
            for i in range(n_out):
                G.add_node(n_in + i, type='output', pos=(1, i))
            
            # Add edges
            edge_index = graph.edge_index.cpu().numpy()
            edge_attr = graph.edge_attr.cpu().numpy()
            
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                weight = edge_attr[i, 0]  # Assuming first feature is the weight
                G.add_edge(src, dst, weight=weight)
            
            # Set node positions
            pos = nx.get_node_attributes(G, 'pos')
            
            # Set node colors
            node_colors = ['skyblue' if G.nodes[n]['type'] == 'input' else 'lightgreen' for n in G.nodes]
            
            # Set edge colors and widths
            edge_colors = []
            edge_widths = []
            
            for u, v, d in G.edges(data=True):
                weight = d['weight']
                edge_colors.append('red' if weight < 0 else 'blue')
                edge_widths.append(abs(weight) * 2)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            plt.title(f"Layer: {graph.layer_name}")
            plt.axis('off')
            
            # Save or show
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            
            return G
            
        except Exception as e:
            logger.error(f"Error visualizing graph: {e}")
            return None


import random  # Import here to avoid circular dependencies

# Test code
if __name__ == "__main__":
    # Create a simple weight matrix for testing
    weight_matrix = torch.randn(5, 3)  # 5 output neurons, 3 input neurons
    
    # Create converter
    converter = WeightToGraph()
    
    # Convert to graph
    graph = converter._weight_matrix_to_graph(weight_matrix, layer_name="test_layer")
    
    # Print graph info
    print(f"Graph: {graph}")
    print(f"Nodes: {graph.x.shape}")
    print(f"Edges: {graph.edge_index.shape}")
    print(f"Edge features: {graph.edge_attr.shape}")
    
    # Apply transforms
    print("\nApplying transforms...")
    perm_graph = converter.apply_permutation_transform(graph)
    scale_graph = converter.apply_scaling_transform(graph)
    dropout_graph = converter.apply_dropout_transform(graph)
    
    print(f"Original graph edges: {graph.edge_index.shape}")
    print(f"Permuted graph edges: {perm_graph.edge_index.shape}")
    print(f"Scaled graph edges: {scale_graph.edge_index.shape}")
    print(f"Dropout graph edges: {dropout_graph.edge_index.shape}")
    
    try:
        # Visualize if matplotlib is available
        orig_graph_nx = converter.visualize_graph(graph, filename="original_graph.png")
        perm_graph_nx = converter.visualize_graph(perm_graph, filename="permuted_graph.png")
        scale_graph_nx = converter.visualize_graph(scale_graph, filename="scaled_graph.png")
        dropout_graph_nx = converter.visualize_graph(dropout_graph, filename="dropout_graph.png")
        print("Visualization saved.")
    except ImportError:
        print("Matplotlib not available, skipping visualization.")
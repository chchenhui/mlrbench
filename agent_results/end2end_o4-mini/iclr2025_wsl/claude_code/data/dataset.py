"""
Dataset classes for permutation-equivariant weight graph embeddings.
"""
import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import torch_geometric.data as geom_data
from torch_geometric.data import Batch as GeometricBatch
import random


class ModelData:
    """Container for a single neural network's weight data."""
    def __init__(self, model_id, architecture, weights, biases, accuracy, task_id=None):
        """
        Args:
            model_id: Unique identifier for the model
            architecture: Architecture name (e.g., 'resnet18', 'vgg16')
            weights: List of weight matrices for each layer [layer_idx][out_neurons, in_neurons]
            biases: List of bias vectors for each layer [layer_idx][out_neurons]
            accuracy: Validation accuracy (0.0-1.0) for task
            task_id: Task identifier (optional)
        """
        self.model_id = model_id
        self.architecture = architecture
        self.weights = weights
        self.biases = biases
        self.accuracy = accuracy
        self.task_id = task_id
        
        # Verify shapes
        assert len(weights) == len(biases), "Number of weight and bias tensors must match"
        for w, b in zip(weights, biases):
            assert w.shape[0] == b.shape[0], f"Weight and bias shapes don't match: {w.shape}, {b.shape}"
    
    @property
    def num_layers(self):
        """Get the number of layers."""
        return len(self.weights)
    
    def to_layer_graphs(self):
        """
        Convert model weights to a list of layer graphs.
        
        Returns:
            List of torch_geometric.data.Data objects, one per layer
        """
        layer_graphs = []
        
        for layer_idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            # Get layer dimensions
            out_neurons, in_neurons = W.shape
            
            # Create node features: [weight_norm, bias_mag, in_dim, out_dim]
            weight_norms = torch.norm(W, dim=1, keepdim=True)  # L2 norm of incoming weights
            bias_mags = torch.abs(b).unsqueeze(1)  # Magnitude of bias
            in_dims = torch.full((out_neurons, 1), in_neurons, dtype=torch.float)
            out_dims = torch.full((out_neurons, 1), out_neurons, dtype=torch.float)
            
            node_features = torch.cat([weight_norms, bias_mags, in_dims, out_dims], dim=1)
            
            # Create edge index (fully connected directed graph)
            rows, cols = [], []
            for i in range(out_neurons):
                for j in range(in_neurons):
                    rows.append(i)  # Destination (output neuron)
                    cols.append(j)  # Source (input neuron)
            
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            
            # Create edge attributes (weight values)
            edge_attr = W.view(-1, 1)  # Flatten weights to [out_neurons * in_neurons, 1]
            
            # Create graph data
            graph = geom_data.Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                layer_idx=layer_idx,
                model_id=self.model_id
            )
            
            layer_graphs.append(graph)
        
        return layer_graphs
    
    def apply_random_permutation(self):
        """
        Apply random neuron permutations and scalings to create a transformed copy.
        
        Returns:
            New ModelData with permuted weights and biases
        """
        perm_weights = []
        perm_biases = []
        
        for layer_idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            out_neurons, in_neurons = W.shape
            
            # Random permutation for output neurons
            out_perm = torch.randperm(out_neurons)
            
            # Random permutation for input neurons
            in_perm = torch.randperm(in_neurons)
            
            # Random scaling factors
            out_scale = torch.rand(out_neurons) + 0.5  # between 0.5 and 1.5
            in_scale = torch.rand(in_neurons) + 0.5
            
            # Apply permutations and scalings
            W_perm = W[out_perm, :]
            W_perm = W_perm[:, in_perm]
            b_perm = b[out_perm]
            
            # Apply scalings (output scalings cancel with input scalings of next layer)
            for i in range(out_neurons):
                W_perm[i, :] *= out_scale[i]
            for j in range(in_neurons):
                W_perm[:, j] *= in_scale[j]
            
            # Scale biases
            b_perm = b_perm * out_scale
            
            perm_weights.append(W_perm)
            perm_biases.append(b_perm)
        
        return ModelData(
            model_id=f"{self.model_id}_perm",
            architecture=self.architecture,
            weights=perm_weights,
            biases=perm_biases,
            accuracy=self.accuracy,
            task_id=self.task_id
        )
    
    def flatten_weights(self, normalize=True):
        """
        Flatten all weights and biases into a single vector.
        
        Args:
            normalize: Whether to normalize each layer's parameters
        
        Returns:
            Flattened vector of all parameters
        """
        flat_tensors = []
        
        for W, b in zip(self.weights, self.biases):
            if normalize:
                # Normalize each layer independently
                w_flat = W.reshape(-1)
                w_norm = torch.norm(w_flat)
                if w_norm > 0:
                    w_flat = w_flat / w_norm
                
                b_flat = b.reshape(-1)
                b_norm = torch.norm(b_flat)
                if b_norm > 0:
                    b_flat = b_flat / b_norm
            else:
                w_flat = W.reshape(-1)
                b_flat = b.reshape(-1)
                
            flat_tensors.append(w_flat)
            flat_tensors.append(b_flat)
        
        return torch.cat(flat_tensors)


class ModelZooDataset(Dataset):
    """Dataset for a collection of neural network weights."""
    def __init__(self, models, transform=None):
        """
        Args:
            models: List of ModelData objects
            transform: Optional transform to apply to the models
        """
        self.models = models
        self.transform = transform
        
        # Collect metadata
        self.architectures = sorted(list(set(m.architecture for m in models)))
        self.arch_to_idx = {arch: i for i, arch in enumerate(self.architectures)}
        
        tasks = sorted(list(set(m.task_id for m in models if m.task_id is not None)))
        self.tasks = tasks
        self.task_to_idx = {task: i for i, task in enumerate(tasks)}
    
    def __len__(self):
        return len(self.models)
    
    def __getitem__(self, idx):
        model = self.models[idx]
        
        # Apply transform if provided
        if self.transform is not None:
            model = self.transform(model)
        
        # Convert to layer graphs
        layer_graphs = model.to_layer_graphs()
        
        # Create metadata
        metadata = {
            'model_id': model.model_id,
            'architecture': model.architecture,
            'arch_idx': self.arch_to_idx[model.architecture],
            'accuracy': model.accuracy,
        }
        
        if model.task_id is not None:
            metadata['task_id'] = model.task_id
            metadata['task_idx'] = self.task_to_idx[model.task_id]
            
        return layer_graphs, metadata


class ContrastiveTransform:
    """Transform that creates positive pairs by applying permutations."""
    def __call__(self, model):
        """
        Args:
            model: ModelData object
        Returns:
            Original model
        """
        # Original model is returned
        # Permuted version will be created in the collate function
        return model


class PermutationPairCollator:
    """
    Collator for creating batches with original and permuted networks as positive pairs.
    """
    def __init__(self, with_permutation=True):
        self.with_permutation = with_permutation
    
    def __call__(self, batch):
        """
        Args:
            batch: List of (layer_graphs, metadata) tuples
        Returns:
            Batch of data with permuted versions
        """
        layer_graphs_list, metadata_list = zip(*batch)
        models = []
        
        # Collect original models
        for layer_graphs, metadata in zip(layer_graphs_list, metadata_list):
            models.append({
                'layer_graphs': layer_graphs,
                'metadata': metadata,
                'is_permuted': False,
                'original_idx': len(models)
            })
        
        # Create permuted versions if required
        if self.with_permutation:
            for i, (layer_graphs, metadata) in enumerate(zip(layer_graphs_list, metadata_list)):
                # Create permuted version by shuffling nodes
                permuted_graphs = []
                for graph in layer_graphs:
                    # Create a copy of the graph
                    perm_graph = geom_data.Data(
                        x=graph.x.clone(),
                        edge_index=graph.edge_index.clone(),
                        edge_attr=graph.edge_attr.clone(),
                        layer_idx=graph.layer_idx,
                        model_id=f"{graph.model_id}_perm"
                    )
                    
                    # Permute nodes
                    num_nodes = graph.x.size(0)
                    perm = torch.randperm(num_nodes)
                    inverse_perm = torch.zeros_like(perm)
                    inverse_perm[perm] = torch.arange(num_nodes)
                    
                    perm_graph.x = perm_graph.x[perm]
                    
                    # Update edge indices to reflect the permutation
                    perm_graph.edge_index[0] = inverse_perm[perm_graph.edge_index[0]]
                    
                    permuted_graphs.append(perm_graph)
                
                # Create permuted metadata
                perm_metadata = metadata.copy()
                perm_metadata['model_id'] = f"{metadata['model_id']}_perm"
                
                models.append({
                    'layer_graphs': permuted_graphs,
                    'metadata': perm_metadata,
                    'is_permuted': True,
                    'original_idx': i
                })
        
        # Combine and collate into a batch
        all_layers = []
        layer_batch_indices = []
        
        # Group layer graphs by layer index
        max_layers = max(len(m['layer_graphs']) for m in models)
        
        # Initialize lists for each layer
        layer_graphs_by_idx = [[] for _ in range(max_layers)]
        layer_batches_by_idx = [[] for _ in range(max_layers)]
        
        # Group graphs by layer index
        for model_idx, model in enumerate(models):
            for layer_idx, graph in enumerate(model['layer_graphs']):
                layer_graphs_by_idx[layer_idx].append(graph)
                layer_batches_by_idx[layer_idx].extend([model_idx] * graph.x.size(0))
        
        # Create batches for each layer
        batched_layers = []
        for layer_idx in range(max_layers):
            if layer_graphs_by_idx[layer_idx]:
                # Convert batch indices to tensor
                batch_indices = torch.tensor(layer_batches_by_idx[layer_idx], dtype=torch.long)
                
                # Concatenate node features
                node_features = torch.cat([g.x for g in layer_graphs_by_idx[layer_idx]], dim=0)
                
                # Combine edge indices with offset
                edge_indices = []
                edge_attrs = []
                offset = 0
                for graph in layer_graphs_by_idx[layer_idx]:
                    edge_index = graph.edge_index.clone()
                    edge_index[0] += offset  # Increment destination node indices
                    edge_indices.append(edge_index)
                    edge_attrs.append(graph.edge_attr)
                    offset += graph.x.size(0)
                
                edge_index = torch.cat(edge_indices, dim=1)
                edge_attr = torch.cat(edge_attrs, dim=0)
                
                batched_layers.append({
                    'node_features': node_features,
                    'edge_index': edge_index,
                    'edge_attr': edge_attr,
                    'batch_indices': batch_indices
                })
        
        # Create batch data
        batch_data = {
            'layer_node_features': [layer['node_features'] for layer in batched_layers],
            'layer_edge_indices': [layer['edge_index'] for layer in batched_layers],
            'layer_edge_attrs': [layer['edge_attr'] for layer in batched_layers],
            'layer_batch_indices': [layer['batch_indices'] for layer in batched_layers],
            'num_layers': torch.tensor([len(m['layer_graphs']) for m in models], dtype=torch.long),
            'metadata': [m['metadata'] for m in models],
            'is_permuted': torch.tensor([m['is_permuted'] for m in models], dtype=torch.bool),
            'original_idx': torch.tensor([m['original_idx'] for m in models], dtype=torch.long)
        }
        
        # Create positives mask for contrastive learning
        batch_size = len(models)
        positives_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool)
        
        if self.with_permutation:
            # Mark permuted versions as positives of their originals
            for i in range(len(batch)):
                orig_idx = i
                perm_idx = i + len(batch)
                positives_mask[orig_idx, perm_idx] = True
                positives_mask[perm_idx, orig_idx] = True
        
        batch_data['positives_mask'] = positives_mask
        
        return batch_data


class ModelZooManager:
    """Manager for creating and handling model zoo datasets."""
    def __init__(self, save_dir):
        """
        Args:
            save_dir: Directory to save/load model data
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.models = []
    
    def generate_synthetic_model_zoo(self, num_models=1000, architectures=None, 
                                     layer_configs=None, tasks=None):
        """
        Generate a synthetic model zoo for experiments.
        
        Args:
            num_models: Number of models to generate
            architectures: List of architecture names (if None, default list is used)
            layer_configs: Dict mapping architecture to layer configs
            tasks: List of task names (if None, default list is used)
        
        Returns:
            List of ModelData objects
        """
        if architectures is None:
            architectures = ['mlp_small', 'mlp_medium', 'mlp_large', 
                           'cnn_small', 'cnn_medium', 'cnn_large']
            
        if tasks is None:
            tasks = ['cifar10', 'cifar100', 'imagenet', 'mnist']
            
        if layer_configs is None:
            # Define default configurations: (input_dim, output_dim) for each layer
            layer_configs = {
                'mlp_small': [(784, 128), (128, 64), (64, 10)],
                'mlp_medium': [(784, 256), (256, 128), (128, 64), (64, 10)],
                'mlp_large': [(784, 512), (512, 256), (256, 128), (128, 64), (64, 10)],
                'cnn_small': [(3, 16), (16, 32), (32, 64), (64, 10)],
                'cnn_medium': [(3, 32), (32, 64), (64, 128), (128, 256), (256, 10)],
                'cnn_large': [(3, 64), (64, 128), (128, 256), (256, 512), (512, 10)]
            }
        
        models = []
        
        for i in range(num_models):
            # Select random architecture and task
            arch = random.choice(architectures)
            task = random.choice(tasks)
            
            # Get layer configuration
            layers = layer_configs[arch]
            
            # Generate random weights and biases
            weights = []
            biases = []
            
            for in_dim, out_dim in layers:
                W = torch.randn(out_dim, in_dim)
                b = torch.randn(out_dim)
                
                # Normalize to simulate trained weights
                W = W / np.sqrt(in_dim)
                
                weights.append(W)
                biases.append(b)
            
            # Generate a random "accuracy" value that depends on architecture and task
            base_acc = {
                'mlp_small': 0.7,
                'mlp_medium': 0.75,
                'mlp_large': 0.8,
                'cnn_small': 0.8,
                'cnn_medium': 0.85,
                'cnn_large': 0.9
            }.get(arch, 0.75)
            
            # Add task-specific offset
            task_offset = {
                'cifar10': 0.05,
                'cifar100': -0.05,
                'imagenet': -0.1,
                'mnist': 0.1
            }.get(task, 0)
            
            # Add randomness
            random_offset = (random.random() - 0.5) * 0.1
            
            accuracy = min(max(base_acc + task_offset + random_offset, 0.5), 0.99)
            
            # Create model data
            model = ModelData(
                model_id=f"model_{i}",
                architecture=arch,
                weights=weights,
                biases=biases,
                accuracy=accuracy,
                task_id=task
            )
            
            models.append(model)
        
        self.models = models
        return models
    
    def save_model_zoo(self, filename='model_zoo.pt'):
        """
        Save the model zoo to disk.
        
        Args:
            filename: Name of the file to save
        """
        path = os.path.join(self.save_dir, filename)
        
        # Convert to serializable format
        serialized_models = []
        for model in self.models:
            serialized = {
                'model_id': model.model_id,
                'architecture': model.architecture,
                'weights': [w.numpy().tolist() for w in model.weights],
                'biases': [b.numpy().tolist() for b in model.biases],
                'accuracy': float(model.accuracy),
                'task_id': model.task_id
            }
            serialized_models.append(serialized)
        
        # Save as JSON
        with open(path, 'w') as f:
            json.dump(serialized_models, f)
    
    def load_model_zoo(self, filename='model_zoo.pt'):
        """
        Load the model zoo from disk.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            List of ModelData objects
        """
        path = os.path.join(self.save_dir, filename)
        
        with open(path, 'r') as f:
            serialized_models = json.load(f)
        
        models = []
        for data in serialized_models:
            # Convert back to tensors
            weights = [torch.tensor(w, dtype=torch.float) for w in data['weights']]
            biases = [torch.tensor(b, dtype=torch.float) for b in data['biases']]
            
            model = ModelData(
                model_id=data['model_id'],
                architecture=data['architecture'],
                weights=weights,
                biases=biases,
                accuracy=data['accuracy'],
                task_id=data['task_id']
            )
            
            models.append(model)
        
        self.models = models
        return models
    
    def create_train_val_test_split(self, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Split the model zoo into training, validation, and test sets.
        
        Args:
            val_ratio: Ratio of validation set
            test_ratio: Ratio of test set
            seed: Random seed for reproducibility
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        assert len(self.models) > 0, "Model zoo is empty"
        
        # Set random seed
        random.seed(seed)
        
        # Shuffle models
        models = self.models.copy()
        random.shuffle(models)
        
        # Calculate split sizes
        test_size = int(len(models) * test_ratio)
        val_size = int(len(models) * val_ratio)
        train_size = len(models) - test_size - val_size
        
        # Split models
        train_models = models[:train_size]
        val_models = models[train_size:train_size+val_size]
        test_models = models[train_size+val_size:]
        
        # Create datasets
        train_dataset = ModelZooDataset(train_models, transform=ContrastiveTransform())
        val_dataset = ModelZooDataset(val_models)
        test_dataset = ModelZooDataset(test_models)
        
        return train_dataset, val_dataset, test_dataset
    
    def create_data_loaders(self, train_dataset, val_dataset, test_dataset, 
                           batch_size=32, num_workers=0):
        """
        Create data loaders for the datasets.
        
        Args:
            train_dataset, val_dataset, test_dataset: Datasets
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=PermutationPairCollator(with_permutation=True)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=PermutationPairCollator(with_permutation=False)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=PermutationPairCollator(with_permutation=False)
        )
        
        return train_loader, val_loader, test_loader
"""
Data generation module for Neural Weight Archeology experiments

This module creates a dataset of neural network models with labeled properties
for training and evaluating weight analysis methods.
"""

import os
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional
import networkx as nx
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Define several model architectures for the dataset
class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron"""
    def __init__(self, input_dim=784, hidden_dims=[128, 64], output_dim=10):
        super().__init__()
        
        # Convert numpy types to Python native types
        input_dim = int(input_dim) if hasattr(input_dim, 'item') else input_dim
        output_dim = int(output_dim) if hasattr(output_dim, 'item') else output_dim
        
        # Convert hidden dims
        processed_hidden_dims = []
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim) if hasattr(hidden_dim, 'item') else hidden_dim
            processed_hidden_dims.append(hidden_dim)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in processed_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class SimpleCNN(nn.Module):
    """Simple Convolutional Neural Network"""
    def __init__(self, input_channels=3, hidden_channels=[16, 32], output_dim=10):
        super().__init__()
        
        # Convert numpy types to Python native types
        input_channels = int(input_channels) if hasattr(input_channels, 'item') else input_channels
        output_dim = int(output_dim) if hasattr(output_dim, 'item') else output_dim
        
        # Convert hidden channels
        processed_hidden_channels = []
        for channels in hidden_channels:
            channels = int(channels) if hasattr(channels, 'item') else channels
            processed_hidden_channels.append(channels)
        
        self.conv_layers = nn.ModuleList()
        prev_channels = input_channels
        
        for channels in processed_hidden_channels:
            self.conv_layers.append(nn.Conv2d(prev_channels, channels, kernel_size=3, padding=1))
            prev_channels = channels
        
        # Calculate the flattened dimension after convolutions and pooling
        # Assuming input is 32x32 and we have 2 pooling layers
        final_size = 32 // (2 ** len(processed_hidden_channels))
        flattened_dim = prev_channels * (final_size ** 2)
        
        self.fc = nn.Linear(flattened_dim, output_dim)
    
    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SimpleRNN(nn.Module):
    """Simple Recurrent Neural Network"""
    def __init__(self, input_dim=300, hidden_dim=128, output_dim=5):
        super().__init__()
        
        # Convert numpy types to Python native types
        input_dim = int(input_dim) if hasattr(input_dim, 'item') else input_dim
        hidden_dim = int(hidden_dim) if hasattr(hidden_dim, 'item') else hidden_dim
        output_dim = int(output_dim) if hasattr(output_dim, 'item') else output_dim
        
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        _, h_n = self.rnn(x)
        x = self.fc(h_n.squeeze(0))
        return x

class SimpleTransformer(nn.Module):
    """Simple Transformer Encoder"""
    def __init__(self, input_dim=512, nhead=8, num_layers=2, output_dim=2):
        super().__init__()
        
        # Convert numpy types to Python native types
        input_dim = int(input_dim) if hasattr(input_dim, 'item') else input_dim
        nhead = int(nhead) if hasattr(nhead, 'item') else nhead
        num_layers = int(num_layers) if hasattr(num_layers, 'item') else num_layers
        output_dim = int(output_dim) if hasattr(output_dim, 'item') else output_dim
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, input_dim]
        x = self.transformer(x)
        x = x.mean(dim=0)  # Average pooling over sequence
        x = self.fc(x)
        return x

def extract_architecture_info(model: nn.Module) -> Dict:
    """Extract architectural information from a PyTorch model"""
    info = {
        'num_params': sum(p.numel() for p in model.parameters()),
        'num_layers': len(list(model.modules())) - 1,  # Subtract 1 for the model itself
        'layer_types': defaultdict(int),
        'activation_types': defaultdict(int),
    }
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.RNN, nn.LSTM, nn.GRU, nn.TransformerEncoder)):
            module_type = module.__class__.__name__
            info['layer_types'][module_type] += 1
        
        if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.GELU)):
            activation_type = module.__class__.__name__
            info['activation_types'][activation_type] += 1
    
    return info

def extract_weight_statistics(model: nn.Module) -> Dict:
    """Extract statistical information about model weights"""
    all_weights = []
    layer_weights = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param.data.cpu().numpy().flatten()
            all_weights.extend(weights)
            layer_weights[name] = weights
    
    all_weights = np.array(all_weights)
    
    stats = {
        'mean': float(np.mean(all_weights)),
        'std': float(np.std(all_weights)),
        'min': float(np.min(all_weights)),
        'max': float(np.max(all_weights)),
        'sparsity': float(np.sum(np.abs(all_weights) < 1e-6) / len(all_weights)),
        'l1_norm': float(np.sum(np.abs(all_weights))),
        'l2_norm': float(np.sqrt(np.sum(all_weights**2))),
    }
    
    # Layer-wise statistics
    layer_stats = {}
    for name, weights in layer_weights.items():
        layer_stats[name] = {
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
            'min': float(np.min(weights)),
            'max': float(np.max(weights)),
            'sparsity': float(np.sum(np.abs(weights) < 1e-6) / len(weights)),
        }
    
    stats['layer_stats'] = layer_stats
    
    return stats

def compute_eigenvalues(model: nn.Module, num_values=10) -> Dict:
    """Compute eigenvalues of weight matrices"""
    eigenvalues = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) == 2:
            # For 2D weight matrices (e.g., Linear layers)
            W = param.data.cpu().numpy()
            
            # Compute singular values (eigenvalues of W^T * W)
            try:
                if W.shape[0] > W.shape[1]:
                    WtW = W.T @ W
                else:
                    WtW = W @ W.T
                
                # Get the top eigenvalues
                evals, _ = np.linalg.eigh(WtW)
                evals = np.sort(evals)[::-1]  # Sort in descending order
                
                # Keep only top N eigenvalues
                eigenvalues[name] = evals[:num_values].tolist()
            except:
                # Skip if computation fails
                continue
    
    return eigenvalues

def create_graph_representation(model: nn.Module) -> nx.Graph:
    """Create a graph representation of the model architecture"""
    G = nx.Graph()
    
    # Add nodes for each parameter tensor
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name:
            G.add_node(
                i, 
                name=name, 
                shape=list(param.shape), 
                size=param.numel(),
                layer_type=name.split('.')[0]
            )
    
    # Add edges between consecutive layers
    nodes = list(G.nodes(data=True))
    for i in range(len(nodes) - 1):
        G.add_edge(
            nodes[i][0], 
            nodes[i+1][0], 
            weight=1.0
        )
    
    return G

def train_model_with_properties(
    model_class, 
    model_args, 
    num_epochs=10, 
    learning_rate=0.001,
    weight_decay=0.0,
    batch_size=64,
    train_noise=0.0,
    seed=42
) -> Tuple[nn.Module, Dict]:
    """Train a model and record its properties"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model
    model = model_class(**model_args)
    
    # Generate synthetic data based on model type
    if model_class == SimpleMLP:
        input_dim = model_args.get('input_dim', 784)
        output_dim = model_args.get('output_dim', 10)
        
        # Create synthetic data for classification
        X = torch.randn(1000, input_dim)
        y = torch.randint(0, output_dim, (1000,))
        
    elif model_class == SimpleCNN:
        input_channels = model_args.get('input_channels', 3)
        output_dim = model_args.get('output_dim', 10)
        
        # Create synthetic image data
        X = torch.randn(1000, input_channels, 32, 32)
        y = torch.randint(0, output_dim, (1000,))
        
    elif model_class == SimpleRNN:
        input_dim = model_args.get('input_dim', 300)
        output_dim = model_args.get('output_dim', 5)
        
        # Create synthetic sequence data
        seq_len = 20
        X = torch.randn(1000, seq_len, input_dim)
        y = torch.randint(0, output_dim, (1000,))
        
    elif model_class == SimpleTransformer:
        input_dim = model_args.get('input_dim', 512)
        output_dim = model_args.get('output_dim', 2)
        
        # Create synthetic sequence data
        seq_len = 10
        X = torch.randn(1000, seq_len, input_dim)
        y = torch.randint(0, output_dim, (1000,))
        
    else:
        raise ValueError(f"Unsupported model class: {model_class.__name__}")
    
    # Split into train and validation sets
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Add noise to training data if specified
    if train_noise > 0:
        X_train = X_train + torch.randn_like(X_train) * train_noise
    
    # Set up training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        
        # Process in batches
        train_loss = 0.0
        correct = 0
        total = 0
        
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        train_loss = train_loss / (len(X_train) / batch_size)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                x_batch = X_val[i:i+batch_size]
                y_batch = y_val[i:i+batch_size]
                
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_loss = val_loss / (len(X_val) / batch_size)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
    
    # Record model properties
    properties = {
        'model_type': model_class.__name__,
        'train_params': {
            'epochs': num_epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'train_noise': train_noise,
            'seed': seed,
        },
        'performance': {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_train_accuracy': train_accuracies[-1],
            'final_val_accuracy': val_accuracies[-1],
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
        },
        'architecture': extract_architecture_info(model),
        'weight_statistics': extract_weight_statistics(model),
        'eigenvalues': compute_eigenvalues(model),
        'graph_representation': create_graph_representation(model),
    }
    
    # Compute training dynamics features
    properties['training_dynamics'] = {
        'loss_decrease_rate': (train_losses[0] - train_losses[-1]) / train_losses[0],
        'accuracy_increase_rate': (train_accuracies[-1] - train_accuracies[0]) / (100 - train_accuracies[0]) if train_accuracies[0] < 100 else 1.0,
        'convergence_epoch': np.argmin(val_losses),
        'overfitting_measure': max(0, val_losses[-1] - np.min(val_losses)) / np.min(val_losses),
    }
    
    return model, properties

class ModelZooDataset(Dataset):
    """Dataset of neural network models with labeled properties"""
    
    def __init__(self, num_models=100, data_dir='./data', create_if_not_exists=True):
        self.data_dir = data_dir
        self.model_dir = os.path.join(data_dir, 'models')
        self.property_dir = os.path.join(data_dir, 'properties')
        self.num_models = num_models
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.property_dir, exist_ok=True)
        
        # Load or create the dataset
        self.models = []
        self.properties = []
        
        # Check if dataset already exists
        if os.path.exists(os.path.join(data_dir, 'dataset_info.json')):
            with open(os.path.join(data_dir, 'dataset_info.json'), 'r') as f:
                dataset_info = json.load(f)
            
            # Load existing dataset
            for i in range(dataset_info['num_models']):
                model_path = os.path.join(self.model_dir, f'model_{i}.pt')
                property_path = os.path.join(self.property_dir, f'properties_{i}.json')
                
                # Load model and properties
                if os.path.exists(model_path) and os.path.exists(property_path):
                    with open(property_path, 'r') as f:
                        properties = json.load(f)
                    
                    self.properties.append(properties)
            
            self.num_models = dataset_info['num_models']
            
            # Set dataset properties
            self._set_dataset_properties()
            
        elif create_if_not_exists:
            # Create new dataset
            self._create_dataset()
            
            # Save dataset info
            dataset_info = {
                'num_models': self.num_models,
                'num_features': self.num_features,
                'num_classes': self.num_classes,
                'num_regression_targets': self.num_regression_targets,
                'class_names': self.class_names,
                'regression_target_names': self.regression_target_names,
            }
            
            with open(os.path.join(data_dir, 'dataset_info.json'), 'w') as f:
                json.dump(dataset_info, f, indent=2)
        
        else:
            raise FileNotFoundError(f"Dataset not found at {data_dir} and create_if_not_exists is False")
    
    def _create_dataset(self):
        """Create a new dataset of models with properties"""
        logger.info(f"Creating dataset with {self.num_models} models...")
        
        model_classes = [SimpleMLP, SimpleCNN, SimpleRNN, SimpleTransformer]
        
        for i in tqdm(range(self.num_models), desc="Generating models"):
            # Randomly select model class and configuration
            model_class = np.random.choice(model_classes)
            
            if model_class == SimpleMLP:
                hidden_dims = np.random.choice([32, 64, 128, 256], size=np.random.randint(1, 4)).tolist()
                model_args = {
                    'input_dim': np.random.choice([100, 300, 784, 1024]),
                    'hidden_dims': hidden_dims,
                    'output_dim': np.random.choice([2, 5, 10]),
                }
            
            elif model_class == SimpleCNN:
                hidden_channels = np.random.choice([8, 16, 32, 64], size=np.random.randint(1, 4)).tolist()
                model_args = {
                    'input_channels': np.random.choice([1, 3]),
                    'hidden_channels': hidden_channels,
                    'output_dim': np.random.choice([2, 5, 10]),
                }
            
            elif model_class == SimpleRNN:
                model_args = {
                    'input_dim': np.random.choice([50, 100, 300]),
                    'hidden_dim': np.random.choice([32, 64, 128, 256]),
                    'output_dim': np.random.choice([2, 5]),
                }
            
            elif model_class == SimpleTransformer:
                model_args = {
                    'input_dim': np.random.choice([64, 128, 256, 512]),
                    'nhead': np.random.choice([4, 8]),
                    'num_layers': np.random.choice([1, 2, 3]),
                    'output_dim': np.random.choice([2, 5]),
                }
            
            # Training settings
            train_params = {
                'num_epochs': np.random.randint(5, 20),
                'learning_rate': 10 ** np.random.uniform(-4, -2),
                'weight_decay': 10 ** np.random.uniform(-6, -3) if np.random.random() > 0.5 else 0.0,
                'batch_size': np.random.choice([16, 32, 64, 128]),
                'train_noise': np.random.uniform(0, 0.1) if np.random.random() > 0.7 else 0.0,
                'seed': np.random.randint(0, 10000),
            }
            
            # Train model and get properties
            model, properties = train_model_with_properties(
                model_class, 
                model_args, 
                **train_params
            )
            
            # Save model
            torch.save(model.state_dict(), os.path.join(self.model_dir, f'model_{i}.pt'))
            
            # Save properties
            with open(os.path.join(self.property_dir, f'properties_{i}.json'), 'w') as f:
                # Convert graph to adjacency list for JSON serialization
                if 'graph_representation' in properties:
                    graph = properties['graph_representation']
                    properties['graph_representation'] = {
                        'nodes': [{
                            'id': int(n),
                            'data': {k: v for k, v in d.items()}
                        } for n, d in graph.nodes(data=True)],
                        'edges': [(int(u), int(v), {k: val for k, val in d.items()}) 
                                 for u, v, d in graph.edges(data=True)]
                    }
                
                # Convert numpy types to Python native types for JSON serialization
                def convert_numpy_types(obj):
                    if isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    elif isinstance(obj, tuple):
                        return tuple(convert_numpy_types(item) for item in obj)
                    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                        return int(obj)
                    elif isinstance(obj, (np.float64, np.float32, np.float16)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return convert_numpy_types(obj.tolist())
                    else:
                        return obj
                
                # Convert all numpy types in properties
                serializable_properties = convert_numpy_types(properties)
                
                json.dump(serializable_properties, f, indent=2)
            
            self.properties.append(properties)
        
        # Set dataset properties
        self._set_dataset_properties()
    
    def _set_dataset_properties(self):
        """Set dataset properties based on the generated models"""
        # Define classification targets
        self.class_names = ['model_type']
        self.class_mapping = {
            'model_type': {
                'SimpleMLP': 0,
                'SimpleCNN': 1,
                'SimpleRNN': 2,
                'SimpleTransformer': 3,
            }
        }
        self.num_classes = {
            'model_type': len(self.class_mapping['model_type']),
        }
        
        # Define regression targets
        self.regression_target_names = [
            'final_val_accuracy',
            'overfitting_measure',
            'convergence_epoch',
            'sparsity',
            'mean_weight',
            'std_weight',
        ]
        self.num_regression_targets = len(self.regression_target_names)
        
        # Define feature extraction
        self.num_features = 50  # Number of features to extract from each model
    
    def _extract_features(self, properties):
        """Extract features from model properties"""
        features = []
        
        # Basic architecture features
        arch = properties['architecture']
        features.extend([
            arch['num_params'] / 1e6,  # Normalize by a million
            arch['num_layers'] / 10,   # Normalize by dividing by 10
            arch['layer_types'].get('Linear', 0) / 5,
            arch['layer_types'].get('Conv2d', 0) / 5,
            arch['layer_types'].get('TransformerEncoder', 0) / 3,
            arch['layer_types'].get('GRU', 0) / 2,
            arch['activation_types'].get('ReLU', 0) / 5,
            arch['activation_types'].get('Tanh', 0) / 5,
            arch['activation_types'].get('GELU', 0) / 5,
        ])
        
        # Weight statistics features
        stats = properties['weight_statistics']
        features.extend([
            stats['mean'],
            stats['std'],
            stats['min'],
            stats['max'],
            stats['sparsity'],
            stats['l1_norm'] / 1000,  # Normalize
            stats['l2_norm'] / 100,   # Normalize
        ])
        
        # Training dynamics features
        dynamics = properties['training_dynamics']
        features.extend([
            dynamics['loss_decrease_rate'],
            dynamics['accuracy_increase_rate'],
            dynamics['convergence_epoch'] / 20,  # Normalize by max epochs
            dynamics['overfitting_measure'],
        ])
        
        # Performance features
        perf = properties['performance']
        features.extend([
            perf['final_train_loss'],
            perf['final_val_loss'],
            perf['final_train_accuracy'] / 100,  # Normalize to [0, 1]
            perf['final_val_accuracy'] / 100,    # Normalize to [0, 1]
        ])
        
        # Training parameters
        train_params = properties['train_params']
        features.extend([
            np.log10(train_params['learning_rate']) + 4,  # Normalize log LR
            np.log10(train_params['weight_decay'] + 1e-10) + 10,  # Normalize log WD
            train_params['batch_size'] / 128,  # Normalize by max batch size
            train_params['train_noise'] * 10,  # Scale up for better signal
            train_params['epochs'] / 20,  # Normalize by max epochs
        ])
        
        # Eigenvalue features (top 5 eigenvalues from first layer)
        if properties['eigenvalues']:
            first_layer = list(properties['eigenvalues'].keys())[0]
            eigenvalues = properties['eigenvalues'][first_layer]
            # Normalize eigenvalues
            if eigenvalues:
                max_eigenvalue = max(eigenvalues)
                normalized_eigenvalues = [e / max_eigenvalue for e in eigenvalues[:5]]
                # Pad with zeros if less than 5
                normalized_eigenvalues += [0] * (5 - len(normalized_eigenvalues))
                features.extend(normalized_eigenvalues)
            else:
                features.extend([0] * 5)
        else:
            features.extend([0] * 5)
        
        # Graph structure features
        graph = properties['graph_representation']
        nodes = graph['nodes']
        edges = graph['edges']
        
        # Node count
        features.append(len(nodes) / 10)  # Normalize by dividing by 10
        
        # Edge count
        features.append(len(edges) / 10)  # Normalize by dividing by 10
        
        # Graph density (if there are nodes)
        if len(nodes) > 1:
            max_edges = len(nodes) * (len(nodes) - 1) / 2
            density = len(edges) / max_edges if max_edges > 0 else 0
            features.append(density)
        else:
            features.append(0)
        
        # Pad or truncate to ensure consistent feature length
        if len(features) < self.num_features:
            features.extend([0] * (self.num_features - len(features)))
        elif len(features) > self.num_features:
            features = features[:self.num_features]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_classification_targets(self, properties):
        """Extract classification targets from model properties"""
        targets = {}
        
        for class_name in self.class_names:
            if class_name == 'model_type':
                model_type = properties['model_type']
                targets[class_name] = self.class_mapping[class_name].get(model_type, 0)
        
        return targets
    
    def _extract_regression_targets(self, properties):
        """Extract regression targets from model properties"""
        targets = {}
        
        for target_name in self.regression_target_names:
            if target_name == 'final_val_accuracy':
                targets[target_name] = properties['performance']['final_val_accuracy'] / 100
            elif target_name == 'overfitting_measure':
                targets[target_name] = properties['training_dynamics']['overfitting_measure']
            elif target_name == 'convergence_epoch':
                targets[target_name] = properties['training_dynamics']['convergence_epoch'] / 20
            elif target_name == 'sparsity':
                targets[target_name] = properties['weight_statistics']['sparsity']
            elif target_name == 'mean_weight':
                targets[target_name] = properties['weight_statistics']['mean']
            elif target_name == 'std_weight':
                targets[target_name] = properties['weight_statistics']['std']
        
        return targets
    
    def __len__(self):
        return len(self.properties)
    
    def __getitem__(self, idx):
        # Extract properties
        properties = self.properties[idx]
        
        # Extract features
        features = self._extract_features(properties)
        
        # Extract classification targets
        classification_targets = self._extract_classification_targets(properties)
        
        # Extract regression targets
        regression_targets = self._extract_regression_targets(properties)
        
        # Combine targets for return
        result = {
            'features': features,
            'model_id': idx,
        }
        
        # Add classification targets
        for class_name in self.class_names:
            result[f'class_{class_name}'] = torch.tensor(classification_targets[class_name], dtype=torch.long)
        
        # Add regression targets
        regression_values = [regression_targets[name] for name in self.regression_target_names]
        result['regression_targets'] = torch.tensor(regression_values, dtype=torch.float32)
        
        return result
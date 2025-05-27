import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import pickle
import random
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class ModelWeightsDataset(Dataset):
    """Dataset for model weights and properties."""
    
    def __init__(
        self,
        data_path: str,
        model_properties: List[str],
        max_models: Optional[int] = None,
        transform: Optional[Callable] = None,
        canonicalization_method: Optional[str] = None,
        tokenization_strategy: str = "neuron_centric",
        max_token_length: int = 4096,
    ):
        """
        Initialize the ModelWeightsDataset.
        
        Args:
            data_path: Path to the directory containing model weights and properties
            model_properties: List of properties to predict (e.g., ['accuracy', 'robustness'])
            max_models: Maximum number of models to include in the dataset
            transform: Optional transform to apply to the weights
            canonicalization_method: Method to use for canonicalizing weights ("activation_sort", 
                                                                              "weight_sort", 
                                                                              "ot", or None)
            tokenization_strategy: Strategy for tokenizing weights ("global", "neuron_centric", "layer_centric")
            max_token_length: Maximum number of tokens to include
        """
        self.data_path = data_path
        self.model_properties = model_properties
        self.max_models = max_models
        self.transform = transform
        self.canonicalization_method = canonicalization_method
        self.tokenization_strategy = tokenization_strategy
        self.max_token_length = max_token_length
        
        self.models_metadata = self._load_metadata()
        self.model_ids = list(self.models_metadata.keys())
        if max_models is not None:
            self.model_ids = self.model_ids[:max_models]
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata for all models in the dataset."""
        metadata_path = os.path.join(self.data_path, "metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            return metadata
        else:
            # For demo purposes, create synthetic metadata
            return self._create_synthetic_metadata()
    
    def _create_synthetic_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Create synthetic metadata for demonstration purposes."""
        logger.info("Creating synthetic metadata for demonstration")
        
        # For this demo, we'll create metadata for 100 models
        model_architectures = ["resnet18", "resnet34", "resnet50", "vgg16", "mobilenet"]
        metadata = {}
        
        for i in range(100):
            model_id = f"model_{i}"
            architecture = random.choice(model_architectures)
            
            # Generate random properties
            properties = {
                "accuracy": random.uniform(0.7, 0.95),
                "robustness": random.uniform(0.5, 0.9),
                "generalization_gap": random.uniform(0.02, 0.15),
                "parameters": random.randint(5000000, 25000000),
                "flops": random.randint(500000000, 5000000000),
                "architecture": architecture
            }
            
            metadata[model_id] = properties
        
        # Save the metadata
        os.makedirs(os.path.dirname(os.path.join(self.data_path, "metadata.pkl")), exist_ok=True)
        with open(os.path.join(self.data_path, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
            
        return metadata
    
    def _load_model_weights(self, model_id: str) -> Dict[str, torch.Tensor]:
        """
        Load weights for a model.
        
        Args:
            model_id: ID of the model to load weights for
            
        Returns:
            Dictionary of weight tensors for the model
        """
        weights_path = os.path.join(self.data_path, f"{model_id}_weights.pt")
        if os.path.exists(weights_path):
            return torch.load(weights_path)
        else:
            # For demo purposes, create synthetic weights
            return self._create_synthetic_weights(model_id)
    
    def _create_synthetic_weights(self, model_id: str) -> Dict[str, torch.Tensor]:
        """Create synthetic weights for demonstration purposes."""
        architecture = self.models_metadata[model_id]["architecture"]
        
        # Create synthetic weights based on architecture
        if architecture == "resnet18":
            layer_sizes = [(64, 3, 3, 3), (64, 64, 3, 3), (128, 64, 3, 3), 
                           (128, 128, 3, 3), (256, 128, 3, 3), (256, 256, 3, 3),
                           (512, 256, 3, 3), (512, 512, 3, 3), (1000, 512)]
        elif architecture == "resnet34":
            layer_sizes = [(64, 3, 3, 3), (64, 64, 3, 3), (64, 64, 3, 3), (64, 64, 3, 3),
                           (128, 64, 3, 3), (128, 128, 3, 3), (128, 128, 3, 3), (128, 128, 3, 3),
                           (256, 128, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3),
                           (512, 256, 3, 3), (512, 512, 3, 3), (512, 512, 3, 3), (1000, 512)]
        elif architecture == "resnet50":
            layer_sizes = [(64, 3, 3, 3), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1),
                           (64, 256, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1),
                           (128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1),
                           (128, 512, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1),
                           (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1),
                           (256, 1024, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1),
                           (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1),
                           (512, 2048, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1),
                           (1000, 2048)]
        elif architecture == "vgg16":
            layer_sizes = [(64, 3, 3, 3), (64, 64, 3, 3), (128, 64, 3, 3), (128, 128, 3, 3),
                           (256, 128, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3),
                           (512, 256, 3, 3), (512, 512, 3, 3), (512, 512, 3, 3),
                           (512, 512, 3, 3), (512, 512, 3, 3), (512, 512, 3, 3),
                           (4096, 512 * 7 * 7), (4096, 4096), (1000, 4096)]
        else:  # mobilenet or default
            layer_sizes = [(32, 3, 3, 3), (64, 32, 1, 1), (128, 64, 1, 1), (128, 128, 1, 1),
                           (256, 128, 1, 1), (256, 256, 1, 1), (512, 256, 1, 1),
                           (512, 512, 1, 1), (512, 512, 1, 1), (512, 512, 1, 1),
                           (512, 512, 1, 1), (512, 512, 1, 1), (1024, 512, 1, 1),
                           (1024, 1024, 1, 1), (1000, 1024)]
        
        weights = {}
        biases = {}
        
        # Generate random weights for each layer
        for i, size in enumerate(layer_sizes):
            if len(size) == 2:  # Linear layer
                out_features, in_features = size
                weights[f"layer_{i}.weight"] = torch.randn(out_features, in_features)
                biases[f"layer_{i}.bias"] = torch.randn(out_features)
            else:  # Conv layer
                out_channels, in_channels, kernel_size1, kernel_size2 = size
                weights[f"layer_{i}.weight"] = torch.randn(out_channels, in_channels, kernel_size1, kernel_size2)
                biases[f"layer_{i}.bias"] = torch.randn(out_channels)
        
        # Save weights
        os.makedirs(os.path.dirname(os.path.join(self.data_path, f"{model_id}_weights.pt")), exist_ok=True)
        combined_weights = {**weights, **biases}
        torch.save(combined_weights, os.path.join(self.data_path, f"{model_id}_weights.pt"))
        
        return combined_weights
    
    def _canonicalize_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Canonicalize the weights to handle permutation symmetry.
        
        Args:
            weights: Dictionary of weight tensors
            
        Returns:
            Dictionary of canonicalized weight tensors
        """
        if self.canonicalization_method is None:
            return weights
        
        canonicalized_weights = {}
        
        for layer_name, weight in weights.items():
            if "weight" not in layer_name:
                canonicalized_weights[layer_name] = weight
                continue
                
            if self.canonicalization_method == "weight_sort":
                # Sort neurons based on L2 norm of weights
                if weight.dim() == 2:  # Linear layer
                    # Sort by incoming weight norm
                    norms = torch.norm(weight, dim=1)
                    sorted_indices = torch.argsort(norms, descending=True)
                    canonicalized_weights[layer_name] = weight[sorted_indices]
                    
                    # Also sort the corresponding bias if it exists
                    bias_name = layer_name.replace("weight", "bias")
                    if bias_name in weights:
                        canonicalized_weights[bias_name] = weights[bias_name][sorted_indices]
                        
                elif weight.dim() == 4:  # Conv layer
                    # Sort by incoming weight norm (across channels)
                    norms = torch.norm(weight.reshape(weight.shape[0], -1), dim=1)
                    sorted_indices = torch.argsort(norms, descending=True)
                    canonicalized_weights[layer_name] = weight[sorted_indices]
                    
                    # Also sort the corresponding bias if it exists
                    bias_name = layer_name.replace("weight", "bias")
                    if bias_name in weights:
                        canonicalized_weights[bias_name] = weights[bias_name][sorted_indices]
                        
            elif self.canonicalization_method == "activation_sort":
                # This would require running inference on a fixed batch of data
                # Simplified version: use random "activations" for demo
                if weight.dim() == 2:  # Linear layer
                    mock_activations = torch.randn(weight.shape[0])
                    sorted_indices = torch.argsort(mock_activations, descending=True)
                    canonicalized_weights[layer_name] = weight[sorted_indices]
                    
                    bias_name = layer_name.replace("weight", "bias")
                    if bias_name in weights:
                        canonicalized_weights[bias_name] = weights[bias_name][sorted_indices]
                        
                elif weight.dim() == 4:  # Conv layer
                    mock_activations = torch.randn(weight.shape[0])
                    sorted_indices = torch.argsort(mock_activations, descending=True)
                    canonicalized_weights[layer_name] = weight[sorted_indices]
                    
                    bias_name = layer_name.replace("weight", "bias")
                    if bias_name in weights:
                        canonicalized_weights[bias_name] = weights[bias_name][sorted_indices]
            
            elif self.canonicalization_method == "ot":
                # Optimal Transport canonicalization would go here
                # For demo, we'll use a simpler approach
                canonicalized_weights[layer_name] = weight
                
            else:
                canonicalized_weights[layer_name] = weight
        
        return canonicalized_weights
    
    def _tokenize_weights(self, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Tokenize the weights according to the specified strategy.
        
        Args:
            weights: Dictionary of weight tensors
            
        Returns:
            Tensor of tokens [num_tokens, token_dim]
        """
        if self.tokenization_strategy == "global":
            # Global flattening: concatenate all weights and chunk
            all_weights = []
            for layer_name, weight in weights.items():
                all_weights.append(weight.flatten())
            
            flat_weights = torch.cat(all_weights)
            
            # Chunk into tokens of fixed size
            token_size = 64  # Arbitrary size for demo
            num_tokens = min(self.max_token_length, (flat_weights.shape[0] + token_size - 1) // token_size)
            
            # Pad or truncate to match token_size * num_tokens
            if flat_weights.shape[0] < token_size * num_tokens:
                flat_weights = torch.cat([flat_weights, torch.zeros(token_size * num_tokens - flat_weights.shape[0])])
            else:
                flat_weights = flat_weights[:token_size * num_tokens]
            
            tokens = flat_weights.reshape(num_tokens, token_size)
            
            # Add layer type and position information
            layer_info = torch.zeros(num_tokens, 3)  # [layer_idx, is_weight, is_bias]
            token_indices = torch.arange(num_tokens).unsqueeze(1)
            
            return torch.cat([tokens, layer_info, token_indices], dim=1)
            
        elif self.tokenization_strategy == "neuron_centric":
            # Neuron-centric tokenization
            tokens = []
            layer_indices = []
            is_weight = []
            is_bias = []
            neuron_indices = []
            
            for layer_idx, (layer_name, weight) in enumerate(weights.items()):
                if "weight" in layer_name:
                    if weight.dim() == 2:  # Linear layer
                        for neuron_idx in range(weight.shape[0]):
                            # For each output neuron, create a token from its incoming weights
                            neuron_weights = weight[neuron_idx]
                            
                            # Find corresponding bias if it exists
                            bias_name = layer_name.replace("weight", "bias")
                            if bias_name in weights:
                                neuron_bias = weights[bias_name][neuron_idx].reshape(1)
                            else:
                                neuron_bias = torch.tensor([0.0])
                            
                            # Combine weights and bias into a token
                            if neuron_weights.shape[0] > 63:
                                # For large neurons, sample or use statistics
                                sampled_indices = torch.randperm(neuron_weights.shape[0])[:63]
                                token_weights = neuron_weights[sampled_indices]
                            else:
                                token_weights = neuron_weights
                                
                            # Combine weights and bias
                            token = torch.cat([token_weights, neuron_bias])
                            
                            # If token is too small, pad it
                            if token.shape[0] < 64:
                                token = torch.cat([token, torch.zeros(64 - token.shape[0])])
                            
                            tokens.append(token)
                            layer_indices.append(layer_idx)
                            is_weight.append(1)
                            is_bias.append(0)
                            neuron_indices.append(neuron_idx)
                            
                    elif weight.dim() == 4:  # Conv layer
                        for filter_idx in range(weight.shape[0]):
                            # For each filter, create a token from its weights
                            filter_weights = weight[filter_idx].flatten()
                            
                            # Find corresponding bias if it exists
                            bias_name = layer_name.replace("weight", "bias")
                            if bias_name in weights:
                                filter_bias = weights[bias_name][filter_idx].reshape(1)
                            else:
                                filter_bias = torch.tensor([0.0])
                            
                            # Combine weights and bias into a token
                            if filter_weights.shape[0] > 63:
                                # For large filters, sample or use statistics
                                sampled_indices = torch.randperm(filter_weights.shape[0])[:63]
                                token_weights = filter_weights[sampled_indices]
                            else:
                                token_weights = filter_weights
                                
                            # Combine weights and bias
                            token = torch.cat([token_weights, filter_bias])
                            
                            # If token is too small, pad it
                            if token.shape[0] < 64:
                                token = torch.cat([token, torch.zeros(64 - token.shape[0])])
                            
                            tokens.append(token)
                            layer_indices.append(layer_idx)
                            is_weight.append(1)
                            is_bias.append(0)
                            neuron_indices.append(filter_idx)
            
            # Convert list of tokens to tensor
            tokens = torch.stack(tokens)
            
            # Limit number of tokens
            if tokens.shape[0] > self.max_token_length:
                indices = torch.randperm(tokens.shape[0])[:self.max_token_length]
                tokens = tokens[indices]
                layer_indices = [layer_indices[i] for i in indices]
                is_weight = [is_weight[i] for i in indices]
                is_bias = [is_bias[i] for i in indices]
                neuron_indices = [neuron_indices[i] for i in indices]
            
            # Add metadata to tokens
            layer_indices_tensor = torch.tensor(layer_indices).unsqueeze(1).float()
            is_weight_tensor = torch.tensor(is_weight).unsqueeze(1).float()
            is_bias_tensor = torch.tensor(is_bias).unsqueeze(1).float()
            neuron_indices_tensor = torch.tensor(neuron_indices).unsqueeze(1).float()
            
            return torch.cat([tokens, layer_indices_tensor, is_weight_tensor, is_bias_tensor, neuron_indices_tensor], dim=1)
            
        elif self.tokenization_strategy == "layer_centric":
            # Layer-centric tokenization
            tokens = []
            layer_indices = []
            is_weight = []
            is_bias = []
            
            for layer_idx, (layer_name, weight) in enumerate(weights.items()):
                if "weight" in layer_name:
                    # Create a token from layer statistics
                    layer_mean = weight.mean().reshape(1)
                    layer_std = weight.std().reshape(1)
                    layer_min = weight.min().reshape(1)
                    layer_max = weight.max().reshape(1)
                    
                    # Sample weights
                    flat_weight = weight.flatten()
                    if flat_weight.shape[0] > 60:
                        sampled_indices = torch.randperm(flat_weight.shape[0])[:60]
                        sampled_weights = flat_weight[sampled_indices]
                    else:
                        sampled_weights = flat_weight
                        
                    # Combine statistics and samples
                    token = torch.cat([layer_mean, layer_std, layer_min, layer_max, sampled_weights])
                    
                    # If token is too small, pad it
                    if token.shape[0] < 64:
                        token = torch.cat([token, torch.zeros(64 - token.shape[0])])
                    
                    tokens.append(token)
                    layer_indices.append(layer_idx)
                    is_weight.append(1)
                    is_bias.append(0)
            
            # Convert list of tokens to tensor
            tokens = torch.stack(tokens)
            
            # Limit number of tokens
            if tokens.shape[0] > self.max_token_length:
                indices = torch.randperm(tokens.shape[0])[:self.max_token_length]
                tokens = tokens[indices]
                layer_indices = [layer_indices[i] for i in indices]
                is_weight = [is_weight[i] for i in indices]
                is_bias = [is_bias[i] for i in indices]
            
            # Add metadata to tokens
            layer_indices_tensor = torch.tensor(layer_indices).unsqueeze(1).float()
            is_weight_tensor = torch.tensor(is_weight).unsqueeze(1).float()
            is_bias_tensor = torch.tensor(is_bias).unsqueeze(1).float()
            
            return torch.cat([tokens, layer_indices_tensor, is_weight_tensor, is_bias_tensor], dim=1)
        
        else:
            raise ValueError(f"Unknown tokenization strategy: {self.tokenization_strategy}")
    
    def __len__(self) -> int:
        """Return the number of models in the dataset."""
        return len(self.model_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Get a model's tokenized weights and properties.
        
        Args:
            idx: Index of the model
            
        Returns:
            Tuple of (tokenized_weights, properties)
        """
        model_id = self.model_ids[idx]
        
        # Load model weights
        weights = self._load_model_weights(model_id)
        
        # Canonicalize weights
        canonicalized_weights = self._canonicalize_weights(weights)
        
        # Tokenize weights
        tokens = self._tokenize_weights(canonicalized_weights)
        
        # Get model properties
        properties = {prop: self.models_metadata[model_id].get(prop, 0.0) 
                      for prop in self.model_properties}
        
        # Convert properties to tensor
        properties_tensor = torch.tensor([properties[prop] for prop in self.model_properties])
        
        # Apply transform if provided
        if self.transform is not None:
            tokens = self.transform(tokens)
        
        return tokens, properties_tensor

def create_dataloaders(
    data_path: str,
    model_properties: List[str],
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    canonicalization_method: Optional[str] = None,
    tokenization_strategy: str = "neuron_centric",
    max_token_length: int = 4096,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for training, validation, and testing.
    
    Args:
        data_path: Path to the directory containing model weights and properties
        model_properties: List of properties to predict
        batch_size: Batch size for the dataloaders
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        canonicalization_method: Method to use for canonicalizing weights
        tokenization_strategy: Strategy for tokenizing weights
        max_token_length: Maximum number of tokens to include
        num_workers: Number of workers for the dataloaders
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create full dataset
    full_dataset = ModelWeightsDataset(
        data_path=data_path,
        model_properties=model_properties,
        canonicalization_method=canonicalization_method,
        tokenization_strategy=tokenization_strategy,
        max_token_length=max_token_length,
    )
    
    # Split dataset
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_split = int(np.floor(train_ratio * dataset_size))
    val_split = int(np.floor((train_ratio + val_ratio) * dataset_size))
    
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    # Create dataset subsets
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_dataloader, val_dataloader, test_dataloader

def generate_synthetic_model_zoo(
    data_path: str,
    num_models: int = 100,
    model_architectures: List[str] = ["resnet18", "resnet34", "resnet50", "vgg16", "mobilenet"],
    properties: List[str] = ["accuracy", "robustness", "generalization_gap"],
    seed: int = 42,
) -> None:
    """
    Generate a synthetic model zoo for demonstration purposes.
    
    Args:
        data_path: Path to save the model zoo
        num_models: Number of models to generate
        model_architectures: List of model architectures to use
        properties: List of properties to generate
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    
    # Generate metadata
    metadata = {}
    for i in tqdm(range(num_models), desc="Generating model zoo"):
        model_id = f"model_{i}"
        architecture = random.choice(model_architectures)
        
        # Generate random properties
        model_props = {
            "accuracy": random.uniform(0.7, 0.95),
            "robustness": random.uniform(0.5, 0.9),
            "generalization_gap": random.uniform(0.02, 0.15),
            "parameters": random.randint(5000000, 25000000),
            "flops": random.randint(500000000, 5000000000),
            "architecture": architecture
        }
        
        metadata[model_id] = model_props
        
        # Generate synthetic weights
        if architecture == "resnet18":
            layer_sizes = [(64, 3, 3, 3), (64, 64, 3, 3), (128, 64, 3, 3), 
                           (128, 128, 3, 3), (256, 128, 3, 3), (256, 256, 3, 3),
                           (512, 256, 3, 3), (512, 512, 3, 3), (1000, 512)]
        elif architecture == "resnet34":
            layer_sizes = [(64, 3, 3, 3), (64, 64, 3, 3), (64, 64, 3, 3), (64, 64, 3, 3),
                           (128, 64, 3, 3), (128, 128, 3, 3), (128, 128, 3, 3), (128, 128, 3, 3),
                           (256, 128, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3),
                           (512, 256, 3, 3), (512, 512, 3, 3), (512, 512, 3, 3), (1000, 512)]
        elif architecture == "resnet50":
            layer_sizes = [(64, 3, 3, 3), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1),
                           (64, 256, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1),
                           (128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1),
                           (128, 512, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1),
                           (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1),
                           (256, 1024, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1),
                           (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1),
                           (512, 2048, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1),
                           (1000, 2048)]
        elif architecture == "vgg16":
            layer_sizes = [(64, 3, 3, 3), (64, 64, 3, 3), (128, 64, 3, 3), (128, 128, 3, 3),
                           (256, 128, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3),
                           (512, 256, 3, 3), (512, 512, 3, 3), (512, 512, 3, 3),
                           (512, 512, 3, 3), (512, 512, 3, 3), (512, 512, 3, 3),
                           (4096, 512 * 7 * 7), (4096, 4096), (1000, 4096)]
        else:  # mobilenet or default
            layer_sizes = [(32, 3, 3, 3), (64, 32, 1, 1), (128, 64, 1, 1), (128, 128, 1, 1),
                           (256, 128, 1, 1), (256, 256, 1, 1), (512, 256, 1, 1),
                           (512, 512, 1, 1), (512, 512, 1, 1), (512, 512, 1, 1),
                           (512, 512, 1, 1), (512, 512, 1, 1), (1024, 512, 1, 1),
                           (1024, 1024, 1, 1), (1000, 1024)]
        
        weights = {}
        biases = {}
        
        # Generate random weights for each layer
        for j, size in enumerate(layer_sizes):
            if len(size) == 2:  # Linear layer
                out_features, in_features = size
                weights[f"layer_{j}.weight"] = torch.randn(out_features, in_features)
                biases[f"layer_{j}.bias"] = torch.randn(out_features)
            else:  # Conv layer
                out_channels, in_channels, kernel_size1, kernel_size2 = size
                weights[f"layer_{j}.weight"] = torch.randn(out_channels, in_channels, kernel_size1, kernel_size2)
                biases[f"layer_{j}.bias"] = torch.randn(out_channels)
        
        # Save weights
        combined_weights = {**weights, **biases}
        torch.save(combined_weights, os.path.join(data_path, f"{model_id}_weights.pt"))
    
    # Save metadata
    with open(os.path.join(data_path, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Generated synthetic model zoo with {num_models} models at {data_path}")

if __name__ == "__main__":
    # Test the dataset
    data_path = "data/raw"
    generate_synthetic_model_zoo(data_path, num_models=10)
    
    dataset = ModelWeightsDataset(
        data_path=data_path,
        model_properties=["accuracy", "robustness", "generalization_gap"],
        canonicalization_method="weight_sort",
        tokenization_strategy="neuron_centric",
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test the first item
    tokens, properties = dataset[0]
    print(f"Tokens shape: {tokens.shape}")
    print(f"Properties: {properties}")
    
    # Test dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=data_path,
        model_properties=["accuracy", "robustness", "generalization_gap"],
        batch_size=2,
        canonicalization_method="weight_sort",
        tokenization_strategy="neuron_centric",
    )
    
    print(f"Train loader size: {len(train_loader)}")
    print(f"Val loader size: {len(val_loader)}")
    print(f"Test loader size: {len(test_loader)}")
    
    # Test a batch
    for batch_tokens, batch_properties in train_loader:
        print(f"Batch tokens shape: {batch_tokens.shape}")
        print(f"Batch properties shape: {batch_properties.shape}")
        break
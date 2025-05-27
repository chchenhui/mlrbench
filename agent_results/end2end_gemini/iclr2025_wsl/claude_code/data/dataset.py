import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import pickle
import random
from tqdm import tqdm
import logging
import torchvision.models as models
from collections import OrderedDict
import math

from data.data_utils import ModelWeightsDataset, create_dataloaders

logger = logging.getLogger(__name__)

def extract_model_weights(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract weights from a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of model weights
    """
    state_dict = model.state_dict()
    return {name: param.clone().detach() for name, param in state_dict.items()}

def generate_synthetic_model_properties(
    model_name: str, 
    num_samples: int = 1,
    use_realistic_correlations: bool = True
) -> List[Dict[str, float]]:
    """
    Generate synthetic properties for a model.
    
    Args:
        model_name: Name of the model
        num_samples: Number of property samples to generate
        use_realistic_correlations: Whether to generate correlated properties that mimic reality
        
    Returns:
        List of dictionaries of properties
    """
    properties_list = []
    
    # Base ranges for properties
    base_ranges = {
        'accuracy': (0.70, 0.95),
        'robustness': (0.50, 0.90),
        'generalization_gap': (0.02, 0.15),
        'parameters': None,  # Will be set based on model
        'flops': None,  # Will be set based on model
    }
    
    # Set model-specific parameter and FLOP counts (rough estimates)
    if 'resnet18' in model_name:
        base_ranges['parameters'] = (11_000_000, 12_000_000)
        base_ranges['flops'] = (1_800_000_000, 2_000_000_000)
    elif 'resnet34' in model_name:
        base_ranges['parameters'] = (21_000_000, 22_000_000)
        base_ranges['flops'] = (3_600_000_000, 3_800_000_000) 
    elif 'resnet50' in model_name:
        base_ranges['parameters'] = (25_000_000, 26_000_000)
        base_ranges['flops'] = (4_000_000_000, 4_200_000_000)
    elif 'resnet101' in model_name:
        base_ranges['parameters'] = (44_000_000, 45_000_000)
        base_ranges['flops'] = (7_500_000_000, 8_000_000_000)
    elif 'resnet152' in model_name:
        base_ranges['parameters'] = (60_000_000, 62_000_000)
        base_ranges['flops'] = (11_000_000_000, 12_000_000_000)
    elif 'vgg' in model_name:
        if '11' in model_name:
            base_ranges['parameters'] = (132_000_000, 133_000_000)
            base_ranges['flops'] = (7_500_000_000, 8_000_000_000)
        elif '13' in model_name:
            base_ranges['parameters'] = (138_000_000, 139_000_000)
            base_ranges['flops'] = (11_000_000_000, 12_000_000_000)
        elif '16' in model_name:
            base_ranges['parameters'] = (138_000_000, 139_000_000)
            base_ranges['flops'] = (15_000_000_000, 16_000_000_000)
        else:  # vgg19
            base_ranges['parameters'] = (143_000_000, 144_000_000)
            base_ranges['flops'] = (19_000_000_000, 20_000_000_000)
    elif 'mobilenet' in model_name:
        base_ranges['parameters'] = (3_500_000, 4_500_000)
        base_ranges['flops'] = (550_000_000, 650_000_000)
    elif 'efficientnet' in model_name:
        if 'b0' in model_name:
            base_ranges['parameters'] = (5_000_000, 6_000_000)
            base_ranges['flops'] = (400_000_000, 500_000_000)
        elif 'b1' in model_name:
            base_ranges['parameters'] = (7_500_000, 8_000_000)
            base_ranges['flops'] = (700_000_000, 800_000_000)
        else:  # b2+
            base_ranges['parameters'] = (9_000_000, 12_000_000)
            base_ranges['flops'] = (1_000_000_000, 1_500_000_000)
    elif 'densenet' in model_name:
        if '121' in model_name:
            base_ranges['parameters'] = (7_900_000, 8_100_000)
            base_ranges['flops'] = (2_800_000_000, 3_000_000_000)
        elif '169' in model_name:
            base_ranges['parameters'] = (14_000_000, 14_500_000)
            base_ranges['flops'] = (3_400_000_000, 3_600_000_000)
        else:  # 201, 264
            base_ranges['parameters'] = (20_000_000, 30_000_000)
            base_ranges['flops'] = (4_000_000_000, 6_000_000_000)
    else:  # Default for unknown models
        base_ranges['parameters'] = (5_000_000, 50_000_000)
        base_ranges['flops'] = (1_000_000_000, 10_000_000_000)
    
    for _ in range(num_samples):
        if use_realistic_correlations:
            # Generate a "model quality" latent variable that affects multiple properties
            model_quality = random.uniform(0, 1)
            
            # Higher model quality generally means higher accuracy, higher robustness,
            # lower generalization gap
            accuracy = base_ranges['accuracy'][0] + model_quality * (base_ranges['accuracy'][1] - base_ranges['accuracy'][0])
            # Add some noise to avoid perfect correlation
            accuracy = min(max(accuracy + random.uniform(-0.05, 0.05), base_ranges['accuracy'][0]), base_ranges['accuracy'][1])
            
            # Robustness is correlated with accuracy but with more noise
            robustness = base_ranges['robustness'][0] + model_quality * (base_ranges['robustness'][1] - base_ranges['robustness'][0])
            robustness = min(max(robustness + random.uniform(-0.1, 0.1), base_ranges['robustness'][0]), base_ranges['robustness'][1])
            
            # Generalization gap is negatively correlated with model quality (better models have smaller gap)
            gen_gap = base_ranges['generalization_gap'][1] - model_quality * (base_ranges['generalization_gap'][1] - base_ranges['generalization_gap'][0])
            gen_gap = min(max(gen_gap + random.uniform(-0.02, 0.02), base_ranges['generalization_gap'][0]), base_ranges['generalization_gap'][1])
            
            # Parameter count and FLOPs are model architecture specific, with slight variations
            params = random.uniform(base_ranges['parameters'][0], base_ranges['parameters'][1])
            flops = random.uniform(base_ranges['flops'][0], base_ranges['flops'][1])
        else:
            # Generate completely uncorrelated random properties
            accuracy = random.uniform(base_ranges['accuracy'][0], base_ranges['accuracy'][1])
            robustness = random.uniform(base_ranges['robustness'][0], base_ranges['robustness'][1])
            gen_gap = random.uniform(base_ranges['generalization_gap'][0], base_ranges['generalization_gap'][1])
            params = random.uniform(base_ranges['parameters'][0], base_ranges['parameters'][1])
            flops = random.uniform(base_ranges['flops'][0], base_ranges['flops'][1])
        
        properties = {
            'accuracy': accuracy,
            'robustness': robustness,
            'generalization_gap': gen_gap,
            'parameters': params,
            'flops': flops,
            'architecture': model_name
        }
        
        properties_list.append(properties)
    
    return properties_list

def create_model_zoo(
    output_dir: str,
    num_models_per_architecture: int = 10,
    architectures: Optional[List[str]] = None,
    generate_variations: bool = True,
    random_seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Create a model zoo with different architectures and variations.
    
    Args:
        output_dir: Directory to save model weights and metadata
        num_models_per_architecture: Number of models to generate for each architecture
        architectures: List of model architectures to include
        generate_variations: Whether to generate variations of each model
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary of model metadata
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if architectures is None:
        # Default set of architectures
        architectures = [
            'resnet18', 'resnet34', 'resnet50', 
            'vgg11', 'vgg16',
            'mobilenet_v2', 
            'densenet121'
        ]
    
    # Dictionary to store metadata for all models
    metadata = {}
    model_id_counter = 0
    
    logger.info(f"Creating model zoo with {len(architectures)} architectures in {output_dir}")
    
    # Create models for each architecture
    for arch in tqdm(architectures, desc="Generating model zoo"):
        for i in range(num_models_per_architecture):
            try:
                # Create model
                if arch == 'resnet18':
                    model = models.resnet18(pretrained=False)
                elif arch == 'resnet34':
                    model = models.resnet34(pretrained=False)
                elif arch == 'resnet50':
                    model = models.resnet50(pretrained=False)
                elif arch == 'vgg11':
                    model = models.vgg11(pretrained=False)
                elif arch == 'vgg16':
                    model = models.vgg16(pretrained=False)
                elif arch == 'mobilenet_v2':
                    model = models.mobilenet_v2(pretrained=False)
                elif arch == 'densenet121':
                    model = models.densenet121(pretrained=False)
                else:
                    logger.warning(f"Unknown architecture: {arch}, skipping")
                    continue
                
                # Initialize model with different random weights
                model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
                
                if generate_variations:
                    # Apply random scaling to weights
                    scale_factor = random.uniform(0.8, 1.2)
                    for param in model.parameters():
                        param.data *= scale_factor
                    
                    # Optionally sparsify some weights
                    if random.random() < 0.5:
                        sparsity_factor = random.uniform(0.1, 0.3)
                        for param in model.parameters():
                            mask = torch.rand_like(param.data) < sparsity_factor
                            param.data[mask] = 0.0
                
                # Extract weights
                weights = extract_model_weights(model)
                
                # Generate model ID
                model_id = f"model_{model_id_counter}"
                model_id_counter += 1
                
                # Generate properties
                properties = generate_synthetic_model_properties(arch)[0]
                properties['architecture'] = arch
                
                # Add to metadata
                metadata[model_id] = properties
                
                # Save weights
                torch.save(weights, os.path.join(output_dir, f"{model_id}_weights.pt"))
                
            except Exception as e:
                logger.error(f"Error generating model for {arch}: {e}")
    
    # Save metadata
    with open(os.path.join(output_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"Created model zoo with {len(metadata)} models")
    
    return metadata

def load_model_zoo(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load metadata for a model zoo.
    
    Args:
        data_dir: Directory containing model weights and metadata
        
    Returns:
        Dictionary of model metadata
    """
    metadata_path = os.path.join(data_dir, "metadata.pkl")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return metadata

def prepare_datasets(
    data_dir: str,
    model_properties: List[str],
    canonicalization_method: Optional[str] = None,
    tokenization_strategy: str = "neuron_centric",
    max_token_length: int = 4096,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_by_architecture: bool = False,
    max_models: Optional[int] = None,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Prepare train, validation, and test datasets.
    
    Args:
        data_dir: Directory containing model weights and metadata
        model_properties: List of properties to predict
        canonicalization_method: Method to use for canonicalizing weights
        tokenization_strategy: Strategy for tokenizing weights
        max_token_length: Maximum number of tokens to include
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        split_by_architecture: Whether to split by architecture
        max_models: Maximum number of models to include
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load metadata
    metadata = load_model_zoo(data_dir)
    model_ids = list(metadata.keys())
    
    if max_models is not None and max_models < len(model_ids):
        model_ids = random.sample(model_ids, max_models)
    
    if split_by_architecture:
        # Group models by architecture
        architecture_groups = {}
        for model_id in model_ids:
            arch = metadata[model_id]['architecture']
            if arch not in architecture_groups:
                architecture_groups[arch] = []
            architecture_groups[arch].append(model_id)
        
        # Reserve some architectures for validation and testing
        architectures = list(architecture_groups.keys())
        random.shuffle(architectures)
        
        num_train_archs = math.ceil(train_ratio * len(architectures))
        num_val_archs = math.ceil(val_ratio * len(architectures))
        
        train_archs = architectures[:num_train_archs]
        val_archs = architectures[num_train_archs:num_train_archs+num_val_archs]
        test_archs = architectures[num_train_archs+num_val_archs:]
        
        # Create dataset splits
        train_ids = [model_id for arch in train_archs for model_id in architecture_groups[arch]]
        val_ids = [model_id for arch in val_archs for model_id in architecture_groups[arch]]
        test_ids = [model_id for arch in test_archs for model_id in architecture_groups[arch]]
        
        logger.info(f"Split by architecture: {len(train_archs)} train, {len(val_archs)} val, {len(test_archs)} test")
        logger.info(f"Model counts: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    else:
        # Randomly split models
        random.shuffle(model_ids)
        
        train_split = int(train_ratio * len(model_ids))
        val_split = int((train_ratio + val_ratio) * len(model_ids))
        
        train_ids = model_ids[:train_split]
        val_ids = model_ids[train_split:val_split]
        test_ids = model_ids[val_split:]
        
        logger.info(f"Random split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    
    # Create datasets
    # Custom ModelWeightsDataset with specific model IDs
    class SubsetModelWeightsDataset(ModelWeightsDataset):
        def __init__(self, data_path, model_properties, model_ids, **kwargs):
            super().__init__(data_path, model_properties, **kwargs)
            self.model_ids = model_ids
    
    train_dataset = SubsetModelWeightsDataset(
        data_path=data_dir,
        model_properties=model_properties,
        model_ids=train_ids,
        canonicalization_method=canonicalization_method,
        tokenization_strategy=tokenization_strategy,
        max_token_length=max_token_length,
    )
    
    val_dataset = SubsetModelWeightsDataset(
        data_path=data_dir,
        model_properties=model_properties,
        model_ids=val_ids,
        canonicalization_method=canonicalization_method,
        tokenization_strategy=tokenization_strategy,
        max_token_length=max_token_length,
    )
    
    test_dataset = SubsetModelWeightsDataset(
        data_path=data_dir,
        model_properties=model_properties,
        model_ids=test_ids,
        canonicalization_method=canonicalization_method,
        tokenization_strategy=tokenization_strategy,
        max_token_length=max_token_length,
    )
    
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test model zoo creation
    data_dir = "data/raw"
    os.makedirs(data_dir, exist_ok=True)
    
    architectures = ['resnet18', 'vgg11', 'mobilenet_v2']
    metadata = create_model_zoo(
        output_dir=data_dir,
        num_models_per_architecture=2,
        architectures=architectures,
        generate_variations=True,
    )
    
    # Test dataset creation
    model_properties = ['accuracy', 'robustness', 'generalization_gap']
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        data_dir=data_dir,
        model_properties=model_properties,
        canonicalization_method="weight_sort",
        tokenization_strategy="neuron_centric",
        split_by_architecture=True,
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=2,
    )
    
    # Test loading a batch
    for tokens, properties in train_loader:
        print(f"Tokens shape: {tokens.shape}")
        print(f"Properties shape: {properties.shape}")
        break
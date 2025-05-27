"""
Utility functions for data handling and transformations.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set all seeds to make results reproducible."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device():
    """Get the device to use for training."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_standard_transforms():
    """Standard image transformations for training and testing."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    return train_transform, test_transform

def load_dataset(dataset_name, data_dir='./data'):
    """Load a dataset by name."""
    if dataset_name == 'cifar10':
        train_transform, test_transform = get_standard_transforms()
        
        train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
        
        return train_dataset, test_dataset
    
    elif dataset_name == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = MNIST(root=data_dir, train=False, download=True, transform=test_transform)
        
        return train_dataset, test_dataset
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

def get_dataloaders(train_dataset, test_dataset, batch_size=128, num_workers=4):
    """Create dataloaders from datasets."""
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader

def create_subset(dataset, indices, transform=None):
    """Create a subset of a dataset with optional transform."""
    subset = Subset(dataset, indices)
    
    if transform:
        subset.dataset.transform = transform
    
    return subset

class TransformedDataset(Dataset):
    """Dataset that applies a transformation function to another dataset."""
    
    def __init__(self, dataset, transform_func):
        self.dataset = dataset
        self.transform_func = transform_func
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        transformed_image = self.transform_func(image)
        return transformed_image, label

def apply_transformation_sequence(image, transformations):
    """Apply a sequence of transformations to an image."""
    if isinstance(image, torch.Tensor):
        # Convert tensor to PIL Image
        image = transforms.ToPILImage()(image)
    
    for transform in transformations:
        image = transform(image)
    
    # Ensure the output is a tensor
    if not isinstance(image, torch.Tensor):
        image = transforms.ToTensor()(image)
    
    return image

def create_adversarial_dataset(dataset, transformation_config):
    """
    Create a dataset with adversarial transformations.
    
    Args:
        dataset: The source dataset
        transformation_config: A list of transformation configurations
    
    Returns:
        A transformed dataset
    """
    def transform_func(image):
        return apply_transformation_sequence(image, transformation_config)
    
    return TransformedDataset(dataset, transform_func)
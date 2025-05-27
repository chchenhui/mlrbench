#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading and processing utilities for CIMRL experiments.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as transforms
from PIL import Image
import json
from pathlib import Path
from transformers import AutoTokenizer

from .synthetic_dataset import create_synthetic_multimodal_dataset
from .waterbirds import WaterbirdsDataset

class MultiModalDataset(Dataset):
    """
    Multi-modal dataset with support for spurious correlations.
    """
    
    def __init__(
        self, 
        data_config, 
        split='train', 
        transform=None, 
        tokenizer=None, 
        max_text_length=128
    ):
        """
        Initialize the multi-modal dataset.
        
        Args:
            data_config: Dictionary containing dataset configuration parameters
            split: Dataset split ('train', 'val', or 'test')
            transform: Image transformation pipeline
            tokenizer: Text tokenizer
            max_text_length: Maximum text length for tokenization
        """
        self.data_config = data_config
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        
        # Load dataset based on config
        dataset_name = data_config['dataset']
        
        if dataset_name == 'synthetic':
            # Create synthetic dataset with controlled spurious correlations
            self.data = create_synthetic_multimodal_dataset(
                data_config=data_config,
                split=split,
                num_samples=data_config.get(f'{split}_samples', 1000)
            )
        elif dataset_name == 'waterbirds':
            # Load Waterbirds dataset (birds with spurious background)
            self.data = WaterbirdsDataset(
                data_dir=data_config['data_dir'],
                split=split,
                transform=transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Set default transform if none provided
        if self.transform is None:
            self.transform = get_default_transform(split)
        
        # Set default tokenizer if none provided and we have text
        if self.tokenizer is None and 'text' in data_config['modalities']:
            tokenizer_name = data_config.get('tokenizer', 'bert-base-uncased')
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data['labels'])
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            sample: Dictionary containing the sample data
        """
        sample = {}
        
        # Get label
        sample['labels'] = torch.tensor(self.data['labels'][idx], dtype=torch.long)
        
        # Get group label if available
        if 'group_labels' in self.data:
            sample['group_labels'] = torch.tensor(self.data['group_labels'][idx], dtype=torch.long)
        
        # Get sample index
        sample['indices'] = idx
        
        # Get vision data if available
        if 'vision' in self.data_config['modalities']:
            if isinstance(self.data['vision'][idx], str):
                # Load image from path
                image = Image.open(self.data['vision'][idx]).convert('RGB')
            elif isinstance(self.data['vision'][idx], np.ndarray):
                # Convert numpy array to PIL Image
                image = Image.fromarray(self.data['vision'][idx].astype(np.uint8))
            else:
                image = self.data['vision'][idx]
            
            # Apply transform
            if self.transform:
                image = self.transform(image)
            
            sample['vision'] = image
        
        # Get text data if available
        if 'text' in self.data_config['modalities']:
            text = self.data['text'][idx]
            
            if self.tokenizer:
                # Tokenize text
                text_encoded = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_text_length,
                    return_tensors='pt'
                )
                
                # Remove batch dimension
                for key in text_encoded:
                    if isinstance(text_encoded[key], torch.Tensor):
                        text_encoded[key] = text_encoded[key].squeeze(0)
                
                sample['text'] = text_encoded
            else:
                sample['text'] = text
        
        # Get perturbed samples if available (for contrastive invariance)
        if 'perturbed' in self.data and self.split == 'train':
            sample['perturbed'] = {}
            
            if 'vision' in self.data_config['modalities'] and 'vision' in self.data['perturbed']:
                perturbed_image = self.data['perturbed']['vision'][idx]
                
                if isinstance(perturbed_image, str):
                    perturbed_image = Image.open(perturbed_image).convert('RGB')
                elif isinstance(perturbed_image, np.ndarray):
                    perturbed_image = Image.fromarray(perturbed_image.astype(np.uint8))
                
                if self.transform:
                    perturbed_image = self.transform(perturbed_image)
                
                sample['perturbed']['vision'] = perturbed_image
            
            if 'text' in self.data_config['modalities'] and 'text' in self.data['perturbed']:
                perturbed_text = self.data['perturbed']['text'][idx]
                
                if self.tokenizer:
                    perturbed_text_encoded = self.tokenizer(
                        perturbed_text,
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_text_length,
                        return_tensors='pt'
                    )
                    
                    # Remove batch dimension
                    for key in perturbed_text_encoded:
                        if isinstance(perturbed_text_encoded[key], torch.Tensor):
                            perturbed_text_encoded[key] = perturbed_text_encoded[key].squeeze(0)
                    
                    sample['perturbed']['text'] = perturbed_text_encoded
                else:
                    sample['perturbed']['text'] = perturbed_text
        
        # Get counterfactual samples if available (for intervention-based fine-tuning)
        if 'counterfactual' in self.data and self.split == 'train':
            sample['counterfactual'] = {}
            
            if 'vision' in self.data_config['modalities'] and 'vision' in self.data['counterfactual']:
                cf_image = self.data['counterfactual']['vision'][idx]
                
                if isinstance(cf_image, str):
                    cf_image = Image.open(cf_image).convert('RGB')
                elif isinstance(cf_image, np.ndarray):
                    cf_image = Image.fromarray(cf_image.astype(np.uint8))
                
                if self.transform:
                    cf_image = self.transform(cf_image)
                
                sample['counterfactual']['vision'] = cf_image
            
            if 'text' in self.data_config['modalities'] and 'text' in self.data['counterfactual']:
                cf_text = self.data['counterfactual']['text'][idx]
                
                if self.tokenizer:
                    cf_text_encoded = self.tokenizer(
                        cf_text,
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_text_length,
                        return_tensors='pt'
                    )
                    
                    # Remove batch dimension
                    for key in cf_text_encoded:
                        if isinstance(cf_text_encoded[key], torch.Tensor):
                            cf_text_encoded[key] = cf_text_encoded[key].squeeze(0)
                    
                    sample['counterfactual']['text'] = cf_text_encoded
                else:
                    sample['counterfactual']['text'] = cf_text
            
            # Copy labels to counterfactual
            sample['counterfactual']['labels'] = sample['labels']
        
        return sample


def get_default_transform(split):
    """
    Get default image transformations based on the dataset split.
    
    Args:
        split: Dataset split ('train', 'val', or 'test')
        
    Returns:
        transform: Image transformation pipeline
    """
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_dataloaders(data_config, batch_size=32, num_workers=4):
    """
    Get data loaders for all splits.
    
    Args:
        data_config: Dictionary containing dataset configuration parameters
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
        ood_test_loader: DataLoader for out-of-distribution test set
    """
    # Get tokenizer if needed
    tokenizer = None
    if 'text' in data_config['modalities']:
        tokenizer_name = data_config.get('tokenizer', 'bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Create datasets for each split
    train_dataset = MultiModalDataset(
        data_config=data_config,
        split='train',
        transform=get_default_transform('train'),
        tokenizer=tokenizer
    )
    
    val_dataset = MultiModalDataset(
        data_config=data_config,
        split='val',
        transform=get_default_transform('val'),
        tokenizer=tokenizer
    )
    
    test_dataset = MultiModalDataset(
        data_config=data_config,
        split='test',
        transform=get_default_transform('test'),
        tokenizer=tokenizer
    )
    
    ood_test_dataset = MultiModalDataset(
        data_config=data_config,
        split='ood_test',
        transform=get_default_transform('test'),
        tokenizer=tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    ood_test_loader = DataLoader(
        ood_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, ood_test_loader
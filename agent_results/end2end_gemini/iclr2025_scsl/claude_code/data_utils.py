"""
Data utilities for loading and preprocessing datasets with spurious correlations.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
from wilds.common.grouper import CombinatorialGrouper

logger = logging.getLogger("LASS.data")

# Constants for Waterbirds dataset
WATERBIRDS_CLASSES = ['landbird', 'waterbird']
WATERBIRDS_BACKGROUNDS = ['land', 'water']

class SpuriousCorrelationDataset:
    """Base class for datasets with spurious correlations."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory of the dataset.
            split: Data split ('train', 'val', or 'test').
            transform: Transforms to be applied to the images.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
    def get_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
        """
        Get DataLoader for the dataset.
        
        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle the data.
            num_workers: Number of worker processes for data loading.
            
        Returns:
            loader: DataLoader object.
        """
        raise NotImplementedError
    
    def get_class_names(self) -> List[str]:
        """
        Get list of class names.
        
        Returns:
            class_names: List of class names.
        """
        raise NotImplementedError
    
    def get_group_names(self) -> List[str]:
        """
        Get list of group names.
        
        Returns:
            group_names: List of group names.
        """
        raise NotImplementedError
    
    def get_group_counts(self) -> Dict[int, int]:
        """
        Get counts for each group.
        
        Returns:
            group_counts: Dictionary mapping group IDs to counts.
        """
        raise NotImplementedError
    
    def get_splits(self) -> Dict[str, Dataset]:
        """
        Get dataset splits.
        
        Returns:
            splits: Dictionary mapping split names to Dataset objects.
        """
        raise NotImplementedError

class WaterbirdsLoader(SpuriousCorrelationDataset):
    """
    Loader for the Waterbirds dataset, which contains spurious correlations
    between bird type (landbird/waterbird) and background (land/water).
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None,
                download: bool = True, metadata_path: Optional[str] = None):
        """
        Initialize the Waterbirds dataset.
        
        Args:
            root_dir: Root directory of the dataset.
            split: Data split ('train', 'val', or 'test').
            transform: Transforms to be applied to the images.
            download: Whether to download the dataset if not found.
            metadata_path: Path to the metadata file.
        """
        super().__init__(root_dir, split, transform)
        
        # Default transforms if none provided
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        
        # Load dataset
        try:
            self.dataset = WaterbirdsDataset(root_dir=root_dir, download=download)
            logger.info(f"Loaded Waterbirds dataset with {len(self.dataset)} samples")
        except Exception as e:
            if download:
                logger.error(f"Failed to download or load Waterbirds dataset: {e}")
            else:
                logger.error(f"Failed to load Waterbirds dataset: {e}")
                logger.info("Consider setting download=True to download the dataset")
            raise
        
        # Create grouper (for spurious attribute combinations)
        self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=['y', 'place'])
        
        # Extract split indices
        if split == 'train':
            self.indices = self.dataset.split_array == 0
        elif split == 'val':
            self.indices = self.dataset.split_array == 1
        elif split == 'test':
            self.indices = self.dataset.split_array == 2
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self.indices = np.where(self.indices)[0]
        
        # Extract labels, groups, and other metadata
        self.y_array = self.dataset.y_array[self.indices]  # Bird type: 0=landbird, 1=waterbird
        self.place_array = self.dataset.metadata_array[self.indices, 0]  # Background: 0=land, 1=water
        self.group_array = self.grouper.metadata_to_group(
            torch.stack([
                torch.LongTensor(self.y_array),
                torch.LongTensor(self.place_array)
            ], dim=1)
        ).numpy()
        
        # Calculate group statistics
        self.group_counts = {int(g): (self.group_array == g).sum() for g in np.unique(self.group_array)}
        self.class_counts = {int(y): (self.y_array == y).sum() for y in np.unique(self.y_array)}
        
        logger.info(f"Waterbirds {split} split:")
        logger.info(f"  Total samples: {len(self.indices)}")
        logger.info(f"  Class counts: {self.class_counts}")
        logger.info(f"  Group counts: {self.group_counts}")
        
    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get sample by index.
        
        Args:
            idx: Sample index.
            
        Returns:
            img: Image tensor.
            label: Class label.
            group: Group label.
        """
        dataset_idx = self.indices[idx]
        img = self.dataset.get_input(dataset_idx)
        
        if self.transform:
            img = self.transform(img)
        
        label = int(self.y_array[idx])
        group = int(self.group_array[idx])
        
        return img, label, group
    
    def get_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
        """
        Get DataLoader for the dataset.
        
        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle the data.
            num_workers: Number of worker processes for data loading.
            
        Returns:
            loader: DataLoader object.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return WATERBIRDS_CLASSES
    
    def get_group_names(self) -> List[str]:
        """Get list of group names (combinations of bird type and background)."""
        return [
            'landbird_land',   # Group 0: landbird on land (correlated)
            'landbird_water',  # Group 1: landbird on water (anti-correlated)
            'waterbird_land',  # Group 2: waterbird on land (anti-correlated)
            'waterbird_water'  # Group 3: waterbird on water (correlated)
        ]
    
    def get_group_counts(self) -> Dict[int, int]:
        """Get counts for each group."""
        return self.group_counts
    
    def get_splits(self) -> Dict[str, 'WaterbirdsLoader']:
        """
        Get all dataset splits.
        
        Returns:
            splits: Dictionary mapping split names to Dataset objects.
        """
        splits = {}
        for split_name in ['train', 'val', 'test']:
            splits[split_name] = WaterbirdsLoader(
                root_dir=self.root_dir,
                split=split_name,
                transform=self.transform if split_name == self.split else None
            )
        return splits

class CelebALoader(SpuriousCorrelationDataset):
    """
    Loader for the CelebA dataset, where we predict "Smiling" with spurious
    correlation to "Blond_Hair".
    
    Note: Implementation placeholder - would need to expand this for full project.
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform)
        logger.warning("CelebA dataset loader not fully implemented yet")

class CivilCommentsLoader(SpuriousCorrelationDataset):
    """
    Loader for the CivilComments dataset, where toxicity is spuriously correlated
    with demographic identity mentions.
    
    Note: Implementation placeholder - would need to expand this for full project.
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform)
        logger.warning("CivilComments dataset loader not fully implemented yet")

def get_dataset_loader(dataset_name: str, root_dir: str, split: str = 'train', 
                     transform=None, **kwargs) -> SpuriousCorrelationDataset:
    """
    Factory function to get the appropriate dataset loader.
    
    Args:
        dataset_name: Name of the dataset.
        root_dir: Root directory of the dataset.
        split: Data split.
        transform: Transforms to be applied to the data.
        **kwargs: Additional arguments specific to each dataset.
        
    Returns:
        dataset: Dataset loader object.
    """
    if dataset_name.lower() == 'waterbirds':
        return WaterbirdsLoader(root_dir, split, transform, **kwargs)
    elif dataset_name.lower() == 'celeba':
        return CelebALoader(root_dir, split, transform, **kwargs)
    elif dataset_name.lower() == 'civilcomments':
        return CivilCommentsLoader(root_dir, split, transform, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def create_synthetic_data(dataset: SpuriousCorrelationDataset, 
                       target_group: int, num_samples: int,
                       class_id: Optional[int] = None) -> List[int]:
    """
    Create synthetic indices to simulate balanced group distribution.
    Used for re-weighting, not for actual data augmentation.
    
    Args:
        dataset: The original dataset.
        target_group: Group to augment.
        num_samples: Number of synthetic samples to create.
        class_id: If specified, only augment samples of this class.
        
    Returns:
        indices: List of sampled indices from the original dataset.
    """
    indices = []
    
    # Filter by group and optionally by class
    for i in range(len(dataset)):
        _, y, g = dataset[i]
        if g == target_group and (class_id is None or y == class_id):
            indices.append(i)
    
    # Sample with replacement if we need more than available
    if num_samples > len(indices):
        synthetic_indices = np.random.choice(indices, size=num_samples, replace=True)
    else:
        synthetic_indices = np.random.choice(indices, size=num_samples, replace=False)
    
    return synthetic_indices.tolist()

def balance_groups(dataset: SpuriousCorrelationDataset, strategy: str = 'upsample',
                  target_count: Optional[int] = None) -> DataLoader:
    """
    Create a balanced dataset with respect to groups.
    
    Args:
        dataset: The original dataset.
        strategy: Balancing strategy ('upsample', 'downsample', or 'reweight').
        target_count: Target count per group (if None, use max or min count).
        
    Returns:
        balanced_loader: DataLoader with balanced group distribution.
    """
    group_counts = dataset.get_group_counts()
    
    if strategy == 'upsample':
        # Upsample minority groups to match the size of the largest group
        max_count = max(group_counts.values()) if target_count is None else target_count
        
        # Collect indices by group
        group_indices = {g: [] for g in group_counts.keys()}
        for i in range(len(dataset)):
            _, _, g = dataset[i]
            group_indices[g].append(i)
        
        # Create balanced dataset by upsampling
        balanced_indices = []
        for g, indices in group_indices.items():
            if len(indices) < max_count:
                # Upsample with replacement
                balanced_indices.extend(np.random.choice(indices, max_count - len(indices), replace=True))
            balanced_indices.extend(indices)
        
        # Create subset dataset
        balanced_dataset = Subset(dataset, balanced_indices)
        
    elif strategy == 'downsample':
        # Downsample majority groups to match the size of the smallest group
        min_count = min(group_counts.values()) if target_count is None else target_count
        
        # Collect indices by group
        group_indices = {g: [] for g in group_counts.keys()}
        for i in range(len(dataset)):
            _, _, g = dataset[i]
            group_indices[g].append(i)
        
        # Create balanced dataset by downsampling
        balanced_indices = []
        for g, indices in group_indices.items():
            if len(indices) > min_count:
                # Downsample without replacement
                balanced_indices.extend(np.random.choice(indices, min_count, replace=False))
            else:
                balanced_indices.extend(indices)
        
        # Create subset dataset
        balanced_dataset = Subset(dataset, balanced_indices)
        
    elif strategy == 'reweight':
        # Return the original dataset with weights for balanced sampling
        weights = torch.zeros(len(dataset))
        
        for i in range(len(dataset)):
            _, _, g = dataset[i]
            # Weight inversely proportional to group count
            weights[i] = 1.0 / group_counts[g]
        
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        # Create weighted sampler
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights, len(weights))
        
        # Create dataloader with the sampler
        return DataLoader(
            dataset,
            batch_size=32,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    
    else:
        raise ValueError(f"Unsupported balancing strategy: {strategy}")
    
    # Create and return the dataloader
    return DataLoader(
        balanced_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

def get_reweighted_loader(dataset: SpuriousCorrelationDataset, 
                        group_weights: Dict[int, float],
                        batch_size: int = 32) -> DataLoader:
    """
    Create a DataLoader with sample weights based on group membership.
    
    Args:
        dataset: The original dataset.
        group_weights: Dictionary mapping group IDs to weights.
        batch_size: Batch size for the loader.
        
    Returns:
        reweighted_loader: DataLoader with weighted sampling.
    """
    weights = torch.zeros(len(dataset))
    
    # Assign weights based on group membership
    for i in range(len(dataset)):
        _, _, g = dataset[i]
        weights[i] = group_weights.get(g, 1.0)
    
    # Create weighted sampler
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, len(weights))
    
    # Create and return the dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
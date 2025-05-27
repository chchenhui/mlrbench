import os
import json
import torch
import numpy as np
import random
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from datasets import load_dataset
import logging

logger = logging.getLogger("influence_space")

# Transform functions for images
def get_transform(is_train: bool = True, image_size: int = 224) -> transforms.Compose:
    """
    Get image transformation pipeline.
    
    Args:
        is_train: Whether to use training or evaluation transformations
        image_size: Size to resize images to
        
    Returns:
        Composition of image transforms
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

class COCODataset(Dataset):
    """
    MS COCO dataset for image-caption pairs.
    """
    def __init__(
        self, 
        split: str = "train", 
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize COCO dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            transform: Image transformation to apply
            max_samples: Maximum number of samples to use (for debugging)
            cache_dir: Directory to cache the dataset
        """
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        
        # Map split names to HuggingFace datasets split names
        split_map = {
            "train": "train",
            "val": "validation",
            "test": "test",
        }
        
        # Load dataset from HuggingFace datasets
        logger.info(f"Loading COCO {split} dataset...")
        try:
            self.dataset = load_dataset("conceptual_captions", split=split_map.get(split, split), cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"Failed to load conceptual_captions: {str(e)}")
            # Fallback to COCO dataset if conceptual_captions is not available
            try:
                self.dataset = load_dataset("Multimodal-Fatima/COCO_2014", split=split_map.get(split, split), cache_dir=cache_dir)
            except Exception as e2:
                logger.warning(f"Failed to load COCO dataset: {str(e2)}")
                # Create a dummy dataset for testing
                logger.warning("Creating dummy dataset for testing")
                import torch
                import random
                
                # Create dummy dataset
                class DummyDataset:
                    def __init__(self, size=1000):
                        self.size = size
                        self.data = [{"image_url": f"image_{i}.jpg", "caption": f"Caption for image {i}"} for i in range(size)]
                    
                    def __len__(self):
                        return self.size
                    
                    def __getitem__(self, idx):
                        return self.data[idx]
                    
                    def select(self, indices):
                        selected_data = [self.data[i] for i in indices]
                        result = DummyDataset(len(selected_data))
                        result.data = selected_data
                        return result
                
                self.dataset = DummyDataset(1000)
        
        # Limit number of samples if max_samples is specified
        if max_samples is not None and max_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(max_samples))
            
        logger.info(f"Loaded {len(self.dataset)} samples from {split} dataset")
        
        # Create dummy demographic attributes for fairness evaluation
        # In a real scenario, these would be inferred from metadata or attribute classifiers
        np.random.seed(42)  # For reproducibility
        self.demographics = {
            "gender": np.random.choice(["male", "female", "other"], size=len(self.dataset)),
            "ethnicity": np.random.choice(["group_a", "group_b", "group_c", "group_d"], size=len(self.dataset))
        }
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing image, caption, and metadata
        """
        item = self.dataset[idx]
        
        # Load and transform image
        try:
            # Check if image_url is an actual file path or a URL
            if os.path.exists(item["image_url"]):
                image = Image.open(item["image_url"]).convert("RGB")
            else:
                # For dummy dataset or non-existing files, create a random image
                # This is just for testing purposes
                random.seed(idx)  # For reproducibility
                image = Image.new('RGB', (224, 224), color=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                ))
            
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            # If there's an error loading the image, return a placeholder
            logger.warning(f"Error loading image at index {idx}: {str(e)}")
            image = torch.zeros(3, 224, 224)
        
        # Get caption
        caption = item["caption"]
        
        # Get demographic attributes
        gender = self.demographics["gender"][idx]
        ethnicity = self.demographics["ethnicity"][idx]
        
        return {
            "image": image,
            "caption": caption,
            "idx": idx,
            "gender": gender,
            "ethnicity": ethnicity
        }

def create_weighted_sampler(weights: List[float], dataset_size: int) -> WeightedRandomSampler:
    """
    Create a weighted sampler for data loading.
    
    Args:
        weights: List of weights for each sample
        dataset_size: Size of the dataset
        
    Returns:
        Weighted random sampler
    """
    # Ensure weights are provided for all samples
    assert len(weights) == dataset_size, "Weights must be provided for all samples"
    
    # Convert weights to tensor
    weights_tensor = torch.DoubleTensor(weights)
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=dataset_size,
        replacement=True
    )
    
    return sampler

def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    weights: Optional[List[float]] = None,
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get dataloaders for training and validation.
    
    Args:
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for data loading
        image_size: Size to resize images to
        max_train_samples: Maximum number of training samples
        max_val_samples: Maximum number of validation samples
        weights: Optional weights for training samples
        cache_dir: Directory to cache the dataset
        
    Returns:
        Training and validation dataloaders
    """
    # Get transforms
    train_transform = get_transform(is_train=True, image_size=image_size)
    val_transform = get_transform(is_train=False, image_size=image_size)
    
    # Create datasets
    train_dataset = COCODataset(
        split="train",
        transform=train_transform,
        max_samples=max_train_samples,
        cache_dir=cache_dir
    )
    
    val_dataset = COCODataset(
        split="val",
        transform=val_transform,
        max_samples=max_val_samples,
        cache_dir=cache_dir
    )
    
    # Create sampler if weights are provided
    sampler = None
    if weights is not None:
        sampler = create_weighted_sampler(weights, len(train_dataset))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
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
    
    return train_loader, val_loader

def get_dataloader_from_indices(
    dataset: Dataset,
    indices: List[int],
    weights: Optional[List[float]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a dataloader from a subset of a dataset.
    
    Args:
        dataset: The full dataset
        indices: Indices of samples to include
        weights: Optional weights for the samples
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader for the subset
    """
    # Create subset
    subset = Subset(dataset, indices)
    
    # Create sampler if weights are provided
    sampler = None
    if weights is not None:
        # Convert to numpy array for easier handling
        weights_np = np.array(weights)
        
        # Check if weights array is large enough for all indices
        if len(weights_np) < max(indices) + 1:
            logger.warning(f"Weights array size ({len(weights_np)}) is smaller than max index ({max(indices)}). Using default weights.")
            # Use default weights of 1.0 for all indices
            subset_weights = np.ones(len(indices))
        else:
            # Extract weights for the subset
            subset_weights = weights_np[indices]
        
        # Create sampler
        sampler = create_weighted_sampler(subset_weights.tolist(), len(subset))
        shuffle = False  # Don't shuffle when using sampler
    
    # Create dataloader
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
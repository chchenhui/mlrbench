"""
Dataset classes for loading and processing SpurGen synthetic data.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Union, Callable

class SpurGenDataset(Dataset):
    """
    Dataset class for loading SpurGen synthetic data.
    
    Supports image-only, text-only, and multimodal (image+text) configurations.
    """
    
    def __init__(
        self, 
        data_dir: str,
        split: str = "train",
        modality: str = "multimodal",
        transform: Optional[Callable] = None,
        max_text_length: int = 77,  # For CLIP-like models
        return_attributes: bool = False,
        return_raw_data: bool = False
    ):
        """
        Initialize the SpurGen dataset.
        
        Args:
            data_dir: Directory containing the dataset files
            split: Dataset split ("train", "val", or "test")
            modality: Data modality ("image", "text", or "multimodal")
            transform: Optional transform to apply to images
            max_text_length: Maximum length for text tokenization
            return_attributes: Whether to return spurious attributes
            return_raw_data: Whether to return the raw data dictionary
        """
        self.data_dir = data_dir
        self.split = split
        self.modality = modality
        self.max_text_length = max_text_length
        self.return_attributes = return_attributes
        self.return_raw_data = return_raw_data
        
        # Set default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Load dataset
        self._load_data()
        
    def _load_data(self):
        """Load dataset from JSON file."""
        # Load metadata
        with open(os.path.join(self.data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
            
        # Load split data
        split_file = os.path.join(self.data_dir, f"{self.split}.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Dataset file {split_file} not found")
            
        with open(split_file, "r") as f:
            self.data = json.load(f)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Union[Dict, Tuple]:
        """
        Return a sample from the dataset.
        
        Args:
            idx: Index of the sample to return
            
        Returns:
            Dictionary or tuple containing the sample data
        """
        sample = self.data[idx]
        
        # Load image if needed
        image = None
        if self.modality in ["image", "multimodal"]:
            img_path = os.path.join(self.data_dir, sample["image_path"])
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
                
        # Process text if needed
        text = None
        if self.modality in ["text", "multimodal"]:
            text = sample["caption"]
            
        # Prepare class label
        label = torch.tensor(sample["class_idx"], dtype=torch.long)
        
        # Return data based on configuration
        if self.return_raw_data:
            return {
                "image": image,
                "text": text,
                "label": label,
                "sample": sample
            }
            
        if self.return_attributes:
            attributes = {
                key: torch.tensor([self.metadata["spurious_channels"][key]["attributes"].index(value)], 
                                  dtype=torch.long)
                for key, value in sample["spurious_attributes"].items()
            }
            
            if self.modality == "image":
                return image, label, attributes
            elif self.modality == "text":
                return text, label, attributes
            else:  # multimodal
                return image, text, label, attributes
        else:
            if self.modality == "image":
                return image, label
            elif self.modality == "text":
                return text, label
            else:  # multimodal
                return image, text, label


def get_data_loaders(
    data_dir: str,
    batch_size: int = 64,
    modality: str = "multimodal",
    transform_train: Optional[Callable] = None,
    transform_eval: Optional[Callable] = None,
    return_attributes: bool = False,
    return_raw_data: bool = False,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create data loaders for SpurGen dataset.
    
    Args:
        data_dir: Directory containing the dataset files
        batch_size: Batch size for data loading
        modality: Data modality ("image", "text", or "multimodal")
        transform_train: Optional transform for training data
        transform_eval: Optional transform for evaluation data
        return_attributes: Whether to return spurious attributes
        return_raw_data: Whether to return the raw data dictionary
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary containing data loaders for train, validation, and test splits
    """
    # Set default transforms if none provided
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    if transform_eval is None:
        transform_eval = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create datasets
    train_dataset = SpurGenDataset(
        data_dir=data_dir,
        split="train",
        modality=modality,
        transform=transform_train,
        return_attributes=return_attributes,
        return_raw_data=return_raw_data
    )
    
    val_dataset = SpurGenDataset(
        data_dir=data_dir,
        split="val",
        modality=modality,
        transform=transform_eval,
        return_attributes=return_attributes,
        return_raw_data=return_raw_data
    )
    
    test_dataset = SpurGenDataset(
        data_dir=data_dir,
        split="test",
        modality=modality,
        transform=transform_eval,
        return_attributes=return_attributes,
        return_raw_data=return_raw_data
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
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


class ShuffledSpurGenDataset(Dataset):
    """
    Dataset for evaluating model sensitivity to spurious features.
    
    Pairs original samples with versions where a specific spurious channel is shuffled.
    """
    
    def __init__(
        self,
        data_dir: str,
        channel: str,
        split: str = "test",
        modality: str = "multimodal",
        transform: Optional[Callable] = None
    ):
        """
        Initialize the shuffled SpurGen dataset.
        
        Args:
            data_dir: Directory containing the dataset files
            channel: Spurious channel to shuffle
            split: Dataset split ("train", "val", or "test")
            modality: Data modality ("image", "text", or "multimodal")
            transform: Optional transform to apply to images
        """
        self.data_dir = data_dir
        self.channel = channel
        self.split = split
        self.modality = modality
        
        # Set default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Load original data
        self._load_data()
        
        # Generate shuffled data
        self._generate_shuffled_data()
        
    def _load_data(self):
        """Load dataset from JSON file."""
        # Load split data
        split_file = os.path.join(self.data_dir, f"{self.split}.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Dataset file {split_file} not found")
            
        with open(split_file, "r") as f:
            self.original_data = json.load(f)
    
    def _generate_shuffled_data(self):
        """
        Generate shuffled versions of the data by changing attributes
        in the specified channel.
        """
        # Load metadata to get channel attributes
        with open(os.path.join(self.data_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        channel_attrs = metadata["spurious_channels"][self.channel]["attributes"]
        
        self.paired_data = []
        
        for sample in self.original_data:
            # Original sample
            orig_sample = sample.copy()
            
            # Create shuffled sample
            shuffled_sample = sample.copy()
            
            # Get current attribute
            current_attr = sample["spurious_attributes"][self.channel]
            
            # Choose a different attribute
            other_attrs = [attr for attr in channel_attrs if attr != current_attr]
            if not other_attrs:
                continue  # Skip if no other attributes available
                
            new_attr = other_attrs[0]  # Deterministic for reproducibility
            
            # Update spurious attributes
            shuffled_attrs = sample["spurious_attributes"].copy()
            shuffled_attrs[self.channel] = new_attr
            shuffled_sample["spurious_attributes"] = shuffled_attrs
            
            # Update image path to point to shuffled image
            shuffled_sample["image_path"] = f"images/shuffled_{sample['id']}_{self.channel}.png"
            
            # Add the pair to the dataset
            self.paired_data.append((orig_sample, shuffled_sample))
    
    def __len__(self) -> int:
        """Return the number of sample pairs in the dataset."""
        return len(self.paired_data)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Return a pair of samples (original and shuffled) from the dataset.
        
        Args:
            idx: Index of the sample pair to return
            
        Returns:
            Tuple containing the original and shuffled samples
        """
        orig_sample, shuffled_sample = self.paired_data[idx]
        
        # Process original sample
        orig_image = None
        if self.modality in ["image", "multimodal"]:
            orig_img_path = os.path.join(self.data_dir, orig_sample["image_path"])
            orig_image = Image.open(orig_img_path).convert("RGB")
            if self.transform:
                orig_image = self.transform(orig_image)
                
        orig_text = None
        if self.modality in ["text", "multimodal"]:
            orig_text = orig_sample["caption"]
            
        # Process shuffled sample
        shuffled_image = None
        if self.modality in ["image", "multimodal"]:
            shuffled_img_path = os.path.join(self.data_dir, shuffled_sample["image_path"])
            shuffled_image = Image.open(shuffled_img_path).convert("RGB")
            if self.transform:
                shuffled_image = self.transform(shuffled_image)
                
        shuffled_text = None
        if self.modality in ["text", "multimodal"]:
            shuffled_text = shuffled_sample["caption"]
            
        # Prepare class label (same for both)
        label = torch.tensor(orig_sample["class_idx"], dtype=torch.long)
        
        # Return data based on modality
        if self.modality == "image":
            return orig_image, shuffled_image, label
        elif self.modality == "text":
            return orig_text, shuffled_text, label
        else:  # multimodal
            return orig_image, orig_text, shuffled_image, shuffled_text, label


def get_shuffled_dataloader(
    data_dir: str,
    channel: str,
    batch_size: int = 64,
    split: str = "test",
    modality: str = "multimodal",
    transform: Optional[Callable] = None,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a data loader for the shuffled SpurGen dataset.
    
    Args:
        data_dir: Directory containing the dataset files
        channel: Spurious channel to shuffle
        batch_size: Batch size for data loading
        split: Dataset split ("train", "val", or "test")
        modality: Data modality ("image", "text", or "multimodal")
        transform: Optional transform to apply to images
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader for the shuffled dataset
    """
    dataset = ShuffledSpurGenDataset(
        data_dir=data_dir,
        channel=channel,
        split=split,
        modality=modality,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader
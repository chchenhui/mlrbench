"""
Dataset utilities for AIFS experiments.

This module implements dataset classes for the AIFS experiments, particularly
focusing on datasets with spurious correlations.
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union, Callable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


# Define standard image transforms
def get_image_transforms(augment: bool = True):
    """
    Get image transforms for training and testing.
    
    Args:
        augment: Whether to include data augmentation
        
    Returns:
        Dictionary with 'train' and 'test' transforms
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    base_transform = [
        transforms.ToTensor(),
        normalize,
    ]
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            *base_transform
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            *base_transform
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        *base_transform
    ])
    
    return {
        'train': train_transform,
        'test': test_transform
    }


class SpuriousCIFAR10(Dataset):
    """
    Modified CIFAR-10 dataset with artificially injected spurious correlations.
    
    This dataset adds a colored border to certain classes to create a spurious correlation.
    """
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = True,
        spurious_correlation_ratio: float = 0.95,
        border_width: int = 4,
        group_label: bool = True
    ):
        """
        Initialize the SpuriousCIFAR10 dataset.
        
        Args:
            root: Dataset root directory
            train: Whether to load training set
            transform: Image transform function
            download: Whether to download the dataset
            spurious_correlation_ratio: Ratio of samples where class and 
                                        spurious feature (border color) are correlated
            border_width: Width of the colored border in pixels
            group_label: Whether to include group labels in the output
        """
        self.base_dataset = CIFAR10(root, train=train, transform=None, download=download)
        self.transform = transform
        self.spurious_correlation_ratio = spurious_correlation_ratio
        self.border_width = border_width
        self.group_label = group_label
        
        # Define colors for spurious feature
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),    # Maroon
            (0, 128, 0),    # Dark Green
            (0, 0, 128),    # Navy
            (128, 128, 0)   # Olive
        ]
        
        # Create assignments of classes to spurious features
        # Each class is predominantly associated with one color
        self._create_spurious_assignments()
    
    def _create_spurious_assignments(self):
        """Create assignments of samples to spurious features."""
        n_samples = len(self.base_dataset)
        n_classes = 10  # CIFAR-10 has 10 classes
        
        self.sample_group = {}  # Maps sample index to group (0: aligned, 1: unaligned)
        self.sample_color = {}  # Maps sample index to color index
        
        # Get targets
        targets = np.array(self.base_dataset.targets)
        
        # For each class
        for class_idx in range(n_classes):
            # Get indices of samples for this class
            class_indices = np.where(targets == class_idx)[0]
            n_class_samples = len(class_indices)
            
            # Calculate how many samples will have the aligned spurious feature
            n_aligned = int(n_class_samples * self.spurious_correlation_ratio)
            
            # Set aligned samples
            aligned_indices = class_indices[:n_aligned]
            for idx in aligned_indices:
                self.sample_color[idx] = class_idx  # Same as class for aligned
                self.sample_group[idx] = 0  # Aligned group
            
            # Set unaligned samples with random other colors
            unaligned_indices = class_indices[n_aligned:]
            for idx in unaligned_indices:
                # Pick a random color different from the class index
                other_colors = [i for i in range(n_classes) if i != class_idx]
                self.sample_color[idx] = np.random.choice(other_colors)
                self.sample_group[idx] = 1  # Unaligned group
    
    def add_spurious_feature(self, img: Image.Image, color_idx: int) -> Image.Image:
        """
        Add a colored border to an image.
        
        Args:
            img: PIL Image
            color_idx: Index of the color to use
            
        Returns:
            Modified PIL Image with colored border
        """
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get image size
        width, height = img.size
        
        # Create a new image with colored border
        border_color = self.colors[color_idx]
        bordered_img = Image.new('RGB', img.size, border_color)
        
        # Paste the original image in the center
        inner_width = width - 2 * self.border_width
        inner_height = height - 2 * self.border_width
        bordered_img.paste(img.resize((inner_width, inner_height)), 
                          (self.border_width, self.border_width))
        
        return bordered_img
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, int]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, label) or (image, label, group)
        """
        img, label = self.base_dataset[idx]
        
        # Add spurious feature (colored border)
        color_idx = self.sample_color[idx]
        img = self.add_spurious_feature(img, color_idx)
        
        # Apply transform if provided
        if self.transform:
            img = self.transform(img)
        
        # Return with or without group label
        if self.group_label:
            group = self.sample_group[idx]
            return img, label, group
        else:
            return img, label


class SpuriousAdultDataset(Dataset):
    """
    Adult dataset (income prediction) with spurious correlations.
    
    This dataset uses the Adult Census Income dataset and amplifies
    existing spurious correlations related to demographic attributes.
    """
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
        amplify_bias: float = 2.0,  # Factor to amplify existing spurious correlations
        group_label: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize the SpuriousAdultDataset.
        
        Args:
            root: Dataset root directory
            train: Whether to load training set
            download: Whether to download the dataset
            amplify_bias: Factor to amplify existing spurious correlations
            group_label: Whether to include group labels in the output
            random_seed: Random seed for reproducibility
        """
        self.root = root
        self.train = train
        self.download = download
        self.amplify_bias = amplify_bias
        self.group_label = group_label
        self.random_seed = random_seed
        
        # Load and process data
        self._load_data()
    
    def _download_data(self):
        """Download Adult dataset if it doesn't exist."""
        os.makedirs(self.root, exist_ok=True)
        data_file = os.path.join(self.root, 'adult.csv')
        
        if not os.path.exists(data_file):
            # URL for the Adult Census dataset
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            
            # Column names for the dataset
            column_names = [
                'age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
            ]
            
            # Download and save data
            print(f"Downloading Adult dataset to {data_file}...")
            df = pd.read_csv(url, header=None, names=column_names, sep=', ', engine='python')
            df.to_csv(data_file, index=False)
            print("Download complete.")
    
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess Adult dataset for classification.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Tuple of (features, labels, group_labels)
        """
        # Clean up data
        df = df.replace(' ?', np.nan).dropna()
        
        # Define sensitive attribute (for spurious correlation)
        # We'll use 'sex' as the sensitive attribute
        df['sensitive'] = (df['sex'] == ' Male').astype(int)
        
        # Define the target
        df['label'] = (df['income'] == ' >50K').astype(int)
        
        # Amplify bias by sampling to create stronger spurious correlation
        np.random.seed(self.random_seed)
        
        # Create groups based on sensitive attribute and label
        df['group'] = df.apply(lambda x: 0 if x['sensitive'] == x['label'] else 1, axis=1)
        
        if self.amplify_bias > 1.0:
            # Undersample the less common combinations to amplify spurious correlation
            aligned = df[df['group'] == 0]
            unaligned = df[df['group'] == 1]
            
            # Undersample unaligned group
            unaligned_sample = unaligned.sample(
                frac=1.0 / self.amplify_bias,
                random_state=self.random_seed
            )
            
            # Combine datasets
            df = pd.concat([aligned, unaligned_sample], ignore_index=True)
        
        # Split into train and test
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=self.random_seed
        )
        
        # Use appropriate split
        df_to_use = train_df if self.train else test_df
        
        # Select features (exclude label, sensitive attribute, and group)
        # Convert categorical variables to one-hot encoding
        categorical_cols = ['workclass', 'education', 'marital-status', 
                           'occupation', 'relationship', 'race', 'native-country']
        
        numerical_cols = ['age', 'fnlwgt', 'education-num', 
                         'capital-gain', 'capital-loss', 'hours-per-week']
        
        # Process categorical features
        cat_data = pd.get_dummies(df_to_use[categorical_cols])
        
        # Process numerical features
        num_data = df_to_use[numerical_cols]
        # Normalize numerical data
        for col in num_data.columns:
            num_data[col] = (num_data[col] - num_data[col].mean()) / num_data[col].std()
        
        # Combine features
        X = pd.concat([num_data, cat_data], axis=1)
        y = df_to_use['label'].values
        groups = df_to_use['group'].values
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        group_tensor = torch.tensor(groups, dtype=torch.long)
        
        return X_tensor, y_tensor, group_tensor
    
    def _load_data(self):
        """Load and process the Adult dataset."""
        # Download data if needed
        if self.download:
            self._download_data()
        
        # Load the data
        data_file = os.path.join(self.root, 'adult.csv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found at {data_file}. "
                                   "Use download=True to download it.")
        
        df = pd.read_csv(data_file)
        
        # Preprocess data
        self.features, self.labels, self.groups = self._preprocess_data(df)
        
        print(f"Loaded {'train' if self.train else 'test'} data: "
              f"{len(self.features)} samples, "
              f"{self.features.shape[1]} features")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, int]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label) or (features, label, group)
        """
        features = self.features[idx]
        label = self.labels[idx].item()
        
        if self.group_label:
            group = self.groups[idx].item()
            return features, label, group
        else:
            return features, label


def get_dataloaders(
    dataset_name: str,
    root: str = './data',
    batch_size: int = 32,
    num_workers: int = 4,
    spurious_correlation_ratio: float = 0.95,
    split_ratio: float = 0.8,
    group_label: bool = True,
    augment: bool = True,
    random_seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Get dataloaders for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset ('spurious_cifar10' or 'spurious_adult')
        root: Dataset root directory
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        spurious_correlation_ratio: Ratio of samples where class and spurious feature are correlated
        split_ratio: Train/validation split ratio
        group_label: Whether to include group labels in the output
        augment: Whether to apply data augmentation
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary of dataloaders for 'train', 'val', and 'test'
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    if dataset_name == 'spurious_cifar10':
        # Set up image transforms
        transforms_dict = get_image_transforms(augment=augment)
        
        # Create datasets
        train_dataset = SpuriousCIFAR10(
            root=root,
            train=True,
            transform=transforms_dict['train'],
            download=True,
            spurious_correlation_ratio=spurious_correlation_ratio,
            group_label=group_label
        )
        
        test_dataset = SpuriousCIFAR10(
            root=root,
            train=False,
            transform=transforms_dict['test'],
            download=True,
            spurious_correlation_ratio=spurious_correlation_ratio,
            group_label=group_label
        )
        
        # Split train into train and validation
        train_size = int(len(train_dataset) * split_ratio)
        val_size = len(train_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        
    elif dataset_name == 'spurious_adult':
        # Create datasets
        train_full_dataset = SpuriousAdultDataset(
            root=root,
            train=True,
            download=True,
            amplify_bias=2.0,
            group_label=group_label,
            random_seed=random_seed
        )
        
        test_dataset = SpuriousAdultDataset(
            root=root,
            train=False,
            download=True,
            amplify_bias=2.0,
            group_label=group_label,
            random_seed=random_seed
        )
        
        # Split train into train and validation
        train_size = int(len(train_full_dataset) * split_ratio)
        val_size = len(train_full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloaders
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
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


# Example usage
# dataloaders = get_dataloaders('spurious_cifar10', batch_size=64)
# train_loader = dataloaders['train']
# val_loader = dataloaders['val']
# test_loader = dataloaders['test']
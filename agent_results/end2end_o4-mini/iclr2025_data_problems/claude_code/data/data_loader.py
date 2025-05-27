"""
Data loading and preprocessing module for the Gradient-Informed Fingerprinting (GIF) method.

This module handles loading datasets from common sources like The Pile, C4, 
and LAION for both textual and multimodal experiments.
"""

import os
import json
import logging
import random
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from datasets import load_dataset, Dataset as HFDataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    dataset_name: str
    subset_name: Optional[str] = None
    text_column: str = "text"
    max_samples: Optional[int] = None
    tokenizer_name: str = "bert-base-uncased"
    max_length: int = 512
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    cache_dir: Optional[str] = None
    use_synthetic: bool = False
    synthetic_samples: int = 1000
    data_dir: str = "data"
    
    def __post_init__(self):
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, \
            "Train, validation, and test ratios must sum to 1.0"


class TextDataset(Dataset):
    """Custom dataset for text data."""
    
    def __init__(
        self, 
        texts: List[str], 
        ids: List[str] = None,
        tokenizer_name: str = "bert-base-uncased", 
        max_length: int = 512
    ):
        self.texts = texts
        self.ids = [str(i) for i in range(len(texts))] if ids is None else ids
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        sample_id = self.ids[idx]
        
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove the batch dimension
        return {
            "id": sample_id,
            "text": text,
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0]
        }


class DataManager:
    """Manager for loading, preprocessing, and splitting datasets."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.datasets = {"train": None, "val": None, "test": None}
        self.data_loaders = {"train": None, "val": None, "test": None}
        
    def load_and_process_data(self) -> Dict[str, Dataset]:
        """Load and process the dataset based on configuration."""
        if self.config.use_synthetic:
            data = self._create_synthetic_data()
        else:
            data = self._load_real_data()
            
        # Create train/val/test splits
        return self._split_data(data)
    
    def _load_real_data(self) -> Dict[str, List]:
        """Load real data from Hugging Face datasets."""
        logger.info(f"Loading {self.config.dataset_name} dataset...")
        
        if self.config.dataset_name.lower() == "pile":
            # For The Pile, we'll use a small subset for demonstration
            dataset = load_dataset(
                "json", 
                data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst",
                split="train",
                cache_dir=self.config.cache_dir
            )
        elif self.config.dataset_name.lower() == "c4":
            # For C4, we'll use the 'en' subset
            dataset = load_dataset(
                "c4", 
                "en",
                split="validation",  # Using validation split to avoid downloading the full train set
                cache_dir=self.config.cache_dir
            )
        elif self.config.dataset_name.lower() == "wikitext":
            # Smaller dataset for quick testing
            dataset = load_dataset(
                "wikitext", 
                "wikitext-103-v1",
                split="validation",
                cache_dir=self.config.cache_dir
            )
        else:
            # Load any other dataset from Hugging Face
            if self.config.subset_name:
                dataset = load_dataset(
                    self.config.dataset_name, 
                    self.config.subset_name, 
                    split="train",
                    cache_dir=self.config.cache_dir
                )
            else:
                dataset = load_dataset(
                    self.config.dataset_name, 
                    split="train",
                    cache_dir=self.config.cache_dir
                )
        
        # Limit number of samples if specified
        if self.config.max_samples and self.config.max_samples < len(dataset):
            dataset = dataset.select(range(self.config.max_samples))
        
        # Extract texts
        texts = dataset[self.config.text_column]
        
        # Generate IDs if not present
        if "id" in dataset.column_names:
            ids = dataset["id"]
        else:
            ids = [str(i) for i in range(len(texts))]
        
        logger.info(f"Loaded {len(texts)} samples from {self.config.dataset_name}")
        
        return {"texts": texts, "ids": ids}
    
    def _create_synthetic_data(self) -> Dict[str, List]:
        """Create synthetic data for testing."""
        logger.info(f"Creating {self.config.synthetic_samples} synthetic samples...")
        
        # Generate simple synthetic texts
        texts = []
        for i in range(self.config.synthetic_samples):
            # Create texts with varying lengths and some repeated patterns
            length = random.randint(50, 200)
            base_text = f"This is synthetic sample {i} for testing."
            repeated = " ".join([base_text] * (length // len(base_text.split()) + 1))
            texts.append(repeated[:length])
        
        ids = [f"synthetic_{i}" for i in range(self.config.synthetic_samples)]
        
        logger.info(f"Created {len(texts)} synthetic samples")
        
        return {"texts": texts, "ids": ids}
    
    def _split_data(self, data: Dict[str, List]) -> Dict[str, Dataset]:
        """Split data into train, validation, and test sets."""
        texts = data["texts"]
        ids = data["ids"]
        
        # Calculate split sizes
        total_size = len(texts)
        train_size = int(total_size * self.config.train_ratio)
        val_size = int(total_size * self.config.val_ratio)
        test_size = total_size - train_size - val_size
        
        # Create dataset
        full_dataset = TextDataset(
            texts, 
            ids,
            self.config.tokenizer_name,
            self.config.max_length
        )
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        logger.info(f"Dataset split into {train_size} train, {val_size} validation, {test_size} test samples")
        
        return {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    
    def get_dataloaders(self, batch_size: int = 32) -> Dict[str, DataLoader]:
        """Create data loaders for train, validation, and test sets."""
        if not self.datasets["train"]:
            self.datasets = self.load_and_process_data()
        
        self.data_loaders = {
            "train": DataLoader(
                self.datasets["train"], 
                batch_size=batch_size, 
                shuffle=True
            ),
            "val": DataLoader(
                self.datasets["val"], 
                batch_size=batch_size, 
                shuffle=False
            ),
            "test": DataLoader(
                self.datasets["test"], 
                batch_size=batch_size, 
                shuffle=False
            )
        }
        
        return self.data_loaders
    
    def save_dataset_stats(self, output_dir: str):
        """Save dataset statistics to a JSON file."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        stats = {
            "dataset_name": self.config.dataset_name,
            "subset_name": self.config.subset_name,
            "total_samples": sum(len(ds) for ds in self.datasets.values() if ds),
            "train_samples": len(self.datasets["train"]) if self.datasets["train"] else 0,
            "val_samples": len(self.datasets["val"]) if self.datasets["val"] else 0,
            "test_samples": len(self.datasets["test"]) if self.datasets["test"] else 0,
            "tokenizer": self.config.tokenizer_name,
            "max_length": self.config.max_length,
            "is_synthetic": self.config.use_synthetic,
        }
        
        with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved dataset statistics to {output_dir}/dataset_stats.json")
        
        return stats


# Command-line interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test data loading and processing")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    
    args = parser.parse_args()
    
    config = DataConfig(
        dataset_name=args.dataset,
        subset_name=args.subset,
        max_samples=args.max_samples,
        use_synthetic=args.synthetic,
        data_dir=args.output_dir
    )
    
    data_manager = DataManager(config)
    datasets = data_manager.load_and_process_data()
    dataloaders = data_manager.get_dataloaders()
    
    # Print a sample
    sample = next(iter(dataloaders["train"]))
    print(f"Sample ID: {sample['id'][0]}")
    print(f"Text: {sample['text'][0][:100]}...")
    
    # Save dataset statistics
    data_manager.save_dataset_stats(args.output_dir)
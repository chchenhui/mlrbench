import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import os

logger = logging.getLogger(__name__)

class LongContextDataset(Dataset):
    """Dataset for long-context language modeling tasks."""
    
    def __init__(self, 
                 dataset_name: str,
                 split: str,
                 tokenizer: AutoTokenizer,
                 max_length: int = 4096,
                 stride: int = 1024,
                 cache_dir: Optional[str] = None,
                 sample_size: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            dataset_name: Name of the dataset to load (e.g., "pg19", "wikitext-103-v1")
            split: Dataset split to use (e.g., "train", "validation", "test")
            tokenizer: Tokenizer for processing text
            max_length: Maximum sequence length
            stride: Stride for overlapping chunks (for long documents)
            cache_dir: Directory to cache the dataset
            sample_size: If set, use only this many samples from the dataset (for quick testing)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.cache_dir = cache_dir
        
        # Load raw dataset
        logger.info(f"Loading dataset {dataset_name} (split: {split})")
        try:
            self.raw_dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise
        
        # Take a sample if requested
        if sample_size is not None:
            self.raw_dataset = self.raw_dataset.select(range(min(sample_size, len(self.raw_dataset))))
        
        # Tokenize and chunk dataset
        logger.info(f"Tokenizing and chunking dataset (max_length={max_length}, stride={stride})")
        self.tokenized_chunks = self._prepare_dataset()
    
    def _prepare_dataset(self) -> List[Dict[str, torch.Tensor]]:
        """
        Tokenize and chunk the dataset.
        
        Returns:
            List of chunked examples with input_ids, attention_mask, and labels
        """
        tokenized_chunks = []
        
        # Process each text in the dataset
        for i, example in enumerate(self.raw_dataset):
            # Get the text field (dataset-specific)
            if self.dataset_name == "pg19":
                text = example["text"]
            elif "wikitext" in self.dataset_name:
                text = example["text"]
            elif "arxiv" in self.dataset_name:
                text = example["abstract"] + "\n\n" + example["article"]
            elif "eli5" in self.dataset_name:
                text = example["question"] + "\n\n" + example["answers"]["text"][0]
            elif "narrativeqa" in self.dataset_name:
                text = example["document"]["text"]
            else:
                # Default behavior
                text = example["text"] if "text" in example else str(example)
            
            # Skip empty texts
            if not text.strip():
                continue
            
            # Tokenize the text
            tokenized = self.tokenizer(text, return_tensors="pt", truncation=False)
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
            
            # Skip if text is too short
            if len(input_ids) < 100:
                continue
            
            # Create chunks with overlap
            for i in range(0, len(input_ids) - 128, self.stride):
                end_idx = min(i + self.max_length, len(input_ids))
                
                # Only add if chunk is long enough
                if end_idx - i >= 128:
                    chunk = {
                        "input_ids": input_ids[i:end_idx].clone(),
                        "attention_mask": attention_mask[i:end_idx].clone(),
                        "labels": input_ids[i:end_idx].clone(),
                    }
                    tokenized_chunks.append(chunk)
                
                # Stop if we've reached the end
                if end_idx == len(input_ids):
                    break
        
        logger.info(f"Created {len(tokenized_chunks)} chunks from dataset")
        return tokenized_chunks
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.tokenized_chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a dataset item."""
        return self.tokenized_chunks[idx]


class SummarizationDataset(Dataset):
    """Dataset for long-context summarization tasks."""
    
    def __init__(self, 
                 dataset_name: str,
                 split: str,
                 tokenizer: AutoTokenizer,
                 max_length: int = 4096,
                 max_target_length: int = 512,
                 cache_dir: Optional[str] = None,
                 sample_size: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            split: Dataset split to use
            tokenizer: Tokenizer for processing text
            max_length: Maximum sequence length for inputs
            max_target_length: Maximum sequence length for targets
            cache_dir: Directory to cache the dataset
            sample_size: If set, use only this many samples from the dataset
        """
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.cache_dir = cache_dir
        
        # Load raw dataset
        logger.info(f"Loading dataset {dataset_name} (split: {split})")
        try:
            self.raw_dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise
        
        # Take a sample if requested
        if sample_size is not None:
            self.raw_dataset = self.raw_dataset.select(range(min(sample_size, len(self.raw_dataset))))
        
        # Tokenize dataset
        logger.info(f"Tokenizing dataset (max_length={max_length}, max_target_length={max_target_length})")
        self.tokenized_data = self._prepare_dataset()
    
    def _prepare_dataset(self) -> List[Dict[str, torch.Tensor]]:
        """
        Tokenize and process the dataset.
        
        Returns:
            List of tokenized examples with input_ids, attention_mask, and labels
        """
        tokenized_data = []
        
        # Process each example
        for i, example in enumerate(self.raw_dataset):
            # Get source and target text (dataset-specific)
            if "arxiv" in self.dataset_name:
                source_text = example["article"]
                target_text = example["abstract"]
            elif "narrativeqa" in self.dataset_name:
                source_text = example["document"]["text"]
                target_text = example["summary"]["text"]
            elif "cnn_dailymail" in self.dataset_name:
                source_text = example["article"]
                target_text = example["highlights"]
            elif "eli5" in self.dataset_name:
                source_text = example["question"]
                target_text = example["answers"]["text"][0] if example["answers"]["text"] else ""
            else:
                # Default behavior
                source_text = example["text"] if "text" in example else example["document"]
                target_text = example["summary"] if "summary" in example else example["abstract"]
            
            # Skip if either text is empty
            if not source_text.strip() or not target_text.strip():
                continue
            
            # Tokenize source and target
            source_tokens = self.tokenizer(
                source_text, 
                max_length=self.max_length, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            
            target_tokens = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Prepare item
            item = {
                "input_ids": source_tokens["input_ids"].squeeze(0),
                "attention_mask": source_tokens["attention_mask"].squeeze(0),
                "labels": target_tokens["input_ids"].squeeze(0),
            }
            tokenized_data.append(item)
        
        logger.info(f"Processed {len(tokenized_data)} examples for summarization")
        return tokenized_data
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.tokenized_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a dataset item."""
        return self.tokenized_data[idx]


def create_dataloader(dataset: Dataset, 
                     batch_size: int, 
                     shuffle: bool = True,
                     num_workers: int = 4) -> DataLoader:
    """
    Create a DataLoader for the given dataset.
    
    Args:
        dataset: Dataset to create loader for
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_dataset(dataset_name: str,
               tokenizer: AutoTokenizer,
               split: str = "train",
               task: str = "language_modeling",
               max_length: int = 4096,
               stride: int = 1024,
               max_target_length: int = 512,
               cache_dir: Optional[str] = None,
               sample_size: Optional[int] = None) -> Dataset:
    """
    Get a dataset for the specified task.
    
    Args:
        dataset_name: Name of the dataset to load
        tokenizer: Tokenizer for processing text
        split: Dataset split to use
        task: Task type ("language_modeling" or "summarization")
        max_length: Maximum sequence length
        stride: Stride for overlapping chunks (for language modeling)
        max_target_length: Maximum target length (for summarization)
        cache_dir: Directory to cache the dataset
        sample_size: If set, use only this many samples from the dataset
        
    Returns:
        Dataset instance
    """
    if task == "language_modeling":
        return LongContextDataset(
            dataset_name=dataset_name,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            stride=stride,
            cache_dir=cache_dir,
            sample_size=sample_size
        )
    elif task == "summarization":
        return SummarizationDataset(
            dataset_name=dataset_name,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            max_target_length=max_target_length,
            cache_dir=cache_dir,
            sample_size=sample_size
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
        
        
def prepare_small_datasets_for_testing():
    """
    Prepare small subsets of datasets for testing and debugging.
    Downloads samples from the datasets and saves them locally.
    """
    cache_dir = os.path.join(os.getcwd(), "data/cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # List of datasets to sample
    datasets_to_sample = [
        ("wikitext", "wikitext-103-v1", "train", 10),
        ("wikitext", "wikitext-103-v1", "validation", 5),
        ("arxiv_dataset", None, "train", 10),
        ("narrativeqa", None, "validation", 5),
    ]
    
    for dataset_name, config, split, num_samples in datasets_to_sample:
        logger.info(f"Downloading sample of {dataset_name} ({config}) - {split} split")
        try:
            if config:
                dataset = load_dataset(dataset_name, config, split=split, cache_dir=cache_dir)
            else:
                dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
                
            # Sample and save
            sample = dataset.select(range(min(num_samples, len(dataset))))
            sample_path = os.path.join(cache_dir, f"{dataset_name}_{config or 'default'}_{split}_sample.pt")
            
            # Convert to dict for saving
            sample_dict = {
                "dataset_name": dataset_name,
                "config": config,
                "split": split,
                "samples": [dict(item) for item in sample]  # Convert dataset items to dict
            }
            
            torch.save(sample_dict, sample_path)
            logger.info(f"Saved {len(sample)} samples to {sample_path}")
            
        except Exception as e:
            logger.error(f"Error sampling {dataset_name}: {e}")
    
    logger.info("Finished preparing test datasets")


def load_test_dataset(dataset_name: str, config: str, split: str) -> List[Dict]:
    """
    Load a small test dataset from disk.
    
    Args:
        dataset_name: Name of the dataset
        config: Dataset configuration
        split: Dataset split
        
    Returns:
        List of dataset examples
    """
    cache_dir = os.path.join(os.getcwd(), "data/cache")
    sample_path = os.path.join(cache_dir, f"{dataset_name}_{config or 'default'}_{split}_sample.pt")
    
    if not os.path.exists(sample_path):
        logger.warning(f"Test dataset not found at {sample_path}. Run prepare_small_datasets_for_testing() first.")
        return []
    
    sample_dict = torch.load(sample_path)
    return sample_dict["samples"]
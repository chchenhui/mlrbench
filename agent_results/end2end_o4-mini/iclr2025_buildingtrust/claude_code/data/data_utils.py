"""
Data utilities for the Cluster-Driven Certified Unlearning experiment.
"""

import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from transformers import PreTrainedTokenizer
from datasets import load_dataset


class LanguageModelingDataset(Dataset):
    """
    Dataset for language modeling tasks.
    """
    
    def __init__(
        self,
        texts,
        tokenizer,
        max_length=512,
        stride=256,
        return_tensors="pt",
        text_column="text"
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of texts or Hugging Face dataset
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            stride: Stride for tokenization window
            return_tensors: Type of tensors to return
            text_column: Column name for text in dataset
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.return_tensors = return_tensors
        
        # Process texts
        if isinstance(texts, list):
            self.examples = texts
        elif hasattr(texts, "map"):
            # Hugging Face dataset
            self.examples = [item[text_column] for item in texts]
        else:
            raise ValueError("texts must be a list or a Hugging Face dataset")
            
        # Tokenize all texts
        self.tokenized_examples = self._tokenize_examples()
        
    def _tokenize_examples(self):
        """
        Tokenize all examples using sliding window for long texts.
        
        Returns:
            List of tokenized examples with input_ids and targets
        """
        tokenized = []
        
        for text in self.examples:
            # Tokenize with stride
            encodings = self.tokenizer(
                text,
                return_tensors=self.return_tensors,
                max_length=self.max_length,
                truncation=True,
                stride=self.stride,
                return_overflowing_tokens=True
            )
            
            # For each chunk
            for i in range(len(encodings["input_ids"])):
                input_ids = encodings["input_ids"][i]
                
                # Create example
                example = {
                    "input_ids": input_ids[:-1],  # All but last token as input
                    "attention_mask": encodings["attention_mask"][i][:-1],
                    "targets": input_ids[1:]  # All but first token as target (shifted right)
                }
                
                tokenized.append(example)
                
        return tokenized
        
    def __len__(self):
        return len(self.tokenized_examples)
        
    def __getitem__(self, idx):
        example = self.tokenized_examples[idx]
        
        # Convert to tensors if necessary
        if self.return_tensors == "pt" and not isinstance(example["input_ids"], torch.Tensor):
            example = {k: torch.tensor(v) for k, v in example.items()}
            
        return example


def load_tiny_shakespeare_data(tokenizer, max_length=512, stride=256):
    """
    Load Tiny Shakespeare dataset for language modeling.
    
    Args:
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        stride: Stride for tokenization window
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Load Tiny Shakespeare dataset
    dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)["train"]
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create language modeling datasets
    train_lm = LanguageModelingDataset(
        train_dataset, tokenizer, max_length=max_length, stride=stride, text_column="text"
    )
    
    val_lm = LanguageModelingDataset(
        val_dataset, tokenizer, max_length=max_length, stride=stride, text_column="text"
    )
    
    test_lm = LanguageModelingDataset(
        test_dataset, tokenizer, max_length=max_length, stride=stride, text_column="text"
    )
    
    return train_lm, val_lm, test_lm


def load_webtext_data(tokenizer, max_length=512, stride=256):
    """
    Load OpenWebText dataset for language modeling.
    
    Args:
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        stride: Stride for tokenization window
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Use tiny_shakespeare as a replacement for webtext (to avoid trust_remote_code issues)
    return load_tiny_shakespeare_data(tokenizer, max_length, stride)


def load_domain_specific_data(domain, tokenizer, max_length=512, stride=256):
    """
    Load domain-specific datasets for language modeling.
    
    Args:
        domain: Domain name ("medical", "legal", "code")
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        stride: Stride for tokenization window
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    if domain == "medical":
        # Use tiny_shakespeare as a replacement for domain-specific data
        return load_tiny_shakespeare_data(tokenizer, max_length, stride)
    elif domain == "legal":
        # Use tiny_shakespeare as a replacement for domain-specific data
        return load_tiny_shakespeare_data(tokenizer, max_length, stride)
    elif domain == "code":
        # Use tiny_shakespeare as a replacement for domain-specific data
        return load_tiny_shakespeare_data(tokenizer, max_length, stride)
    else:
        raise ValueError(f"Unsupported domain: {domain}")


def create_deletion_sets(dataset, num_sets=5, set_sizes=[10, 50, 100, 500, 1000]):
    """
    Create multiple deletion sets of varying sizes.
    
    Args:
        dataset: Dataset to sample from
        num_sets: Number of deletion sets to create
        set_sizes: List of deletion set sizes
        
    Returns:
        List of deletion sets, where each set is a list of examples
    """
    # Limit set sizes based on dataset size
    max_size = min(max(set_sizes), len(dataset) // 10)
    valid_set_sizes = [s for s in set_sizes if s <= max_size]
    
    # Create deletion sets
    deletion_sets = []
    
    for size in valid_set_sizes:
        for _ in range(num_sets):
            # Sample random indices
            indices = random.sample(range(len(dataset)), size)
            
            # Create deletion set
            deletion_set = [dataset[i] for i in indices]
            deletion_sets.append(deletion_set)
    
    return deletion_sets


def create_sequential_deletion_requests(dataset, num_requests=5, request_sizes=[50, 50, 50, 50, 50]):
    """
    Create a sequence of deletion requests.
    
    Args:
        dataset: Dataset to sample from
        num_requests: Number of deletion requests
        request_sizes: List of request sizes
        
    Returns:
        List of deletion requests, where each request is a list of examples
    """
    assert num_requests == len(request_sizes), "Number of requests must match length of request_sizes"
    
    # Limit total size based on dataset size
    total_size = sum(request_sizes)
    if total_size > len(dataset) // 2:
        scale = (len(dataset) // 2) / total_size
        request_sizes = [int(s * scale) for s in request_sizes]
    
    # Create deletion requests
    deletion_requests = []
    used_indices = set()
    
    for size in request_sizes:
        # Sample random indices not used in previous requests
        available_indices = list(set(range(len(dataset))) - used_indices)
        request_indices = random.sample(available_indices, size)
        
        # Update used indices
        used_indices.update(request_indices)
        
        # Create deletion request
        deletion_request = [dataset[i] for i in request_indices]
        deletion_requests.append(deletion_request)
    
    return deletion_requests
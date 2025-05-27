import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

class SyntheticDataset(Dataset):
    """Dataset that generates synthetic data for LM testing."""
    
    def __init__(self, 
                 tokenizer,
                 num_samples: int = 10,
                 seq_length: int = 512,
                 vocab_size: Optional[int] = None):
        """
        Initialize synthetic dataset.
        
        Args:
            tokenizer: Tokenizer to use for vocab size
            num_samples: Number of samples to generate
            seq_length: Sequence length for each sample
            vocab_size: Vocabulary size (if None, use tokenizer's vocab size)
        """
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size or len(tokenizer)
        
        logger.info(f"Creating synthetic dataset with {num_samples} samples, "
                   f"sequence length {seq_length}, vocab size {self.vocab_size}")
        
        # Generate synthetic data
        self.samples = []
        for _ in range(num_samples):
            # Generate random token IDs (ensure they're in valid vocabulary range)
            input_ids = torch.randint(2, min(self.vocab_size, 20000), (seq_length,))
            attention_mask = torch.ones_like(input_ids)
            
            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone()
            })
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a dataset item."""
        return self.samples[idx]


class SyntheticSummarizationDataset(Dataset):
    """Dataset that generates synthetic data for summarization testing."""
    
    def __init__(self, 
                 tokenizer,
                 num_samples: int = 10,
                 input_length: int = 512,
                 output_length: int = 128,
                 vocab_size: Optional[int] = None):
        """
        Initialize synthetic summarization dataset.
        
        Args:
            tokenizer: Tokenizer to use for vocab size
            num_samples: Number of samples to generate
            input_length: Input sequence length
            output_length: Output sequence length
            vocab_size: Vocabulary size (if None, use tokenizer's vocab size)
        """
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.input_length = input_length
        self.output_length = output_length
        self.vocab_size = vocab_size or len(tokenizer)
        
        logger.info(f"Creating synthetic summarization dataset with {num_samples} samples, "
                   f"input length {input_length}, output length {output_length}")
        
        # Generate synthetic data
        self.samples = []
        for _ in range(num_samples):
            # Generate random token IDs for input
            input_ids = torch.randint(2, min(self.vocab_size, 20000), (input_length,))
            attention_mask = torch.ones_like(input_ids)
            
            # Generate random token IDs for output/summary
            # Make the summary tokens partially based on the input to simulate a real summary
            input_subset = input_ids[:output_length // 4]  # Use some tokens from input
            random_subset = torch.randint(2, min(self.vocab_size, 20000), (output_length - output_length // 4,))
            labels = torch.cat([input_subset, random_subset])
            
            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a dataset item."""
        return self.samples[idx]
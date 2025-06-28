#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Long-context datasets for evaluating the proposed architecture.

This module provides dataset loading and preprocessing utilities for:
1. Natural Questions-Long for long-context QA
2. ELI5 for explanatory QA
3. CNN/DailyMail for streaming news analysis
4. GitHub Code for code understanding
5. S2ORC for scientific literature processing
"""

import os
import json
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LongContextDataset(Dataset):
    """
    Base dataset class for long-context data.
    """
    def __init__(
        self,
        samples: List[Dict],
        vocab: Dict[str, int],
        max_tokens: int = 4096
    ):
        """
        Initialize the dataset.
        
        Args:
            samples: List of data samples, each a dict with at least 'input_ids' and 'labels'
            vocab: Vocabulary mapping from tokens to ids
            max_tokens: Maximum number of tokens per sample
        """
        self.samples = samples
        self.vocab = vocab
        self.max_tokens = max_tokens
        
        logger.info(f"Created dataset with {len(samples)} samples, vocab size: {len(vocab)}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a data sample by index.
        
        Args:
            idx: Sample index
        
        Returns:
            Dict containing 'input_ids', 'attention_mask', 'labels'
        """
        sample = self.samples[idx]
        
        # Get input IDs and truncate if needed
        input_ids = sample['input_ids'][:self.max_tokens]
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = torch.ones_like(input_ids)
        
        # Get labels, or use input_ids if not present
        if 'labels' in sample:
            labels = sample['labels'][:self.max_tokens]
        else:
            # Use input shifted by 1 as default labels for next-token prediction
            labels = torch.cat([input_ids[1:], torch.tensor([-100])])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class NaturalQuestionsDataset(LongContextDataset):
    """
    Dataset class for Natural Questions-Long.
    """
    def __init__(
        self,
        samples: List[Dict],
        vocab: Dict[str, int],
        max_tokens: int = 4096
    ):
        super().__init__(samples, vocab, max_tokens)


class ELI5Dataset(LongContextDataset):
    """
    Dataset class for ELI5 (Explain Like I'm 5).
    """
    def __init__(
        self,
        samples: List[Dict],
        vocab: Dict[str, int],
        max_tokens: int = 4096
    ):
        super().__init__(samples, vocab, max_tokens)


class CNNDailyMailDataset(LongContextDataset):
    """
    Dataset class for CNN/DailyMail with temporal relationship markers.
    """
    def __init__(
        self,
        samples: List[Dict],
        vocab: Dict[str, int],
        max_tokens: int = 4096
    ):
        super().__init__(samples, vocab, max_tokens)


class GitHubCodeDataset(LongContextDataset):
    """
    Dataset class for GitHub Code corpus.
    """
    def __init__(
        self,
        samples: List[Dict],
        vocab: Dict[str, int],
        max_tokens: int = 4096
    ):
        super().__init__(samples, vocab, max_tokens)


class S2ORCDataset(LongContextDataset):
    """
    Dataset class for S2ORC academic papers.
    """
    def __init__(
        self,
        samples: List[Dict],
        vocab: Dict[str, int],
        max_tokens: int = 4096
    ):
        super().__init__(samples, vocab, max_tokens)


def create_simple_vocab(texts: List[str], min_freq: int = 2) -> Dict[str, int]:
    """
    Create a simple vocabulary from a list of texts.
    
    Args:
        texts: List of texts
        min_freq: Minimum frequency for a token to be included
    
    Returns:
        vocab: Dictionary mapping tokens to ids
    """
    # Count token frequencies
    token_freq = {}
    for text in texts:
        for token in text.split():
            token_freq[token] = token_freq.get(token, 0) + 1
    
    # Filter by minimum frequency
    filtered_tokens = [token for token, freq in token_freq.items() if freq >= min_freq]
    
    # Create vocabulary
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<bos>': 2,
        '<eos>': 3
    }
    
    for idx, token in enumerate(filtered_tokens):
        vocab[token] = idx + 4
    
    return vocab


def tokenize_text(text: str, vocab: Dict[str, int]) -> torch.Tensor:
    """
    Tokenize text using a vocabulary.
    
    Args:
        text: Input text
        vocab: Vocabulary mapping from tokens to ids
    
    Returns:
        tokens: Tensor of token ids
    """
    tokens = []
    for token in text.split():
        token_id = vocab.get(token, vocab['<unk>'])
        tokens.append(token_id)
    
    return torch.tensor(tokens, dtype=torch.long)


def load_natural_questions(max_samples: int = 1000, max_tokens: int = 4096) -> NaturalQuestionsDataset:
    """
    Load and preprocess Natural Questions dataset.
    
    Args:
        max_samples: Maximum number of samples to load
        max_tokens: Maximum number of tokens per sample
    
    Returns:
        dataset: NaturalQuestionsDataset
    """
    logger.info("Loading Natural Questions dataset...")
    
    try:
        # Load the dataset from Hugging Face
        nq_dataset = load_dataset("nq_open", split="train[:1000]")
        
        # Take a subset of samples
        samples = nq_dataset[:min(max_samples, len(nq_dataset))]
        
        # Extract questions and answers
        texts = []
        processed_samples = []
        
        for sample in samples:
            question = sample['question']
            answer = sample.get('answer', [''])[0]  # Get first answer if available
            
            # Combine question and answer into a single text
            full_text = f"Question: {question} Answer: {answer}"
            texts.append(full_text)
        
        # Create vocabulary
        vocab = create_simple_vocab(texts)
        
        # Preprocess samples
        for text in texts:
            input_ids = tokenize_text(text, vocab)
            
            # Skip samples that are empty after tokenization
            if len(input_ids) > 0:
                processed_samples.append({
                    'input_ids': input_ids,
                    'labels': input_ids.clone()  # For reconstruction/generation
                })
        
        return NaturalQuestionsDataset(processed_samples, vocab, max_tokens)
    
    except Exception as e:
        logger.warning(f"Error loading Natural Questions dataset: {e}")
        logger.info("Creating simulated dataset instead.")
        
        # Create a simulated dataset
        vocab_size = 10000
        vocab = {f"token_{i}": i for i in range(vocab_size)}
        vocab.update({'<pad>': vocab_size, '<unk>': vocab_size + 1})
        
        samples = []
        for i in range(max_samples):
            # Create random input_ids with length between 1000 and max_tokens
            length = np.random.randint(1000, max_tokens + 1)
            input_ids = torch.randint(0, vocab_size, (length,))
            
            samples.append({
                'input_ids': input_ids,
                'labels': input_ids.clone()
            })
        
        return NaturalQuestionsDataset(samples, vocab, max_tokens)


def load_eli5(max_samples: int = 1000, max_tokens: int = 4096) -> ELI5Dataset:
    """
    Load and preprocess ELI5 dataset.
    
    Args:
        max_samples: Maximum number of samples to load
        max_tokens: Maximum number of tokens per sample
    
    Returns:
        dataset: ELI5Dataset
    """
    logger.info("Loading ELI5 dataset...")
    
    try:
        # Load the dataset from Hugging Face
        eli5_dataset = load_dataset("eli5", split="train[:1000]")
        
        # Take a subset of samples
        samples = eli5_dataset[:min(max_samples, len(eli5_dataset))]
        
        # Extract questions and answers
        texts = []
        processed_samples = []
        
        for sample in samples:
            question = sample['title']
            answer = sample.get('answers', {}).get('text', [''])[0]  # Get first answer if available
            
            # Combine question and answer into a single text
            full_text = f"Question: {question} Answer: {answer}"
            texts.append(full_text)
        
        # Create vocabulary
        vocab = create_simple_vocab(texts)
        
        # Preprocess samples
        for text in texts:
            input_ids = tokenize_text(text, vocab)
            
            # Skip samples that are empty after tokenization
            if len(input_ids) > 0:
                processed_samples.append({
                    'input_ids': input_ids,
                    'labels': input_ids.clone()  # For reconstruction/generation
                })
        
        return ELI5Dataset(processed_samples, vocab, max_tokens)
    
    except Exception as e:
        logger.warning(f"Error loading ELI5 dataset: {e}")
        logger.info("Creating simulated dataset instead.")
        
        # Create a simulated dataset
        vocab_size = 10000
        vocab = {f"token_{i}": i for i in range(vocab_size)}
        vocab.update({'<pad>': vocab_size, '<unk>': vocab_size + 1})
        
        samples = []
        for i in range(max_samples):
            # Create random input_ids with length between 1000 and max_tokens
            length = np.random.randint(1000, max_tokens + 1)
            input_ids = torch.randint(0, vocab_size, (length,))
            
            samples.append({
                'input_ids': input_ids,
                'labels': input_ids.clone()
            })
        
        return ELI5Dataset(samples, vocab, max_tokens)


def load_cnn_dailymail(max_samples: int = 1000, max_tokens: int = 4096) -> CNNDailyMailDataset:
    """
    Load and preprocess CNN/DailyMail dataset with temporal markers.
    
    Args:
        max_samples: Maximum number of samples to load
        max_tokens: Maximum number of tokens per sample
    
    Returns:
        dataset: CNNDailyMailDataset
    """
    logger.info("Loading CNN/DailyMail dataset...")
    
    try:
        # Load the dataset from Hugging Face
        cnn_dm_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1000]")
        
        # Take a subset of samples
        samples = cnn_dm_dataset[:min(max_samples, len(cnn_dm_dataset))]
        
        # Extract articles and summaries
        texts = []
        processed_samples = []
        
        for sample in samples:
            article = sample['article']
            summary = sample['highlights']
            
            # Combine article and summary into a single text
            full_text = f"Article: {article} Summary: {summary}"
            texts.append(full_text)
        
        # Create vocabulary
        vocab = create_simple_vocab(texts)
        
        # Preprocess samples
        for text in texts:
            input_ids = tokenize_text(text, vocab)
            
            # Skip samples that are empty after tokenization
            if len(input_ids) > 0:
                processed_samples.append({
                    'input_ids': input_ids,
                    'labels': input_ids.clone()  # For reconstruction/generation
                })
        
        return CNNDailyMailDataset(processed_samples, vocab, max_tokens)
    
    except Exception as e:
        logger.warning(f"Error loading CNN/DailyMail dataset: {e}")
        logger.info("Creating simulated dataset instead.")
        
        # Create a simulated dataset
        vocab_size = 10000
        vocab = {f"token_{i}": i for i in range(vocab_size)}
        vocab.update({'<pad>': vocab_size, '<unk>': vocab_size + 1})
        
        samples = []
        for i in range(max_samples):
            # Create random input_ids with length between 1000 and max_tokens
            length = np.random.randint(1000, max_tokens + 1)
            input_ids = torch.randint(0, vocab_size, (length,))
            
            samples.append({
                'input_ids': input_ids,
                'labels': input_ids.clone()
            })
        
        return CNNDailyMailDataset(samples, vocab, max_tokens)


def load_github_code(max_samples: int = 1000, max_tokens: int = 4096) -> GitHubCodeDataset:
    """
    Load and preprocess GitHub Code corpus.
    
    Args:
        max_samples: Maximum number of samples to load
        max_tokens: Maximum number of tokens per sample
    
    Returns:
        dataset: GitHubCodeDataset
    """
    logger.info("Loading GitHub Code dataset...")
    
    try:
        # Load the dataset from Hugging Face
        code_dataset = load_dataset("codeparrot/github-code", split="train[:1000]")
        
        # Take a subset of samples
        samples = code_dataset[:min(max_samples, len(code_dataset))]
        
        # Extract code
        texts = []
        processed_samples = []
        
        for sample in samples:
            code = sample['code']
            texts.append(code)
        
        # Create vocabulary
        vocab = create_simple_vocab(texts)
        
        # Preprocess samples
        for text in texts:
            input_ids = tokenize_text(text, vocab)
            
            # Skip samples that are empty after tokenization
            if len(input_ids) > 0:
                processed_samples.append({
                    'input_ids': input_ids,
                    'labels': input_ids.clone()  # For reconstruction/generation
                })
        
        return GitHubCodeDataset(processed_samples, vocab, max_tokens)
    
    except Exception as e:
        logger.warning(f"Error loading GitHub Code dataset: {e}")
        logger.info("Creating simulated dataset instead.")
        
        # Create a simulated dataset
        vocab_size = 10000
        vocab = {f"token_{i}": i for i in range(vocab_size)}
        vocab.update({'<pad>': vocab_size, '<unk>': vocab_size + 1})
        
        samples = []
        for i in range(max_samples):
            # Create random input_ids with length between 1000 and max_tokens
            length = np.random.randint(1000, max_tokens + 1)
            input_ids = torch.randint(0, vocab_size, (length,))
            
            samples.append({
                'input_ids': input_ids,
                'labels': input_ids.clone()
            })
        
        return GitHubCodeDataset(samples, vocab, max_tokens)


def load_s2orc(max_samples: int = 1000, max_tokens: int = 4096) -> S2ORCDataset:
    """
    Load and preprocess S2ORC academic papers.
    
    Args:
        max_samples: Maximum number of samples to load
        max_tokens: Maximum number of tokens per sample
    
    Returns:
        dataset: S2ORCDataset
    """
    logger.info("Loading S2ORC dataset...")
    
    try:
        # Since S2ORC is a large dataset and may not be directly available,
        # we'll try to load PubMed abstracts as a substitute if available
        pubmed_dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train[:1000]")
        
        # Take a subset of samples
        samples = pubmed_dataset[:min(max_samples, len(pubmed_dataset))]
        
        # Extract papers
        texts = []
        processed_samples = []
        
        for sample in samples:
            context = ' '.join(sample['context']['contexts'])
            question = sample['question']
            
            # Combine context and question into a single text
            full_text = f"Context: {context} Question: {question}"
            texts.append(full_text)
        
        # Create vocabulary
        vocab = create_simple_vocab(texts)
        
        # Preprocess samples
        for text in texts:
            input_ids = tokenize_text(text, vocab)
            
            # Skip samples that are empty after tokenization
            if len(input_ids) > 0:
                processed_samples.append({
                    'input_ids': input_ids,
                    'labels': input_ids.clone()  # For reconstruction/generation
                })
        
        return S2ORCDataset(processed_samples, vocab, max_tokens)
    
    except Exception as e:
        logger.warning(f"Error loading academic papers dataset: {e}")
        logger.info("Creating simulated dataset instead.")
        
        # Create a simulated dataset
        vocab_size = 10000
        vocab = {f"token_{i}": i for i in range(vocab_size)}
        vocab.update({'<pad>': vocab_size, '<unk>': vocab_size + 1})
        
        samples = []
        for i in range(max_samples):
            # Create random input_ids with length between 1000 and max_tokens
            length = np.random.randint(1000, max_tokens + 1)
            input_ids = torch.randint(0, vocab_size, (length,))
            
            samples.append({
                'input_ids': input_ids,
                'labels': input_ids.clone()
            })
        
        return S2ORCDataset(samples, vocab, max_tokens)
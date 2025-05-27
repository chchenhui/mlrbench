#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processing functionality for the Attribution-Guided Training experiments.
This module handles dataset loading, preprocessing, and preparation.
"""

import os
import json
import logging
import random
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AttributionDataset(Dataset):
    """Dataset class for attribution experiments.
    
    Each example has text and source attribution information.
    """
    
    def __init__(
        self, 
        texts: List[str], 
        source_ids: List[int],
        tokenizer,
        max_length: int = 512,
        source_metadata: Optional[Dict[int, Dict]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text examples
            source_ids: List of source identifiers for each example
            tokenizer: Tokenizer to use for encoding texts
            max_length: Maximum sequence length
            source_metadata: Optional metadata for each source (author, license, etc.)
        """
        self.texts = texts
        self.source_ids = source_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_metadata = source_metadata or {}
        
        # Ensure source_ids are 0-indexed for embedding lookup
        self.unique_sources = sorted(list(set(source_ids)))
        self.source_to_idx = {src: idx for idx, src in enumerate(self.unique_sources)}
        self.idx_to_source = {idx: src for src, idx in self.source_to_idx.items()}
        
        # Convert source_ids to indices
        self.source_indices = [self.source_to_idx[src] for src in source_ids]
        self.num_sources = len(self.unique_sources)
        
        logger.info(f"Created dataset with {len(texts)} examples from {self.num_sources} sources")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        source_idx = self.source_indices[idx]
        source_id = self.source_ids[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Remove the batch dimension added by the tokenizer
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add source information
        encoding['source_idx'] = torch.tensor(source_idx, dtype=torch.long)
        encoding['source_id'] = source_id
        
        return encoding

def load_gutenberg_dataset(tokenizer, max_length: int = 512, dataset_size: int = 5000):
    """
    Load and preprocess the Project Gutenberg dataset.
    Each book is treated as a separate source.
    
    Args:
        tokenizer: Tokenizer to use for encoding texts
        max_length: Maximum sequence length
        dataset_size: Number of examples to include (for debugging/testing)
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    logger.info("Loading Gutenberg dataset")
    
    # Load a subset of Project Gutenberg books
    dataset = load_dataset("gutenberg", split="train", streaming=False)
    
    # Filter English texts and take a manageable sample
    english_books = [item for item in dataset if item["language"] == "en"]
    
    if len(english_books) > dataset_size:
        logger.info(f"Sampling {dataset_size} books from {len(english_books)} available")
        english_books = random.sample(english_books, k=min(dataset_size, len(english_books)))
    
    # Extract metadata for each source
    source_metadata = {}
    texts = []
    source_ids = []
    
    for book_idx, book in enumerate(english_books):
        source_id = book_idx
        source_metadata[source_id] = {
            "title": book.get("title", "Unknown"),
            "author": book.get("author", "Unknown"),
            "license": "Public Domain",
            "publication_date": book.get("publication_date", "Unknown")
        }
        
        # Split book into chunks of approximately 1000 characters
        text = book["text"]
        if text and len(text) > 1000:
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            # Take up to 10 chunks per book to keep dataset balanced
            for chunk in chunks[:10]:
                if len(chunk.strip()) > 100:  # Only include non-empty chunks
                    texts.append(chunk)
                    source_ids.append(source_id)
    
    logger.info(f"Extracted {len(texts)} text chunks from {len(source_metadata)} books")
    
    # Split into train, validation, and test
    indices = list(range(len(texts)))
    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
    
    # Create datasets
    train_dataset = AttributionDataset(
        [texts[i] for i in train_indices],
        [source_ids[i] for i in train_indices],
        tokenizer,
        max_length,
        source_metadata
    )
    
    val_dataset = AttributionDataset(
        [texts[i] for i in val_indices],
        [source_ids[i] for i in val_indices],
        tokenizer,
        max_length,
        source_metadata
    )
    
    test_dataset = AttributionDataset(
        [texts[i] for i in test_indices],
        [source_ids[i] for i in test_indices],
        tokenizer,
        max_length,
        source_metadata
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, source_metadata

def load_scientific_papers_dataset(tokenizer, max_length: int = 512, dataset_size: int = 5000):
    """
    Load and preprocess a scientific papers dataset.
    Each paper is treated as a separate source.
    
    Args:
        tokenizer: Tokenizer to use for encoding texts
        max_length: Maximum sequence length
        dataset_size: Number of examples to include (for debugging/testing)
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    logger.info("Loading scientific papers dataset")
    
    # Load arXiv papers dataset
    try:
        dataset = load_dataset("arxiv_dataset", split="train", streaming=False)
    except Exception as e:
        logger.warning(f"Failed to load arxiv_dataset: {e}")
        logger.info("Falling back to ccdv/arxiv-summarization")
        try:
            dataset = load_dataset("ccdv/arxiv-summarization", split="train", streaming=False)
        except Exception as e:
            logger.warning(f"Failed to load ccdv/arxiv-summarization: {e}")
            logger.info("Falling back to abstract_datasets")
            dataset = load_dataset("brentrichards/abstract_datasets", "arxiv", split="train", streaming=False)
    
    # Take a random sample to keep dataset size manageable
    if len(dataset) > dataset_size:
        logger.info(f"Sampling {dataset_size} papers from {len(dataset)} available")
        indices = random.sample(range(len(dataset)), k=dataset_size)
        dataset = dataset.select(indices)
    
    # Extract metadata for each source
    source_metadata = {}
    texts = []
    source_ids = []
    
    for paper_idx, paper in enumerate(dataset):
        source_id = paper_idx
        
        # Extract metadata depending on dataset structure
        if "title" in paper:
            title = paper["title"]
        elif "article_title" in paper:
            title = paper["article_title"]
        else:
            title = "Unknown Title"
            
        if "authors" in paper:
            author = paper["authors"] if isinstance(paper["authors"], str) else str(paper["authors"])
        else:
            author = "Unknown Authors"
            
        if "abstract" in paper:
            text = paper["abstract"]
        elif "summary" in paper:
            text = paper["summary"]
        elif "article_abstract" in paper:
            text = paper["article_abstract"]
        else:
            # Skip if no usable text
            continue
            
        source_metadata[source_id] = {
            "title": title,
            "author": author,
            "license": "Academic",
            "publication_date": paper.get("update_date", "Unknown")
        }
        
        # Include the text if it's not empty
        if text and len(text.strip()) > 100:
            texts.append(text)
            source_ids.append(source_id)
    
    logger.info(f"Extracted {len(texts)} abstracts from {len(source_metadata)} papers")
    
    # Split into train, validation, and test
    indices = list(range(len(texts)))
    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
    
    # Create datasets
    train_dataset = AttributionDataset(
        [texts[i] for i in train_indices],
        [source_ids[i] for i in train_indices],
        tokenizer,
        max_length,
        source_metadata
    )
    
    val_dataset = AttributionDataset(
        [texts[i] for i in val_indices],
        [source_ids[i] for i in val_indices],
        tokenizer,
        max_length,
        source_metadata
    )
    
    test_dataset = AttributionDataset(
        [texts[i] for i in test_indices],
        [source_ids[i] for i in test_indices],
        tokenizer,
        max_length,
        source_metadata
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, source_metadata

def create_combined_dataset(tokenizer, max_length: int = 512, dataset_size: int = 5000):
    """
    Create a combined dataset from multiple sources.
    
    Args:
        tokenizer: Tokenizer to use for encoding texts
        max_length: Maximum sequence length
        dataset_size: Target number of examples to include per dataset type
        
    Returns:
        Combined train, validation, and test datasets
    """
    logger.info("Creating combined dataset")
    
    # Load component datasets (using smaller sizes to keep total manageable)
    component_size = dataset_size // 2
    gutenberg_train, gutenberg_val, gutenberg_test, gutenberg_metadata = load_gutenberg_dataset(
        tokenizer, max_length, component_size
    )
    scientific_train, scientific_val, scientific_test, scientific_metadata = load_scientific_papers_dataset(
        tokenizer, max_length, component_size
    )
    
    # Merge source metadata, offsetting scientific source IDs to avoid collisions
    offset = max(gutenberg_metadata.keys()) + 1 if gutenberg_metadata else 0
    
    # Combine datasets
    combined_train_texts = gutenberg_train.texts + scientific_train.texts
    combined_train_sources = gutenberg_train.source_ids + [s + offset for s in scientific_train.source_ids]
    
    combined_val_texts = gutenberg_val.texts + scientific_val.texts
    combined_val_sources = gutenberg_val.source_ids + [s + offset for s in scientific_val.source_ids]
    
    combined_test_texts = gutenberg_test.texts + scientific_test.texts
    combined_test_sources = gutenberg_test.source_ids + [s + offset for s in scientific_test.source_ids]
    
    # Combine metadata
    combined_metadata = gutenberg_metadata.copy()
    for src_id, metadata in scientific_metadata.items():
        combined_metadata[src_id + offset] = metadata
    
    # Create combined datasets
    combined_train = AttributionDataset(
        combined_train_texts,
        combined_train_sources,
        tokenizer,
        max_length,
        combined_metadata
    )
    
    combined_val = AttributionDataset(
        combined_val_texts,
        combined_val_sources,
        tokenizer,
        max_length,
        combined_metadata
    )
    
    combined_test = AttributionDataset(
        combined_test_texts,
        combined_test_sources,
        tokenizer,
        max_length,
        combined_metadata
    )
    
    logger.info(f"Combined - Train: {len(combined_train)}, Val: {len(combined_val)}, Test: {len(combined_test)}")
    logger.info(f"Number of unique sources: {combined_train.num_sources}")
    
    return combined_train, combined_val, combined_test, combined_metadata

def create_adversarial_test_set(test_dataset, tokenizer, max_length=512):
    """
    Create an adversarial test set with paraphrased content to challenge 
    the attribution mechanism.
    
    Args:
        test_dataset: Original test dataset
        tokenizer: Tokenizer to use for encoding texts
        max_length: Maximum sequence length
        
    Returns:
        Adversarial test dataset
    """
    logger.info("Creating adversarial test set")
    
    # Define simple paraphrasing rules (this is a simplified approach;
    # in a real implementation, we would use a more sophisticated paraphrasing model)
    def simple_paraphrase(text):
        # Replace some common words with synonyms
        replacements = {
            "small": "little",
            "big": "large",
            "fast": "quick",
            "slow": "gradual",
            "good": "excellent",
            "bad": "poor",
            "happy": "joyful",
            "sad": "unhappy",
            "important": "significant",
            "problem": "issue",
            "result": "outcome",
            "method": "approach",
            "use": "utilize",
            "create": "develop",
            "find": "discover",
            "show": "demonstrate"
        }
        
        paraphrased = text
        for original, replacement in replacements.items():
            # Replace with 50% probability to maintain some original text
            if random.random() > 0.5:
                paraphrased = paraphrased.replace(f" {original} ", f" {replacement} ")
                
        return paraphrased
    
    # Paraphrase the texts while keeping source information
    adversarial_texts = [simple_paraphrase(text) for text in test_dataset.texts]
    
    # Create adversarial dataset with same source information
    adversarial_dataset = AttributionDataset(
        adversarial_texts,
        test_dataset.source_ids,
        tokenizer,
        max_length,
        test_dataset.source_metadata
    )
    
    logger.info(f"Created adversarial test set with {len(adversarial_dataset)} examples")
    
    return adversarial_dataset

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    """
    Create DataLoaders for the datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for DataLoaders
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader

def save_metadata(metadata, save_path):
    """
    Save source metadata to disk.
    
    Args:
        metadata: Source metadata dictionary
        save_path: Path to save the metadata
    """
    # Convert source IDs to strings for JSON serialization
    json_metadata = {str(k): v for k, v in metadata.items()}
    
    with open(save_path, 'w') as f:
        json.dump(json_metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {save_path}")

def load_metadata(load_path):
    """
    Load source metadata from disk.
    
    Args:
        load_path: Path to load the metadata from
        
    Returns:
        Source metadata dictionary
    """
    with open(load_path, 'r') as f:
        json_metadata = json.load(f)
    
    # Convert source IDs back to integers
    metadata = {int(k): v for k, v in json_metadata.items()}
    
    logger.info(f"Loaded metadata from {load_path}")
    
    return metadata

def prepare_datasets(model_name, max_length=512, batch_size=16, dataset_size=5000):
    """
    Prepare all datasets for experiments.
    
    Args:
        model_name: Name of the pretrained model to use (for tokenizer)
        max_length: Maximum sequence length
        batch_size: Batch size for DataLoaders
        dataset_size: Number of examples to include (per dataset type)
        
    Returns:
        Dictionary with all prepared datasets and dataloaders
    """
    # Create output directory if it doesn't exist
    data_dir = os.path.join("data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create combined dataset
    train_dataset, val_dataset, test_dataset, source_metadata = create_combined_dataset(
        tokenizer, max_length, dataset_size
    )
    
    # Create adversarial test set
    adversarial_dataset = create_adversarial_test_set(test_dataset, tokenizer, max_length)
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )
    adversarial_loader = DataLoader(
        adversarial_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Save metadata
    metadata_path = os.path.join(data_dir, "source_metadata.json")
    save_metadata(source_metadata, metadata_path)
    
    # Save dataset statistics
    stats = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "adversarial_size": len(adversarial_dataset),
        "num_sources": train_dataset.num_sources,
        "model_name": model_name,
        "max_length": max_length,
        "batch_size": batch_size
    }
    
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved dataset statistics to {stats_path}")
    
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "adversarial_dataset": adversarial_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "adversarial_loader": adversarial_loader,
        "tokenizer": tokenizer,
        "source_metadata": source_metadata,
        "stats": stats
    }

if __name__ == "__main__":
    # Test data processing
    datasets = prepare_datasets(
        model_name="distilroberta-base",  # Smaller model for testing
        max_length=128,
        batch_size=16,
        dataset_size=1000  # Small size for testing
    )
    
    logger.info("Data processing test complete")
    logger.info(f"Train dataset size: {len(datasets['train_dataset'])}")
    logger.info(f"Number of sources: {datasets['train_dataset'].num_sources}")
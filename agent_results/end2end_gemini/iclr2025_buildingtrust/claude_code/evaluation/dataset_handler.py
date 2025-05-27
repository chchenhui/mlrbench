"""
Dataset Handler Module

This module handles downloading, preprocessing, and managing evaluation datasets.
"""

import os
import json
import logging
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from datasets import load_dataset

from ..utils.logging_utils import timeit
from ..utils.data_utils import (
    download_huggingface_dataset,
    prepare_hotpotqa_samples,
    prepare_gsm8k_samples,
    prepare_strategyqa_samples,
    split_dataset,
    get_dataset_stats
)

logger = logging.getLogger(__name__)

class DatasetHandler:
    """
    Class to handle dataset operations for concept graph experiments.
    
    This class provides functionality for:
    1. Downloading and preprocessing datasets
    2. Splitting data into train/val/test sets
    3. Analyzing and summarizing dataset statistics
    4. Managing data storage and retrieval
    """
    
    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize the DatasetHandler.
        
        Args:
            data_dir: Directory to store datasets
            cache_dir: Optional directory to cache downloaded datasets
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.seed = seed
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize dataset containers
        self.datasets = {}
        self.splits = {}
        self.stats = {}
    
    @timeit
    def download_hotpotqa(
        self,
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Download and prepare HotpotQA dataset.
        
        Args:
            max_samples: Maximum number of samples to download
            
        Returns:
            Prepared HotpotQA samples
        """
        logger.info(f"Downloading HotpotQA dataset (max_samples={max_samples})")
        
        # Check if already downloaded
        cache_path = os.path.join(self.data_dir, 'hotpotqa.json')
        if os.path.exists(cache_path):
            logger.info(f"Loading HotpotQA from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                samples = json.load(f)
            
            if max_samples and len(samples) > max_samples:
                samples = samples[:max_samples]
            
            return samples
        
        # Download from HuggingFace
        dataset = download_huggingface_dataset('hotpot_qa', 'fullwiki', 'train', self.cache_dir, max_samples)
        
        if dataset is None:
            logger.error("Failed to download HotpotQA dataset")
            return []
        
        # Prepare samples
        samples = prepare_hotpotqa_samples(dataset, max_samples)
        
        # Save to cache
        with open(cache_path, 'w') as f:
            json.dump(samples, f)
        
        logger.info(f"Downloaded and prepared {len(samples)} HotpotQA samples")
        
        # Store in instance
        self.datasets['hotpotqa'] = samples
        
        return samples
    
    @timeit
    def download_gsm8k(
        self,
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Download and prepare GSM8K dataset.
        
        Args:
            max_samples: Maximum number of samples to download
            
        Returns:
            Prepared GSM8K samples
        """
        logger.info(f"Downloading GSM8K dataset (max_samples={max_samples})")
        
        # Check if already downloaded
        cache_path = os.path.join(self.data_dir, 'gsm8k.json')
        if os.path.exists(cache_path):
            logger.info(f"Loading GSM8K from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                samples = json.load(f)
            
            if max_samples and len(samples) > max_samples:
                samples = samples[:max_samples]
            
            return samples
        
        # Download from HuggingFace
        dataset = download_huggingface_dataset('gsm8k', 'main', 'train', self.cache_dir, max_samples)
        
        if dataset is None:
            logger.error("Failed to download GSM8K dataset")
            return []
        
        # Prepare samples
        samples = prepare_gsm8k_samples(dataset, max_samples)
        
        # Save to cache
        with open(cache_path, 'w') as f:
            json.dump(samples, f)
        
        logger.info(f"Downloaded and prepared {len(samples)} GSM8K samples")
        
        # Store in instance
        self.datasets['gsm8k'] = samples
        
        return samples
    
    @timeit
    def download_strategyqa(
        self,
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Download and prepare StrategyQA dataset.
        
        Args:
            max_samples: Maximum number of samples to download
            
        Returns:
            Prepared StrategyQA samples
        """
        logger.info(f"Downloading StrategyQA dataset (max_samples={max_samples})")
        
        # Check if already downloaded
        cache_path = os.path.join(self.data_dir, 'strategyqa.json')
        if os.path.exists(cache_path):
            logger.info(f"Loading StrategyQA from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                samples = json.load(f)
            
            if max_samples and len(samples) > max_samples:
                samples = samples[:max_samples]
            
            return samples
        
        # Download from HuggingFace
        dataset = download_huggingface_dataset('metaeval/strategy-qa', split='train', cache_dir=self.cache_dir)
        
        if dataset is None:
            logger.error("Failed to download StrategyQA dataset")
            return []
        
        # Prepare samples
        samples = prepare_strategyqa_samples(dataset, max_samples)
        
        # Save to cache
        with open(cache_path, 'w') as f:
            json.dump(samples, f)
        
        logger.info(f"Downloaded and prepared {len(samples)} StrategyQA samples")
        
        # Store in instance
        self.datasets['strategyqa'] = samples
        
        return samples
    
    def create_dataset_splits(
        self,
        dataset_name: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create train/val/test splits for a dataset.
        
        Args:
            dataset_name: Name of the dataset to split
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            
        Returns:
            Dictionary with train, val, and test splits
        """
        logger.info(f"Creating dataset splits for {dataset_name}")
        
        if dataset_name not in self.datasets:
            logger.error(f"Dataset {dataset_name} not found")
            return {'train': [], 'val': [], 'test': []}
        
        # Get the dataset
        dataset = self.datasets[dataset_name]
        
        # Create splits
        splits = split_dataset(dataset, train_ratio, val_ratio, test_ratio, self.seed)
        
        # Store splits
        self.splits[dataset_name] = splits
        
        # Log split sizes
        logger.info(f"Created splits for {dataset_name}: "
                   f"train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, "
                   f"test={len(splits['test'])}")
        
        return splits
    
    def analyze_dataset(
        self,
        dataset_name: str
    ) -> Dict[str, Any]:
        """
        Analyze a dataset and compute statistics.
        
        Args:
            dataset_name: Name of the dataset to analyze
            
        Returns:
            Dictionary of dataset statistics
        """
        logger.info(f"Analyzing dataset: {dataset_name}")
        
        if dataset_name not in self.datasets:
            logger.error(f"Dataset {dataset_name} not found")
            return {}
        
        # Get the dataset
        dataset = self.datasets[dataset_name]
        
        # Compute statistics
        stats = get_dataset_stats(dataset)
        
        # If splits exist, compute stats for each split
        if dataset_name in self.splits:
            splits = self.splits[dataset_name]
            stats['splits'] = {
                'train': get_dataset_stats(splits['train']),
                'val': get_dataset_stats(splits['val']),
                'test': get_dataset_stats(splits['test'])
            }
        
        # Store statistics
        self.stats[dataset_name] = stats
        
        return stats
    
    def get_sample_by_id(
        self,
        dataset_name: str,
        sample_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific sample by ID.
        
        Args:
            dataset_name: Name of the dataset
            sample_id: ID of the sample to retrieve
            
        Returns:
            Sample dictionary or None if not found
        """
        if dataset_name not in self.datasets:
            logger.error(f"Dataset {dataset_name} not found")
            return None
        
        # Get the dataset
        dataset = self.datasets[dataset_name]
        
        # Find the sample
        for sample in dataset:
            if sample.get('id') == sample_id:
                return sample
        
        logger.warning(f"Sample {sample_id} not found in {dataset_name}")
        return None
    
    def get_random_samples(
        self,
        dataset_name: str,
        num_samples: int,
        split: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get random samples from a dataset.
        
        Args:
            dataset_name: Name of the dataset
            num_samples: Number of samples to retrieve
            split: Optional split to sample from ('train', 'val', 'test')
            
        Returns:
            List of random samples
        """
        # Determine the source dataset
        if split and dataset_name in self.splits:
            if split in self.splits[dataset_name]:
                source = self.splits[dataset_name][split]
            else:
                logger.error(f"Split {split} not found for {dataset_name}")
                return []
        elif dataset_name in self.datasets:
            source = self.datasets[dataset_name]
        else:
            logger.error(f"Dataset {dataset_name} not found")
            return []
        
        # Sample random indices
        indices = random.sample(range(len(source)), min(num_samples, len(source)))
        
        # Get the samples
        return [source[i] for i in indices]
    
    def save_all_datasets(self) -> None:
        """Save all datasets to disk."""
        logger.info("Saving all datasets to disk")
        
        for name, dataset in self.datasets.items():
            # Save dataset
            dataset_path = os.path.join(self.data_dir, f"{name}.json")
            with open(dataset_path, 'w') as f:
                json.dump(dataset, f)
            
            logger.info(f"Saved dataset {name} to {dataset_path}")
            
            # Save splits if they exist
            if name in self.splits:
                splits_dir = os.path.join(self.data_dir, f"{name}_splits")
                os.makedirs(splits_dir, exist_ok=True)
                
                for split_name, split_data in self.splits[name].items():
                    split_path = os.path.join(splits_dir, f"{split_name}.json")
                    with open(split_path, 'w') as f:
                        json.dump(split_data, f)
                
                logger.info(f"Saved splits for {name} to {splits_dir}")
            
            # Save stats if they exist
            if name in self.stats:
                stats_path = os.path.join(self.data_dir, f"{name}_stats.json")
                with open(stats_path, 'w') as f:
                    json.dump(self.stats[name], f)
                
                logger.info(f"Saved stats for {name} to {stats_path}")
    
    def load_all_datasets(self) -> None:
        """Load all datasets from disk."""
        logger.info("Loading all datasets from disk")
        
        # Find all dataset files
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json') and not filename.endswith('_stats.json'):
                dataset_name = filename.split('.')[0]
                dataset_path = os.path.join(self.data_dir, filename)
                
                # Load dataset
                with open(dataset_path, 'r') as f:
                    dataset = json.load(f)
                
                self.datasets[dataset_name] = dataset
                logger.info(f"Loaded dataset {dataset_name} with {len(dataset)} samples")
                
                # Load splits if they exist
                splits_dir = os.path.join(self.data_dir, f"{dataset_name}_splits")
                if os.path.exists(splits_dir):
                    splits = {}
                    
                    for split_name in ['train', 'val', 'test']:
                        split_path = os.path.join(splits_dir, f"{split_name}.json")
                        if os.path.exists(split_path):
                            with open(split_path, 'r') as f:
                                splits[split_name] = json.load(f)
                    
                    self.splits[dataset_name] = splits
                    logger.info(f"Loaded splits for {dataset_name}")
                
                # Load stats if they exist
                stats_path = os.path.join(self.data_dir, f"{dataset_name}_stats.json")
                if os.path.exists(stats_path):
                    with open(stats_path, 'r') as f:
                        self.stats[dataset_name] = json.load(f)
                    
                    logger.info(f"Loaded stats for {dataset_name}")
    
    def get_datasets_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all loaded datasets.
        
        Returns:
            Dictionary with dataset summaries
        """
        summary = {}
        
        for name, dataset in self.datasets.items():
            dataset_summary = {
                'num_samples': len(dataset),
                'has_splits': name in self.splits,
                'has_stats': name in self.stats
            }
            
            if name in self.splits:
                splits = self.splits[name]
                dataset_summary['split_sizes'] = {
                    split: len(data) for split, data in splits.items()
                }
            
            if name in self.stats:
                # Include some basic stats, but not everything to keep it concise
                stats = self.stats[name]
                dataset_summary['stats'] = {
                    'avg_question_length': stats.get('avg_question_length', None),
                    'avg_answer_length': stats.get('avg_answer_length', None)
                }
            
            summary[name] = dataset_summary
        
        return summary
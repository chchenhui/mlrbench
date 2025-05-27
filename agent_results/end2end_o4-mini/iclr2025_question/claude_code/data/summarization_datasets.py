"""
Summarization dataset loading utilities for the SCEC experiments.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SummarizationDatasetLoader:
    """
    Loader for abstractive summarization datasets from the Hugging Face datasets library.
    
    Supported datasets:
    - XSum
    - CNN/DailyMail
    - PubMed
    """
    
    SUPPORTED_DATASETS = {
        "xsum": {
            "hf_name": "xsum",
            "split_mapping": {"train": "train", "validation": "validation", "test": "test[:500]"},
            "document_column": "document",
            "summary_column": "summary",
        },
        "cnn_dailymail": {
            "hf_name": "cnn_dailymail",
            "config": "3.0.0",
            "split_mapping": {"train": "train", "validation": "validation", "test": "test[:500]"},
            "document_column": "article",
            "summary_column": "highlights",
        },
        "pubmed": {
            "hf_name": "ccdv/pubmed-summarization",
            "split_mapping": {"train": "train", "validation": "validation", "test": "test[:500]"},
            "document_column": "article",
            "summary_column": "abstract",
        }
    }
    
    def __init__(
        self, 
        dataset_name: str, 
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        max_document_length: Optional[int] = 1024,
        seed: int = 42,
    ):
        """
        Initialize the summarization dataset loader.
        
        Args:
            dataset_name: Name of the dataset to load ('xsum', 'cnn_dailymail', 'pubmed')
            cache_dir: Directory to cache the dataset
            max_samples: Maximum number of samples to load per split (for debugging)
            max_document_length: Maximum document length in tokens to keep
            seed: Random seed for reproducibility
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(self.SUPPORTED_DATASETS.keys())}")
        
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.max_document_length = max_document_length
        self.seed = seed
        self.dataset_info = self.SUPPORTED_DATASETS[dataset_name]
        self.dataset = None
        
        np.random.seed(seed)
    
    def load(self) -> DatasetDict:
        """Load and preprocess the dataset."""
        logger.info(f"Loading {self.dataset_name} dataset...")
        
        if "config" in self.dataset_info:
            raw_dataset = load_dataset(
                self.dataset_info["hf_name"],
                self.dataset_info["config"],
                cache_dir=self.cache_dir,
            )
        else:
            raw_dataset = load_dataset(
                self.dataset_info["hf_name"],
                cache_dir=self.cache_dir,
            )
        
        # Create a new DatasetDict with our custom splits
        dataset = DatasetDict()
        for split_name, hf_split in self.dataset_info["split_mapping"].items():
            if ":" in hf_split:  # Handle slices
                split_parts = hf_split.split(":")
                base_split = split_parts[0]
                slice_str = split_parts[1].strip("[]")
                if slice_str:
                    slice_val = int(slice_str)
                    split_data = raw_dataset[base_split].select(range(slice_val))
                else:
                    split_data = raw_dataset[base_split]
            else:
                split_data = raw_dataset[hf_split]
            
            # Apply max_samples if specified
            if self.max_samples and len(split_data) > self.max_samples:
                indices = np.random.choice(len(split_data), self.max_samples, replace=False)
                split_data = split_data.select(indices)
            
            dataset[split_name] = self._preprocess_split(split_data)
        
        self.dataset = dataset
        return dataset
    
    def _preprocess_split(self, split: Dataset) -> Dataset:
        """Preprocess a dataset split to standardize the format."""
        document_col = self.dataset_info["document_column"]
        summary_col = self.dataset_info["summary_column"]
        
        # Function to preprocess a single example
        def preprocess_example(example):
            # Get the document and summary
            document = example[document_col]
            summary = example[summary_col]
            
            # Truncate document if needed
            if self.max_document_length and len(document.split()) > self.max_document_length:
                document = " ".join(document.split()[:self.max_document_length])
            
            # Clean and format the data
            if self.dataset_name == "cnn_dailymail":
                # For CNN/DailyMail, clean up the highlights format
                summary = summary.replace("\n", " ").strip()
            
            return {
                "document": document,
                "summary": summary,
                "id": example.get("id", f"{self.dataset_name}-{np.random.randint(0, 100000)}")
            }
            
        # Apply preprocessing to all examples
        mapped_split = split.map(
            preprocess_example,
            remove_columns=split.column_names,
            desc=f"Preprocessing {self.dataset_name}"
        )
        
        return mapped_split
    
    def save_to_json(self, output_dir: str):
        """Save the preprocessed dataset to JSON files."""
        if self.dataset is None:
            self.load()
            
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, split_data in self.dataset.items():
            output_file = os.path.join(output_dir, f"{self.dataset_name}_{split_name}.json")
            with open(output_file, 'w') as f:
                json.dump(split_data.to_dict(), f, indent=2)
            logger.info(f"Saved {split_name} split to {output_file}")
    
    def get_sample_prompt(self, index: int = 0, split: str = "test") -> Tuple[str, str]:
        """Get a formatted prompt for a sample in the dataset."""
        if self.dataset is None:
            self.load()
            
        sample = self.dataset[split][index]
        prompt = f"Document: {sample['document']}\n\nSummarize the above document in a concise way:"
        return prompt, sample["summary"]


class SummarizationSubsetDataset:
    """Create a small subset of summarization datasets for faster experimentation."""
    
    def __init__(
        self,
        output_dir: str,
        num_samples: int = 50,
        datasets: List[str] = ["xsum", "cnn_dailymail"],
        max_document_length: int = 1024,
        seed: int = 42,
    ):
        """
        Initialize the summarization subset creator.
        
        Args:
            output_dir: Directory to save the subset dataset
            num_samples: Number of samples per dataset to include in the subset
            datasets: List of dataset names to include
            max_document_length: Maximum document length in tokens to keep
            seed: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.datasets = datasets
        self.max_document_length = max_document_length
        self.seed = seed
        
        os.makedirs(output_dir, exist_ok=True)
        np.random.seed(seed)
    
    def create_subsets(self):
        """Create subsets of the specified datasets."""
        all_samples = []
        
        for dataset_name in self.datasets:
            logger.info(f"Creating subset for {dataset_name}")
            loader = SummarizationDatasetLoader(
                dataset_name, 
                max_samples=self.num_samples, 
                max_document_length=self.max_document_length,
                seed=self.seed
            )
            dataset = loader.load()
            
            # Add dataset samples to the combined dataset
            for sample in dataset["test"]:
                sample["dataset"] = dataset_name
                all_samples.append(sample)
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(all_samples)
        
        # Save the combined subset
        output_file = os.path.join(self.output_dir, "summarization_subset.json")
        df.to_json(output_file, orient="records", indent=2)
        logger.info(f"Saved combined summarization subset with {len(df)} samples to {output_file}")
        
        return df


def load_summarization_subset(path: str) -> List[Dict]:
    """Load a summarization subset from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    output_dir = "data/summarization_subsets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create small subsets for experimentation
    subset_creator = SummarizationSubsetDataset(
        output_dir, 
        num_samples=50,
        max_document_length=1024
    )
    subset_data = subset_creator.create_subsets()
    
    print(f"Created summarization subset with {len(subset_data)} samples")
    # Sample document from the subset
    print(f"Sample document (truncated): {subset_data.iloc[0]['document'][:100]}...")
    print(f"Sample summary: {subset_data.iloc[0]['summary']}")
"""
QA dataset loading utilities for the SCEC experiments.
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

class QADatasetLoader:
    """
    Loader for question answering datasets from the Hugging Face datasets library.
    
    Supported datasets:
    - Natural Questions
    - TriviaQA
    - WebQuestions
    """
    
    SUPPORTED_DATASETS = {
        "natural_questions": {
            "hf_name": "natural_questions",
            "split_mapping": {"train": "train", "validation": "validation", "test": "validation[:500]"},
            "question_column": "question",
            "answer_column": "answer",
        },
        "trivia_qa": {
            "hf_name": "trivia_qa",
            "config": "rc.nocontext",
            "split_mapping": {"train": "train", "validation": "validation", "test": "validation[:500]"},
            "question_column": "question",
            "answer_column": "answer",
        },
        "web_questions": {
            "hf_name": "web_questions",
            "split_mapping": {"train": "train", "validation": "test", "test": "test[:500]"},
            "question_column": "question",
            "answer_column": "answers",
        }
    }
    
    def __init__(
        self, 
        dataset_name: str, 
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize the QA dataset loader.
        
        Args:
            dataset_name: Name of the dataset to load ('natural_questions', 'trivia_qa', 'web_questions')
            cache_dir: Directory to cache the dataset
            max_samples: Maximum number of samples to load per split (for debugging)
            seed: Random seed for reproducibility
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(self.SUPPORTED_DATASETS.keys())}")
        
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.max_samples = max_samples
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
        question_col = self.dataset_info["question_column"]
        answer_col = self.dataset_info["answer_column"]
        
        # Function to preprocess a single example
        def preprocess_example(example):
            # Get the question
            question = example[question_col]
            
            # Handle different answer formats
            if self.dataset_name == "natural_questions":
                # For Natural Questions, we need to extract short answers
                short_answers = example["annotations"]["short_answers"][0]
                if short_answers:
                    answers = [example["document"]["tokens"][a["start"]:a["end"]] 
                              for a in short_answers]
                    answers = [" ".join(tokens) for tokens in answers]
                else:
                    answers = []
            elif self.dataset_name == "trivia_qa":
                # For TriviaQA, we use the answer aliases
                answers = example["answer"]["aliases"]
            else:
                # For other datasets, use the answer column directly
                if isinstance(example[answer_col], list):
                    answers = example[answer_col]
                else:
                    answers = [example[answer_col]]
            
            return {
                "question": question,
                "answers": answers,
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
    
    def get_sample_prompt(self, index: int = 0, split: str = "test") -> str:
        """Get a formatted prompt for a sample in the dataset."""
        if self.dataset is None:
            self.load()
            
        sample = self.dataset[split][index]
        prompt = f"Question: {sample['question']}\nAnswer:"
        return prompt, sample["answers"]


class QASubsetDataset:
    """Create a small subset of QA datasets for faster experimentation."""
    
    def __init__(
        self,
        output_dir: str,
        num_samples: int = 50,
        datasets: List[str] = ["natural_questions", "trivia_qa", "web_questions"],
        seed: int = 42,
    ):
        """
        Initialize the QA subset creator.
        
        Args:
            output_dir: Directory to save the subset dataset
            num_samples: Number of samples per dataset to include in the subset
            datasets: List of dataset names to include
            seed: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.datasets = datasets
        self.seed = seed
        
        os.makedirs(output_dir, exist_ok=True)
        np.random.seed(seed)
    
    def create_subsets(self):
        """Create subsets of the specified datasets."""
        all_samples = []
        
        for dataset_name in self.datasets:
            logger.info(f"Creating subset for {dataset_name}")
            loader = QADatasetLoader(dataset_name, max_samples=self.num_samples, seed=self.seed)
            dataset = loader.load()
            
            # Add dataset samples to the combined dataset
            for sample in dataset["test"]:
                sample["dataset"] = dataset_name
                all_samples.append(sample)
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(all_samples)
        
        # Save the combined subset
        output_file = os.path.join(self.output_dir, "qa_subset.json")
        df.to_json(output_file, orient="records", indent=2)
        logger.info(f"Saved combined QA subset with {len(df)} samples to {output_file}")
        
        return df


def load_qa_subset(path: str) -> List[Dict]:
    """Load a QA subset from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    output_dir = "data/qa_subsets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create small subsets for experimentation
    subset_creator = QASubsetDataset(output_dir, num_samples=50)
    subset_data = subset_creator.create_subsets()
    
    print(f"Created QA subset with {len(subset_data)} samples")
    # Sample question from the subset
    print(f"Sample question: {subset_data.iloc[0]['question']}")
    print(f"Sample answers: {subset_data.iloc[0]['answers']}")
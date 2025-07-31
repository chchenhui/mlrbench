"""
Data loaders for the Self-Correcting Language Model experiment.
"""
import os
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datasets import load_dataset, Dataset

from config import DATASET_CONFIG, DATA_DIR, EXPERIMENT_CONFIG, logger

class DatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, dataset_name: str, max_samples: Optional[int] = None):
        """
        Initialize dataset loader.
        
        Args:
            dataset_name: Name of the dataset to load
            max_samples: Maximum number of samples to load
        """
        self.dataset_name = dataset_name
        self.max_samples = max_samples or EXPERIMENT_CONFIG["max_samples"]
        self.dataset_config = DATASET_CONFIG.get(dataset_name, {})
        
        # Validate dataset config
        if not self.dataset_config:
            raise ValueError(f"Dataset {dataset_name} not found in config")
        
        # Cache path for processed dataset
        self.cache_path = DATA_DIR / f"{dataset_name}_processed.json"
        
        # Initialize dataset
        self.dataset = None
        self.processed_data = None
        
    def load(self) -> Dataset:
        """Load dataset from Hugging Face."""
        logger.info(f"Loading dataset {self.dataset_name} from Hugging Face")
        try:
            # Extract configuration parameters
            huggingface_id = self.dataset_config["huggingface_id"]
            config = self.dataset_config.get("config", None)
            split = self.dataset_config.get("split", "validation")
            trust_remote_code = self.dataset_config.get("trust_remote_code", False)
            
            # Load the dataset with appropriate parameters
            if config:
                dataset = load_dataset(
                    huggingface_id,
                    config,
                    split=split,
                    trust_remote_code=trust_remote_code
                )
            else:
                dataset = load_dataset(
                    huggingface_id,
                    split=split,
                    trust_remote_code=trust_remote_code
                )
            
            # Limit the number of samples if specified
            if self.max_samples and len(dataset) > self.max_samples:
                dataset = dataset.select(range(self.max_samples))
            
            self.dataset = dataset
            logger.info(f"Loaded {len(dataset)} samples from {self.dataset_name}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise
    
    def process(self) -> List[Dict[str, Any]]:
        """
        Process dataset for the experiment.
        
        Returns:
            Processed data as a list of dictionaries
        """
        raise NotImplementedError("Subclasses must implement process()")
    
    def get_data(self) -> List[Dict[str, Any]]:
        """
        Get processed data.
        
        Returns:
            Processed data as a list of dictionaries
        """
        if self.processed_data is not None:
            return self.processed_data
        
        # Try to load from cache
        if os.path.exists(self.cache_path):
            logger.info(f"Loading processed data from cache: {self.cache_path}")
            try:
                import json
                with open(self.cache_path, 'r') as f:
                    self.processed_data = json.load(f)
                return self.processed_data
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        # If not cached or failed to load, process the data
        if self.dataset is None:
            self.load()
        
        self.processed_data = self.process()
        
        # Cache the processed data
        try:
            import json
            with open(self.cache_path, 'w') as f:
                json.dump(self.processed_data, f, indent=2)
            logger.info(f"Cached processed data to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache processed data: {e}")
        
        return self.processed_data


class TruthfulQALoader(DatasetLoader):
    """Data loader for TruthfulQA dataset."""
    
    def __init__(self, max_samples: Optional[int] = None):
        """
        Initialize TruthfulQA loader.
        
        Args:
            max_samples: Maximum number of samples to load
        """
        super().__init__("truthfulqa", max_samples)
    
    def process(self) -> List[Dict[str, Any]]:
        """
        Process TruthfulQA dataset for the experiment.
        
        Returns:
            Processed data as a list of dictionaries
        """
        if self.dataset is None:
            self.load()
        
        logger.info("Processing TruthfulQA dataset")
        
        processed_data = []
        for idx, sample in enumerate(self.dataset):
            # Extract question, correct answers, and incorrect answers
            question = sample["question"]
            correct_answers = sample["correct_answers"]
            incorrect_answers = sample.get("incorrect_answers", [])
            
            # Process into our standard format
            processed_sample = {
                "id": f"truthfulqa_{idx}",
                "question": question,
                "correct_answers": correct_answers,
                "incorrect_answers": incorrect_answers,
                "context": "",  # No context in TruthfulQA
                "category": sample.get("category", ""),
                "type": "factual_qa"
            }
            processed_data.append(processed_sample)
        
        logger.info(f"Processed {len(processed_data)} samples from TruthfulQA")
        return processed_data


class FEVERLoader(DatasetLoader):
    """Data loader for FEVER dataset."""
    
    def __init__(self, max_samples: Optional[int] = None):
        """
        Initialize FEVER loader.
        
        Args:
            max_samples: Maximum number of samples to load
        """
        super().__init__("fever", max_samples)
    
    def process(self) -> List[Dict[str, Any]]:
        """
        Process FEVER dataset for the experiment.
        
        Returns:
            Processed data as a list of dictionaries
        """
        if self.dataset is None:
            self.load()
        
        logger.info("Processing FEVER dataset")
        
        processed_data = []
        for idx, sample in enumerate(self.dataset):
            # Extract claim, label, and evidence
            claim = sample["claim"]
            label = sample["label"]  # SUPPORTS, REFUTES, or NOT ENOUGH INFO
            
            # Process evidence
            evidence = []
            if "evidence" in sample and isinstance(sample["evidence"], list):
                for ev_group in sample["evidence"]:
                    if isinstance(ev_group, list):
                        for ev in ev_group:
                            if len(ev) >= 5:  # Check if evidence has the expected format
                                doc_id = ev[2]
                                text = ev[3]
                                evidence.append(f"{doc_id}: {text}")
            
            # Process into our standard format
            processed_sample = {
                "id": f"fever_{idx}",
                "question": claim,  # Treating the claim as a question
                "correct_answers": ["True"] if label == "SUPPORTS" else ["False"] if label == "REFUTES" else ["Uncertain"],
                "incorrect_answers": [],
                "context": "\n".join(evidence) if evidence else "",
                "label": label,
                "type": "fact_verification"
            }
            processed_data.append(processed_sample)
        
        logger.info(f"Processed {len(processed_data)} samples from FEVER")
        return processed_data


def get_dataset_loader(dataset_name: str, max_samples: Optional[int] = None) -> DatasetLoader:
    """
    Factory function to get the appropriate dataset loader.
    
    Args:
        dataset_name: Name of the dataset
        max_samples: Maximum number of samples to load
    
    Returns:
        Dataset loader instance
    """
    if dataset_name.lower() == "truthfulqa":
        return TruthfulQALoader(max_samples)
    elif dataset_name.lower() == "fever":
        return FEVERLoader(max_samples)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
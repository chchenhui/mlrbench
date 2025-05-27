"""
Data Utilities

This module provides utilities for data handling, processing, and loading.
"""

import os
import json
import pickle
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datasets import load_dataset

logger = logging.getLogger(__name__)

def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save the JSON file
        indent: Indentation level for pretty printing
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)
    
    logger.debug(f"Saved JSON data to {filepath}")

def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.debug(f"Loaded JSON data from {filepath}")
    return data

def save_pickle(data: Any, filepath: str) -> None:
    """
    Save data to a pickle file.
    
    Args:
        data: Data to save
        filepath: Path to save the pickle file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    logger.debug(f"Saved pickle data to {filepath}")

def load_pickle(filepath: str) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Loaded data
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return None
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    logger.debug(f"Loaded pickle data from {filepath}")
    return data

def download_huggingface_dataset(
    dataset_name: str,
    subset_name: Optional[str] = None,
    split: str = "train",
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Any:
    """
    Download a dataset from the Hugging Face Hub.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        subset_name: Optional subset name
        split: Dataset split to load (e.g., "train", "validation", "test")
        cache_dir: Optional cache directory
        max_samples: Maximum number of samples to load
        
    Returns:
        The downloaded dataset
    """
    logger.info(f"Downloading dataset: {dataset_name}" + 
               (f" (subset: {subset_name})" if subset_name else ""))
    
    try:
        if subset_name:
            dataset = load_dataset(dataset_name, subset_name, split=split, cache_dir=cache_dir)
        else:
            dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            logger.info(f"Limited dataset to {max_samples} samples")
        
        logger.info(f"Successfully loaded dataset with {len(dataset)} samples")
        return dataset
    
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return None

def prepare_hotpotqa_samples(
    dataset: Any,
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Prepare samples from the HotpotQA dataset for concept graph experiments.
    
    Args:
        dataset: HotpotQA dataset from HuggingFace
        max_samples: Maximum number of samples to prepare
        
    Returns:
        List of prepared samples
    """
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    prepared_samples = []
    
    for item in dataset:
        # Format the context from supporting facts
        supporting_facts = []
        for title, sent_id in item['supporting_facts']:
            for context in item['context']:
                if context[0] == title:
                    if sent_id < len(context[1]):
                        supporting_facts.append(f"{title}: {context[1][sent_id]}")
        
        # Combine all supporting facts into a context string
        context_str = " ".join(supporting_facts)
        
        # Create a formatted prompt
        prompt = f"Context:\n{context_str}\n\nQuestion: {item['question']}\n\nAnswer:"
        
        prepared_sample = {
            'id': item['_id'],
            'question': item['question'],
            'answer': item['answer'],
            'context': context_str,
            'supporting_facts': supporting_facts,
            'prompt': prompt,
            'level': item['level'],  # easy or hard
            'type': item['type']     # comparison, bridge, etc.
        }
        
        prepared_samples.append(prepared_sample)
    
    return prepared_samples

def prepare_gsm8k_samples(
    dataset: Any,
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Prepare samples from the GSM8K dataset for concept graph experiments.
    
    Args:
        dataset: GSM8K dataset from HuggingFace
        max_samples: Maximum number of samples to prepare
        
    Returns:
        List of prepared samples
    """
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    prepared_samples = []
    
    for item in dataset:
        # Extract the final answer from the solution
        solution_lines = item['answer'].strip().split('\n')
        final_answer = solution_lines[-1]
        
        # Check if the final answer contains "The answer is"
        if "The answer is" in final_answer:
            answer_value = final_answer.split("The answer is")[-1].strip()
        else:
            answer_value = final_answer
        
        # Prepare sample with structured information
        prepared_sample = {
            'id': str(len(prepared_samples)),
            'question': item['question'],
            'solution': item['answer'],
            'solution_steps': solution_lines[:-1],  # All lines except the last one
            'final_answer': answer_value,
            'prompt': f"Question: {item['question']}\n\nStep-by-step solution:"
        }
        
        prepared_samples.append(prepared_sample)
    
    return prepared_samples

def prepare_strategyqa_samples(
    dataset: Any,
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Prepare samples from the StrategyQA dataset for concept graph experiments.
    
    Args:
        dataset: StrategyQA dataset from HuggingFace
        max_samples: Maximum number of samples to prepare
        
    Returns:
        List of prepared samples
    """
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    prepared_samples = []
    
    for item in dataset:
        # Extract the reasoning steps
        reasoning_steps = []
        if 'facts' in item:
            reasoning_steps = item['facts']
        
        # Prepare the prompt
        prompt = f"Question: {item['question']}\n\nThink through this step-by-step:"
        
        prepared_sample = {
            'id': str(len(prepared_samples)),
            'question': item['question'],
            'answer': "Yes" if item['answer'] else "No",
            'reasoning_steps': reasoning_steps,
            'prompt': prompt
        }
        
        prepared_samples.append(prepared_sample)
    
    return prepared_samples

def split_dataset(
    dataset: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        dataset: List of dataset items
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with train, val, and test splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    
    train_end = int(train_ratio * len(dataset))
    val_end = train_end + int(val_ratio * len(dataset))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    return {
        'train': [dataset[i] for i in train_indices],
        'val': [dataset[i] for i in val_indices],
        'test': [dataset[i] for i in test_indices]
    }

def get_dataset_stats(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics for a dataset.
    
    Args:
        dataset: List of dataset items
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'num_samples': len(dataset)
    }
    
    # Sample-specific statistics
    if len(dataset) > 0:
        # Check if the dataset contains different types of questions
        if 'type' in dataset[0]:
            type_counts = {}
            for item in dataset:
                item_type = item['type']
                type_counts[item_type] = type_counts.get(item_type, 0) + 1
            
            stats['type_distribution'] = type_counts
        
        # Check if the dataset has difficulty levels
        if 'level' in dataset[0]:
            level_counts = {}
            for item in dataset:
                level = item['level']
                level_counts[level] = level_counts.get(level, 0) + 1
            
            stats['level_distribution'] = level_counts
        
        # Compute average question length
        if 'question' in dataset[0]:
            question_lengths = [len(item['question'].split()) for item in dataset]
            stats['avg_question_length'] = sum(question_lengths) / len(question_lengths)
            stats['min_question_length'] = min(question_lengths)
            stats['max_question_length'] = max(question_lengths)
        
        # Compute average answer length for QA datasets
        if 'answer' in dataset[0] and isinstance(dataset[0]['answer'], str):
            answer_lengths = [len(item['answer'].split()) for item in dataset]
            stats['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths)
            stats['min_answer_length'] = min(answer_lengths)
            stats['max_answer_length'] = max(answer_lengths)
    
    return stats
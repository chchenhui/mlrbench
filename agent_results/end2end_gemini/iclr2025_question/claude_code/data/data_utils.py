"""
Data utilities for the AUG-RAG experiments.
"""

import os
import json
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from typing import Dict, List, Tuple, Union, Any, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def setup_logging(log_file: str = None, level: int = logging.INFO):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def load_truthfulqa_dataset(split: str = "validation") -> Dataset:
    """
    Load the TruthfulQA dataset.

    Args:
        split: The dataset split to load ("train", "validation", or "test").

    Returns:
        The loaded dataset.
    """
    try:
        dataset = load_dataset("truthful_qa", "multiple_choice", split=split)
        logger.info(f"Loaded TruthfulQA dataset ({split} split) with {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.error(f"Error loading TruthfulQA dataset: {e}")
        # Create a minimal dummy dataset for testing
        dummy_data = {
            "question": ["Is the Earth flat?", "Who was the first person on the Moon?"],
            "mc1_targets": [["No"], ["Neil Armstrong"]],
            "mc2_targets": [["The Earth is approximately spherical."],
                           ["Neil Armstrong was the first person to walk on the Moon in 1969."]],
            "mc1_labels": [[1], [1]],
            "mc2_labels": [[1], [1]]
        }
        return Dataset.from_dict(dummy_data)

def load_halueval_dataset(split: str = "test") -> Dataset:
    """
    Load the HaluEval dataset.
    
    Args:
        split: The dataset split to load ("train", "validation", or "test").
    
    Returns:
        The loaded dataset.
    """
    try:
        dataset = load_dataset("halucination/halueval", split=split)
        logger.info(f"Loaded HaluEval dataset ({split} split) with {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.error(f"Error loading HaluEval dataset: {e}")
        # Create a minimal dummy dataset for testing
        dummy_data = {
            "question": ["What is the capital of France?", "Who wrote the novel 'War and Peace'?"],
            "reference": ["Paris", "Leo Tolstoy"],
            "hallucination": ["Paris is the city of love and is known for its beautiful architecture.",
                             "Leo Tolstoy wrote 'War and Peace' in 1869, which is considered one of the greatest novels of all time."]
        }
        return Dataset.from_dict(dummy_data)

def load_nq_dataset(split: str = "validation") -> Dataset:
    """
    Load the Natural Questions dataset.
    
    Args:
        split: The dataset split to load ("train", "validation", or "test").
    
    Returns:
        The loaded dataset.
    """
    try:
        dataset = load_dataset("natural_questions", split=split)
        logger.info(f"Loaded Natural Questions dataset ({split} split) with {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.error(f"Error loading Natural Questions dataset: {e}")
        # Create a minimal dummy dataset for testing
        dummy_data = {
            "question": ["What is the capital of France?", "Who wrote the novel 'War and Peace'?"],
            "answer": ["Paris", "Leo Tolstoy"]
        }
        return Dataset.from_dict(dummy_data)

def create_knowledge_base(dataset_name: str, output_path: str) -> str:
    """
    Create a simple knowledge base from a dataset.
    
    Args:
        dataset_name: The name of the dataset to use.
        output_path: The path to save the knowledge base.
    
    Returns:
        The path to the created knowledge base.
    """
    if dataset_name == "truthfulqa":
        dataset = load_truthfulqa_dataset("validation")
        knowledge = []
        for item in dataset:
            for target in item["mc2_targets"]:
                if isinstance(target, list):
                    target = target[0]  # Take the first target if it's a list
                knowledge.append({"id": len(knowledge), "text": target})
    
    elif dataset_name == "nq":
        dataset = load_nq_dataset("validation")
        knowledge = []
        for item in dataset:
            if isinstance(item["answer"], list) and len(item["answer"]) > 0:
                knowledge.append({"id": len(knowledge), "text": item["answer"][0]})
    
    else:
        # Create a dummy knowledge base
        knowledge = [
            {"id": 0, "text": "The Earth is approximately spherical, not flat."},
            {"id": 1, "text": "Neil Armstrong was the first person to walk on the Moon in 1969."},
            {"id": 2, "text": "Paris is the capital of France."},
            {"id": 3, "text": "Leo Tolstoy wrote the novel 'War and Peace'."}
        ]
    
    # Save the knowledge base
    kb_path = os.path.join(output_path, f"{dataset_name}_kb.json")
    with open(kb_path, 'w') as f:
        json.dump(knowledge, f, indent=2)
    
    logger.info(f"Created knowledge base with {len(knowledge)} items at {kb_path}")
    return kb_path

def preprocess_for_evaluation(
    dataset: Dataset, 
    model_outputs: List[str], 
    mode: str = "factuality"
) -> Tuple[List[str], List[str]]:
    """
    Preprocess model outputs and dataset for evaluation.
    
    Args:
        dataset: The dataset containing ground truth answers.
        model_outputs: List of model outputs.
        mode: The evaluation mode ("factuality", "hallucination", etc.).
    
    Returns:
        A tuple of (references, predictions).
    """
    if mode == "factuality":
        # For factual QA, compare model outputs with reference answers
        references = []
        for item in dataset:
            if "answer" in item:
                references.append(item["answer"])
            elif "mc1_targets" in item:
                references.append(item["mc1_targets"][0])
            else:
                references.append("")
        
        return references, model_outputs
    
    elif mode == "hallucination":
        # For hallucination evaluation, we need to check if model outputs contain
        # factually incorrect information compared to references
        references = []
        for item in dataset:
            if "reference" in item:
                references.append(item["reference"])
            else:
                references.append("")
        
        return references, model_outputs
    
    else:
        logger.warning(f"Unknown evaluation mode: {mode}. Returning raw outputs.")
        return [], model_outputs

def prepare_dataset_for_training(
    dataset: Dataset, 
    tokenizer: Any, 
    max_length: int = 512
) -> Dict[str, torch.Tensor]:
    """
    Prepare a dataset for training by tokenizing inputs and outputs.
    
    Args:
        dataset: The dataset to prepare.
        tokenizer: The tokenizer to use.
        max_length: The maximum sequence length.
    
    Returns:
        A dictionary of tensors for training.
    """
    inputs = []
    outputs = []
    
    for item in dataset:
        if "question" in item:
            inputs.append(item["question"])
        else:
            inputs.append("")
        
        if "answer" in item:
            outputs.append(item["answer"])
        elif "mc1_targets" in item:
            outputs.append(item["mc1_targets"][0])
        else:
            outputs.append("")
    
    # Tokenize inputs
    input_encodings = tokenizer(
        inputs,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Tokenize outputs
    output_encodings = tokenizer(
        outputs,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Return tensors
    return {
        "input_ids": input_encodings.input_ids,
        "attention_mask": input_encodings.attention_mask,
        "labels": output_encodings.input_ids
    }
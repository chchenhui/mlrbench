#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for the LLM-TAC experiment.
"""

import logging
import os
import random
import numpy as np
import torch
import re
from typing import List, Dict, Any, Tuple, Optional

def setup_logging(log_file: str, level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to the log file
        level: Logging level
    """
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def format_coq_goal_state(goal: str, hypotheses: List[str]) -> str:
    """
    Format Coq goal state for input to LLM.
    
    Args:
        goal: The goal statement
        hypotheses: List of hypotheses
        
    Returns:
        Formatted goal state
    """
    formatted_hypotheses = "\n".join([f"H{i}: {hyp}" for i, hyp in enumerate(hypotheses)])
    return f"===== HYPOTHESES =====\n{formatted_hypotheses}\n===== GOAL =====\n{goal}"

def parse_tactic_sequence(tactics_str: str) -> List[str]:
    """
    Parse a sequence of tactics from a string.
    
    Args:
        tactics_str: String containing tactic sequences
        
    Returns:
        List of individual tactics
    """
    # Remove comments
    tactics_str = re.sub(r'\(\*.*?\*\)', '', tactics_str)
    
    # Split by semicolons and periods
    tactics = re.split(r'[;.]', tactics_str)
    
    # Clean and filter empty tactics
    tactics = [tactic.strip() for tactic in tactics if tactic.strip()]
    
    return tactics

def calculate_tactic_accuracy(predicted: List[str], ground_truth: List[str]) -> float:
    """
    Calculate the accuracy of predicted tactics compared to ground truth.
    
    Args:
        predicted: List of predicted tactics
        ground_truth: List of ground truth tactics
        
    Returns:
        Accuracy score between 0 and 1
    """
    if not ground_truth:
        return 0.0
    
    correct = 0
    for p_tactic, gt_tactic in zip(predicted, ground_truth):
        if p_tactic.strip() == gt_tactic.strip():
            correct += 1
    
    return correct / len(ground_truth)

def calculate_reduction_in_manual_writing(automated_tactics: int, total_tactics: int) -> float:
    """
    Calculate the reduction in manual tactic writing.
    
    Args:
        automated_tactics: Number of tactics successfully automated
        total_tactics: Total number of tactics in the proof
        
    Returns:
        Percentage reduction in manual writing (0-100)
    """
    if total_tactics == 0:
        return 0.0
    
    return (automated_tactics / total_tactics) * 100.0

def save_dict_as_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save a dictionary as a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the JSON file
    """
    import json
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json_as_dict(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file as a dictionary.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary loaded from the JSON file
    """
    import json
    
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_proof_completion_time(start_time: float, end_time: float) -> float:
    """
    Calculate the proof completion time in seconds.
    
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Completion time in seconds
    """
    return end_time - start_time

def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    half_length = max_length // 2
    return text[:half_length] + " ... [truncated] ... " + text[-half_length:]
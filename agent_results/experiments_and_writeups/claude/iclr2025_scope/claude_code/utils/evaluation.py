#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation utilities for the proposed architecture.

This module provides functions to evaluate the performance of the models:
1. Task performance metrics (ROUGE-L, BLEU, Exact Match, F1)
2. Efficiency metrics (throughput, memory usage, token efficiency)
3. Adaptation metrics (information retention, temporal consistency)
"""

import torch
import numpy as np
import re
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_exact_match(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Calculate exact match score.
    
    Args:
        predictions: Tensor of shape [batch_size, seq_length] containing predicted tokens
        targets: Tensor of shape [batch_size, seq_length] containing target tokens
        ignore_index: Token index to ignore in evaluation
    
    Returns:
        exact_match: Exact match score
    """
    # Create a mask for tokens to consider
    mask = targets != ignore_index
    
    # Count exact matches per sample
    exact_matches = 0
    batch_size = predictions.size(0)
    
    for i in range(batch_size):
        sample_mask = mask[i]
        
        if sample_mask.sum() == 0:
            continue
        
        sample_preds = predictions[i, sample_mask]
        sample_targets = targets[i, sample_mask]
        
        if torch.all(sample_preds == sample_targets):
            exact_matches += 1
    
    return exact_matches / batch_size if batch_size > 0 else 0.0


def calculate_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Calculate token-level F1 score.
    
    Args:
        predictions: Tensor of shape [batch_size, seq_length] containing predicted tokens
        targets: Tensor of shape [batch_size, seq_length] containing target tokens
        ignore_index: Token index to ignore in evaluation
    
    Returns:
        f1: F1 score
    """
    # Create a mask for tokens to consider
    mask = targets != ignore_index
    
    # Count exact matches per sample
    total_f1 = 0.0
    batch_size = predictions.size(0)
    
    for i in range(batch_size):
        sample_mask = mask[i]
        
        if sample_mask.sum() == 0:
            continue
        
        sample_preds = predictions[i, sample_mask].cpu().numpy()
        sample_targets = targets[i, sample_mask].cpu().numpy()
        
        # Create counters for tokens
        pred_counter = Counter(sample_preds)
        target_counter = Counter(sample_targets)
        
        # Calculate TP, FP, FN
        true_positives = sum((pred_counter & target_counter).values())
        false_positives = sum(pred_counter.values()) - true_positives
        false_negatives = sum(target_counter.values()) - true_positives
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
        
        # Calculate F1
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        total_f1 += f1
    
    return total_f1 / batch_size if batch_size > 0 else 0.0


def calculate_rouge_l(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    vocab: Optional[Dict[int, str]] = None
) -> float:
    """
    Calculate ROUGE-L score.
    
    Args:
        predictions: Tensor of shape [batch_size, seq_length] containing predicted tokens
        targets: Tensor of shape [batch_size, seq_length] containing target tokens
        ignore_index: Token index to ignore in evaluation
        vocab: Optional vocabulary mapping from token IDs to strings
    
    Returns:
        rouge_l: ROUGE-L score
    """
    # Create a mask for tokens to consider
    mask = targets != ignore_index
    
    # Count scores per sample
    total_rouge_l = 0.0
    batch_size = predictions.size(0)
    
    for i in range(batch_size):
        sample_mask = mask[i]
        
        if sample_mask.sum() == 0:
            continue
        
        sample_preds = predictions[i, sample_mask].cpu().numpy()
        sample_targets = targets[i, sample_mask].cpu().numpy()
        
        # Find the length of the longest common subsequence
        lcs_length = _lcs_length(sample_preds, sample_targets)
        
        # Calculate precision and recall
        precision = lcs_length / len(sample_preds) if len(sample_preds) > 0 else 0.0
        recall = lcs_length / len(sample_targets) if len(sample_targets) > 0 else 0.0
        
        # Calculate ROUGE-L
        rouge_l = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        total_rouge_l += rouge_l
    
    return total_rouge_l / batch_size if batch_size > 0 else 0.0


def _lcs_length(
    a: np.ndarray,
    b: np.ndarray
) -> int:
    """
    Calculate the length of the longest common subsequence.
    
    Args:
        a: First sequence
        b: Second sequence
    
    Returns:
        length: Length of the LCS
    """
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    
    return dp[m, n]


def calculate_bleu(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    n_gram: int = 4
) -> float:
    """
    Calculate BLEU score.
    
    Args:
        predictions: Tensor of shape [batch_size, seq_length] containing predicted tokens
        targets: Tensor of shape [batch_size, seq_length] containing target tokens
        ignore_index: Token index to ignore in evaluation
        n_gram: Maximum n-gram to consider
    
    Returns:
        bleu: BLEU score
    """
    # Create a mask for tokens to consider
    mask = targets != ignore_index
    
    # Count scores per sample
    total_bleu = 0.0
    batch_size = predictions.size(0)
    
    for i in range(batch_size):
        sample_mask = mask[i]
        
        if sample_mask.sum() == 0:
            continue
        
        sample_preds = predictions[i, sample_mask].cpu().numpy().tolist()
        sample_targets = targets[i, sample_mask].cpu().numpy().tolist()
        
        # Calculate BLEU score for this sample
        bleu = _calculate_bleu_score(sample_preds, sample_targets, n_gram)
        total_bleu += bleu
    
    return total_bleu / batch_size if batch_size > 0 else 0.0


def _calculate_bleu_score(
    prediction: List[int],
    reference: List[int],
    n_gram: int = 4
) -> float:
    """
    Calculate BLEU score for a single sample.
    
    Args:
        prediction: List of predicted tokens
        reference: List of reference tokens
        n_gram: Maximum n-gram to consider
    
    Returns:
        bleu: BLEU score
    """
    # Short circuit if prediction is empty
    if len(prediction) == 0:
        return 0.0
    
    # Calculate brevity penalty
    if len(prediction) < len(reference):
        brevity_penalty = np.exp(1 - len(reference) / len(prediction))
    else:
        brevity_penalty = 1.0
    
    # Calculate n-gram precision
    precisions = []
    
    for n in range(1, min(n_gram + 1, len(prediction) + 1)):
        # Count prediction n-grams
        pred_ngrams = _count_ngrams(prediction, n)
        ref_ngrams = _count_ngrams(reference, n)
        
        # Calculate precision for this n
        matches = 0
        total = 0
        
        for ngram, count in pred_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
            total += count
        
        precision = matches / total if total > 0 else 0.0
        precisions.append(precision)
    
    # Calculate geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0
    
    log_precisions = [np.log(p) for p in precisions]
    avg_log_precision = sum(log_precisions) / len(log_precisions)
    
    bleu = brevity_penalty * np.exp(avg_log_precision)
    
    return bleu


def _count_ngrams(
    sequence: List[int],
    n: int
) -> Dict[Tuple[int, ...], int]:
    """
    Count n-grams in a sequence.
    
    Args:
        sequence: List of tokens
        n: N-gram size
    
    Returns:
        counts: Dictionary mapping n-grams to counts
    """
    ngrams = {}
    
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i:i + n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    
    return ngrams


def evaluate_task_performance(
    outputs: Any,
    batch: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Evaluate task performance.
    
    Args:
        outputs: Model outputs
        batch: Input batch
    
    Returns:
        metrics: Dictionary of task performance metrics
    """
    # Get predictions and targets
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    labels = batch['labels']
    
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)
    
    # Calculate metrics
    exact_match = calculate_exact_match(predictions, labels)
    f1 = calculate_f1(predictions, labels)
    rouge_l = calculate_rouge_l(predictions, labels)
    bleu = calculate_bleu(predictions, labels)
    
    return {
        'exact_match': exact_match,
        'f1': f1,
        'rouge_l': rouge_l,
        'bleu': bleu
    }


def evaluate_efficiency(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model efficiency.
    
    Args:
        model: Model to evaluate
        batch: Input batch
        device: Device to run evaluation on
    
    Returns:
        metrics: Dictionary of efficiency metrics
    """
    # Prepare inputs
    batch = {k: v.to(device) for k, v in batch.items()}
    batch_size, seq_length = batch['input_ids'].shape
    
    # Reset GPU memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Measure inference time and memory usage
    start_time = time.time()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**batch)
    
    # Synchronize to ensure all operations are complete
    torch.cuda.synchronize()
    
    # Calculate metrics
    elapsed_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    
    # Calculate throughput
    tokens_per_second = batch_size * seq_length / elapsed_time
    
    # Calculate token efficiency if available
    token_efficiency = model.get_token_efficiency() if hasattr(model, 'get_token_efficiency') else 1.0
    
    # Calculate latency
    latency = elapsed_time / batch_size  # seconds per sample
    
    return {
        'throughput': tokens_per_second,
        'memory_usage': peak_memory,
        'token_efficiency': token_efficiency,
        'latency': latency
    }


def evaluate_adaptation(
    model: torch.nn.Module,
    streaming_data: List[Dict[str, torch.Tensor]],
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model adaptation capabilities.
    
    Args:
        model: Model to evaluate
        streaming_data: List of batches representing a stream of data
        device: Device to run evaluation on
    
    Returns:
        metrics: Dictionary of adaptation metrics
    """
    # Check if model has adaptation capabilities
    if not hasattr(model, 'evaluate_adaptation'):
        return {
            'information_retention': 0.0,
            'temporal_consistency': 0.0,
            'adaptation_speed': 0.0
        }
    
    # Process streaming data
    model.eval()
    
    initial_outputs = None
    final_outputs = None
    
    # Process each batch in the stream
    for i, batch in enumerate(streaming_data):
        # Prepare inputs
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**batch)
        
        # Store initial and final outputs
        if i == 0:
            initial_outputs = outputs
        final_outputs = outputs
    
    # Get adaptation metrics from the model
    adaptation_metrics = model.evaluate_adaptation()
    
    return adaptation_metrics
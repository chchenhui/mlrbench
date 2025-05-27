"""
Evaluation utilities for the AUG-RAG experiments.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Union, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import re
from collections import Counter

logger = logging.getLogger(__name__)

# Initialize NLTK if needed
try:
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    logger.warning("NLTK not installed. Some metrics won't be available.")


def exact_match_score(predictions: List[str], references: List[str]) -> float:
    """
    Calculate the exact match score.
    
    Args:
        predictions: List of predicted answers.
        references: List of reference answers.
    
    Returns:
        The exact match score.
    """
    if len(predictions) != len(references):
        logger.warning(f"Length mismatch: predictions ({len(predictions)}) vs references ({len(references)})")
        return 0.0
    
    # Normalize and compare
    matches = sum(1 for p, r in zip(predictions, references) 
                  if normalize_answer(p) == normalize_answer(r))
    return matches / len(predictions) if predictions else 0.0


def normalize_answer(text: str) -> str:
    """
    Normalize text for exact match evaluation.
    
    Args:
        text: Text to normalize.
    
    Returns:
        Normalized text.
    """
    # Check if text is already a string
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    
    # Convert to lowercase and remove punctuation, articles and extra whitespace
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def f1_token_score(predictions: List[str], references: List[str]) -> float:
    """
    Calculate the F1 score based on token overlap.
    
    Args:
        predictions: List of predicted answers.
        references: List of reference answers.
    
    Returns:
        The F1 score.
    """
    if len(predictions) != len(references):
        logger.warning(f"Length mismatch: predictions ({len(predictions)}) vs references ({len(references)})")
        return 0.0
    
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = normalize_answer(pred).split()
        ref_tokens = normalize_answer(ref).split()
        
        common_tokens = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common_tokens.values())
        
        if num_common == 0 or not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue
            
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def bleu_score(predictions: List[str], references: List[str]) -> float:
    """
    Calculate the BLEU score.
    
    Args:
        predictions: List of predicted answers.
        references: List of reference answers.
    
    Returns:
        The BLEU score.
    """
    if len(predictions) != len(references):
        logger.warning(f"Length mismatch: predictions ({len(predictions)}) vs references ({len(references)})")
        return 0.0
    
    bleu_scores = []
    smoothing = SmoothingFunction().method1
    
    for pred, ref in zip(predictions, references):
        # Tokenize
        pred_tokens = word_tokenize(pred.lower())
        ref_tokens = word_tokenize(ref.lower())
        
        # Skip empty sequences
        if not pred_tokens or not ref_tokens:
            continue
        
        # Calculate BLEU score
        try:
            score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
            bleu_scores.append(score)
        except Exception as e:
            logger.warning(f"Error calculating BLEU score: {e}")
            bleu_scores.append(0.0)
    
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0


def truthfulness_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate a simplified truthfulness score for TruthfulQA.
    
    Args:
        predictions: List of predicted answers.
        references: List of reference answers (truthful answers).
    
    Returns:
        Dictionary with truthfulness metrics.
    """
    if len(predictions) != len(references):
        logger.warning(f"Length mismatch: predictions ({len(predictions)}) vs references ({len(references)})")
        return {"truthful_percent": 0.0, "informative_percent": 0.0}
    
    # For demonstration, use token overlap as a proxy for truthfulness
    truthful_count = 0
    informative_count = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = set(normalize_answer(pred).split())
        ref_tokens = set(normalize_answer(ref).split())
        
        # Consider truthful if there's significant overlap
        overlap = len(pred_tokens.intersection(ref_tokens))
        if overlap > 0 and overlap / len(ref_tokens) > 0.5:
            truthful_count += 1
        
        # Consider informative if the answer is substantive
        if len(pred_tokens) >= 3:
            informative_count += 1
    
    return {
        "truthful_percent": truthful_count / len(predictions) if predictions else 0.0,
        "informative_percent": informative_count / len(predictions) if predictions else 0.0
    }


def self_contradiction_rate(texts: List[str]) -> float:
    """
    Calculate a simplified self-contradiction rate in model outputs.
    
    Args:
        texts: List of model-generated texts.
    
    Returns:
        The self-contradiction rate.
    """
    contradiction_patterns = [
        (r"yes.*no", r"no.*yes"),
        (r"true.*false", r"false.*true"),
        (r"correct.*incorrect", r"incorrect.*correct"),
        (r"is.*is not", r"is not.*is")
    ]
    
    contradiction_count = 0
    
    for text in texts:
        has_contradiction = False
        text_lower = text.lower()
        
        for pattern_pair in contradiction_patterns:
            if (re.search(pattern_pair[0], text_lower) or 
                re.search(pattern_pair[1], text_lower)):
                has_contradiction = True
                break
        
        if has_contradiction:
            contradiction_count += 1
    
    return contradiction_count / len(texts) if texts else 0.0


def expected_calibration_error(
    confidences: List[float],
    correctness: List[bool],
    num_bins: int = 10
) -> float:
    """
    Calculate the Expected Calibration Error (ECE).
    
    Args:
        confidences: List of model confidence scores.
        correctness: List of boolean values indicating correctness.
        num_bins: Number of bins for calibration analysis.
    
    Returns:
        The ECE score.
    """
    if len(confidences) != len(correctness):
        logger.warning(f"Length mismatch: confidences ({len(confidences)}) vs correctness ({len(correctness)})")
        return 1.0  # Worst possible ECE
    
    # Convert to numpy arrays
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=int)
    
    # Create bins and bin assignments
    bin_indices = np.digitize(confidences, np.linspace(0, 1, num_bins + 1)[1:-1])
    
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    # Compute bin statistics
    for i in range(len(confidences)):
        bin_idx = bin_indices[i]
        bin_accuracies[bin_idx] += correctness[i]
        bin_confidences[bin_idx] += confidences[i]
        bin_counts[bin_idx] += 1
    
    # Compute averages
    for i in range(num_bins):
        if bin_counts[i] > 0:
            bin_accuracies[i] /= bin_counts[i]
            bin_confidences[i] /= bin_counts[i]
    
    # Compute ECE
    ece = 0
    total_samples = len(confidences)
    
    for i in range(num_bins):
        if bin_counts[i] > 0:
            bin_weight = bin_counts[i] / total_samples
            bin_error = abs(bin_accuracies[i] - bin_confidences[i])
            ece += bin_weight * bin_error
    
    return ece


def evaluate_uncertainty_calibration(
    uncertainty_scores: List[float],
    has_hallucination: List[bool]
) -> Dict[str, float]:
    """
    Evaluate how well uncertainty scores predict hallucinations.
    
    Args:
        uncertainty_scores: List of uncertainty scores.
        has_hallucination: List of boolean values indicating hallucination presence.
    
    Returns:
        Dictionary with calibration metrics.
    """
    if len(uncertainty_scores) != len(has_hallucination):
        logger.warning(f"Length mismatch: uncertainty ({len(uncertainty_scores)}) vs hallucination ({len(has_hallucination)})")
        return {
            "auroc": 0.5,
            "auprc": 0.5,
            "ece": 1.0
        }
    
    # Convert to numpy arrays
    uncertainty_scores = np.array(uncertainty_scores)
    has_hallucination = np.array(has_hallucination, dtype=int)
    
    try:
        # Calculate AUROC
        auroc = roc_auc_score(has_hallucination, uncertainty_scores)
        
        # Calculate AUPRC
        auprc = average_precision_score(has_hallucination, uncertainty_scores)
        
        # Calculate ECE (treat uncertainty as probability of hallucination)
        ece = expected_calibration_error(uncertainty_scores, has_hallucination)
        
        return {
            "auroc": auroc,
            "auprc": auprc,
            "ece": ece
        }
    except Exception as e:
        logger.error(f"Error calculating uncertainty calibration metrics: {e}")
        return {
            "auroc": 0.5,
            "auprc": 0.5,
            "ece": 1.0
        }


def diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Calculate diversity metrics for generated texts.
    
    Args:
        texts: List of generated texts.
    
    Returns:
        Dictionary with diversity metrics.
    """
    if not texts:
        return {
            "unique_1grams": 0.0,
            "unique_2grams": 0.0,
            "mean_length": 0.0
        }
    
    # Tokenize texts
    tokenized_texts = [text.lower().split() for text in texts]
    
    # Calculate n-gram diversity
    all_1grams = [gram for text in tokenized_texts for gram in text]
    all_2grams = [(text[i], text[i+1]) for text in tokenized_texts 
                  for i in range(len(text)-1)]
    
    unique_1grams = len(set(all_1grams)) / (len(all_1grams) + 1e-10)
    unique_2grams = len(set(all_2grams)) / (len(all_2grams) + 1e-10)
    
    # Calculate mean text length
    mean_length = sum(len(text) for text in tokenized_texts) / len(tokenized_texts)
    
    return {
        "unique_1grams": unique_1grams,
        "unique_2grams": unique_2grams,
        "mean_length": mean_length
    }


def knowledge_f1_score(
    generations: List[str],
    retrieved_contexts: List[List[str]]
) -> float:
    """
    Calculate F1 score for knowledge usage from retrieved contexts.
    
    Args:
        generations: List of generated texts.
        retrieved_contexts: List of lists of retrieved contexts for each generation.
    
    Returns:
        F1 score for knowledge usage.
    """
    if len(generations) != len(retrieved_contexts):
        logger.warning(f"Length mismatch: generations ({len(generations)}) vs contexts ({len(retrieved_contexts)})")
        return 0.0
    
    f1_scores = []
    
    for gen, contexts in zip(generations, retrieved_contexts):
        # Extract key facts from contexts (simple approach: use all nouns and named entities)
        context_facts = set()
        for ctx in contexts:
            # Simplified approach: just use words as facts
            words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]+\b', ctx)
            context_facts.update(words)
        
        # Extract facts from generation
        gen_words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]+\b', gen)
        gen_facts = set(gen_words)
        
        # Skip if no facts in either
        if not context_facts or not gen_facts:
            continue
        
        # Calculate precision, recall, F1
        overlap = len(gen_facts.intersection(context_facts))
        precision = overlap / len(gen_facts)
        recall = overlap / len(context_facts)
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
    
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def evaluate_model_outputs(
    predictions: List[str],
    references: List[str],
    uncertainty_scores: Optional[List[float]] = None,
    hallucination_labels: Optional[List[bool]] = None,
    retrieved_contexts: Optional[List[List[str]]] = None,
    mode: str = "factuality"
) -> Dict[str, float]:
    """
    Evaluate model outputs with multiple metrics.
    
    Args:
        predictions: List of model predictions.
        references: List of reference answers.
        uncertainty_scores: Optional list of uncertainty scores.
        hallucination_labels: Optional list of hallucination labels.
        retrieved_contexts: Optional list of retrieved contexts.
        mode: Evaluation mode ("factuality", "hallucination", etc.).
    
    Returns:
        Dictionary with evaluation metrics.
    """
    metrics = {}
    
    # Basic metrics for all modes
    metrics["num_samples"] = len(predictions)
    if len(predictions) == 0:
        logger.warning("No predictions to evaluate")
        return metrics
    
    # Calculate metrics based on mode
    if mode == "factuality":
        metrics["exact_match"] = exact_match_score(predictions, references)
        metrics["f1_score"] = f1_token_score(predictions, references)
        metrics["bleu"] = bleu_score(predictions, references)
    
    elif mode == "hallucination":
        # Add truthfulness metrics
        truthfulness = truthfulness_score(predictions, references)
        metrics.update(truthfulness)
        
        # Add self-contradiction rate
        metrics["self_contradiction_rate"] = self_contradiction_rate(predictions)
    
    # Add diversity metrics
    diversity = diversity_metrics(predictions)
    metrics.update(diversity)
    
    # Add uncertainty calibration if provided
    if uncertainty_scores is not None and hallucination_labels is not None:
        calibration = evaluate_uncertainty_calibration(uncertainty_scores, hallucination_labels)
        metrics.update(calibration)
    
    # Add knowledge usage metrics if contexts provided
    if retrieved_contexts is not None:
        metrics["knowledge_f1"] = knowledge_f1_score(predictions, retrieved_contexts)
    
    return metrics
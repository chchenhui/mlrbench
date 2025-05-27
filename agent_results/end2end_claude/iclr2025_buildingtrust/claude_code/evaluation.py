"""
Evaluation Module for TrustPath.

This module implements evaluation metrics and procedures for assessing
the performance of TrustPath and baseline methods.
"""

import json
import logging
import os
import time
from typing import Dict, List, Tuple, Any, Optional

import nltk
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from config import EVAL_CONFIG, RESULTS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources for BLEU score
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

class TrustPathEvaluator:
    """
    Evaluator for the TrustPath framework and baseline methods.
    
    This class measures the performance of error detection, correction quality,
    and system efficiency for TrustPath and baselines.
    """
    
    def __init__(self):
        """
        Initialize the evaluator.
        """
        self.metrics = EVAL_CONFIG["metrics"]
        self.random_seed = EVAL_CONFIG["random_seed"]
        self.n_runs = EVAL_CONFIG["n_runs"]
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        logger.info(f"Initialized TrustPathEvaluator with metrics: {', '.join(self.metrics)}")
    
    def evaluate_error_detection(self, 
                                detected_errors: List[Dict[str, Any]], 
                                ground_truth_errors: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate error detection performance.
        
        Args:
            detected_errors: List of detected errors
            ground_truth_errors: List of ground truth errors
            
        Returns:
            A dictionary with evaluation metrics
        """
        logger.info(f"Evaluating error detection with {len(detected_errors)} detected errors and {len(ground_truth_errors)} ground truth errors")
        
        # If there are no ground truth errors and no detected errors, perfect accuracy
        if not ground_truth_errors and not detected_errors:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "accuracy": 1.0
            }
        
        # If there are no ground truth errors but there are detected errors, all false positives
        if not ground_truth_errors:
            return {
                "precision": 0.0,
                "recall": 1.0,  # Correctly identified all (zero) true errors
                "f1": 0.0,
                "accuracy": 0.0
            }
        
        # If there are ground truth errors but no detected errors, all false negatives
        if not detected_errors:
            return {
                "precision": 1.0,  # No false positives
                "recall": 0.0,  # Missed all true errors
                "f1": 0.0,
                "accuracy": 0.0
            }
        
        # Extract error content for comparison
        detected_contents = [error.get("content", "") for error in detected_errors]
        ground_truth_contents = [error.get("content", "") for error in ground_truth_errors]
        
        # For each detected error, check if it matches any ground truth error
        true_positives = 0
        for detected in detected_contents:
            # Check for an approximate match with any ground truth error
            for truth in ground_truth_contents:
                if self._text_overlap(detected, truth):
                    true_positives += 1
                    break
        
        # Calculate metrics
        if true_positives == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            precision = true_positives / len(detected_contents)
            recall = true_positives / len(ground_truth_contents)
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": true_positives / max(len(detected_contents), len(ground_truth_contents))
        }
    
    def evaluate_correction_quality(self, 
                                   corrections: List[Dict[str, Any]], 
                                   ground_truth_corrections: List[str]) -> Dict[str, float]:
        """
        Evaluate the quality of corrections.
        
        Args:
            corrections: List of generated corrections
            ground_truth_corrections: List of ground truth corrections
            
        Returns:
            A dictionary with evaluation metrics
        """
        logger.info(f"Evaluating correction quality with {len(corrections)} corrections and {len(ground_truth_corrections)} ground truth corrections")
        
        # If there are no corrections to evaluate, return null metrics
        if not corrections or not ground_truth_corrections:
            return {
                "bleu": 0.0,
                "rouge1_f": 0.0,
                "rouge2_f": 0.0,
                "rougeL_f": 0.0,
                "exact_match_ratio": 0.0
            }
        
        # Extract correction content
        correction_texts = [corr.get("content", "") for corr in corrections]
        
        # Make sure we only evaluate matching pairs
        min_len = min(len(correction_texts), len(ground_truth_corrections))
        correction_texts = correction_texts[:min_len]
        ground_truth_corrections = ground_truth_corrections[:min_len]
        
        # Calculate BLEU scores
        bleu_scores = []
        for i in range(min_len):
            reference = [ground_truth_corrections[i].split()]
            hypothesis = correction_texts[i].split()
            
            # Skip if empty
            if not hypothesis or not reference[0]:
                continue
                
            try:
                bleu = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis)
                bleu_scores.append(bleu)
            except Exception as e:
                logger.warning(f"Error calculating BLEU score: {e}")
        
        # Calculate ROUGE scores
        rouge_scores = {
            "rouge1_f": [],
            "rouge2_f": [],
            "rougeL_f": []
        }
        
        for i in range(min_len):
            try:
                scores = self.rouge_scorer.score(ground_truth_corrections[i], correction_texts[i])
                rouge_scores["rouge1_f"].append(scores["rouge1"].fmeasure)
                rouge_scores["rouge2_f"].append(scores["rouge2"].fmeasure)
                rouge_scores["rougeL_f"].append(scores["rougeL"].fmeasure)
            except Exception as e:
                logger.warning(f"Error calculating ROUGE scores: {e}")
        
        # Calculate exact match ratio
        exact_matches = sum(1 for i in range(min_len) 
                          if self._normalize_text(correction_texts[i]) == self._normalize_text(ground_truth_corrections[i]))
        exact_match_ratio = exact_matches / min_len if min_len > 0 else 0.0
        
        # Aggregate metrics
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_rouge1 = np.mean(rouge_scores["rouge1_f"]) if rouge_scores["rouge1_f"] else 0.0
        avg_rouge2 = np.mean(rouge_scores["rouge2_f"]) if rouge_scores["rouge2_f"] else 0.0
        avg_rougeL = np.mean(rouge_scores["rougeL_f"]) if rouge_scores["rougeL_f"] else 0.0
        
        return {
            "bleu": avg_bleu,
            "rouge1_f": avg_rouge1,
            "rouge2_f": avg_rouge2,
            "rougeL_f": avg_rougeL,
            "exact_match_ratio": exact_match_ratio
        }
    
    def evaluate_system_efficiency(self, time_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate system efficiency in terms of latency and resource usage.
        
        Args:
            time_metrics: Dictionary with execution times
            
        Returns:
            A dictionary with efficiency metrics
        """
        return {
            "total_time": time_metrics.get("total_time", 0.0),
            "average_processing_time": time_metrics.get("average_processing_time", 0.0),
            "average_detection_time": time_metrics.get("average_detection_time", 0.0),
            "average_correction_time": time_metrics.get("average_correction_time", 0.0)
        }
    
    def evaluate_trust_metrics(self, 
                              system_results: Dict[str, Any], 
                              ground_truth: Dict[str, Any],
                              transparency_score: float) -> Dict[str, float]:
        """
        Evaluate trust calibration and other trust-related metrics.
        
        Args:
            system_results: Results from the system
            ground_truth: Ground truth data
            transparency_score: Score for transparency features
            
        Returns:
            A dictionary with trust metrics
        """
        # In a real experiment, this would use real user trust ratings
        # Here we simulate trust calibration based on system accuracy and transparency
        
        # Calculate system accuracy
        error_detection_metrics = self.evaluate_error_detection(
            system_results.get("detected_errors", []),
            ground_truth.get("errors", [])
        )
        
        system_accuracy = error_detection_metrics["f1"]
        
        # Simulate trust calibration
        # Perfect trust calibration would be 1.0 (user trust = system accuracy)
        # We simulate this by using transparency as a factor
        trust_calibration = 1.0 - abs(transparency_score - system_accuracy)
        
        # Simulate other trust metrics
        explanation_satisfaction = transparency_score * 0.8 + 0.2  # Simulated user satisfaction with explanations
        
        return {
            "trust_calibration": trust_calibration,
            "explanation_satisfaction": explanation_satisfaction,
            "transparency_score": transparency_score
        }
    
    def evaluate_results(self, 
                         system_results: Dict[str, Any], 
                         ground_truth: Dict[str, Any],
                         time_metrics: Dict[str, float],
                         transparency_score: float) -> Dict[str, Any]:
        """
        Evaluate all aspects of system performance.
        
        Args:
            system_results: Results from the system
            ground_truth: Ground truth data
            time_metrics: Dictionary with execution times
            transparency_score: Score for transparency features
            
        Returns:
            A dictionary with comprehensive evaluation metrics
        """
        # Evaluate error detection
        error_detection_metrics = self.evaluate_error_detection(
            system_results.get("detected_errors", []),
            ground_truth.get("errors", [])
        )
        
        # Evaluate correction quality
        correction_metrics = self.evaluate_correction_quality(
            system_results.get("suggested_corrections", []),
            ground_truth.get("corrections", [])
        )
        
        # Evaluate system efficiency
        efficiency_metrics = self.evaluate_system_efficiency(time_metrics)
        
        # Evaluate trust metrics
        trust_metrics = self.evaluate_trust_metrics(
            system_results, ground_truth, transparency_score
        )
        
        return {
            "error_detection": error_detection_metrics,
            "correction_quality": correction_metrics,
            "system_efficiency": efficiency_metrics,
            "trust_metrics": trust_metrics,
            "overall_score": self._calculate_overall_score(
                error_detection_metrics, correction_metrics, efficiency_metrics, trust_metrics
            )
        }
    
    def evaluate_method_on_dataset(self,
                                  method_results: List[Dict[str, Any]],
                                  dataset: List[Dict[str, Any]],
                                  time_metrics: Dict[str, float],
                                  transparency_score: float,
                                  method_name: str) -> Dict[str, Any]:
        """
        Evaluate a method on a dataset.
        
        Args:
            method_results: Results from the method on each sample
            dataset: The dataset with ground truth
            time_metrics: Dictionary with execution times
            transparency_score: Score for transparency features
            method_name: Name of the method
            
        Returns:
            A dictionary with evaluation results
        """
        logger.info(f"Evaluating {method_name} on {len(dataset)} samples")
        
        # Initialize aggregated metrics
        aggregated_metrics = {
            "error_detection": {
                "precision": [],
                "recall": [],
                "f1": [],
                "accuracy": []
            },
            "correction_quality": {
                "bleu": [],
                "rouge1_f": [],
                "rouge2_f": [],
                "rougeL_f": [],
                "exact_match_ratio": []
            },
            "system_efficiency": time_metrics,
            "trust_metrics": {
                "trust_calibration": [],
                "explanation_satisfaction": [],
                # transparency_score is used directly, not aggregated
            }
        }
        
        # Evaluate each sample
        for i, (result, sample) in enumerate(zip(method_results, dataset)):
            ground_truth = sample.get("ground_truth", {"errors": [], "corrections": []})
            
            # Evaluate error detection
            error_detection_metrics = self.evaluate_error_detection(
                result.get("detected_errors", []),
                ground_truth.get("errors", [])
            )
            
            # Evaluate correction quality
            correction_metrics = self.evaluate_correction_quality(
                result.get("suggested_corrections", []),
                ground_truth.get("corrections", [])
            )
            
            # Evaluate trust metrics
            trust_metrics = self.evaluate_trust_metrics(
                result, ground_truth, transparency_score
            )
            
            # Aggregate metrics
            for metric_name, value in error_detection_metrics.items():
                aggregated_metrics["error_detection"][metric_name].append(value)
                
            for metric_name, value in correction_metrics.items():
                aggregated_metrics["correction_quality"][metric_name].append(value)
                
            for metric_name, value in trust_metrics.items():
                if metric_name in aggregated_metrics["trust_metrics"]:
                    aggregated_metrics["trust_metrics"][metric_name].append(value)
        
        # Calculate average metrics
        averaged_metrics = {
            "error_detection": {
                metric_name: np.mean(values) for metric_name, values in aggregated_metrics["error_detection"].items()
            },
            "correction_quality": {
                metric_name: np.mean(values) for metric_name, values in aggregated_metrics["correction_quality"].items()
            },
            "system_efficiency": time_metrics,
            "trust_metrics": {
                metric_name: np.mean(values) if isinstance(values, list) and values else value
                for metric_name, values in aggregated_metrics["trust_metrics"].items()
            }
        }
        
        # Add transparency score directly
        averaged_metrics["trust_metrics"]["transparency_score"] = transparency_score
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            averaged_metrics["error_detection"],
            averaged_metrics["correction_quality"],
            averaged_metrics["system_efficiency"],
            averaged_metrics["trust_metrics"]
        )
        
        return {
            **averaged_metrics,
            "overall_score": overall_score,
            "method_name": method_name,
            "num_samples": len(dataset)
        }
    
    def _calculate_overall_score(self,
                                error_detection_metrics: Dict[str, float],
                                correction_metrics: Dict[str, float],
                                efficiency_metrics: Dict[str, float],
                                trust_metrics: Dict[str, float]) -> float:
        """
        Calculate an overall score based on all metrics.
        
        Args:
            error_detection_metrics: Error detection metrics
            correction_metrics: Correction quality metrics
            efficiency_metrics: System efficiency metrics
            trust_metrics: Trust-related metrics
            
        Returns:
            A combined overall score
        """
        # Weights for different components
        weights = {
            "error_detection": 0.4,
            "correction_quality": 0.3,
            "trust_metrics": 0.2,
            "system_efficiency": 0.1
        }
        
        # Score for error detection (F1 score)
        error_detection_score = error_detection_metrics.get("f1", 0.0)
        
        # Score for correction quality (average of BLEU and ROUGE-L)
        correction_score = (
            correction_metrics.get("bleu", 0.0) + 
            correction_metrics.get("rougeL_f", 0.0)
        ) / 2
        
        # Score for trust metrics (trust calibration)
        trust_score = trust_metrics.get("trust_calibration", 0.0)
        
        # Score for system efficiency (normalized inverse of processing time)
        processing_time = efficiency_metrics.get("average_processing_time", 1.0)
        # Normalize to 0-1 range (assuming reasonable bounds for processing time)
        efficiency_score = 1.0 / (1.0 + processing_time / 10.0)  # 10 seconds as reference
        
        # Calculate weighted sum
        overall_score = (
            weights["error_detection"] * error_detection_score +
            weights["correction_quality"] * correction_score +
            weights["trust_metrics"] * trust_score +
            weights["system_efficiency"] * efficiency_score
        )
        
        return overall_score
    
    def _text_overlap(self, text1: str, text2: str, threshold: float = 0.5) -> bool:
        """
        Check if two text strings overlap significantly.
        
        Args:
            text1: First text string
            text2: Second text string
            threshold: Overlap threshold (0-1)
            
        Returns:
            Boolean indicating significant overlap
        """
        # Normalize texts
        norm_text1 = self._normalize_text(text1)
        norm_text2 = self._normalize_text(text2)
        
        # If either is empty, no overlap
        if not norm_text1 or not norm_text2:
            return False
        
        # Check if one is contained in the other
        if norm_text1 in norm_text2 or norm_text2 in norm_text1:
            return True
        
        # Check word overlap
        words1 = set(norm_text1.split())
        words2 = set(norm_text2.split())
        
        if not words1 or not words2:
            return False
        
        common_words = words1.intersection(words2)
        overlap_ratio = len(common_words) / max(len(words1), len(words2))
        
        return overlap_ratio >= threshold
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: The text to normalize
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        for punct in '.,;:!?"\'()[]{}':
            text = text.replace(punct, ' ')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def save_evaluation_results(self, results: Dict[str, Any], filename: str) -> str:
        """
        Save evaluation results to a file.
        
        Args:
            results: The evaluation results
            filename: The filename to save to
            
        Returns:
            The path to the saved file
        """
        file_path = RESULTS_DIR / filename
        
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {file_path}")
        return str(file_path)
    
    def load_evaluation_results(self, filename: str) -> Dict[str, Any]:
        """
        Load evaluation results from a file.
        
        Args:
            filename: The filename to load from
            
        Returns:
            The loaded evaluation results
        """
        file_path = RESULTS_DIR / filename
        
        try:
            with open(file_path, "r") as f:
                results = json.load(f)
            
            logger.info(f"Loaded evaluation results from {file_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading evaluation results from {file_path}: {e}")
            return {}
    
    def results_to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert evaluation results to a pandas DataFrame for easier analysis.
        
        Args:
            results: The evaluation results
            
        Returns:
            A pandas DataFrame
        """
        # Extract results for each method
        methods_data = []
        
        for method_name, method_results in results.items():
            # Extract metrics
            error_detection = method_results.get("error_detection", {})
            correction_quality = method_results.get("correction_quality", {})
            system_efficiency = method_results.get("system_efficiency", {})
            trust_metrics = method_results.get("trust_metrics", {})
            
            # Create a row for the method
            method_data = {
                "method": method_name,
                "precision": error_detection.get("precision", 0.0),
                "recall": error_detection.get("recall", 0.0),
                "f1": error_detection.get("f1", 0.0),
                "accuracy": error_detection.get("accuracy", 0.0),
                "bleu": correction_quality.get("bleu", 0.0),
                "rouge1_f": correction_quality.get("rouge1_f", 0.0),
                "rouge2_f": correction_quality.get("rouge2_f", 0.0),
                "rougeL_f": correction_quality.get("rougeL_f", 0.0),
                "exact_match_ratio": correction_quality.get("exact_match_ratio", 0.0),
                "total_time": system_efficiency.get("total_time", 0.0),
                "average_processing_time": system_efficiency.get("average_processing_time", 0.0),
                "trust_calibration": trust_metrics.get("trust_calibration", 0.0),
                "explanation_satisfaction": trust_metrics.get("explanation_satisfaction", 0.0),
                "transparency_score": trust_metrics.get("transparency_score", 0.0),
                "overall_score": method_results.get("overall_score", 0.0),
                "num_samples": method_results.get("num_samples", 0)
            }
            
            methods_data.append(method_data)
        
        return pd.DataFrame(methods_data)

if __name__ == "__main__":
    # Simple test of the evaluator
    print("Testing evaluator...")
    
    # Create a simple test case
    ground_truth = {
        "errors": [
            {"content": "The Eiffel Tower was built in 1878"},
            {"content": "The Eiffel Tower is located in Lyon"}
        ],
        "corrections": [
            "The Eiffel Tower was built in 1889",
            "The Eiffel Tower is located in Paris"
        ]
    }
    
    system_results = {
        "detected_errors": [
            {"content": "The Eiffel Tower was built in 1878", "confidence_score": 0.8},
            {"content": "The tower is 124 meters tall", "confidence_score": 0.7}
        ],
        "suggested_corrections": [
            {"content": "The Eiffel Tower was built in 1889", "confidence_score": 0.9},
            {"content": "The tower is 324 meters tall", "confidence_score": 0.8}
        ]
    }
    
    time_metrics = {
        "total_time": 5.0,
        "average_processing_time": 2.5,
        "average_detection_time": 1.5,
        "average_correction_time": 1.0
    }
    
    transparency_score = 0.8
    
    evaluator = TrustPathEvaluator()
    results = evaluator.evaluate_results(system_results, ground_truth, time_metrics, transparency_score)
    
    print(json.dumps(results, indent=2))
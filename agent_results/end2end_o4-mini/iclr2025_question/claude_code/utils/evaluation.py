"""
Evaluation metrics for SCEC experiments.

Implements:
- Calibration metrics: Expected Calibration Error (ECE), Brier score
- Hallucination detection metrics: Precision, recall, F1
- Task performance metrics: Exact Match (EM), F1 score, ROUGE, BERTScore
- Diversity metrics: Distinct-n, Self-BLEU
- Efficiency metrics: Wall-clock time
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
from bert_score import score as bert_score

logger = logging.getLogger(__name__)

# Try to import NLTK
try:
    import nltk
    nltk.download('punkt', quiet=True)
except:
    logger.warning("Failed to import NLTK. Some metrics may not work.")


class CalibrationMetrics:
    """Calculate calibration metrics for uncertainty estimation."""
    
    @staticmethod
    def bin_predictions(
        confidence_scores: List[float],
        correctness: List[bool],
        num_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Bin predictions by confidence for calibration metrics.
        
        Args:
            confidence_scores: List of model confidence scores (0-1)
            correctness: List of boolean values indicating if predictions were correct
            num_bins: Number of bins to use
            
        Returns:
            Dictionary with binning information
        """
        if len(confidence_scores) != len(correctness):
            raise ValueError(f"Length of confidence_scores ({len(confidence_scores)}) must match length of correctness ({len(correctness)})")
        
        # Create bins
        bin_size = 1.0 / num_bins
        bins = []
        
        # Initialize bins
        for i in range(num_bins):
            lower = i * bin_size
            upper = (i + 1) * bin_size
            
            # Ensure the last bin includes 1.0
            if i == num_bins - 1:
                upper = 1.0 + 1e-10
                
            bins.append({
                "lower": lower,
                "upper": upper,
                "confidence_sum": 0.0,
                "accuracy_sum": 0.0,
                "count": 0,
            })
        
        # Assign predictions to bins
        for conf, correct in zip(confidence_scores, correctness):
            # Find the right bin
            bin_idx = min(int(conf / bin_size), num_bins - 1)
            
            # Update bin
            bins[bin_idx]["confidence_sum"] += conf
            bins[bin_idx]["accuracy_sum"] += float(correct)
            bins[bin_idx]["count"] += 1
        
        # Calculate bin statistics
        for b in bins:
            if b["count"] > 0:
                b["avg_confidence"] = b["confidence_sum"] / b["count"]
                b["accuracy"] = b["accuracy_sum"] / b["count"]
            else:
                b["avg_confidence"] = 0.0
                b["accuracy"] = 0.0
        
        return {
            "bins": bins,
            "num_samples": len(confidence_scores),
        }
    
    @staticmethod
    def expected_calibration_error(
        confidence_scores: List[float],
        correctness: List[bool],
        num_bins: int = 10,
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            confidence_scores: List of model confidence scores (0-1)
            correctness: List of boolean values indicating if predictions were correct
            num_bins: Number of bins to use
            
        Returns:
            ECE score (lower is better)
        """
        binning = CalibrationMetrics.bin_predictions(confidence_scores, correctness, num_bins)
        bins = binning["bins"]
        num_samples = binning["num_samples"]
        
        # Calculate ECE
        ece = 0.0
        for b in bins:
            if b["count"] > 0:
                # Weight by fraction of samples in the bin
                weight = b["count"] / num_samples
                
                # Add weighted absolute difference between confidence and accuracy
                ece += weight * abs(b["avg_confidence"] - b["accuracy"])
        
        return ece
    
    @staticmethod
    def brier_score(
        confidence_scores: List[float],
        correctness: List[bool],
    ) -> float:
        """
        Calculate Brier score.
        
        Args:
            confidence_scores: List of model confidence scores (0-1)
            correctness: List of boolean values indicating if predictions were correct
            
        Returns:
            Brier score (lower is better)
        """
        # Convert bool to float
        correctness_float = [float(c) for c in correctness]
        
        # Calculate mean squared error
        brier = mean_squared_error(correctness_float, confidence_scores)
        return brier
    
    @staticmethod
    def calculate_calibration_metrics(
        confidence_scores: List[float],
        correctness: List[bool],
        num_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Calculate all calibration metrics.
        
        Args:
            confidence_scores: List of model confidence scores (0-1)
            correctness: List of boolean values indicating if predictions were correct
            num_bins: Number of bins to use
            
        Returns:
            Dictionary with calibration metrics
        """
        ece = CalibrationMetrics.expected_calibration_error(
            confidence_scores, correctness, num_bins
        )
        
        brier = CalibrationMetrics.brier_score(confidence_scores, correctness)
        
        return {
            "ece": ece,
            "brier_score": brier,
        }


class HallucinationMetrics:
    """Calculate hallucination detection metrics."""
    
    @staticmethod
    def binary_classification_metrics(
        predicted_hallucinations: List[bool],
        true_hallucinations: List[bool],
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 for hallucination detection.
        
        Args:
            predicted_hallucinations: List of booleans indicating predicted hallucinations
            true_hallucinations: List of booleans indicating true hallucinations
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_hallucinations,
            predicted_hallucinations,
            average='binary',
            zero_division=0,
        )
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    
    @staticmethod
    def hallucination_detection_at_threshold(
        uncertainty_scores: List[float],
        true_hallucinations: List[bool],
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Calculate hallucination detection metrics at a specific threshold.
        
        Args:
            uncertainty_scores: List of uncertainty scores (0-1)
            true_hallucinations: List of booleans indicating true hallucinations
            threshold: Threshold for classifying as hallucination
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        # Convert uncertainty scores to binary predictions
        predicted_hallucinations = [u >= threshold for u in uncertainty_scores]
        
        # Calculate metrics
        metrics = HallucinationMetrics.binary_classification_metrics(
            predicted_hallucinations, true_hallucinations
        )
        
        return metrics
    
    @staticmethod
    def hallucination_detection_best_threshold(
        uncertainty_scores: List[float],
        true_hallucinations: List[bool],
        thresholds: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Find the best threshold for hallucination detection.
        
        Args:
            uncertainty_scores: List of uncertainty scores (0-1)
            true_hallucinations: List of booleans indicating true hallucinations
            thresholds: List of thresholds to try (default: 0.05 to 0.95 by 0.05)
            
        Returns:
            Dictionary with best threshold and corresponding metrics
        """
        if thresholds is None:
            thresholds = np.arange(0.05, 1.0, 0.05).tolist()
        
        best_f1 = -1
        best_threshold = None
        best_metrics = None
        
        # Try different thresholds
        threshold_results = []
        for threshold in thresholds:
            metrics = HallucinationMetrics.hallucination_detection_at_threshold(
                uncertainty_scores, true_hallucinations, threshold
            )
            
            threshold_results.append({
                "threshold": threshold,
                **metrics
            })
            
            # Track best F1
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = threshold
                best_metrics = metrics
        
        return {
            "best_threshold": best_threshold,
            "best_metrics": best_metrics,
            "all_thresholds": threshold_results
        }


class QAMetrics:
    """Calculate metrics for question answering tasks."""
    
    @staticmethod
    def normalize_answer(s: str) -> str:
        """
        Normalize answer string for comparison.
        
        Args:
            s: Input string
            
        Returns:
            Normalized string
        """
        # Convert to lowercase
        s = s.lower()
        
        # Remove punctuation
        import string
        for p in string.punctuation:
            s = s.replace(p, '')
        
        # Remove articles
        s = ' '.join([token for token in s.split() if token not in ['a', 'an', 'the']])
        
        # Remove extra whitespace
        s = ' '.join(s.split())
        
        return s
    
    @staticmethod
    def exact_match(prediction: str, reference: Union[str, List[str]]) -> float:
        """
        Calculate exact match score.
        
        Args:
            prediction: Predicted answer
            reference: Reference answer(s)
            
        Returns:
            1.0 if match, 0.0 otherwise
        """
        # Normalize prediction
        norm_prediction = QAMetrics.normalize_answer(prediction)
        
        # Handle single reference
        if isinstance(reference, str):
            reference = [reference]
        
        # Check if prediction matches any reference
        for ref in reference:
            norm_ref = QAMetrics.normalize_answer(ref)
            if norm_prediction == norm_ref:
                return 1.0
        
        return 0.0
    
    @staticmethod
    def f1_score(prediction: str, reference: Union[str, List[str]]) -> float:
        """
        Calculate token-level F1 score.
        
        Args:
            prediction: Predicted answer
            reference: Reference answer(s)
            
        Returns:
            F1 score (0-1)
        """
        # Normalize and tokenize prediction
        norm_prediction = QAMetrics.normalize_answer(prediction)
        prediction_tokens = set(norm_prediction.split())
        
        # Handle single reference
        if isinstance(reference, str):
            reference = [reference]
        
        # Calculate F1 against each reference and take the max
        max_f1 = 0.0
        
        for ref in reference:
            # Normalize and tokenize reference
            norm_ref = QAMetrics.normalize_answer(ref)
            ref_tokens = set(norm_ref.split())
            
            # Calculate intersection
            intersection = prediction_tokens.intersection(ref_tokens)
            
            # Calculate precision and recall
            if len(prediction_tokens) == 0:
                precision = 0.0
            else:
                precision = len(intersection) / len(prediction_tokens)
            
            if len(ref_tokens) == 0:
                recall = 0.0
            else:
                recall = len(intersection) / len(ref_tokens)
            
            # Calculate F1
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            # Update max F1
            max_f1 = max(max_f1, f1)
        
        return max_f1
    
    @staticmethod
    def calculate_qa_metrics(
        predictions: List[str],
        references: List[Union[str, List[str]]],
    ) -> Dict[str, float]:
        """
        Calculate QA metrics for a set of predictions.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers (each can be a string or list of strings)
            
        Returns:
            Dictionary with QA metrics
        """
        if len(predictions) != len(references):
            raise ValueError(f"Length of predictions ({len(predictions)}) must match length of references ({len(references)})")
        
        # Calculate metrics for each example
        exact_matches = []
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            em = QAMetrics.exact_match(pred, ref)
            f1 = QAMetrics.f1_score(pred, ref)
            
            exact_matches.append(em)
            f1_scores.append(f1)
        
        # Calculate aggregate metrics
        return {
            "exact_match": float(np.mean(exact_matches)),
            "f1": float(np.mean(f1_scores)),
        }


class SummarizationMetrics:
    """Calculate metrics for summarization tasks."""
    
    @staticmethod
    def calculate_rouge(
        predictions: List[str],
        references: List[str],
        rouge_types: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            rouge_types: Types of ROUGE to calculate (default: ['rouge1', 'rouge2', 'rougeL'])
            
        Returns:
            Dictionary with ROUGE scores
        """
        if len(predictions) != len(references):
            raise ValueError(f"Length of predictions ({len(predictions)}) must match length of references ({len(references)})")
        
        if rouge_types is None:
            rouge_types = ['rouge1', 'rouge2', 'rougeL']
        
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        
        # Calculate ROUGE for each example
        scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            rouge_scores = scorer.score(ref, pred)
            
            for rouge_type, score in rouge_scores.items():
                scores[f"{rouge_type}_precision"].append(score.precision)
                scores[f"{rouge_type}_recall"].append(score.recall)
                scores[f"{rouge_type}_fmeasure"].append(score.fmeasure)
        
        # Calculate aggregate metrics
        results = {}
        for metric, values in scores.items():
            results[metric] = float(np.mean(values))
        
        return results
    
    @staticmethod
    def calculate_bertscore(
        predictions: List[str],
        references: List[str],
        lang: str = "en",
        model_type: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Calculate BERTScore.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            lang: Language of the texts
            model_type: Model to use for BERTScore
            
        Returns:
            Dictionary with BERTScore metrics
        """
        if len(predictions) != len(references):
            raise ValueError(f"Length of predictions ({len(predictions)}) must match length of references ({len(references)})")
        
        try:
            # Calculate BERTScore
            P, R, F1 = bert_score(
                predictions,
                references,
                lang=lang,
                model_type=model_type,
                verbose=False
            )
            
            # Convert to numpy arrays
            P = P.numpy()
            R = R.numpy()
            F1 = F1.numpy()
            
            return {
                "bertscore_precision": float(np.mean(P)),
                "bertscore_recall": float(np.mean(R)),
                "bertscore_f1": float(np.mean(F1)),
            }
        except:
            logger.warning("Failed to calculate BERTScore. Returning zeros.")
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
            }
    
    @staticmethod
    def calculate_summarization_metrics(
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Calculate all summarization metrics.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with summarization metrics
        """
        # Calculate ROUGE scores
        rouge_scores = SummarizationMetrics.calculate_rouge(
            predictions, references
        )
        
        # Calculate BERTScore (if available)
        try:
            bertscore = SummarizationMetrics.calculate_bertscore(
                predictions, references
            )
        except:
            logger.warning("Failed to calculate BERTScore. Skipping.")
            bertscore = {}
        
        # Combine metrics
        return {
            **rouge_scores,
            **bertscore,
        }


class DiversityMetrics:
    """Calculate diversity metrics for generated text."""
    
    @staticmethod
    def distinct_n(
        texts: List[str],
        n: int = 1,
    ) -> float:
        """
        Calculate Distinct-n metric.
        
        Args:
            texts: List of generated texts
            n: n-gram size
            
        Returns:
            Distinct-n score (higher = more diverse)
        """
        all_ngrams = []
        
        for text in texts:
            # Tokenize
            tokens = text.split()
            
            # Extract n-grams
            text_ngrams = list(ngrams(tokens, n))
            all_ngrams.extend(text_ngrams)
        
        # Count unique n-grams
        unique_ngrams = set(all_ngrams)
        
        # Calculate diversity
        if len(all_ngrams) == 0:
            return 0.0
        
        return len(unique_ngrams) / len(all_ngrams)
    
    @staticmethod
    def self_bleu(
        texts: List[str],
        n: int = 4,
    ) -> float:
        """
        Calculate Self-BLEU metric.
        
        Args:
            texts: List of generated texts
            n: Maximum n-gram size
            
        Returns:
            Self-BLEU score (lower = more diverse)
        """
        if len(texts) <= 1:
            return 0.0
        
        # Tokenize all texts
        tokenized_texts = [text.split() for text in texts]
        
        # Calculate BLEU for each text against all others
        bleu_scores = []
        
        for i, target in enumerate(tokenized_texts):
            # Create a list of references (all texts except the target)
            references = [tokenized_texts[j] for j in range(len(tokenized_texts)) if j != i]
            
            # Calculate BLEU score
            smoothing = SmoothingFunction().method1
            weights = [1/n] * n  # Equal weights for all n-grams
            
            bleu = sentence_bleu(
                references,
                target,
                weights=weights,
                smoothing_function=smoothing
            )
            
            bleu_scores.append(bleu)
        
        # Return average Self-BLEU
        return float(np.mean(bleu_scores))
    
    @staticmethod
    def calculate_diversity_metrics(
        texts: List[str],
    ) -> Dict[str, float]:
        """
        Calculate all diversity metrics.
        
        Args:
            texts: List of generated texts
            
        Returns:
            Dictionary with diversity metrics
        """
        # Calculate Distinct-n for different n
        distinct_1 = DiversityMetrics.distinct_n(texts, n=1)
        distinct_2 = DiversityMetrics.distinct_n(texts, n=2)
        distinct_3 = DiversityMetrics.distinct_n(texts, n=3)
        
        # Calculate Self-BLEU
        self_bleu = DiversityMetrics.self_bleu(texts)
        
        return {
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "distinct_3": distinct_3,
            "self_bleu": self_bleu,
        }


class EfficiencyMetrics:
    """Calculate efficiency metrics."""
    
    @staticmethod
    def time_function(
        func: Callable,
        *args,
        **kwargs
    ) -> Tuple[Any, float]:
        """
        Time a function call.
        
        Args:
            func: Function to time
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            Tuple of (function result, elapsed time in seconds)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        return result, elapsed_time


class EvaluationRunner:
    """Run evaluation for SCEC experiments."""
    
    def __init__(
        self,
        output_dir: str,
    ):
        """
        Initialize the evaluation runner.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_qa(
        self,
        method_results: Dict[str, List[Dict[str, Any]]],
        references: List[Union[str, List[str]]],
        true_hallucinations: Optional[List[bool]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate QA results for multiple methods.
        
        Args:
            method_results: Dictionary mapping method names to lists of result dictionaries
            references: List of reference answers
            true_hallucinations: Optional list of booleans indicating true hallucinations
            
        Returns:
            Dictionary with evaluation results for each method
        """
        eval_results = {}
        
        for method_name, results in method_results.items():
            # Extract predictions and confidence scores
            predictions = [r["text"] for r in results]
            confidence_scores = [1.0 - r.get("uncertainty_score", 0.0) for r in results]
            
            # Calculate QA metrics
            qa_metrics = QAMetrics.calculate_qa_metrics(predictions, references)
            
            # Check if answers are correct (for calibration)
            correctness = []
            for pred, ref in zip(predictions, references):
                # Check if prediction matches any reference
                em = QAMetrics.exact_match(pred, ref)
                correctness.append(em > 0.5)  # Convert to boolean
            
            # Calculate calibration metrics
            calibration_metrics = CalibrationMetrics.calculate_calibration_metrics(
                confidence_scores, correctness
            )
            
            # Calculate hallucination metrics if true hallucinations are provided
            hallucination_metrics = {}
            if true_hallucinations is not None:
                uncertainty_scores = [r.get("uncertainty_score", 0.0) for r in results]
                
                # Find best threshold
                hallucination_metrics = HallucinationMetrics.hallucination_detection_best_threshold(
                    uncertainty_scores, true_hallucinations
                )
            
            # Calculate diversity metrics
            diversity_metrics = DiversityMetrics.calculate_diversity_metrics(predictions)
            
            # Combine all metrics
            eval_results[method_name] = {
                "qa_metrics": qa_metrics,
                "calibration_metrics": calibration_metrics,
                "hallucination_metrics": hallucination_metrics,
                "diversity_metrics": diversity_metrics,
            }
        
        return eval_results
    
    def evaluate_summarization(
        self,
        method_results: Dict[str, List[Dict[str, Any]]],
        references: List[str],
        true_hallucinations: Optional[List[bool]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate summarization results for multiple methods.
        
        Args:
            method_results: Dictionary mapping method names to lists of result dictionaries
            references: List of reference summaries
            true_hallucinations: Optional list of booleans indicating true hallucinations
            
        Returns:
            Dictionary with evaluation results for each method
        """
        eval_results = {}
        
        for method_name, results in method_results.items():
            # Extract predictions and confidence scores
            predictions = [r["text"] for r in results]
            confidence_scores = [1.0 - r.get("uncertainty_score", 0.0) for r in results]
            
            # Calculate summarization metrics
            summarization_metrics = SummarizationMetrics.calculate_summarization_metrics(
                predictions, references
            )
            
            # For calibration, we need a notion of correctness
            # We'll use ROUGE-L F1 > 0.3 as a proxy for correctness
            rouge_scores = SummarizationMetrics.calculate_rouge(
                predictions, references, rouge_types=["rougeL"]
            )
            
            correctness = [score > 0.3 for score in rouge_scores["rougeL_fmeasure"]]
            
            # Calculate calibration metrics
            calibration_metrics = CalibrationMetrics.calculate_calibration_metrics(
                confidence_scores, correctness
            )
            
            # Calculate hallucination metrics if true hallucinations are provided
            hallucination_metrics = {}
            if true_hallucinations is not None:
                uncertainty_scores = [r.get("uncertainty_score", 0.0) for r in results]
                
                # Find best threshold
                hallucination_metrics = HallucinationMetrics.hallucination_detection_best_threshold(
                    uncertainty_scores, true_hallucinations
                )
            
            # Calculate diversity metrics
            diversity_metrics = DiversityMetrics.calculate_diversity_metrics(predictions)
            
            # Combine all metrics
            eval_results[method_name] = {
                "summarization_metrics": summarization_metrics,
                "calibration_metrics": calibration_metrics,
                "hallucination_metrics": hallucination_metrics,
                "diversity_metrics": diversity_metrics,
            }
        
        return eval_results
    
    def save_evaluation_results(
        self,
        eval_results: Dict[str, Dict[str, Any]],
        task_type: str,
        filename: Optional[str] = None,
    ) -> str:
        """
        Save evaluation results to a JSON file.
        
        Args:
            eval_results: Evaluation results dictionary
            task_type: Type of task ('qa' or 'summarization')
            filename: Optional filename (default: {task_type}_eval_results.json)
            
        Returns:
            Path to the saved results file
        """
        if filename is None:
            filename = f"{task_type}_eval_results.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Convert numpy arrays and other non-serializable objects to Python types
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        # Convert results for JSON serialization
        serializable_results = convert_for_json(eval_results)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {output_path}")
        
        return output_path


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage (commented out since it requires actual results)
    """
    # QA evaluation example
    method_results = {
        "vanilla": [
            {"text": "Paris", "uncertainty_score": 0.1},
            {"text": "Berlin", "uncertainty_score": 0.2},
        ],
        "scec": [
            {"text": "Paris", "uncertainty_score": 0.1},
            {"text": "London", "uncertainty_score": 0.8},
        ],
    }
    
    references = [
        ["Paris", "City of Paris"],
        ["London", "City of London"],
    ]
    
    true_hallucinations = [False, True]
    
    # Create evaluator
    evaluator = EvaluationRunner("results")
    
    # Run QA evaluation
    qa_eval_results = evaluator.evaluate_qa(
        method_results, references, true_hallucinations
    )
    
    # Save results
    evaluator.save_evaluation_results(qa_eval_results, "qa")
    """
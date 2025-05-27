"""
Evaluation metrics and utilities for the Self-Correcting Language Model experiment.
"""
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import re

from config import EVAL_CONFIG, logger

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {e}")


class Evaluator:
    """Base evaluator class."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EVAL_CONFIG
    
    def evaluate(self, predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate predictions against references.
        
        Args:
            predictions: List of prediction dictionaries
            references: List of reference dictionaries
            
        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate()")


class FactualQAEvaluator(Evaluator):
    """Evaluator for factual QA tasks (e.g., TruthfulQA)."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize factual QA evaluator.
        
        Args:
            config: Evaluation configuration
        """
        super().__init__(config)
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def _prepare_data(self, predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Tuple[List, List]:
        """
        Prepare data for evaluation.
        
        Args:
            predictions: List of prediction dictionaries
            references: List of reference dictionaries
            
        Returns:
            Tuple of prepared predictions and references
        """
        prepared_predictions = []
        prepared_references = []
        
        for pred, ref in zip(predictions, references):
            # Process the prediction
            prediction_text = pred.get("final_text", "")
            
            # Process the reference answers
            reference_answers = ref.get("correct_answers", [])
            
            prepared_predictions.append(prediction_text)
            prepared_references.append(reference_answers)
        
        return prepared_predictions, prepared_references
    
    def _calculate_factuality(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Calculate factuality metrics.
        
        Args:
            predictions: List of prediction texts
            references: List of reference answer lists
            
        Returns:
            Dictionary of factuality metrics
        """
        factuality_metrics = {}
        
        # Simulate factuality scoring
        # In a real implementation, this would involve a more sophisticated approach
        correct_count = 0
        for pred, refs in zip(predictions, references):
            # Check if any reference answer is contained in the prediction
            is_correct = any(ref.lower() in pred.lower() for ref in refs)
            correct_count += 1 if is_correct else 0
        
        factuality_metrics["accuracy"] = correct_count / len(predictions) if predictions else 0.0
        
        return factuality_metrics
    
    def _calculate_hallucination_rate(self, predictions: List[Dict[str, Any]]) -> float:
        """
        Calculate hallucination rate.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Hallucination rate
        """
        total_corrections = sum(len(pred.get("corrections", [])) for pred in predictions)
        return total_corrections / len(predictions) if predictions else 0.0
    
    def _calculate_fluency(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Calculate fluency metrics (BLEU, ROUGE).
        
        Args:
            predictions: List of prediction texts
            references: List of reference answer lists
            
        Returns:
            Dictionary of fluency metrics
        """
        fluency_metrics = {}
        
        # Calculate BLEU
        bleu_scores = []
        for pred, refs in zip(predictions, references):
            # Tokenize
            pred_tokens = nltk.word_tokenize(pred.lower())
            refs_tokens = [nltk.word_tokenize(ref.lower()) for ref in refs]
            
            # Calculate BLEU with smoothing
            try:
                if refs_tokens:
                    bleu = sentence_bleu(refs_tokens, pred_tokens, 
                                        smoothing_function=SmoothingFunction().method1)
                    bleu_scores.append(bleu)
                else:
                    bleu_scores.append(0.0)
            except Exception as e:
                logger.warning(f"Failed to calculate BLEU: {e}")
                bleu_scores.append(0.0)
        
        fluency_metrics["bleu"] = np.mean(bleu_scores) if bleu_scores else 0.0
        
        # Calculate ROUGE
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, refs in zip(predictions, references):
            # Calculate ROUGE for each reference and take the best
            best_rouge1 = 0.0
            best_rouge2 = 0.0
            best_rougeL = 0.0
            
            for ref in refs:
                try:
                    rouge_scores = self.rouge_scorer.score(ref, pred)
                    
                    best_rouge1 = max(best_rouge1, rouge_scores['rouge1'].fmeasure)
                    best_rouge2 = max(best_rouge2, rouge_scores['rouge2'].fmeasure)
                    best_rougeL = max(best_rougeL, rouge_scores['rougeL'].fmeasure)
                except Exception as e:
                    logger.warning(f"Failed to calculate ROUGE: {e}")
            
            rouge1_scores.append(best_rouge1)
            rouge2_scores.append(best_rouge2)
            rougeL_scores.append(best_rougeL)
        
        fluency_metrics["rouge1"] = np.mean(rouge1_scores) if rouge1_scores else 0.0
        fluency_metrics["rouge2"] = np.mean(rouge2_scores) if rouge2_scores else 0.0
        fluency_metrics["rougeL"] = np.mean(rougeL_scores) if rougeL_scores else 0.0
        
        return fluency_metrics
    
    def _calculate_efficiency(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate efficiency metrics.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary of efficiency metrics
        """
        efficiency_metrics = {}
        
        # Extract latency values
        latencies = [pred.get("metrics", {}).get("latency", 0.0) for pred in predictions]
        
        # Average latency
        efficiency_metrics["latency"] = np.mean(latencies) if latencies else 0.0
        
        # Average number of iterations
        iterations = [pred.get("metrics", {}).get("num_iterations", 0) for pred in predictions]
        efficiency_metrics["avg_iterations"] = np.mean(iterations) if iterations else 0.0
        
        # Average confidence improvement
        conf_improvements = [pred.get("metrics", {}).get("confidence_improvement", 0.0) 
                            for pred in predictions]
        efficiency_metrics["avg_confidence_improvement"] = np.mean(conf_improvements) if conf_improvements else 0.0
        
        return efficiency_metrics
    
    def evaluate(self, predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate predictions against references.
        
        Args:
            predictions: List of prediction dictionaries
            references: List of reference dictionaries
            
        Returns:
            Dictionary of evaluation metrics
        """
        start_time = time.time()
        
        # Prepare data
        prepared_predictions, prepared_references = self._prepare_data(predictions, references)
        
        # Calculate metrics
        metrics = {}
        
        # Factuality metrics
        factuality_metrics = self._calculate_factuality(prepared_predictions, prepared_references)
        metrics.update(factuality_metrics)
        
        # Hallucination rate
        metrics["hallucination_rate"] = self._calculate_hallucination_rate(predictions)
        
        # Fluency metrics
        fluency_metrics = self._calculate_fluency(prepared_predictions, prepared_references)
        metrics.update(fluency_metrics)
        
        # Efficiency metrics
        efficiency_metrics = self._calculate_efficiency(predictions)
        metrics.update(efficiency_metrics)
        
        # Record evaluation time
        metrics["eval_time"] = time.time() - start_time
        
        return metrics


class FactVerificationEvaluator(Evaluator):
    """Evaluator for fact verification tasks (e.g., FEVER)."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize fact verification evaluator.
        
        Args:
            config: Evaluation configuration
        """
        super().__init__(config)
    
    def _extract_claim_verification(self, text: str) -> str:
        """
        Extract claim verification (SUPPORTS, REFUTES, NOT ENOUGH INFO) from text.
        
        Args:
            text: Generated text
            
        Returns:
            Extracted verification (supports, refutes, nei)
        """
        text = text.lower()
        
        # Look for explicit verification statements
        if re.search(r'(support|true|correct|accurate|valid|confirmed)', text):
            return "supports"
        elif re.search(r'(refute|false|incorrect|inaccurate|invalid|wrong)', text):
            return "refutes"
        elif re.search(r'(not enough|insufficient|uncertain|unclear|cannot)', text):
            return "nei"
        
        # Default to "nei" if no clear indication
        return "nei"
    
    def _prepare_data(self, predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Tuple[List, List]:
        """
        Prepare data for evaluation.
        
        Args:
            predictions: List of prediction dictionaries
            references: List of reference dictionaries
            
        Returns:
            Tuple of prepared predictions and references
        """
        prepared_predictions = []
        prepared_references = []
        
        for pred, ref in zip(predictions, references):
            # Process the prediction
            prediction_text = pred.get("final_text", "")
            prediction_label = self._extract_claim_verification(prediction_text)
            
            # Process the reference
            reference_label = ref.get("label", "NOT ENOUGH INFO").lower()
            if reference_label == "supports":
                reference_label = "supports"
            elif reference_label == "refutes":
                reference_label = "refutes"
            else:
                reference_label = "nei"
            
            prepared_predictions.append(prediction_label)
            prepared_references.append(reference_label)
        
        return prepared_predictions, prepared_references
    
    def _calculate_accuracy(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate accuracy metrics.
        
        Args:
            predictions: List of prediction labels
            references: List of reference labels
            
        Returns:
            Dictionary of accuracy metrics
        """
        accuracy_metrics = {}
        
        # Calculate accuracy
        accuracy = accuracy_score(references, predictions)
        accuracy_metrics["accuracy"] = accuracy
        
        # Calculate F1 score (macro-averaged)
        try:
            f1 = f1_score(references, predictions, average='macro', zero_division=0)
            accuracy_metrics["f1"] = f1
        except Exception as e:
            logger.warning(f"Failed to calculate F1: {e}")
            accuracy_metrics["f1"] = 0.0
        
        return accuracy_metrics
    
    def _calculate_confusion_matrix(self, predictions: List[str], references: List[str]) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            predictions: List of prediction labels
            references: List of reference labels
            
        Returns:
            Confusion matrix
        """
        # Get unique classes
        classes = sorted(set(predictions + references))
        
        # Calculate confusion matrix
        cm = confusion_matrix(references, predictions, labels=classes)
        
        return cm, classes
    
    def evaluate(self, predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate predictions against references.
        
        Args:
            predictions: List of prediction dictionaries
            references: List of reference dictionaries
            
        Returns:
            Dictionary of evaluation metrics
        """
        start_time = time.time()
        
        # Prepare data
        prepared_predictions, prepared_references = self._prepare_data(predictions, references)
        
        # Calculate metrics
        metrics = {}
        
        # Accuracy metrics
        accuracy_metrics = self._calculate_accuracy(prepared_predictions, prepared_references)
        metrics.update(accuracy_metrics)
        
        # Confusion matrix
        cm, classes = self._calculate_confusion_matrix(prepared_predictions, prepared_references)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["classes"] = classes
        
        # Hallucination rate (number of corrections per sample)
        total_corrections = sum(len(pred.get("corrections", [])) for pred in predictions)
        metrics["hallucination_rate"] = total_corrections / len(predictions) if predictions else 0.0
        
        # Efficiency metrics (same as FactualQAEvaluator)
        latencies = [pred.get("metrics", {}).get("latency", 0.0) for pred in predictions]
        metrics["latency"] = np.mean(latencies) if latencies else 0.0
        
        iterations = [pred.get("metrics", {}).get("num_iterations", 0) for pred in predictions]
        metrics["avg_iterations"] = np.mean(iterations) if iterations else 0.0
        
        conf_improvements = [pred.get("metrics", {}).get("confidence_improvement", 0.0) 
                            for pred in predictions]
        metrics["avg_confidence_improvement"] = np.mean(conf_improvements) if conf_improvements else 0.0
        
        # Record evaluation time
        metrics["eval_time"] = time.time() - start_time
        
        return metrics


def get_evaluator(dataset_name: str, config: Dict[str, Any] = None) -> Evaluator:
    """
    Factory function to get the appropriate evaluator.
    
    Args:
        dataset_name: Name of the dataset
        config: Evaluation configuration
    
    Returns:
        Evaluator instance
    """
    if dataset_name.lower() == "truthfulqa":
        return FactualQAEvaluator(config)
    elif dataset_name.lower() == "fever":
        return FactVerificationEvaluator(config)
    else:
        logger.warning(f"No specific evaluator for {dataset_name}, using FactualQAEvaluator as default")
        return FactualQAEvaluator(config)
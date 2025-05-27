"""
Uncertainty quantification module for Reasoning Uncertainty Networks (RUNs) experiment.

This module implements various uncertainty estimation methods for LLMs:
1. SelfCheckGPT - Consistency-based uncertainty estimation
2. Multi-dimensional UQ - Tensor decomposition based uncertainty estimation
3. Calibration-based approaches - Post-hoc calibration of LLM probabilities
4. HuDEx - Explanation-enhanced hallucination detection
5. MetaQA - Metamorphic relation-based hallucination detection
"""
import os
import json
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import random
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer

from config import BASELINE_CONFIG, LLM_CONFIG, MODELS_DIR
from model import LLMInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SelfCheckGPT:
    """
    Implementation of SelfCheckGPT for hallucination detection.
    
    Reference: "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for 
               Generative Large Language Models" (Manakul et al., 2023)
    """
    
    def __init__(self, llm_interface: LLMInterface = None, config: Dict = None):
        """
        Initialize SelfCheckGPT.
        
        Args:
            llm_interface: Interface for LLM interactions
            config: Configuration for SelfCheckGPT
        """
        if config is None:
            config = BASELINE_CONFIG["selfcheckgpt"]
        
        self.config = config
        self.llm = llm_interface or LLMInterface()
        
        # Initialize embedding model for semantic similarity
        self.embedding_model = SentenceTransformer(LLM_CONFIG["embedding_model"]["name"])
        logger.info(f"Initialized SelfCheckGPT with {self.config['num_samples']} samples")
    
    def generate_sample_responses(self, question: str, context: str = "") -> List[str]:
        """
        Generate multiple responses for the same question.
        
        Args:
            question: Question to generate responses for
            context: Additional context
            
        Returns:
            List of sample responses
        """
        prompt = f"""
Question: {question}

Context: {context}

Answer the question based on the provided context.
"""
        
        samples = []
        for i in range(self.config["num_samples"]):
            response = self.llm.generate(
                prompt, 
                temperature=self.config["temperature"]
            )
            samples.append(response)
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
        
        return samples
    
    def compute_similarity_matrix(self, samples: List[str]) -> np.ndarray:
        """
        Compute pairwise similarity matrix for sample responses.
        
        Args:
            samples: List of sample responses
            
        Returns:
            Similarity matrix
        """
        # Get embeddings for all samples
        embeddings = self.embedding_model.encode(samples)
        
        # Compute pairwise cosine similarities
        similarity_matrix = np.zeros((len(samples), len(samples)))
        
        for i in range(len(samples)):
            for j in range(len(samples)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def compute_hallucination_score(self, similarity_matrix: np.ndarray) -> float:
        """
        Compute hallucination score based on similarity matrix.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            
        Returns:
            Hallucination score (0-1, higher means more likely to be a hallucination)
        """
        # Extract upper triangular part (excluding diagonal)
        upper_indices = np.triu_indices(similarity_matrix.shape[0], k=1)
        similarities = similarity_matrix[upper_indices]
        
        # Calculate statistics of similarities
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Compute hallucination score
        # Lower similarity -> higher hallucination score
        hallucination_score = 1 - (mean_sim - self.config["similarity_threshold"]) / (1 - self.config["similarity_threshold"])
        hallucination_score = min(max(hallucination_score, 0.0), 1.0)  # Clip to [0, 1]
        
        return hallucination_score
    
    def detect_hallucination(self, question: str, context: str = "") -> Tuple[bool, float, Dict]:
        """
        Detect hallucination in response to a question.
        
        Args:
            question: Question to check
            context: Additional context
            
        Returns:
            Tuple of (is_hallucination, hallucination_score, details)
        """
        # Generate sample responses
        samples = self.generate_sample_responses(question, context)
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(samples)
        
        # Compute hallucination score
        hallucination_score = self.compute_hallucination_score(similarity_matrix)
        
        # Determine if it's a hallucination
        is_hallucination = hallucination_score > 0.5
        
        # Collect details
        details = {
            "num_samples": len(samples),
            "similarity_matrix": similarity_matrix.tolist(),
            "mean_similarity": float(np.mean(similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)])),
            "std_similarity": float(np.std(similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)])),
            "samples": samples
        }
        
        return is_hallucination, hallucination_score, details


class MultiDimensionalUQ:
    """
    Implementation of Multi-dimensional Uncertainty Quantification for LLMs.
    
    Reference: "Uncertainty Quantification of Large Language Models through 
                Multi-Dimensional Responses" (Chen et al., 2025)
    """
    
    def __init__(self, llm_interface: LLMInterface = None, config: Dict = None):
        """
        Initialize Multi-dimensional UQ.
        
        Args:
            llm_interface: Interface for LLM interactions
            config: Configuration for Multi-dimensional UQ
        """
        if config is None:
            config = BASELINE_CONFIG["multidim_uq"]
        
        self.config = config
        self.llm = llm_interface or LLMInterface()
        
        # Initialize embedding model for semantic similarity
        self.embedding_model = SentenceTransformer(LLM_CONFIG["embedding_model"]["name"])
        logger.info(f"Initialized MultiDimensionalUQ with {self.config['num_responses']} responses and {self.config['num_dimensions']} dimensions")
    
    def generate_responses(self, question: str, context: str = "") -> List[str]:
        """
        Generate multiple responses for the same question.
        
        Args:
            question: Question to generate responses for
            context: Additional context
            
        Returns:
            List of responses
        """
        prompt = f"""
Question: {question}

Context: {context}

Answer the question based on the provided context.
"""
        
        responses = []
        for i in range(self.config["num_responses"]):
            response = self.llm.generate(
                prompt, 
                temperature=0.7  # Higher temperature for diversity
            )
            responses.append(response)
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
        
        return responses
    
    def compute_similarity_tensor(self, responses: List[str]) -> np.ndarray:
        """
        Compute similarity tensor based on different dimensions of similarity.
        
        Args:
            responses: List of responses
            
        Returns:
            Similarity tensor
        """
        num_responses = len(responses)
        num_dimensions = self.config["num_dimensions"]
        similarity_tensor = np.zeros((num_responses, num_responses, num_dimensions))
        
        # Get embeddings
        embeddings = self.embedding_model.encode(responses)
        
        # Dimension 1: Semantic similarity
        for i in range(num_responses):
            for j in range(num_responses):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarity_tensor[i, j, 0] = similarity
        
        # Dimension 2: Length similarity (normalized difference in length)
        lengths = np.array([len(resp) for resp in responses])
        length_diffs = np.abs(lengths.reshape(-1, 1) - lengths.reshape(1, -1))
        max_length_diff = np.max(length_diffs) if np.max(length_diffs) > 0 else 1
        length_similarities = 1 - length_diffs / max_length_diff
        
        for i in range(num_responses):
            for j in range(num_responses):
                similarity_tensor[i, j, 1] = length_similarities[i, j]
        
        # Dimension 3: Structure similarity (simplified, based on sentence count)
        sentence_counts = np.array([len(resp.split('.')) for resp in responses])
        sentence_diffs = np.abs(sentence_counts.reshape(-1, 1) - sentence_counts.reshape(1, -1))
        max_sentence_diff = np.max(sentence_diffs) if np.max(sentence_diffs) > 0 else 1
        structure_similarities = 1 - sentence_diffs / max_sentence_diff
        
        for i in range(num_responses):
            for j in range(num_responses):
                similarity_tensor[i, j, 2] = structure_similarities[i, j]
        
        return similarity_tensor
    
    def compute_tensor_decomposition(self, similarity_tensor: np.ndarray) -> np.ndarray:
        """
        Compute tensor decomposition for uncertainty estimation.
        
        Args:
            similarity_tensor: 3D similarity tensor
            
        Returns:
            Uncertainty representation vector
        """
        # For simplicity, we'll use a simplified approach:
        # Average the similarity matrices across dimensions
        avg_similarity = np.mean(similarity_tensor, axis=2)
        
        # Extract upper triangular part (excluding diagonal)
        upper_indices = np.triu_indices(avg_similarity.shape[0], k=1)
        similarities = avg_similarity[upper_indices]
        
        # Compute statistics as uncertainty representation
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        
        uncertainty_repr = np.array([mean_sim, std_sim, min_sim, max_sim])
        
        return uncertainty_repr
    
    def compute_hallucination_score(self, uncertainty_repr: np.ndarray) -> float:
        """
        Compute hallucination score from uncertainty representation.
        
        Args:
            uncertainty_repr: Uncertainty representation vector
            
        Returns:
            Hallucination score (0-1, higher means more likely to be a hallucination)
        """
        # Extract components
        mean_sim, std_sim, min_sim, max_sim = uncertainty_repr
        
        # Compute hallucination score
        # Lower mean similarity and higher std -> higher hallucination score
        hallucination_score = (1 - mean_sim) * (1 + std_sim)
        hallucination_score = min(max(hallucination_score, 0.0), 1.0)  # Clip to [0, 1]
        
        return hallucination_score
    
    def detect_hallucination(self, question: str, context: str = "") -> Tuple[bool, float, Dict]:
        """
        Detect hallucination in response to a question.
        
        Args:
            question: Question to check
            context: Additional context
            
        Returns:
            Tuple of (is_hallucination, hallucination_score, details)
        """
        # Generate responses
        responses = self.generate_responses(question, context)
        
        # Compute similarity tensor
        similarity_tensor = self.compute_similarity_tensor(responses)
        
        # Compute tensor decomposition
        uncertainty_repr = self.compute_tensor_decomposition(similarity_tensor)
        
        # Compute hallucination score
        hallucination_score = self.compute_hallucination_score(uncertainty_repr)
        
        # Determine if it's a hallucination
        is_hallucination = hallucination_score > 0.5
        
        # Collect details
        details = {
            "num_responses": len(responses),
            "uncertainty_representation": uncertainty_repr.tolist(),
            "responses": responses
        }
        
        return is_hallucination, hallucination_score, details


class CalibrationBasedUQ:
    """
    Implementation of Calibration-based Uncertainty Quantification for LLMs.
    
    Uses temperature scaling and isotonic regression for post-hoc calibration.
    """
    
    def __init__(self, llm_interface: LLMInterface = None, config: Dict = None):
        """
        Initialize Calibration-based UQ.
        
        Args:
            llm_interface: Interface for LLM interactions
            config: Configuration for Calibration
        """
        if config is None:
            config = BASELINE_CONFIG["calibration"]
        
        self.config = config
        self.llm = llm_interface or LLMInterface()
        
        # Initialize calibrators
        self.temperature = 1.0  # Default temperature
        self.isotonic_regressor = IsotonicRegression(out_of_bounds="clip")
        self.calibrated = False
        
        logger.info(f"Initialized CalibrationBasedUQ with method: {self.config['method']}")
    
    def calibrate(self, validation_data: List[Tuple[str, str, bool]]) -> None:
        """
        Calibrate the model using validation data.
        
        Args:
            validation_data: List of tuples (question, context, is_hallucination)
        """
        logger.info(f"Calibrating with {len(validation_data)} validation examples")
        
        # Get raw confidence scores
        raw_scores = []
        true_labels = []
        
        for question, context, is_hallucination in tqdm(validation_data, desc="Calibrating"):
            confidence = self._get_raw_confidence(question, context)
            raw_scores.append(confidence)
            true_labels.append(int(is_hallucination))
        
        # Apply calibration method
        if self.config["method"] == "temperature_scaling":
            self._calibrate_temperature(raw_scores, true_labels)
        elif self.config["method"] == "isotonic_regression":
            self._calibrate_isotonic(raw_scores, true_labels)
        else:
            logger.warning(f"Unknown calibration method: {self.config['method']}")
            return
        
        self.calibrated = True
    
    def _get_raw_confidence(self, question: str, context: str = "") -> float:
        """
        Get raw confidence from the LLM.
        
        Args:
            question: Question to check
            context: Additional context
            
        Returns:
            Raw confidence score
        """
        prompt = f"""
Question: {question}

Context: {context}

First, answer the question based on the provided context.
Then, on a scale from 0 to 100, rate your confidence in your answer.
Format your response as:

Answer: [your answer]
Confidence: [0-100]
"""
        
        response = self.llm.generate(prompt, temperature=0.0)
        
        # Extract confidence from response
        try:
            if "Confidence:" in response:
                confidence_line = [line for line in response.split('\n') if "Confidence:" in line][0]
                confidence_str = confidence_line.split("Confidence:")[1].strip().replace('%', '')
                confidence = float(confidence_str) / 100.0
                return min(max(confidence, 0.0), 1.0)  # Ensure in [0, 1]
            else:
                return 0.5  # Default confidence
        except:
            return 0.5  # Default confidence
    
    def _calibrate_temperature(self, raw_scores: List[float], true_labels: List[int]) -> None:
        """
        Calibrate using temperature scaling.
        
        Args:
            raw_scores: List of raw confidence scores
            true_labels: List of true labels (1 for hallucination, 0 for not)
        """
        # Convert to numpy arrays
        raw_scores = np.array(raw_scores)
        true_labels = np.array(true_labels)
        
        # For hallucination detection, we need to adjust:
        # Higher score should mean higher probability of hallucination
        # So we need to use 1 - raw_scores if raw_scores represent confidence
        adjusted_scores = 1 - raw_scores
        
        # Grid search for optimal temperature
        best_temp = 1.0
        best_loss = float('inf')
        
        for temp in np.linspace(0.1, 10.0, 100):
            # Apply temperature scaling
            calibrated_scores = 1 / (1 + np.exp(-np.log(adjusted_scores / (1 - adjusted_scores)) / temp))
            
            # Compute loss (negative log likelihood)
            eps = 1e-7  # Small epsilon to avoid log(0)
            calibrated_scores = np.clip(calibrated_scores, eps, 1 - eps)
            loss = -np.mean(true_labels * np.log(calibrated_scores) + (1 - true_labels) * np.log(1 - calibrated_scores))
            
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
        
        self.temperature = best_temp
        logger.info(f"Optimal temperature: {self.temperature:.4f}")
    
    def _calibrate_isotonic(self, raw_scores: List[float], true_labels: List[int]) -> None:
        """
        Calibrate using isotonic regression.
        
        Args:
            raw_scores: List of raw confidence scores
            true_labels: List of true labels (1 for hallucination, 0 for not)
        """
        # Convert to numpy arrays
        raw_scores = np.array(raw_scores)
        true_labels = np.array(true_labels)
        
        # For hallucination detection, we need to adjust:
        # Higher score should mean higher probability of hallucination
        adjusted_scores = 1 - raw_scores
        
        # Fit isotonic regression
        self.isotonic_regressor.fit(adjusted_scores, true_labels)
        logger.info("Fitted isotonic regression calibrator")
    
    def get_calibrated_score(self, raw_score: float) -> float:
        """
        Get calibrated score from raw score.
        
        Args:
            raw_score: Raw confidence score
            
        Returns:
            Calibrated score
        """
        if not self.calibrated:
            logger.warning("Model not calibrated yet, returning raw score")
            return raw_score
        
        # Adjust raw score
        adjusted_score = 1 - raw_score
        
        # Apply calibration
        if self.config["method"] == "temperature_scaling":
            calibrated = 1 / (1 + np.exp(-np.log(adjusted_score / (1 - adjusted_score)) / self.temperature))
        elif self.config["method"] == "isotonic_regression":
            calibrated = self.isotonic_regressor.predict([adjusted_score])[0]
        else:
            calibrated = adjusted_score
        
        return min(max(calibrated, 0.0), 1.0)  # Ensure in [0, 1]
    
    def detect_hallucination(self, question: str, context: str = "") -> Tuple[bool, float, Dict]:
        """
        Detect hallucination in response to a question.
        
        Args:
            question: Question to check
            context: Additional context
            
        Returns:
            Tuple of (is_hallucination, hallucination_score, details)
        """
        # Get raw confidence
        raw_confidence = self._get_raw_confidence(question, context)
        
        # Convert to hallucination score (higher score = higher chance of hallucination)
        raw_hallucination_score = 1 - raw_confidence
        
        # Get calibrated score
        calibrated_score = self.get_calibrated_score(raw_confidence)
        
        # Determine if it's a hallucination
        is_hallucination = calibrated_score > 0.5
        
        # Collect details
        details = {
            "raw_confidence": raw_confidence,
            "raw_hallucination_score": raw_hallucination_score,
            "calibrated": self.calibrated,
            "calibration_method": self.config["method"],
            "temperature": self.temperature if self.config["method"] == "temperature_scaling" else None
        }
        
        return is_hallucination, calibrated_score, details


class HuDEx:
    """
    Implementation of HuDEx: Hallucination Detection with Explanations for LLMs.
    
    Reference: "HuDEx: Integrating Hallucination Detection and Explainability for 
                Enhancing the Reliability of LLM responses" (Lee et al., 2025)
    """
    
    def __init__(self, llm_interface: LLMInterface = None, config: Dict = None):
        """
        Initialize HuDEx.
        
        Args:
            llm_interface: Interface for LLM interactions
            config: Configuration for HuDEx
        """
        if config is None:
            config = BASELINE_CONFIG["hudex"]
        
        self.config = config
        self.llm = llm_interface or LLMInterface()
        
        # For explanation model, we'll use the same LLM for simplicity
        # In a real implementation, this might be a separate model
        self.explanation_llm = self.llm
        
        logger.info(f"Initialized HuDEx with threshold: {self.config['threshold']}")
    
    def generate_explanation(self, question: str, context: str, answer: str) -> str:
        """
        Generate an explanation for why an answer might be hallucinated.
        
        Args:
            question: Original question
            context: Provided context
            answer: Generated answer to evaluate
            
        Returns:
            Explanation of potential hallucinations
        """
        prompt = f"""
You are a critical evaluator analyzing whether an answer to a question contains hallucinations based on the provided context.

Question: {question}

Context: {context}

Answer to evaluate: {answer}

Please explain whether the answer contains any hallucinations (information not supported by the context).
Focus on specific claims in the answer and whether they are directly supported by the context.
Be specific about which parts might be hallucinated and why.

Your explanation:
"""
        
        explanation = self.explanation_llm.generate(prompt, temperature=0.1)
        return explanation
    
    def detect_hallucination_from_explanation(self, explanation: str) -> Tuple[bool, float]:
        """
        Detect hallucination based on the generated explanation.
        
        Args:
            explanation: Generated explanation
            
        Returns:
            Tuple of (is_hallucination, hallucination_score)
        """
        prompt = f"""
Based on the following explanation, determine whether the answer contains hallucinations.
Provide a score from 0 to 100, where:
- 0 means definitely no hallucinations
- 100 means definitely contains hallucinations
Reply with ONLY a number between 0 and 100, no other text.

Explanation: {explanation}

Hallucination score (0-100):
"""
        
        try:
            response = self.llm.generate(prompt, temperature=0.0)
            score = float(response.strip().replace('%', '')) / 100.0
            score = min(max(score, 0.0), 1.0)  # Ensure in [0, 1]
            
            is_hallucination = score > self.config["threshold"]
            return is_hallucination, score
        except:
            logger.warning("Failed to extract hallucination score from explanation")
            return False, 0.0
    
    def extract_evidence(self, question: str, context: str) -> List[str]:
        """
        Extract relevant evidence from the context.
        
        Args:
            question: Question to answer
            context: Context to extract evidence from
            
        Returns:
            List of evidence statements
        """
        prompt = f"""
Extract the key pieces of information from the context that are relevant to answering the question.
List each piece of information as a separate point.

Question: {question}

Context: {context}

Relevant information (list each point separately):
"""
        
        response = self.llm.generate(prompt, temperature=0.1)
        
        # Extract evidence points
        evidence = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('- ')):
                # Remove the leading number/bullet and any trailing/leading whitespace
                clean_line = line.lstrip('0123456789.- \t')
                if clean_line:
                    evidence.append(clean_line)
        
        return evidence
    
    def detect_hallucination(self, question: str, context: str = "") -> Tuple[bool, float, Dict]:
        """
        Detect hallucination in response to a question.
        
        Args:
            question: Question to check
            context: Additional context
            
        Returns:
            Tuple of (is_hallucination, hallucination_score, details)
        """
        # First, generate an answer
        prompt = f"""
Question: {question}

Context: {context}

Please answer the question based on the provided context.
"""
        
        answer = self.llm.generate(prompt, temperature=0.2)
        
        # Extract evidence from context
        evidence = self.extract_evidence(question, context)
        
        # Generate explanation
        explanation = self.generate_explanation(question, context, answer)
        
        # Detect hallucination from explanation
        is_hallucination, hallucination_score = self.detect_hallucination_from_explanation(explanation)
        
        # Collect details
        details = {
            "answer": answer,
            "evidence": evidence,
            "explanation": explanation
        }
        
        return is_hallucination, hallucination_score, details


class MetaQA:
    """
    Implementation of MetaQA: Metamorphic relation-based hallucination detection for LLMs.
    
    Reference: "Hallucination Detection in Large Language Models with 
                Metamorphic Relations" (Yang et al., 2025)
    """
    
    def __init__(self, llm_interface: LLMInterface = None, config: Dict = None):
        """
        Initialize MetaQA.
        
        Args:
            llm_interface: Interface for LLM interactions
            config: Configuration for MetaQA
        """
        if config is None:
            config = BASELINE_CONFIG["metaqa"]
        
        self.config = config
        self.llm = llm_interface or LLMInterface()
        
        # Initialize embedding model for semantic similarity
        self.embedding_model = SentenceTransformer(LLM_CONFIG["embedding_model"]["name"])
        
        logger.info(f"Initialized MetaQA with {self.config['num_mutations']} mutations")
    
    def generate_mutations(self, question: str) -> List[str]:
        """
        Generate mutations of the original question.
        
        Args:
            question: Original question
            
        Returns:
            List of mutated questions
        """
        prompt = f"""
I will give you a question, and I want you to generate {self.config['num_mutations']} variations of this question.
Each variation should ask for the same information but be worded differently.
The variations should preserve the original meaning and intent of the question.

Original question: {question}

Generate {self.config['num_mutations']} variations, labeled as 1., 2., etc.:
"""
        
        response = self.llm.generate(prompt)
        
        # Parse mutations from response
        mutations = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('- ')):
                # Remove the leading number/bullet and any trailing/leading whitespace
                clean_line = line.lstrip('0123456789.- \t')
                if clean_line:
                    mutations.append(clean_line)
        
        # If we didn't get enough mutations, pad with the original
        while len(mutations) < self.config['num_mutations']:
            mutations.append(question)
        
        # If we got too many, truncate
        mutations = mutations[:self.config['num_mutations']]
        
        return mutations
    
    def generate_answers(self, questions: List[str], context: str = "") -> List[str]:
        """
        Generate answers for a list of questions.
        
        Args:
            questions: List of questions
            context: Additional context
            
        Returns:
            List of answers
        """
        answers = []
        
        for question in questions:
            prompt = f"""
Question: {question}

Context: {context}

Please answer the question based on the provided context.
"""
            
            answer = self.llm.generate(prompt, temperature=0.1)
            answers.append(answer)
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
        
        return answers
    
    def compute_answer_consistency(self, answers: List[str]) -> float:
        """
        Compute consistency score among answers.
        
        Args:
            answers: List of answers to compare
            
        Returns:
            Consistency score (0-1)
        """
        # Get embeddings for answers
        embeddings = self.embedding_model.encode(answers)
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(answers)):
            for j in range(i+1, len(answers)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(similarity)
        
        # Compute average similarity
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            return float(avg_similarity)
        else:
            return 1.0  # Perfect consistency if only one answer
    
    def detect_hallucination(self, question: str, context: str = "") -> Tuple[bool, float, Dict]:
        """
        Detect hallucination in response to a question.
        
        Args:
            question: Question to check
            context: Additional context
            
        Returns:
            Tuple of (is_hallucination, hallucination_score, details)
        """
        # Generate question mutations
        mutations = self.generate_mutations(question)
        all_questions = [question] + mutations
        
        # Generate answers for all questions
        answers = self.generate_answers(all_questions, context)
        
        # Compute consistency score
        consistency = self.compute_answer_consistency(answers)
        
        # Convert consistency to hallucination score (lower consistency -> higher hallucination score)
        hallucination_score = 1 - (consistency - self.config["similarity_threshold"]) / (1 - self.config["similarity_threshold"])
        hallucination_score = min(max(hallucination_score, 0.0), 1.0)  # Clip to [0, 1]
        
        # Determine if it's a hallucination
        is_hallucination = hallucination_score > 0.5
        
        # Collect details
        details = {
            "mutations": mutations,
            "answers": answers,
            "consistency_score": consistency
        }
        
        return is_hallucination, hallucination_score, details


# Factory function to create uncertainty quantification methods
def create_uq_method(method_name: str, llm_interface: LLMInterface = None) -> Any:
    """
    Create an uncertainty quantification method.
    
    Args:
        method_name: Name of the method to create
        llm_interface: Interface for LLM interactions
        
    Returns:
        Uncertainty quantification method instance
    """
    if method_name == "selfcheckgpt":
        return SelfCheckGPT(llm_interface)
    elif method_name == "multidim_uq":
        return MultiDimensionalUQ(llm_interface)
    elif method_name == "calibration":
        return CalibrationBasedUQ(llm_interface)
    elif method_name == "hudex":
        return HuDEx(llm_interface)
    elif method_name == "metaqa":
        return MetaQA(llm_interface)
    else:
        raise ValueError(f"Unknown UQ method: {method_name}")


# Example usage
if __name__ == "__main__":
    # Test the uncertainty quantification methods
    print("Testing uncertainty quantification methods...")
    
    # Initialize LLM interface
    llm = LLMInterface()
    
    # Test question and context
    question = "What is the capital of France?"
    context = "France is a country in Western Europe. It is known for its cuisine, culture, and history."
    
    # SelfCheckGPT
    print("\nTesting SelfCheckGPT...")
    selfcheckgpt = SelfCheckGPT(llm, {"num_samples": 3, "temperature": 0.7, "similarity_threshold": 0.8})
    is_hall1, score1, details1 = selfcheckgpt.detect_hallucination(question, context)
    print(f"Is hallucination: {is_hall1}, Score: {score1:.4f}")
    
    # MultiDimensionalUQ
    print("\nTesting MultiDimensionalUQ...")
    multidim = MultiDimensionalUQ(llm, {"num_responses": 3, "num_dimensions": 3})
    is_hall2, score2, details2 = multidim.detect_hallucination(question, context)
    print(f"Is hallucination: {is_hall2}, Score: {score2:.4f}")
    
    # CalibrationBasedUQ
    print("\nTesting CalibrationBasedUQ...")
    calibration = CalibrationBasedUQ(llm, {"method": "temperature_scaling", "validation_size": 10})
    # We would normally calibrate with validation data, but we'll skip that for the test
    is_hall3, score3, details3 = calibration.detect_hallucination(question, context)
    print(f"Is hallucination: {is_hall3}, Score: {score3:.4f}")
    
    # HuDEx
    print("\nTesting HuDEx...")
    hudex = HuDEx(llm, {"explanation_model": "claude-3.7-sonnet", "threshold": 0.65})
    is_hall4, score4, details4 = hudex.detect_hallucination(question, context)
    print(f"Is hallucination: {is_hall4}, Score: {score4:.4f}")
    print(f"Explanation: {details4['explanation'][:100]}...")
    
    # MetaQA
    print("\nTesting MetaQA...")
    metaqa = MetaQA(llm, {"num_mutations": 3, "similarity_threshold": 0.75})
    is_hall5, score5, details5 = metaqa.detect_hallucination(question, context)
    print(f"Is hallucination: {is_hall5}, Score: {score5:.4f}")
    
    print("\nUncertainty quantification test complete.")
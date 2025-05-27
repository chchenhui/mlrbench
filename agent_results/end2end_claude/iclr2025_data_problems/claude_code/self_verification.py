#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Self-verification techniques for Attribution-Guided Training.
Implements methods for models to verify their own attributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import re
import json

logger = logging.getLogger(__name__)

class AttributionVerifier:
    """
    Base class for attribution verification.
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize the attribution verifier.
        
        Args:
            threshold: Confidence threshold for attribution
        """
        self.threshold = threshold
        
        logger.info(f"Initialized AttributionVerifier with threshold {threshold}")
    
    def verify_attributions(
        self,
        model_attributions: Dict[str, Any],
        source_texts: List[str],
        generated_text: str
    ) -> Dict[str, Any]:
        """
        Verify attributions produced by a model.
        
        Args:
            model_attributions: Attribution claims from the model
            source_texts: List of source texts
            generated_text: Generated text
            
        Returns:
            Dictionary with verification results
        """
        raise NotImplementedError("Subclasses must implement verify_attributions")

class SelfConsistencyVerifier(AttributionVerifier):
    """
    Verifier that checks the consistency of attributions across
    multiple generations from the same model.
    """
    
    def __init__(
        self,
        num_generations: int = 3,
        threshold: float = 0.7
    ):
        """
        Initialize the self-consistency verifier.
        
        Args:
            num_generations: Number of generations to compare
            threshold: Confidence threshold for attribution
        """
        super().__init__(threshold)
        self.num_generations = num_generations
        
        logger.info(f"Initialized SelfConsistencyVerifier with {num_generations} generations")
    
    def verify_attributions(
        self,
        model_attributions_list: List[Dict[str, Any]],
        source_texts: List[str],
        generated_texts: List[str]
    ) -> Dict[str, Any]:
        """
        Verify attributions by checking consistency across multiple generations.
        
        Args:
            model_attributions_list: List of attribution claims from multiple generations
            source_texts: List of source texts
            generated_texts: List of generated texts
            
        Returns:
            Dictionary with verification results
        """
        # Ensure we have enough generations
        if len(model_attributions_list) < self.num_generations:
            logger.warning(f"Expected {self.num_generations} generations, got {len(model_attributions_list)}")
            
        # Extract claimed sources from each generation
        all_claimed_sources = []
        
        for attribution in model_attributions_list:
            claimed_sources = attribution.get("sources", [])
            all_claimed_sources.append(set(claimed_sources))
        
        # Calculate agreement between generations
        agreements = []
        
        for i in range(len(all_claimed_sources)):
            for j in range(i+1, len(all_claimed_sources)):
                sources_i = all_claimed_sources[i]
                sources_j = all_claimed_sources[j]
                
                if not sources_i and not sources_j:
                    # Both empty, perfect agreement
                    agreement = 1.0
                elif not sources_i or not sources_j:
                    # One empty, zero agreement
                    agreement = 0.0
                else:
                    # Calculate Jaccard similarity
                    intersection = len(sources_i.intersection(sources_j))
                    union = len(sources_i.union(sources_j))
                    agreement = intersection / union
                
                agreements.append(agreement)
        
        # Calculate average agreement
        avg_agreement = np.mean(agreements) if agreements else 0.0
        
        # Determine consistent attributions (appear in majority of generations)
        source_counts = {}
        
        for sources in all_claimed_sources:
            for source in sources:
                if source in source_counts:
                    source_counts[source] += 1
                else:
                    source_counts[source] = 1
        
        consistent_sources = [
            source for source, count in source_counts.items()
            if count >= len(all_claimed_sources) / 2
        ]
        
        return {
            "consistency_score": avg_agreement,
            "consistent_sources": consistent_sources,
            "num_generations": len(model_attributions_list),
            "verified": avg_agreement >= self.threshold
        }

class LanguageModelVerifier(AttributionVerifier):
    """
    Verifier that uses a language model to verify attributions.
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_length: int = 512,
        threshold: float = 0.7,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the language model verifier.
        
        Args:
            model_name: Name of the language model to use
            max_length: Maximum sequence length
            threshold: Confidence threshold for attribution
            device: Device to use for computation
        """
        super().__init__(threshold)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Load model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Check if it's a causal LM or seq2seq
            if "t5" in model_name.lower():
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
                self.model_type = "seq2seq"
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
                self.model_type = "causal"
                
            self.model_loaded = True
            
        except Exception as e:
            logger.warning(f"Failed to load language model: {e}")
            logger.info("Falling back to simpler verification methods")
            self.model_loaded = False
        
        logger.info(f"Initialized LanguageModelVerifier with model {model_name}")
    
    def _create_verification_prompt(
        self,
        generated_text: str,
        claimed_source_texts: List[str]
    ) -> str:
        """
        Create a prompt for verification.
        
        Args:
            generated_text: Generated text to verify
            claimed_source_texts: Texts from claimed sources
            
        Returns:
            Verification prompt
        """
        prompt = "Task: Determine whether the generated text below contains content that should be attributed to any of the provided source texts.\n\n"
        
        prompt += "Generated text:\n" + generated_text + "\n\n"
        
        prompt += "Source texts:\n"
        for i, source in enumerate(claimed_source_texts):
            prompt += f"Source {i+1}: {source}\n\n"
        
        prompt += "For each piece of content in the generated text that is similar to any source text, identify the source number. " \
                 "If the generated text contains no content requiring attribution, say 'No attribution needed'.\n\n" \
                 "Answer: "
        
        return prompt
    
    def _parse_verification_result(self, result: str) -> Dict[str, Any]:
        """
        Parse the result from the language model.
        
        Args:
            result: Output from the language model
            
        Returns:
            Parsed verification result
        """
        # Check for "No attribution needed"
        if "no attribution" in result.lower():
            return {
                "verified_sources": [],
                "requires_attribution": False,
                "confidence": 1.0
            }
        
        # Extract source numbers using regex
        source_pattern = r'Source\s+(\d+)'
        matches = re.findall(source_pattern, result)
        
        # Convert to integers (1-indexed to 0-indexed)
        verified_sources = [int(match) - 1 for match in matches]
        
        # Calculate confidence based on consistency of language model output
        # This is a simplified heuristic
        confidence = 0.8  # Default reasonable confidence
        
        # Adjust based on phrasing
        if "definitely" in result.lower() or "clearly" in result.lower():
            confidence = 0.9
        elif "possibly" in result.lower() or "might" in result.lower():
            confidence = 0.6
        
        return {
            "verified_sources": verified_sources,
            "requires_attribution": len(verified_sources) > 0,
            "confidence": confidence
        }
    
    def verify_attributions(
        self,
        model_attributions: Dict[str, Any],
        source_texts: List[str],
        generated_text: str
    ) -> Dict[str, Any]:
        """
        Verify attributions using a language model.
        
        Args:
            model_attributions: Attribution claims from the model
            source_texts: List of source texts
            generated_text: Generated text
            
        Returns:
            Dictionary with verification results
        """
        if not self.model_loaded:
            logger.warning("Language model not loaded, returning unverified attributions")
            return {
                "original_attributions": model_attributions,
                "verified": False,
                "verification_method": "none"
            }
        
        # Get claimed sources
        claimed_sources = model_attributions.get("sources", [])
        
        # Get texts for claimed sources
        claimed_source_texts = [source_texts[src] for src in claimed_sources if src < len(source_texts)]
        
        # Create verification prompt
        prompt = self._create_verification_prompt(generated_text, claimed_source_texts)
        
        # Generate verification
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                if self.model_type == "seq2seq":
                    outputs = self.model.generate(
                        **inputs, 
                        max_length=150,
                        temperature=0.7,
                        do_sample=False
                    )
                    result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    # For causal models, we need to generate continuation
                    outputs = self.model.generate(
                        inputs.input_ids, 
                        attention_mask=inputs.attention_mask,
                        max_length=inputs.input_ids.shape[1] + 150,
                        temperature=0.7,
                        do_sample=False
                    )
                    # Extract only the new tokens
                    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
                    result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Parse result
            parsed = self._parse_verification_result(result)
            
            # Compare against claimed sources
            claimed_set = set(claimed_sources)
            verified_set = set(parsed["verified_sources"])
            
            correct_claims = claimed_set.intersection(verified_set)
            incorrect_claims = claimed_set - verified_set
            missing_claims = verified_set - claimed_set
            
            # Calculate verification score
            if not claimed_set and not verified_set:
                # Both empty, perfect agreement
                verification_score = 1.0
            elif not claimed_set or not verified_set:
                # One empty, zero agreement
                verification_score = 0.0
            else:
                # Calculate Jaccard similarity
                verification_score = len(correct_claims) / len(claimed_set.union(verified_set))
            
            return {
                "original_attributions": model_attributions,
                "verified_sources": list(verified_set),
                "correct_claims": list(correct_claims),
                "incorrect_claims": list(incorrect_claims),
                "missing_claims": list(missing_claims),
                "verification_score": verification_score,
                "verified": verification_score >= self.threshold,
                "verification_method": "language_model",
                "lm_confidence": parsed["confidence"],
                "lm_output": result
            }
        
        except Exception as e:
            logger.error(f"Error in language model verification: {e}")
            return {
                "original_attributions": model_attributions,
                "verified": False,
                "verification_method": "failed",
                "error": str(e)
            }

class ContrastiveVerifier(AttributionVerifier):
    """
    Verifier that uses contrastive testing to verify attributions.
    It checks if changing sources affects the attributions.
    """
    
    def __init__(
        self,
        attribution_model,
        num_contrasts: int = 3,
        threshold: float = 0.7,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the contrastive verifier.
        
        Args:
            attribution_model: Model that produces attributions
            num_contrasts: Number of contrasts to generate
            threshold: Confidence threshold for attribution
            device: Device to use for computation
        """
        super().__init__(threshold)
        self.attribution_model = attribution_model
        self.num_contrasts = num_contrasts
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        logger.info(f"Initialized ContrastiveVerifier with {num_contrasts} contrasts")
    
    def _create_contrastive_sources(
        self,
        source_texts: List[str],
        claimed_sources: List[int]
    ) -> List[List[str]]:
        """
        Create contrasts by modifying sources.
        
        Args:
            source_texts: Original source texts
            claimed_sources: Claimed source indices
            
        Returns:
            List of contrasted source lists
        """
        contrasts = []
        
        for _ in range(self.num_contrasts):
            # Create a contrast by changing claimed sources
            modified_sources = source_texts.copy()
            
            for source_idx in claimed_sources:
                if source_idx < len(modified_sources):
                    # Modify the source by shuffling sentences or replacing words
                    original = modified_sources[source_idx]
                    
                    # Simple modification: shuffle words
                    words = original.split()
                    np.random.shuffle(words)
                    modified = " ".join(words)
                    
                    modified_sources[source_idx] = modified
            
            contrasts.append(modified_sources)
        
        return contrasts
    
    def verify_attributions(
        self,
        model_attributions: Dict[str, Any],
        source_texts: List[str],
        generated_text: str
    ) -> Dict[str, Any]:
        """
        Verify attributions using contrastive testing.
        
        Args:
            model_attributions: Attribution claims from the model
            source_texts: List of source texts
            generated_text: Generated text
            
        Returns:
            Dictionary with verification results
        """
        # Get claimed sources
        claimed_sources = model_attributions.get("sources", [])
        
        if not claimed_sources:
            # No sources claimed, nothing to verify
            return {
                "original_attributions": model_attributions,
                "verification_score": 1.0,
                "verified": True,
                "verification_method": "contrastive"
            }
        
        # Create contrasts
        contrasted_sources = self._create_contrastive_sources(source_texts, claimed_sources)
        
        # For each contrast, run attribution
        contrast_results = []
        
        for contrast_sources in contrasted_sources:
            # Run attribution
            try:
                contrast_attributions = self.attribution_model.attribute(
                    generated_text, contrast_sources
                )
                
                contrast_claimed = contrast_attributions.get("sources", [])
                
                # Calculate agreement
                original_set = set(claimed_sources)
                contrast_set = set(contrast_claimed)
                
                # We expect low agreement with contrasts
                if not original_set and not contrast_set:
                    # Both empty, no difference
                    agreement = 1.0
                elif not original_set or not contrast_set:
                    # One empty, perfect difference
                    agreement = 0.0
                else:
                    # Calculate Jaccard similarity (lower is better)
                    agreement = len(original_set.intersection(contrast_set)) / len(original_set.union(contrast_set))
                
                # We want lower agreement for contrasts
                contrast_score = 1.0 - agreement
                
                contrast_results.append({
                    "contrast_attributions": contrast_attributions,
                    "contrast_score": contrast_score
                })
                
            except Exception as e:
                logger.error(f"Error in contrastive attribution: {e}")
                contrast_results.append({
                    "error": str(e),
                    "contrast_score": 0.0
                })
        
        # Calculate average contrast score
        avg_contrast_score = np.mean([r["contrast_score"] for r in contrast_results])
        
        return {
            "original_attributions": model_attributions,
            "contrast_results": contrast_results,
            "verification_score": avg_contrast_score,
            "verified": avg_contrast_score >= self.threshold,
            "verification_method": "contrastive"
        }

class ConfidenceBasedVerifier(AttributionVerifier):
    """
    Verifier that relies on the confidence scores from the model.
    """
    
    def verify_attributions(
        self,
        model_attributions: Dict[str, Any],
        source_texts: List[str],
        generated_text: str
    ) -> Dict[str, Any]:
        """
        Verify attributions based on confidence scores.
        
        Args:
            model_attributions: Attribution claims from the model
            source_texts: List of source texts
            generated_text: Generated text
            
        Returns:
            Dictionary with verification results
        """
        # Get claimed sources and scores
        claimed_sources = model_attributions.get("sources", [])
        scores = model_attributions.get("scores", [])
        
        # If no scores available, can't verify
        if not scores:
            return {
                "original_attributions": model_attributions,
                "verified": False,
                "verification_method": "confidence"
            }
        
        # Filter sources based on threshold
        verified_sources = []
        verified_scores = []
        
        for source, score in zip(claimed_sources, scores):
            if score >= self.threshold:
                verified_sources.append(source)
                verified_scores.append(score)
        
        # Calculate average confidence
        avg_confidence = np.mean(scores) if scores else 0.0
        
        return {
            "original_attributions": model_attributions,
            "verified_sources": verified_sources,
            "verified_scores": verified_scores,
            "avg_confidence": avg_confidence,
            "verified": avg_confidence >= self.threshold,
            "verification_method": "confidence"
        }

class EnsembleVerifier(AttributionVerifier):
    """
    Verifier that combines multiple verification methods.
    """
    
    def __init__(
        self,
        verifiers: List[AttributionVerifier],
        weights: Optional[List[float]] = None,
        threshold: float = 0.7
    ):
        """
        Initialize the ensemble verifier.
        
        Args:
            verifiers: List of verifiers to ensemble
            weights: Weights for each verifier (defaults to equal weighting)
            threshold: Confidence threshold for attribution
        """
        super().__init__(threshold)
        self.verifiers = verifiers
        
        # Set weights
        if weights is None:
            weights = [1.0 / len(verifiers)] * len(verifiers)
        
        if len(weights) != len(verifiers):
            raise ValueError("Number of weights must match number of verifiers")
        
        self.weights = weights
        
        logger.info(f"Initialized EnsembleVerifier with {len(verifiers)} verifiers")
    
    def verify_attributions(
        self,
        model_attributions: Dict[str, Any],
        source_texts: List[str],
        generated_text: str
    ) -> Dict[str, Any]:
        """
        Verify attributions using ensemble of methods.
        
        Args:
            model_attributions: Attribution claims from the model
            source_texts: List of source texts
            generated_text: Generated text
            
        Returns:
            Dictionary with verification results
        """
        # Run all verifiers
        verification_results = []
        
        for verifier in self.verifiers:
            try:
                result = verifier.verify_attributions(
                    model_attributions, source_texts, generated_text
                )
                verification_results.append(result)
            except Exception as e:
                logger.error(f"Error in verifier {verifier.__class__.__name__}: {e}")
                verification_results.append({
                    "error": str(e),
                    "verified": False
                })
        
        # Aggregate verification scores
        weighted_scores = []
        
        for i, result in enumerate(verification_results):
            if "verification_score" in result:
                weighted_scores.append(result["verification_score"] * self.weights[i])
            elif "avg_confidence" in result:
                weighted_scores.append(result["avg_confidence"] * self.weights[i])
            else:
                # Default score of 0 if no score available
                weighted_scores.append(0.0)
        
        ensemble_score = sum(weighted_scores)
        
        # Aggregate verified sources
        all_verified_sources = set()
        
        for result in verification_results:
            if "verified_sources" in result:
                all_verified_sources.update(result["verified_sources"])
        
        # Calculate verification counts for each source
        source_counts = {}
        
        for result in verification_results:
            if "verified_sources" in result:
                for source in result["verified_sources"]:
                    if source in source_counts:
                        source_counts[source] += 1
                    else:
                        source_counts[source] = 1
        
        # Sources verified by majority of verifiers
        majority_threshold = len(self.verifiers) / 2
        majority_verified = [
            source for source, count in source_counts.items()
            if count >= majority_threshold
        ]
        
        return {
            "original_attributions": model_attributions,
            "verifier_results": verification_results,
            "ensemble_score": ensemble_score,
            "all_verified_sources": list(all_verified_sources),
            "majority_verified_sources": majority_verified,
            "verified": ensemble_score >= self.threshold,
            "verification_method": "ensemble"
        }

if __name__ == "__main__":
    # Test the verifiers
    logging.basicConfig(level=logging.INFO)
    
    # Mock attribution model for testing
    class MockAttributionModel:
        def attribute(self, text, sources):
            return {
                "sources": [0, 1],
                "scores": [0.9, 0.8]
            }
    
    # Sample data
    source_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence."
    ]
    
    generated_text = "The quick brown fox is fast. Python is great for machine learning."
    
    model_attributions = {
        "sources": [0, 1],
        "scores": [0.9, 0.7]
    }
    
    # Test ConfidenceBasedVerifier
    print("Testing ConfidenceBasedVerifier...")
    confidence_verifier = ConfidenceBasedVerifier(threshold=0.8)
    confidence_result = confidence_verifier.verify_attributions(
        model_attributions, source_texts, generated_text
    )
    print(f"Confidence verification result: {json.dumps(confidence_result, indent=2)}")
    
    # Test SelfConsistencyVerifier
    print("\nTesting SelfConsistencyVerifier...")
    consistency_verifier = SelfConsistencyVerifier(num_generations=2)
    consistency_result = consistency_verifier.verify_attributions(
        [model_attributions, {"sources": [0], "scores": [0.9]}],
        source_texts,
        [generated_text, generated_text]
    )
    print(f"Consistency verification result: {json.dumps(consistency_result, indent=2)}")
    
    # Test ContrastiveVerifier (requires a model)
    print("\nTesting ContrastiveVerifier...")
    mock_model = MockAttributionModel()
    contrastive_verifier = ContrastiveVerifier(mock_model, num_contrasts=2)
    try:
        contrastive_result = contrastive_verifier.verify_attributions(
            model_attributions, source_texts, generated_text
        )
        print(f"Contrastive verification result: {json.dumps(contrastive_result, indent=2)}")
    except Exception as e:
        print(f"Contrastive verification error: {e}")
    
    # Test EnsembleVerifier
    print("\nTesting EnsembleVerifier...")
    verifiers = [confidence_verifier]
    ensemble_verifier = EnsembleVerifier(verifiers)
    ensemble_result = ensemble_verifier.verify_attributions(
        model_attributions, source_texts, generated_text
    )
    print(f"Ensemble verification result: {json.dumps(ensemble_result, indent=2)}")
    
    print("\nAll tests completed!")
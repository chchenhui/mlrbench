"""
Baseline methods for comparison with SCEC.

Implements:
- Vanilla LLM decoding (no UQ)
- Semantic Entropy Probes (Kossen et al., 2024)
- Uncertainty-Aware Fusion (Dey et al., 2025)
- Claim Conditioned Probability (Fadeeva et al., 2024)
- MetaQA (Yang et al., 2025)
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from .llm_interface import LLMInterface, get_llm_interface

logger = logging.getLogger(__name__)

class BaselineMethod(ABC):
    """Abstract base class for baseline methods."""
    
    @abstractmethod
    def compute_uncertainty(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Compute uncertainty for a response to a prompt.
        
        Args:
            prompt: Input prompt
            response: Model response
            
        Returns:
            Dictionary with uncertainty metrics
        """
        pass
    
    @abstractmethod
    def generate_with_uncertainty(
        self, 
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Generate a response and compute its uncertainty.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary with generation result and uncertainty metrics
        """
        pass


class VanillaBaseline(BaselineMethod):
    """
    Vanilla LLM decoding (no uncertainty quantification).
    
    This baseline simply uses the LLM without any UQ mechanism,
    and assigns a fixed confidence score of 1.0 for comparison purposes.
    """
    
    def __init__(self, llm: Union[LLMInterface, str], **llm_kwargs):
        """
        Initialize the vanilla baseline.
        
        Args:
            llm: LLM interface or model name
            **llm_kwargs: Additional arguments for LLM interface
        """
        if isinstance(llm, str):
            self.llm = get_llm_interface(llm, **llm_kwargs)
        else:
            self.llm = llm
    
    def compute_uncertainty(self, prompt: str, response: str) -> Dict[str, Any]:
        """Assign fixed uncertainty of 0.0 (i.e., full confidence)."""
        return {
            "uncertainty_score": 0.0,
            "confidence_score": 1.0,
            "method": "vanilla",
        }
    
    def generate_with_uncertainty(
        self, 
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response with fixed confidence score."""
        # Generate response
        generation_result = self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        # Add fixed uncertainty metrics
        uncertainty_result = self.compute_uncertainty(prompt, generation_result["text"])
        
        # Combine results
        result = {
            **generation_result,
            **uncertainty_result,
        }
        
        return result


class SemanticEntropyProbes(BaselineMethod):
    """
    Semantic Entropy Probes (SEP) implementation.
    
    Based on Kossen et al. (2024), this method approximates semantic entropy
    from hidden states of a single generation without requiring multiple samples.
    """
    
    def __init__(
        self,
        llm: Union[LLMInterface, str],
        uncertainty_model_name: str = "bert-base-uncased",
        device: str = "auto",
        **llm_kwargs
    ):
        """
        Initialize the SEP baseline.
        
        Args:
            llm: LLM interface or model name
            uncertainty_model_name: Model to use for uncertainty probing
            device: Device to run uncertainty model on
            **llm_kwargs: Additional arguments for LLM interface
        """
        if isinstance(llm, str):
            self.llm = get_llm_interface(llm, **llm_kwargs)
        else:
            self.llm = llm
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load uncertainty model
        logger.info(f"Loading SEP uncertainty model {uncertainty_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(uncertainty_model_name)
        self.model = AutoModel.from_pretrained(uncertainty_model_name).to(self.device)
        
        # Initialize SEP probe (a simple MLP head)
        self.probe = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Note: In a real implementation, this probe would be trained on a dataset
        # of text with uncertainty annotations. Here we use a placeholder.
    
    def _get_uncertainty_from_embeddings(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Extract uncertainty scores from token embeddings using the SEP probe.
        
        Args:
            embeddings: Token embeddings from the model
            
        Returns:
            Array of token-level uncertainty scores
        """
        # Apply the probe to get token-level uncertainty scores
        with torch.no_grad():
            # Note: In a real implementation, this would use the trained probe.
            # Here we just simulate some uncertainty scores.
            uncertainty_scores = self.probe(embeddings).squeeze(-1)
        
        return uncertainty_scores.cpu().numpy()
    
    def compute_uncertainty(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Compute uncertainty for a response using Semantic Entropy Probes.
        
        Args:
            prompt: Input prompt
            response: Model response
            
        Returns:
            Dictionary with uncertainty metrics
        """
        # Tokenize the response
        inputs = self.tokenizer(response, return_tensors="pt", truncation=True).to(self.device)
        
        # Get model embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
        
        # Get token-level uncertainty scores
        token_uncertainties = self._get_uncertainty_from_embeddings(hidden_states)[0]
        
        # Compute aggregate uncertainty score
        sequence_uncertainty = float(np.mean(token_uncertainties))
        
        return {
            "uncertainty_score": sequence_uncertainty,
            "confidence_score": 1.0 - sequence_uncertainty,
            "token_uncertainties": token_uncertainties.tolist(),
            "method": "semantic_entropy_probes",
        }
    
    def generate_with_uncertainty(
        self, 
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response and compute its uncertainty using SEP.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generation result and uncertainty metrics
        """
        # Generate response
        generation_result = self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        # Compute uncertainty
        uncertainty_result = self.compute_uncertainty(prompt, generation_result["text"])
        
        # Combine results
        result = {
            **generation_result,
            **uncertainty_result,
        }
        
        return result


class UncertaintyAwareFusion(BaselineMethod):
    """
    Uncertainty-Aware Fusion (UAF) implementation.
    
    Based on Dey et al. (2025), this method combines multiple LLMs
    based on their accuracy and self-assessment capabilities to reduce hallucinations.
    """
    
    def __init__(
        self,
        llms: List[Union[LLMInterface, str]],
        weights: Optional[List[float]] = None,
        **llm_kwargs
    ):
        """
        Initialize the UAF baseline.
        
        Args:
            llms: List of LLM interfaces or model names
            weights: Weights for each LLM (if None, equal weights are used)
            **llm_kwargs: Additional arguments for LLM interfaces
        """
        # Initialize LLMs
        self.llms = []
        for llm in llms:
            if isinstance(llm, str):
                self.llms.append(get_llm_interface(llm, **llm_kwargs))
            else:
                self.llms.append(llm)
        
        # Set weights
        if weights is None:
            # Equal weights
            self.weights = [1.0 / len(self.llms)] * len(self.llms)
        else:
            if len(weights) != len(self.llms):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of LLMs ({len(self.llms)})")
            
            # Normalize weights to sum to 1
            self.weights = [w / sum(weights) for w in weights]
    
    def compute_uncertainty(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Compute uncertainty by comparing responses from multiple LLMs.
        
        Args:
            prompt: Input prompt
            response: Model response (not used directly, as we generate from all LLMs)
            
        Returns:
            Dictionary with uncertainty metrics
        """
        # Generate responses from all LLMs
        responses = []
        for llm in self.llms:
            result = llm.generate(prompt=prompt, max_tokens=500, temperature=0.0)
            responses.append(result["text"])
        
        # Compute pairwise similarities between responses
        # (Simplified - in practice, would use more sophisticated metrics)
        similarities = []
        for i, resp_i in enumerate(responses):
            for j, resp_j in enumerate(responses[i+1:], i+1):
                # Simple Jaccard similarity of token sets
                tokens_i = set(resp_i.lower().split())
                tokens_j = set(resp_j.lower().split())
                
                intersection = len(tokens_i.intersection(tokens_j))
                union = len(tokens_i.union(tokens_j))
                
                if union > 0:
                    similarity = intersection / union
                else:
                    similarity = 0.0
                
                similarities.append(similarity)
        
        # Average similarity as confidence score
        if similarities:
            confidence_score = float(np.mean(similarities))
        else:
            confidence_score = 0.5  # Default mid-point if no similarities
        
        # Map confidence to uncertainty
        uncertainty_score = 1.0 - confidence_score
        
        return {
            "uncertainty_score": uncertainty_score,
            "confidence_score": confidence_score,
            "model_responses": responses,
            "method": "uncertainty_aware_fusion",
        }
    
    def generate_with_uncertainty(
        self, 
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response using UAF ensemble and compute uncertainty.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generation result and uncertainty metrics
        """
        # Generate responses from all LLMs
        responses = []
        confidences = []
        
        for i, llm in enumerate(self.llms):
            # Generate response
            result = llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            # Get confidence from token logprobs (if available)
            if "token_logprobs" in result:
                confidence = float(np.exp(np.mean(result["token_logprobs"])))
            else:
                # Default confidence based on model weight
                confidence = self.weights[i]
            
            responses.append(result["text"])
            confidences.append(confidence)
        
        # Normalize confidences
        if sum(confidences) > 0:
            normalized_confidences = [c / sum(confidences) for c in confidences]
        else:
            normalized_confidences = self.weights
        
        # Compute aggregate confidence score
        confidence_score = float(sum(c * w for c, w in zip(confidences, self.weights)))
        uncertainty_score = 1.0 - confidence_score
        
        # Select response based on highest confidence
        best_idx = np.argmax(confidences)
        selected_response = responses[best_idx]
        
        return {
            "text": selected_response,
            "uncertainty_score": uncertainty_score,
            "confidence_score": confidence_score,
            "model_responses": responses,
            "model_confidences": confidences,
            "normalized_confidences": normalized_confidences,
            "selected_model": best_idx,
            "method": "uncertainty_aware_fusion",
        }


class ClaimConditionedProbability(BaselineMethod):
    """
    Claim Conditioned Probability (CCP) implementation.
    
    Based on Fadeeva et al. (2024), this method measures the uncertainty
    of specific claims expressed by the model.
    """
    
    def __init__(
        self,
        llm: Union[LLMInterface, str],
        verifier_model_name: str = "facebook/bart-large-mnli",
        device: str = "auto",
        **llm_kwargs
    ):
        """
        Initialize the CCP baseline.
        
        Args:
            llm: LLM interface or model name
            verifier_model_name: Model for claim verification
            device: Device to run models on
            **llm_kwargs: Additional arguments for LLM interface
        """
        if isinstance(llm, str):
            self.llm = get_llm_interface(llm, **llm_kwargs)
        else:
            self.llm = llm
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load claim verification model
        logger.info(f"Loading CCP verifier model {verifier_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(verifier_model_name)
        self.verifier = AutoModelForSequenceClassification.from_pretrained(
            verifier_model_name
        ).to(self.device)
    
    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text.
        
        Args:
            text: Input text to extract claims from
            
        Returns:
            List of extracted claim strings
        """
        # Simple claim extraction based on sentences
        # In a real implementation, this would use more sophisticated NLP
        sentences = text.split(".")
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims
    
    def _verify_claim(self, claim: str, context: str) -> float:
        """
        Verify a claim against a context.
        
        Args:
            claim: Claim to verify
            context: Context to verify against
            
        Returns:
            Verification score (0-1)
        """
        # Use NLI model to check entailment
        inputs = self.tokenizer(
            premise=context,
            hypothesis=claim,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.verifier(**inputs)
            scores = outputs.logits.softmax(dim=-1)
            
            # Get entailment probability
            # Typically: 0 = contradiction, 1 = neutral, 2 = entailment
            entailment_score = scores[0, 2].item()
            
            # Higher entailment = higher verification
            return entailment_score
    
    def compute_uncertainty(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Compute uncertainty using Claim Conditioned Probability.
        
        Args:
            prompt: Input prompt
            response: Model response
            
        Returns:
            Dictionary with uncertainty metrics
        """
        # Extract claims from response
        claims = self._extract_claims(response)
        
        # For empty responses or no claims
        if not claims:
            return {
                "uncertainty_score": 0.5,  # Default mid-point
                "confidence_score": 0.5,
                "claims": [],
                "claim_scores": [],
                "method": "claim_conditioned_probability",
            }
        
        # Verify each claim against the prompt
        claim_scores = []
        for claim in claims:
            verification_score = self._verify_claim(claim, prompt)
            claim_scores.append(verification_score)
        
        # Compute aggregate verification score
        verification_score = float(np.mean(claim_scores))
        
        # Map verification to uncertainty
        uncertainty_score = 1.0 - verification_score
        
        return {
            "uncertainty_score": uncertainty_score,
            "confidence_score": verification_score,
            "claims": claims,
            "claim_scores": claim_scores,
            "method": "claim_conditioned_probability",
        }
    
    def generate_with_uncertainty(
        self, 
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response and compute its uncertainty using CCP.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generation result and uncertainty metrics
        """
        # Generate response
        generation_result = self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        # Compute uncertainty
        uncertainty_result = self.compute_uncertainty(prompt, generation_result["text"])
        
        # Combine results
        result = {
            **generation_result,
            **uncertainty_result,
        }
        
        return result


class MetaQA(BaselineMethod):
    """
    MetaQA implementation.
    
    Based on Yang et al. (2025), this method uses metamorphic relations
    and prompt mutation to detect hallucinations without external resources.
    """
    
    def __init__(
        self,
        llm: Union[LLMInterface, str],
        num_mutations: int = 3,
        **llm_kwargs
    ):
        """
        Initialize the MetaQA baseline.
        
        Args:
            llm: LLM interface or model name
            num_mutations: Number of prompt mutations to generate
            **llm_kwargs: Additional arguments for LLM interface
        """
        if isinstance(llm, str):
            self.llm = get_llm_interface(llm, **llm_kwargs)
        else:
            self.llm = llm
        
        self.num_mutations = num_mutations
    
    def _generate_mutations(self, prompt: str) -> List[str]:
        """
        Generate metamorphic mutations of a prompt.
        
        Args:
            prompt: Original prompt
            
        Returns:
            List of mutated prompts
        """
        # Note: In practice, this would implement the actual mutation strategies
        # from the MetaQA paper. Here we use simple mutations for illustration.
        
        mutations = []
        
        # Mutation 1: Rephrase the question
        mutations.append(f"Rephrase and answer this question: {prompt}")
        
        # Mutation 2: Ask for step-by-step reasoning
        mutations.append(f"Think step by step to answer: {prompt}")
        
        # Mutation 3: Ask for verification
        mutations.append(f"Verify your knowledge about this question and answer: {prompt}")
        
        # Use a subset of mutations up to num_mutations
        return mutations[:self.num_mutations]
    
    def compute_uncertainty(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Compute uncertainty using metamorphic relations.
        
        Args:
            prompt: Input prompt
            response: Model response
            
        Returns:
            Dictionary with uncertainty metrics
        """
        # Generate prompt mutations
        mutations = self._generate_mutations(prompt)
        
        # Get responses for mutations
        mutation_responses = []
        for mutation in mutations:
            result = self.llm.generate(prompt=mutation, max_tokens=500, temperature=0.0)
            mutation_responses.append(result["text"])
        
        # Compare original response with mutation responses
        # (Simplified - in practice, would use more sophisticated metrics)
        similarities = []
        for mut_resp in mutation_responses:
            # Simple Jaccard similarity of token sets
            tokens_orig = set(response.lower().split())
            tokens_mut = set(mut_resp.lower().split())
            
            intersection = len(tokens_orig.intersection(tokens_mut))
            union = len(tokens_orig.union(tokens_mut))
            
            if union > 0:
                similarity = intersection / union
            else:
                similarity = 0.0
            
            similarities.append(similarity)
        
        # Average similarity as consistency score
        if similarities:
            consistency_score = float(np.mean(similarities))
        else:
            consistency_score = 0.5  # Default mid-point if no similarities
        
        # Map consistency to uncertainty (higher consistency = lower uncertainty)
        uncertainty_score = 1.0 - consistency_score
        
        return {
            "uncertainty_score": uncertainty_score,
            "confidence_score": consistency_score,
            "mutations": mutations,
            "mutation_responses": mutation_responses,
            "similarities": similarities,
            "method": "metaqa",
        }
    
    def generate_with_uncertainty(
        self, 
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response and compute its uncertainty using MetaQA.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generation result and uncertainty metrics
        """
        # Generate response
        generation_result = self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        # Compute uncertainty
        uncertainty_result = self.compute_uncertainty(prompt, generation_result["text"])
        
        # Combine results
        result = {
            **generation_result,
            **uncertainty_result,
        }
        
        return result


def get_baseline_method(
    method_name: str,
    llm: Union[LLMInterface, str],
    **kwargs
) -> BaselineMethod:
    """
    Factory function to create a baseline method.
    
    Args:
        method_name: Name of the baseline method
        llm: LLM interface or model name
        **kwargs: Additional arguments for the baseline method
        
    Returns:
        Instance of a BaselineMethod subclass
    """
    method_name = method_name.lower()
    
    if method_name == "vanilla":
        return VanillaBaseline(llm, **kwargs)
    elif method_name in ["sep", "semantic_entropy_probes"]:
        return SemanticEntropyProbes(llm, **kwargs)
    elif method_name in ["uaf", "uncertainty_aware_fusion"]:
        # For UAF, llm should be a list of LLMs
        if not isinstance(llm, list):
            llm = [llm]
        return UncertaintyAwareFusion(llm, **kwargs)
    elif method_name in ["ccp", "claim_conditioned_probability"]:
        return ClaimConditionedProbability(llm, **kwargs)
    elif method_name in ["metaqa"]:
        return MetaQA(llm, **kwargs)
    else:
        raise ValueError(f"Unknown baseline method: {method_name}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage (commented out since it requires actual LLM instances)
    """
    from models.llm_interface import get_llm_interface
    
    # Initialize LLM
    llm = get_llm_interface("claude-3-sonnet")
    
    # Create baseline method
    baseline = get_baseline_method("vanilla", llm)
    
    # Generate text with uncertainty estimation
    prompt = "What is the capital of France?"
    result = baseline.generate_with_uncertainty(prompt)
    
    print(f"Generated text: {result['text']}")
    print(f"Uncertainty score: {result['uncertainty_score']}")
    """
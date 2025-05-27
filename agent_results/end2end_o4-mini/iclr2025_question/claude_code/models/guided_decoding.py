"""
Uncertainty-guided decoding with hallucination penalties for SCEC.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

from .llm_interface import LLMInterface, get_llm_interface
from .self_consistency import SelfConsistencySampler
from .evidence_retrieval import EvidenceAligner, ClaimExtractor, EntailmentScorer
from .uncertainty_scoring import UncertaintyScorer

logger = logging.getLogger(__name__)

class GuidedDecoder:
    """
    Base class for decoding with hallucination penalties based on uncertainty.
    
    This implements the core SCEC guided decoding algorithm:
    
    p̃_t(w) ∝ p_t(w)·exp(-β·u_t(w))
    
    where:
    - p̃_t(w) is the adjusted probability for token w at time t
    - p_t(w) is the original probability from the base LLM
    - u_t(w) is the uncertainty score for token w
    - β > 0 is a temperature-like coefficient controlling penalty strength
    """
    
    def __init__(
        self,
        uncertainty_scorer: UncertaintyScorer,
        beta: float = 0.1,
    ):
        """
        Initialize the guided decoder.
        
        Args:
            uncertainty_scorer: UncertaintyScorer instance
            beta: Strength of hallucination penalty (higher = stronger penalty)
        """
        self.uncertainty_scorer = uncertainty_scorer
        self.beta = beta
    
    def adjust_probabilities(
        self,
        original_probs: np.ndarray,
        uncertainty_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Apply uncertainty-based adjustment to token probabilities.
        
        Args:
            original_probs: Original token probabilities
            uncertainty_scores: Uncertainty scores for each token
            
        Returns:
            Adjusted token probabilities
        """
        # Apply penalty: p̃(w) ∝ p(w)·exp(-β·u(w))
        penalty = np.exp(-self.beta * uncertainty_scores)
        adjusted_probs = original_probs * penalty
        
        # Renormalize to sum to 1
        if np.sum(adjusted_probs) > 0:
            adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
        else:
            # If all probabilities are zero after adjustment, revert to original
            adjusted_probs = original_probs
        
        return adjusted_probs
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response with uncertainty-guided decoding.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            **kwargs: Additional generation parameters
            
        Returns:
            Generation result dictionary
        """
        raise NotImplementedError("Subclasses must implement generate")


class APIGuidedDecoder(GuidedDecoder):
    """
    Guided decoder for API-based LLMs.
    
    Since we cannot directly modify the decoding algorithm of API-based models like OpenAI or Anthropic,
    this class implements a multi-step approach:
    
    1. Generate sample responses with different temps
    2. Analyze uncertainty using SCEC
    3. Select the response with lowest overall uncertainty
    4. For any uncertainty above threshold, query the model again with a modified prompt
    """
    
    def __init__(
        self,
        llm: Union[LLMInterface, str],
        uncertainty_scorer: UncertaintyScorer,
        beta: float = 0.1,
        uncertainty_threshold: float = 0.7,
        **llm_kwargs
    ):
        """
        Initialize the API-based guided decoder.
        
        Args:
            llm: LLM interface or model name string
            uncertainty_scorer: UncertaintyScorer instance
            beta: Strength of hallucination penalty
            uncertainty_threshold: Threshold for flagging high uncertainty
            **llm_kwargs: Additional arguments for the LLM interface
        """
        super().__init__(uncertainty_scorer, beta)
        
        if isinstance(llm, str):
            self.llm = get_llm_interface(llm, **llm_kwargs)
        else:
            self.llm = llm
        
        self.uncertainty_threshold = uncertainty_threshold
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_candidates: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response with uncertainty-guided decoding.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            num_candidates: Number of candidate responses to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generation result dictionary
        """
        start_time = time.time()
        
        # Step 1: Generate multiple candidate responses
        candidates = []
        
        # Generate candidates with different temperature settings
        temps = np.linspace(0.1, temperature, num_candidates)
        
        for i, temp in enumerate(temps):
            result = self.llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                **kwargs
            )
            candidates.append(result)
        
        # Step 2: Analyze each candidate for uncertainty
        uncertainty_results = []
        
        for candidate in tqdm(candidates, desc="Analyzing candidates"):
            # Get token-level uncertainty from the SCEC pipeline
            uncertainty = self.uncertainty_scorer.get_token_uncertainty(prompt)
            uncertainty_results.append(uncertainty)
            
            # Add uncertainty score to candidate
            candidate["uncertainty_score"] = uncertainty["sequence_uncertainty"]
        
        # Step 3: Select the candidate with lowest uncertainty
        candidates_with_scores = sorted(candidates, key=lambda c: c["uncertainty_score"])
        best_candidate = candidates_with_scores[0]
        best_uncertainty = best_candidate["uncertainty_score"]
        
        # Step 4: If uncertainty is still too high, try refined prompting
        if best_uncertainty > self.uncertainty_threshold and len(prompt.split()) < 500:
            logger.info(f"Best candidate has high uncertainty ({best_uncertainty:.3f}), attempting refined prompt")
            
            # Add a request for confidence and verification
            refined_prompt = (
                f"{prompt}\n\n"
                "Important: Please be factual and accurate in your response. "
                "If you're uncertain about any details, please state so clearly. "
                "Only include information you're confident about and can support with evidence."
            )
            
            # Generate with the refined prompt and lower temperature
            refined_result = self.llm.generate(
                prompt=refined_prompt,
                max_tokens=max_tokens,
                temperature=min(0.5, temperature),  # Lower temperature for refinement
                top_p=top_p,
                **kwargs
            )
            
            # Analyze refined result
            refined_uncertainty = self.uncertainty_scorer.get_token_uncertainty(refined_prompt)
            refined_result["uncertainty_score"] = refined_uncertainty["sequence_uncertainty"]
            
            # Use refined result if it has lower uncertainty
            if refined_result["uncertainty_score"] < best_uncertainty:
                best_candidate = refined_result
                best_uncertainty = refined_result["uncertainty_score"]
        
        elapsed_time = time.time() - start_time
        
        # Create final result
        result = {
            "text": best_candidate["text"],
            "uncertainty_score": best_uncertainty,
            "uncertainty_level": "high" if best_uncertainty > self.uncertainty_threshold else "medium" if best_uncertainty > 0.3 else "low",
            "elapsed_time": elapsed_time,
            "num_candidates": num_candidates,
            "all_candidates": [
                {
                    "text": c["text"],
                    "uncertainty_score": c.get("uncertainty_score", float('inf')),
                }
                for c in candidates_with_scores
            ]
        }
        
        return result


class HuggingFaceGuidedDecoder(GuidedDecoder):
    """
    Guided decoder for local HuggingFace models.
    
    This implementation directly modifies the token probabilities during beam search
    or sampling decoding, applying the hallucination penalty at each generation step.
    
    Note: This is a simplified implementation and would need additional work to
    properly integrate with the full HuggingFace generation pipeline in practice.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        uncertainty_scorer: UncertaintyScorer,
        beta: float = 0.1,
        device: str = "auto",
    ):
        """
        Initialize the HuggingFace guided decoder.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            uncertainty_scorer: UncertaintyScorer instance
            beta: Strength of hallucination penalty
            device: Device to run model on
        """
        super().__init__(uncertainty_scorer, beta)
        
        self.model = model
        self.tokenizer = tokenizer
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
    
    def _custom_logits_processor(self, uncertainty_map):
        """
        Create a custom logits processor that applies uncertainty penalties.
        
        Args:
            uncertainty_map: Mapping of token IDs to uncertainty scores
            
        Returns:
            A callable that modifies logits based on uncertainty
        """
        def process_logits(input_ids, scores):
            # Get the current token position
            current_pos = input_ids.shape[-1] - 1
            
            # Get logits for current position
            current_logits = scores.clone()
            
            # Convert logits to probabilities
            probs = torch.softmax(current_logits, dim=-1)
            
            # Get uncertainty scores for each token
            # This is a simplified mapping - in practice, you'd need to
            # predict uncertainty for each possible next token
            uncertainties = torch.zeros_like(probs)
            for token_id, uncertainty in uncertainty_map.items():
                uncertainties[token_id] = uncertainty
            
            # Apply penalty
            penalty = torch.exp(-self.beta * uncertainties)
            adjusted_probs = probs * penalty
            
            # Renormalize
            adjusted_probs = adjusted_probs / adjusted_probs.sum()
            
            # Convert back to logits
            adjusted_logits = torch.log(adjusted_probs + 1e-10)
            
            return adjusted_logits
        
        return process_logits
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response with uncertainty-guided decoding.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            **kwargs: Additional generation parameters
            
        Returns:
            Generation result dictionary
        """
        start_time = time.time()
        
        # Step 1: Get uncertainty predictions from sampler
        # This is a simplification - in practice, you'd need to develop
        # a more sophisticated approach to predict per-token uncertainty
        # for every possible next token at each step
        uncertainty_result = self.uncertainty_scorer.get_token_uncertainty(prompt)
        
        # Create a simplified uncertainty map based on prior samples
        # In practice, this would be much more sophisticated
        uncertainty_map = {}  # token_id -> uncertainty_score
        
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Create logits processor list with our custom processor
        logits_processor = self._custom_logits_processor(uncertainty_map)
        
        # Generate with custom logits processor
        with torch.no_grad():
            # Note: This is a simplified version. In practice, you would use
            # the HuggingFace generate method with custom logits_processor
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
        
        # Decode the outputs
        generated_text = self.tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        elapsed_time = time.time() - start_time
        
        # Create final result
        result = {
            "text": generated_text,
            "elapsed_time": elapsed_time,
        }
        
        return result


class SCECPipeline:
    """
    Complete SCEC pipeline combining all components.
    
    This class provides a unified interface for the entire SCEC process:
    1. Self-consistency sampling
    2. Evidence retrieval and alignment
    3. Uncertainty scoring
    4. Guided decoding with hallucination penalties
    """
    
    def __init__(
        self,
        llm: Union[LLMInterface, str],
        alpha: float = 0.5,
        beta: float = 0.1,
        num_samples: int = 10,
        evidence_retriever_type: str = "wikipedia",
        corpus_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the SCEC pipeline.
        
        Args:
            llm: LLM interface or model name
            alpha: Weight for balancing variance and evidence alignment
            beta: Strength of hallucination penalty
            num_samples: Number of samples for self-consistency
            evidence_retriever_type: Type of evidence retriever
            corpus_path: Path to corpus file (for BM25 or Dense retrievers)
            cache_dir: Directory to cache results
            **kwargs: Additional arguments for components
        """
        # Initialize LLM interface
        if isinstance(llm, str):
            self.llm = get_llm_interface(llm)
        else:
            self.llm = llm
        
        # Cache directory
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize self-consistency sampler
        self.sampler = SelfConsistencySampler(
            llm=self.llm,
            num_samples=num_samples,
            temperature=kwargs.get("temperature", 0.7),
            use_cot=kwargs.get("use_cot", True),
            cot_prompt=kwargs.get("cot_prompt", "Let's think through this step-by-step:"),
            seed=kwargs.get("seed", 42),
        )
        
        # Initialize evidence components
        self.claim_extractor = ClaimExtractor(use_spacy=kwargs.get("use_spacy", True))
        self.entailment_scorer = EntailmentScorer(
            model_name=kwargs.get("entailment_model", "facebook/bart-large-mnli"),
            device=kwargs.get("device", "auto"),
        )
        
        # Initialize evidence retriever based on type
        from .evidence_retrieval import (
            WikipediaRetriever,
            BM25Retriever,
            DenseRetriever,
        )
        
        if evidence_retriever_type == "wikipedia":
            self.retriever = WikipediaRetriever(
                cache_dir=os.path.join(cache_dir, "wikipedia_cache") if cache_dir else None,
                language=kwargs.get("language", "en"),
                user_agent=kwargs.get("user_agent", "SCEC-Experiment/1.0"),
            )
        elif evidence_retriever_type == "bm25":
            if not corpus_path:
                raise ValueError("corpus_path is required for BM25 retriever")
            
            self.retriever = BM25Retriever(
                corpus_path=corpus_path,
                cache_dir=os.path.join(cache_dir, "bm25_cache") if cache_dir else None,
            )
        elif evidence_retriever_type == "dense":
            if not corpus_path:
                raise ValueError("corpus_path is required for Dense retriever")
            
            self.retriever = DenseRetriever(
                corpus_path=corpus_path,
                model_name=kwargs.get("retriever_model", "facebook/dpr-question_encoder-single-nq-base"),
                context_model_name=kwargs.get("context_model", None),
                cache_dir=os.path.join(cache_dir, "dense_cache") if cache_dir else None,
                device=kwargs.get("device", "auto"),
            )
        else:
            raise ValueError(f"Unknown retriever type: {evidence_retriever_type}")
        
        # Initialize evidence aligner
        self.aligner = EvidenceAligner(
            retriever=self.retriever,
            claim_extractor=self.claim_extractor,
            entailment_scorer=self.entailment_scorer,
        )
        
        # Initialize uncertainty scorer
        self.scorer = UncertaintyScorer(
            self_consistency_sampler=self.sampler,
            evidence_aligner=self.aligner,
            alpha=alpha,
        )
        
        # Initialize guided decoder
        # Choose decoder based on LLM type
        if isinstance(self.llm, get_llm_interface("huggingface-model").__class__):
            # This is a placeholder - in practice, would need to extract model and tokenizer
            model = None
            tokenizer = None
            self.decoder = HuggingFaceGuidedDecoder(
                model=model,
                tokenizer=tokenizer,
                uncertainty_scorer=self.scorer,
                beta=beta,
                device=kwargs.get("device", "auto"),
            )
        else:
            # Use API-based decoder for OpenAI/Anthropic models
            self.decoder = APIGuidedDecoder(
                llm=self.llm,
                uncertainty_scorer=self.scorer,
                beta=beta,
                uncertainty_threshold=kwargs.get("uncertainty_threshold", 0.7),
            )
    
    def run(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_uncertainty: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the full SCEC pipeline on a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            return_uncertainty: Whether to include uncertainty details in result
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generation results and optional uncertainty details
        """
        # Step 1: Generate a response with guided decoding
        result = self.decoder.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        # Step 2: If requested, include detailed uncertainty analysis
        if return_uncertainty:
            # Get full uncertainty analysis
            uncertainty_result = self.scorer.get_token_uncertainty(prompt)
            
            # Add to result
            result["uncertainty_details"] = {
                "sequence_uncertainty": uncertainty_result["sequence_uncertainty"],
                "variance_component": uncertainty_result["variance_component"],
                "evidence_component": uncertainty_result["evidence_component"],
            }
            
            if "token_uncertainties" in uncertainty_result:
                result["uncertainty_details"]["token_uncertainties"] = uncertainty_result["token_uncertainties"]
        
        return result


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # This would be a minimal example of the guided decoder
    # You would need instances of UncertaintyScorer, etc.
    
    # Example usage (commented out since it requires actual instances)
    """
    from models.llm_interface import get_llm_interface
    from models.self_consistency import SelfConsistencySampler
    from models.evidence_retrieval import EvidenceAligner, BM25Retriever, ClaimExtractor, EntailmentScorer
    from models.uncertainty_scoring import UncertaintyScorer
    
    # Initialize LLM
    llm = get_llm_interface("claude-3-sonnet")
    
    # Initialize components
    sampler = SelfConsistencySampler(llm, num_samples=5)
    
    # Create a BM25 retriever with a local corpus
    corpus_path = "data/synthetic_corpus.json"
    retriever = BM25Retriever(corpus_path=corpus_path, cache_dir="cache")
    
    # Initialize claim extractor and entailment scorer
    claim_extractor = ClaimExtractor()
    entailment_scorer = EntailmentScorer()
    
    # Initialize evidence aligner
    aligner = EvidenceAligner(retriever, claim_extractor, entailment_scorer)
    
    # Create uncertainty scorer
    scorer = UncertaintyScorer(sampler, aligner, alpha=0.5)
    
    # Create guided decoder
    decoder = APIGuidedDecoder(llm, scorer, beta=0.1)
    
    # Generate text with hallucination penalty
    prompt = "What is the capital of France?"
    result = decoder.generate(prompt)
    
    print(f"Generated text: {result['text']}")
    print(f"Uncertainty score: {result['uncertainty_score']}")
    """
"""
Self-consistency sampling implementation for SCEC.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import torch
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from .llm_interface import LLMInterface, get_llm_interface

logger = logging.getLogger(__name__)

class SelfConsistencySampler:
    """
    Implements self-consistency sampling for LLMs.
    
    Generates multiple diverse chains of thought for a given prompt and
    analyzes the variance in model responses to quantify uncertainty.
    """
    
    def __init__(
        self,
        llm: Union[LLMInterface, str],
        num_samples: int = 10,
        temperature: float = 0.7,
        use_cot: bool = True,
        cot_prompt: str = "Let's think through this step-by-step:",
        seed: int = 42,
        **llm_kwargs
    ):
        """
        Initialize the self-consistency sampler.
        
        Args:
            llm: LLM interface or model name string
            num_samples: Number of samples to generate per prompt
            temperature: Temperature for generation
            use_cot: Whether to use chain-of-thought prompting
            cot_prompt: Prompt suffix for chain-of-thought
            seed: Random seed for reproducibility
            **llm_kwargs: Additional arguments for the LLM interface
        """
        if isinstance(llm, str):
            self.llm = get_llm_interface(llm, **llm_kwargs)
        else:
            self.llm = llm
        
        self.num_samples = num_samples
        self.temperature = temperature
        self.use_cot = use_cot
        self.cot_prompt = cot_prompt
        self.seed = seed
        
        # For storing generated samples
        self.samples = {}
        
        np.random.seed(seed)
    
    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt with chain-of-thought if enabled."""
        if self.use_cot and self.cot_prompt:
            return f"{prompt}\n\n{self.cot_prompt}"
        return prompt
    
    def generate_samples(
        self, 
        prompt: str, 
        max_tokens: int = 500,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple diverse samples for a given prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            force_refresh: Force regeneration of samples even if they exist
            
        Returns:
            List of sample dictionaries with generation info
        """
        # Check if we already have samples for this prompt
        if prompt in self.samples and not force_refresh:
            return self.samples[prompt]
        
        formatted_prompt = self._format_prompt(prompt)
        
        # Generate multiple samples
        samples = self.llm.generate_multiple(
            prompt=formatted_prompt,
            n=self.num_samples,
            temperature=self.temperature,
            max_tokens=max_tokens,
            seed=self.seed,
        )
        
        # Store the samples
        self.samples[prompt] = samples
        
        return samples
    
    def compute_token_variance(
        self, 
        prompt: str, 
        samples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Compute token-level variance metrics from generated samples.
        
        Args:
            prompt: The input prompt
            samples: Optional list of sample dictionaries (if not provided, will use cached samples)
            
        Returns:
            Dictionary of token-level variance metrics
        """
        # Get samples if not provided
        if samples is None:
            if prompt not in self.samples:
                samples = self.generate_samples(prompt)
            else:
                samples = self.samples[prompt]
        
        # Extract logprobs from samples
        token_logprobs_list = []
        for sample in samples:
            if "token_logprobs" in sample:
                token_logprobs_list.append(sample["token_logprobs"])
        
        # Compute variance metrics
        result = {
            "samples": samples,
            "num_samples": len(samples),
        }
        
        # Only compute variance if we have token-level logprobs
        if token_logprobs_list:
            # Pad sequences to the same length
            max_length = max(len(lp) for lp in token_logprobs_list)
            padded_logprobs = []
            for lp in token_logprobs_list:
                padded = lp + [0.0] * (max_length - len(lp))
                padded_logprobs.append(padded)
            
            # Compute variance at each token position
            token_variances = np.var(padded_logprobs, axis=0)
            result["token_variances"] = token_variances.tolist()
            result["mean_token_variance"] = np.mean(token_variances)
            result["max_token_variance"] = np.max(token_variances)
        
        return result
    
    def compute_sequence_divergence(
        self, 
        prompt: str, 
        samples: Optional[List[Dict[str, Any]]] = None,
        divergence_type: str = "js"
    ) -> Dict[str, Any]:
        """
        Compute divergence metrics between generated sequences.
        
        Args:
            prompt: The input prompt
            samples: Optional list of sample dictionaries
            divergence_type: Type of divergence metric ('js' for Jensen-Shannon, 'kl' for KL)
            
        Returns:
            Dictionary of sequence-level divergence metrics
        """
        # Get samples if not provided
        if samples is None:
            if prompt not in self.samples:
                samples = self.generate_samples(prompt)
            else:
                samples = self.samples[prompt]
        
        # Extract generated texts
        texts = [sample["text"] for sample in samples]
        
        # Compute pairwise divergences
        n = len(texts)
        divergences = np.zeros((n, n))
        
        # Simple character-level frequency representation
        # (This is a simplification - in practice, we might use embeddings or more sophisticated representations)
        char_distributions = []
        for text in texts:
            # Count character frequencies
            char_counts = {}
            for char in text.lower():
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # Create a normalized distribution
            total = sum(char_counts.values())
            char_dist = {char: count / total for char, count in char_counts.items()}
            char_distributions.append(char_dist)
        
        # Compute all character keys
        all_chars = set()
        for dist in char_distributions:
            all_chars.update(dist.keys())
        all_chars = sorted(list(all_chars))
        
        # Create normalized distributions for comparison
        distributions = []
        for dist in char_distributions:
            full_dist = [dist.get(char, 0.0) for char in all_chars]
            distributions.append(full_dist)
        
        # Compute pairwise divergences
        for i in range(n):
            for j in range(i+1, n):
                if divergence_type == "js":
                    div = jensenshannon(distributions[i], distributions[j])
                else:  # KL divergence
                    # Add smoothing to avoid division by zero
                    p = np.array(distributions[i]) + 1e-10
                    q = np.array(distributions[j]) + 1e-10
                    p /= p.sum()
                    q /= q.sum()
                    div = entropy(p, q)
                
                divergences[i, j] = div
                divergences[j, i] = div
        
        # Compute summary statistics
        mean_divergence = np.mean(divergences[np.triu_indices(n, k=1)])
        max_divergence = np.max(divergences[np.triu_indices(n, k=1)])
        
        result = {
            "divergence_matrix": divergences.tolist(),
            "mean_divergence": float(mean_divergence),
            "max_divergence": float(max_divergence),
            "divergence_type": divergence_type,
        }
        
        return result
    
    def analyze_samples(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        """
        Generate samples and compute all self-consistency metrics.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with all self-consistency metrics
        """
        # Generate samples if needed
        if prompt not in self.samples:
            samples = self.generate_samples(prompt, max_tokens=max_tokens)
        else:
            samples = self.samples[prompt]
        
        # Compute token variance
        token_variance = self.compute_token_variance(prompt, samples)
        
        # Compute sequence divergence
        sequence_divergence = self.compute_sequence_divergence(prompt, samples)
        
        # Combine results
        result = {
            "prompt": prompt,
            "samples": samples,
            "token_variance": token_variance,
            "sequence_divergence": sequence_divergence,
        }
        
        return result
    
    def save_results(self, output_dir: str, filename: str = "self_consistency_results.json"):
        """
        Save self-consistency analysis results to a file.
        
        Args:
            output_dir: Directory to save results
            filename: Filename for results
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # Create a serializable version of results
        serializable_results = {}
        for prompt, samples in self.samples.items():
            # Add analysis results for each prompt
            serializable_results[prompt] = {
                "samples": [
                    {
                        "text": sample["text"],
                        "token_logprobs": sample.get("token_logprobs", []),
                        "finish_reason": sample.get("finish_reason", "unknown"),
                    }
                    for sample in samples
                ],
                "token_variance": self.compute_token_variance(prompt, samples),
                "sequence_divergence": self.compute_sequence_divergence(prompt, samples),
            }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved self-consistency results to {output_path}")


def compute_chain_similarities(
    chains: List[str],
    method: str = "character_ngrams"
) -> np.ndarray:
    """
    Compute pairwise similarities between chains of thought.
    
    Args:
        chains: List of chain-of-thought strings
        method: Method for computing similarity ('character_ngrams', 'word_ngrams', or 'embedding')
        
    Returns:
        n x n similarity matrix
    """
    n = len(chains)
    similarity_matrix = np.zeros((n, n))
    
    # Compute similarity based on character n-grams
    if method == "character_ngrams":
        # Extract character n-grams (n=3) for each chain
        def get_char_ngrams(text, n=3):
            text = text.lower()
            return [text[i:i+n] for i in range(len(text) - n + 1)]
        
        ngrams_list = [set(get_char_ngrams(chain)) for chain in chains]
        
        # Compute Jaccard similarity between n-gram sets
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    ngrams_i = ngrams_list[i]
                    ngrams_j = ngrams_list[j]
                    intersection = len(ngrams_i.intersection(ngrams_j))
                    union = len(ngrams_i.union(ngrams_j))
                    if union > 0:
                        similarity_matrix[i, j] = intersection / union
    
    # Compute similarity based on word n-grams
    elif method == "word_ngrams":
        # Extract word n-grams (n=2) for each chain
        def get_word_ngrams(text, n=2):
            words = text.lower().split()
            return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        ngrams_list = [set(get_word_ngrams(chain)) for chain in chains]
        
        # Compute Jaccard similarity between n-gram sets
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    ngrams_i = ngrams_list[i]
                    ngrams_j = ngrams_list[j]
                    intersection = len(ngrams_i.intersection(ngrams_j))
                    union = len(ngrams_i.union(ngrams_j))
                    if union > 0:
                        similarity_matrix[i, j] = intersection / union
    
    # For embedding method, we would need to import sentence-transformers
    # This is a placeholder that would be implemented in practice
    elif method == "embedding":
        # Placeholder for embedding-based similarity
        # In practice, we would use a sentence embedding model like SBERT
        similarity_matrix = np.eye(n)
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    return similarity_matrix


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    model_name = "claude-3-7-sonnet-20250219"  # Change to an available model
    sampler = SelfConsistencySampler(
        llm=model_name,
        num_samples=5,
        temperature=0.7,
        use_cot=True,
    )
    
    prompt = "What is the capital of France?"
    results = sampler.analyze_samples(prompt, max_tokens=300)
    
    print(f"Generated {len(results['samples'])} samples")
    print(f"Mean token variance: {results['token_variance'].get('mean_token_variance', 'N/A')}")
    print(f"Mean sequence divergence: {results['sequence_divergence']['mean_divergence']}")
    
    # Save results
    sampler.save_results("results", "self_consistency_example.json")
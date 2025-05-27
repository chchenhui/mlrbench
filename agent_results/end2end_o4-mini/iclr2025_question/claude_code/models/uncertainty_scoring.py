"""
Uncertainty score calculation module for SCEC.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
from tqdm import tqdm

from .self_consistency import SelfConsistencySampler
from .evidence_retrieval import EvidenceAligner

logger = logging.getLogger(__name__)

class UncertaintyScorer:
    """
    Calculate uncertainty scores combining self-consistency variance and evidence agreement.
    
    This implements the core SCEC uncertainty scoring algorithm:
    
    u_t = α·u^var_t + (1-α)·[1 - (1/k)·∑_{i=1}^k s^{(i)}_t]
    
    where:
    - u_t is the composite uncertainty at token t
    - u^var_t is the variance in model outputs at token t
    - s^{(i)}_t is the evidence agreement score for token t in sample i
    - α balances variance and evidence misalignment (α ∈ [0,1])
    """
    
    def __init__(
        self,
        self_consistency_sampler: SelfConsistencySampler,
        evidence_aligner: EvidenceAligner,
        alpha: float = 0.5,
    ):
        """
        Initialize the uncertainty scorer.
        
        Args:
            self_consistency_sampler: Self-consistency sampler instance
            evidence_aligner: Evidence aligner instance
            alpha: Weight parameter balancing variance and evidence alignment (0-1)
        """
        self.self_consistency_sampler = self_consistency_sampler
        self.evidence_aligner = evidence_aligner
        self.alpha = alpha
        
        # Validate alpha
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    
    def get_token_uncertainty(
        self,
        prompt: str,
        max_tokens: int = 500,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate token-level uncertainty scores.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            force_refresh: Force regeneration of samples
            
        Returns:
            Dictionary with token-level uncertainty scores and related data
        """
        # Step 1: Generate samples using self-consistency
        samples = self.self_consistency_sampler.generate_samples(
            prompt=prompt,
            max_tokens=max_tokens,
            force_refresh=force_refresh,
        )
        
        # Step 2: Get token variance metrics
        variance_results = self.self_consistency_sampler.compute_token_variance(prompt, samples)
        
        # Step 3: Calculate evidence agreement for each sample
        evidence_results = []
        for sample in samples:
            # Extract the generated text
            text = sample["text"]
            
            # Get evidence alignment
            alignment = self.evidence_aligner.align(text)
            evidence_results.append(alignment)
            
            # Add agreement score to the sample
            sample["evidence_agreement"] = alignment["overall_agreement"]
        
        # Step 4: Prepare token-level arrays
        token_uncertainties = []
        
        # If we have token-level info
        if "token_variances" in variance_results:
            token_variances = variance_results["token_variances"]
            
            # Create a mapping from tokens to claims for evidence scoring
            # This is a simplification - in practice, we'd need more sophisticated
            # alignment between tokens and claims
            for i, sample in enumerate(samples):
                if "tokens" not in sample:
                    continue
                    
                tokens = sample["tokens"]
                evidence_agreement = sample["evidence_agreement"]
                
                # Assign the overall agreement score to all tokens as a simplification
                token_agreements = [evidence_agreement] * len(tokens)
                
                # If we have token logprobs, use that length instead
                if "token_logprobs" in sample:
                    token_agreements = [evidence_agreement] * len(sample["token_logprobs"])
                
                # Create token-level uncertainty scores
                sample_token_uncertainties = []
                for j, (var, agr) in enumerate(zip(token_variances, token_agreements)):
                    # Apply the uncertainty formula: α·variance + (1-α)·(1-agreement)
                    uncertainty = self.alpha * var + (1 - self.alpha) * (1 - agr)
                    sample_token_uncertainties.append(uncertainty)
                
                token_uncertainties.append(sample_token_uncertainties)
        
        # Step 5: Compute aggregate uncertainty metrics
        # Average token-level uncertainty across samples
        if token_uncertainties:
            # Pad to same length
            max_length = max(len(u) for u in token_uncertainties)
            padded_uncertainties = []
            for u in token_uncertainties:
                padded = u + [0.0] * (max_length - len(u))
                padded_uncertainties.append(padded)
            
            # Compute average uncertainty at each position
            avg_token_uncertainties = np.mean(padded_uncertainties, axis=0)
            max_token_uncertainties = np.max(padded_uncertainties, axis=0)
            
            # Find positions with highest uncertainty
            highest_uncertainty_pos = np.argsort(avg_token_uncertainties)[-5:][::-1]
        else:
            avg_token_uncertainties = []
            max_token_uncertainties = []
            highest_uncertainty_pos = []
        
        # Compute segment-level uncertainty using self-consistency divergence
        sequence_divergence = self.self_consistency_sampler.compute_sequence_divergence(prompt, samples)
        
        # Average evidence agreement
        avg_evidence_agreement = np.mean([s["evidence_agreement"] for s in evidence_results])
        
        # Step 6: Create final uncertainty score
        # Combine sequence divergence with evidence agreement
        sequence_uncertainty = self.alpha * sequence_divergence["mean_divergence"] + \
                              (1 - self.alpha) * (1 - avg_evidence_agreement)
        
        # Compile all results
        result = {
            "prompt": prompt,
            "alpha": self.alpha,
            "token_uncertainties": {
                "values": avg_token_uncertainties.tolist() if isinstance(avg_token_uncertainties, np.ndarray) else [],
                "max_values": max_token_uncertainties.tolist() if isinstance(max_token_uncertainties, np.ndarray) else [],
                "highest_positions": highest_uncertainty_pos.tolist() if isinstance(highest_uncertainty_pos, np.ndarray) else [],
            },
            "sequence_uncertainty": float(sequence_uncertainty),
            "variance_component": float(sequence_divergence["mean_divergence"]),
            "evidence_component": float(1 - avg_evidence_agreement),
            "samples": [
                {
                    "text": s["text"],
                    "evidence_agreement": s.get("evidence_agreement", 0.0),
                }
                for s in samples
            ]
        }
        
        return result
    
    def get_uncertainty_threshold(self, uncertainty_score: float) -> str:
        """
        Convert an uncertainty score to a qualitative threshold.
        
        Args:
            uncertainty_score: Numeric uncertainty score
            
        Returns:
            String representing the uncertainty level ("low", "medium", "high")
        """
        if uncertainty_score < 0.3:
            return "low"
        elif uncertainty_score < 0.7:
            return "medium"
        else:
            return "high"
    
    def format_uncertainty_report(self, uncertainty_result: Dict[str, Any]) -> str:
        """
        Format an uncertainty result into a human-readable report.
        
        Args:
            uncertainty_result: Result from get_token_uncertainty
            
        Returns:
            Formatted report string
        """
        sequence_uncertainty = uncertainty_result["sequence_uncertainty"]
        uncertainty_level = self.get_uncertainty_threshold(sequence_uncertainty)
        
        report = [
            f"Uncertainty Report:",
            f"Overall uncertainty: {sequence_uncertainty:.3f} ({uncertainty_level})",
            f"Variance component: {uncertainty_result['variance_component']:.3f}",
            f"Evidence component: {uncertainty_result['evidence_component']:.3f}",
            f"",
            f"Sample responses:",
        ]
        
        for i, sample in enumerate(uncertainty_result["samples"][:3]):  # Show first 3 samples
            report.append(f"Sample {i+1} (Evidence agreement: {sample['evidence_agreement']:.3f}):")
            report.append(f"  {sample['text'][:100]}...")
            report.append("")
        
        return "\n".join(report)


class TokenLevelUncertaintyVisualizer:
    """Visualize token-level uncertainty in LLM outputs."""
    
    def __init__(self, uncertainty_scorer: UncertaintyScorer):
        """
        Initialize the uncertainty visualizer.
        
        Args:
            uncertainty_scorer: UncertaintyScorer instance
        """
        self.uncertainty_scorer = uncertainty_scorer
    
    def visualize_token_uncertainties(
        self, 
        uncertainty_result: Dict[str, Any],
        threshold_low: float = 0.3,
        threshold_high: float = 0.7,
    ) -> str:
        """
        Create a visualization of token-level uncertainties.
        
        Args:
            uncertainty_result: Result from uncertainty_scorer.get_token_uncertainty
            threshold_low: Threshold for low uncertainty
            threshold_high: Threshold for high uncertainty
            
        Returns:
            String with ANSI color-coded tokens based on uncertainty
        """
        # Get a representative sample
        if not uncertainty_result["samples"]:
            return "No samples available for visualization"
        
        sample = uncertainty_result["samples"][0]
        text = sample["text"]
        
        token_uncertainties = uncertainty_result["token_uncertainties"]["values"]
        
        # If we don't have token-level uncertainties, return the raw text
        if not token_uncertainties:
            return text
        
        # Tokenize the text (simplistic approach - in practice, should use the model's tokenizer)
        tokens = text.split()
        
        # Truncate or pad token uncertainties to match token count
        n_tokens = len(tokens)
        n_uncertainties = len(token_uncertainties)
        
        if n_uncertainties < n_tokens:
            # Pad with zeros
            token_uncertainties = token_uncertainties + [0.0] * (n_tokens - n_uncertainties)
        elif n_uncertainties > n_tokens:
            # Truncate
            token_uncertainties = token_uncertainties[:n_tokens]
        
        # Create colored visualization
        # Green: low uncertainty, Yellow: medium, Red: high
        colored_tokens = []
        for token, uncertainty in zip(tokens, token_uncertainties):
            if uncertainty < threshold_low:
                # Green for low uncertainty
                colored_token = f"\033[92m{token}\033[0m"
            elif uncertainty < threshold_high:
                # Yellow for medium uncertainty
                colored_token = f"\033[93m{token}\033[0m"
            else:
                # Red for high uncertainty
                colored_token = f"\033[91m{token}\033[0m"
            
            colored_tokens.append(colored_token)
        
        # Join tokens with spaces
        colored_text = " ".join(colored_tokens)
        
        # Add legend
        legend = (
            "\nLegend:\n"
            "\033[92mGreen: Low uncertainty\033[0m\n"
            "\033[93mYellow: Medium uncertainty\033[0m\n"
            "\033[91mRed: High uncertainty\033[0m\n"
        )
        
        return colored_text + legend
    
    def html_visualization(
        self, 
        uncertainty_result: Dict[str, Any],
        threshold_low: float = 0.3,
        threshold_high: float = 0.7,
    ) -> str:
        """
        Create an HTML visualization of token-level uncertainties.
        
        Args:
            uncertainty_result: Result from uncertainty_scorer.get_token_uncertainty
            threshold_low: Threshold for low uncertainty
            threshold_high: Threshold for high uncertainty
            
        Returns:
            HTML string with color-coded tokens based on uncertainty
        """
        # Get a representative sample
        if not uncertainty_result["samples"]:
            return "<p>No samples available for visualization</p>"
        
        sample = uncertainty_result["samples"][0]
        text = sample["text"]
        
        token_uncertainties = uncertainty_result["token_uncertainties"]["values"]
        
        # If we don't have token-level uncertainties, return the raw text
        if not token_uncertainties:
            return f"<p>{text}</p>"
        
        # Tokenize the text (simplistic approach - in practice, should use the model's tokenizer)
        tokens = text.split()
        
        # Truncate or pad token uncertainties to match token count
        n_tokens = len(tokens)
        n_uncertainties = len(token_uncertainties)
        
        if n_uncertainties < n_tokens:
            # Pad with zeros
            token_uncertainties = token_uncertainties + [0.0] * (n_tokens - n_uncertainties)
        elif n_uncertainties > n_tokens:
            # Truncate
            token_uncertainties = token_uncertainties[:n_tokens]
        
        # Create colored visualization
        # Green: low uncertainty, Yellow: medium, Red: high
        html_tokens = []
        for token, uncertainty in zip(tokens, token_uncertainties):
            if uncertainty < threshold_low:
                # Green for low uncertainty
                html_token = f'<span style="color: green;">{token}</span>'
            elif uncertainty < threshold_high:
                # Yellow for medium uncertainty
                html_token = f'<span style="color: orange;">{token}</span>'
            else:
                # Red for high uncertainty
                html_token = f'<span style="color: red;">{token}</span>'
            
            html_tokens.append(html_token)
        
        # Join tokens with spaces
        html_text = " ".join(html_tokens)
        
        # Add legend and wrap in div
        legend = (
            '<div class="uncertainty-legend">'
            '<p><span style="color: green;">■</span> Low uncertainty</p>'
            '<p><span style="color: orange;">■</span> Medium uncertainty</p>'
            '<p><span style="color: red;">■</span> High uncertainty</p>'
            '</div>'
        )
        
        html = (
            '<div class="uncertainty-visualization">'
            f'<div class="uncertainty-text">{html_text}</div>'
            f'{legend}'
            '</div>'
        )
        
        return html


class UncertaintyAwareSCEC:
    """
    Combines all SCEC components into a unified pipeline.
    
    This class integrates:
    1. Self-consistency sampling
    2. Evidence retrieval and alignment
    3. Uncertainty calculation
    4. Visualization
    
    It provides a simplified interface for using SCEC.
    """
    
    def __init__(
        self,
        sampler: SelfConsistencySampler,
        aligner: EvidenceAligner,
        alpha: float = 0.5,
    ):
        """
        Initialize the SCEC pipeline.
        
        Args:
            sampler: Self-consistency sampler instance
            aligner: Evidence aligner instance
            alpha: Weight parameter balancing variance and evidence alignment (0-1)
        """
        self.sampler = sampler
        self.aligner = aligner
        self.scorer = UncertaintyScorer(sampler, aligner, alpha)
        self.visualizer = TokenLevelUncertaintyVisualizer(self.scorer)
    
    def analyze(
        self,
        prompt: str,
        max_tokens: int = 500,
        visualize: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full SCEC pipeline on a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            visualize: Whether to include visualization
            
        Returns:
            Dictionary with analysis results and uncertainty metrics
        """
        # Calculate uncertainty
        uncertainty_result = self.scorer.get_uncertainty_score(
            prompt=prompt,
            max_tokens=max_tokens,
        )
        
        # Add formatted report
        uncertainty_result["report"] = self.scorer.format_uncertainty_report(uncertainty_result)
        
        # Add visualization if requested
        if visualize:
            uncertainty_result["visualization"] = self.visualizer.visualize_token_uncertainties(
                uncertainty_result
            )
            uncertainty_result["html_visualization"] = self.visualizer.html_visualization(
                uncertainty_result
            )
        
        return uncertainty_result
    
    def save_results(self, results: Dict[str, Any], output_dir: str, filename: str = "scec_results.json"):
        """
        Save SCEC analysis results to a file.
        
        Args:
            results: Analysis results from analyze()
            output_dir: Directory to save results
            filename: Filename for results
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # Create a serializable version of results
        serializable_results = {
            "prompt": results["prompt"],
            "alpha": results["alpha"],
            "sequence_uncertainty": results["sequence_uncertainty"],
            "variance_component": results["variance_component"],
            "evidence_component": results["evidence_component"],
            "samples": results["samples"],
            "report": results["report"],
        }
        
        # Add token uncertainties if available
        if "token_uncertainties" in results:
            serializable_results["token_uncertainties"] = results["token_uncertainties"]
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved SCEC results to {output_path}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # This would be a minimal example of the uncertainty scorer
    # You would need instances of SelfConsistencySampler and EvidenceAligner
    
    # Example usage (commented out since it requires actual instances)
    """
    from models.self_consistency import SelfConsistencySampler
    from models.evidence_retrieval import EvidenceAligner, BM25Retriever, ClaimExtractor, EntailmentScorer
    
    # Initialize components
    sampler = SelfConsistencySampler("claude-3-sonnet", num_samples=5)
    
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
    
    # Analyze a prompt
    prompt = "What is the capital of France?"
    uncertainty_result = scorer.get_token_uncertainty(prompt)
    
    # Print report
    print(scorer.format_uncertainty_report(uncertainty_result))
    
    # Visualize token uncertainties
    visualizer = TokenLevelUncertaintyVisualizer(scorer)
    colored_text = visualizer.visualize_token_uncertainties(uncertainty_result)
    print(colored_text)
    """
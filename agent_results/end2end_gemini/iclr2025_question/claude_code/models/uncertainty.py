"""
Implementation of various uncertainty estimation methods for LLMs.
"""

import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional
from .base_model import BaseModel, APIBasedModel

logger = logging.getLogger(__name__)

class UncertaintyEstimationModule:
    """Base class for uncertainty estimation modules."""
    
    def __init__(self, model: Union[BaseModel, APIBasedModel]):
        """
        Initialize the uncertainty estimation module.
        
        Args:
            model: The base model to estimate uncertainty for.
        """
        self.model = model
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def estimate_uncertainty(
        self,
        input_text: str,
        **kwargs
    ) -> float:
        """
        Estimate the uncertainty for the next token generation.
        
        Args:
            input_text: The input text context.
            **kwargs: Additional parameters for uncertainty estimation.
        
        Returns:
            Uncertainty score in the range [0, 1].
        """
        raise NotImplementedError("Subclasses must implement this method")


class EntropyBasedUncertainty(UncertaintyEstimationModule):
    """Uncertainty estimation based on the entropy of the next token distribution."""
    
    def estimate_uncertainty(
        self,
        input_text: str,
        top_k: int = None,
        normalized: bool = True,
        **kwargs
    ) -> float:
        """
        Estimate uncertainty using entropy of the predictive distribution.

        Args:
            input_text: The input text context.
            top_k: Only consider top-k tokens for entropy calculation (if None, use all).
            normalized: Whether to normalize the entropy to [0, 1].
            **kwargs: Additional parameters.

        Returns:
            Entropy-based uncertainty score.
        """
        # For API-based models, use a simplified approach
        if isinstance(self.model, APIBasedModel):
            # If the model supports token probabilities
            if hasattr(self.model, 'get_token_probabilities'):
                try:
                    token_probs = self.model.get_token_probabilities(input_text)
                    if token_probs and token_probs[0]:
                        # Convert logprobs to probs
                        api_probs = {token: np.exp(logprob) for token, logprob in token_probs[0].items()}
                        total_prob = sum(api_probs.values())
                        api_probs = {k: v/total_prob for k, v in api_probs.items()}

                        # Calculate entropy
                        entropy = -sum(p * np.log2(p + 1e-10) for p in api_probs.values())

                        # Normalize to [0, 1] if requested
                        if normalized:
                            max_entropy = np.log2(len(api_probs))
                            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5
                            return float(normalized_entropy)
                        return float(entropy)
                except Exception as e:
                    logger.warning(f"Error using API token probabilities: {e}")

            # For API models without token probabilities support, use a heuristic approach
            temperature = kwargs.get('temperature', 0.7)
            input_length = len(input_text.split())
            # Simple heuristic based on temperature and input length
            heuristic_uncertainty = 0.3 + (temperature * 0.4) + (min(1.0, input_length / 100) * 0.3)
            return min(1.0, max(0.0, heuristic_uncertainty))

        # For local models with tokenizers and logits
        try:
            # Get logits for the next token
            logits = self.model.get_logits(input_text)

            # Take the logits for the last token prediction
            last_token_logits = logits[0, -1, :]

            # Apply softmax to get probabilities
            probs = torch.softmax(last_token_logits, dim=0)

            # If using top-k, keep only the top-k probabilities and renormalize
            if top_k is not None:
                top_k_probs, _ = torch.topk(probs, min(top_k, probs.size(0)))
                probs = top_k_probs / top_k_probs.sum()

            # Calculate entropy: -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log2(probs + 1e-10))

            # Normalize to [0, 1] if requested
            if normalized:
                # Max entropy for uniform distribution over vocabulary size
                vocab_size = self.model.tokenizer.vocab_size if hasattr(self.model, 'tokenizer') else 50257  # Default for GPT-2
                max_entropy = np.log2(min(top_k, vocab_size) if top_k else vocab_size)
                normalized_entropy = entropy / max_entropy
                return float(normalized_entropy)

            return float(entropy)

        except Exception as e:
            logger.warning(f"Error calculating entropy-based uncertainty: {e}")
            # Fallback to a default uncertainty value
            return 0.5


class MCDropoutUncertainty(UncertaintyEstimationModule):
    """Uncertainty estimation based on Monte Carlo Dropout."""
    
    def __init__(self, model: BaseModel, num_samples: int = 5):
        """
        Initialize the MC Dropout uncertainty module.
        
        Args:
            model: The base model to estimate uncertainty for.
            num_samples: Number of MC samples to take.
        """
        super().__init__(model)
        self.num_samples = num_samples
        
        # Enable dropout during inference if the model supports it
        if hasattr(self.model.model, 'config') and hasattr(self.model.model.config, 'dropout'):
            self.original_dropout = self.model.model.config.dropout
        else:
            self.original_dropout = None
            logger.warning("Model may not support dropout for MC uncertainty estimation")
    
    def estimate_uncertainty(
        self,
        input_text: str,
        use_logits_variance: bool = False,
        **kwargs
    ) -> float:
        """
        Estimate uncertainty using Monte Carlo Dropout.

        Args:
            input_text: The input text context.
            use_logits_variance: Whether to use variance of logits instead of variance of probs.
            **kwargs: Additional parameters.

        Returns:
            MC Dropout uncertainty score.
        """
        # For API-based models, MC Dropout isn't applicable in the traditional sense
        # So fall back to entropy-based method or a simpler approach
        if isinstance(self.model, APIBasedModel):
            # Fall back to entropy-based uncertainty
            fallback = EntropyBasedUncertainty(self.model)
            return fallback.estimate_uncertainty(input_text, **kwargs)

        # If the model doesn't support dropout, fall back to entropy
        if self.original_dropout is None or self.original_dropout == 0:
            fallback = EntropyBasedUncertainty(self.model)
            return fallback.estimate_uncertainty(input_text, **kwargs)

        try:
            # Enable dropout for inference
            if hasattr(self.model.model, 'eval'):
                self.model.model.train()  # Set to training mode to enable dropout

            # Collect logits from multiple forward passes
            all_logits = []
            for _ in range(self.num_samples):
                logits = self.model.get_logits(input_text)
                last_token_logits = logits[0, -1, :]
                all_logits.append(last_token_logits)

            # Stack all logits
            stacked_logits = torch.stack(all_logits)

            if use_logits_variance:
                # Calculate variance of logits
                logits_variance = torch.var(stacked_logits, dim=0)
                # Average variance across all tokens
                uncertainty = torch.mean(logits_variance).item()
            else:
                # Convert logits to probabilities
                all_probs = torch.softmax(stacked_logits, dim=1)
                # Calculate variance of probabilities
                probs_variance = torch.var(all_probs, dim=0)
                # Average variance across all tokens
                uncertainty = torch.mean(probs_variance).item()

            # Reset model to evaluation mode
            if hasattr(self.model.model, 'eval'):
                self.model.model.eval()

            # Normalize uncertainty score to [0, 1]
            normalized_uncertainty = min(1.0, max(0.0, uncertainty * 10))  # Heuristic scaling

            return float(normalized_uncertainty)

        except Exception as e:
            logger.warning(f"Error calculating MC Dropout uncertainty: {e}")
            # Fallback to a default uncertainty value
            return 0.5


class TokenConfidenceUncertainty(UncertaintyEstimationModule):
    """Uncertainty estimation based on the confidence score of the sampled token."""
    
    def estimate_uncertainty(
        self,
        input_text: str,
        **kwargs
    ) -> float:
        """
        Estimate uncertainty using the probability of the sampled token.

        Args:
            input_text: The input text context.
            **kwargs: Additional parameters.

        Returns:
            Token confidence-based uncertainty score.
        """
        # For API-based models, use a simplified approach
        if isinstance(self.model, APIBasedModel):
            # If the model supports token probabilities
            if hasattr(self.model, 'get_token_probabilities'):
                try:
                    token_probs = self.model.get_token_probabilities(input_text)
                    if token_probs and token_probs[0]:
                        # Find max probability
                        max_prob = max(np.exp(logprob) for _, logprob in token_probs[0].items())
                        uncertainty = 1.0 - max_prob
                        return float(uncertainty)
                except Exception as e:
                    logger.warning(f"Error using API token probabilities: {e}")

            # For API models without token probabilities support, use a heuristic approach
            temperature = kwargs.get('temperature', 0.7)
            # Higher temperature usually indicates higher uncertainty
            heuristic_uncertainty = 0.3 + (temperature * 0.5)
            return min(1.0, max(0.0, heuristic_uncertainty))

        # For local models with tokenizers and logits
        try:
            # Get logits for the next token
            logits = self.model.get_logits(input_text)

            # Take the logits for the last token prediction
            last_token_logits = logits[0, -1, :]

            # Apply softmax to get probabilities
            probs = torch.softmax(last_token_logits, dim=0)

            # Get the highest probability (confidence in the most likely token)
            max_prob = torch.max(probs).item()

            # Convert confidence to uncertainty (1 - confidence)
            uncertainty = 1.0 - max_prob

            return float(uncertainty)

        except Exception as e:
            logger.warning(f"Error calculating token confidence uncertainty: {e}")
            # Fallback to a default uncertainty value
            return 0.5


class SPUQInspiredUncertainty(UncertaintyEstimationModule):
    """
    Uncertainty estimation inspired by SPUQ (Perturbation-Based UQ for LLMs).
    A simplified version where we perturb the input with small variations.
    """
    
    def __init__(self, model: Union[BaseModel, APIBasedModel], num_perturbations: int = 3):
        """
        Initialize the SPUQ-inspired uncertainty module.
        
        Args:
            model: The base model to estimate uncertainty for.
            num_perturbations: Number of input perturbations to generate.
        """
        super().__init__(model)
        self.num_perturbations = num_perturbations
    
    def generate_perturbations(self, input_text: str) -> List[str]:
        """
        Generate perturbed versions of the input text.
        
        Args:
            input_text: The original input text.
        
        Returns:
            List of perturbed input texts.
        """
        perturbations = [input_text]  # Include original
        
        # Basic perturbations: paraphrasing, minor word changes
        if " " in input_text:
            words = input_text.split()
            if len(words) > 3:
                # Swap two adjacent words
                for i in range(len(words) - 1):
                    words_copy = words.copy()
                    words_copy[i], words_copy[i+1] = words_copy[i+1], words_copy[i]
                    perturbations.append(" ".join(words_copy))
                    if len(perturbations) >= self.num_perturbations + 1:
                        break
        
        # Add a small amount of punctuation or capitalization variation
        if len(perturbations) < self.num_perturbations + 1:
            if input_text.endswith("."):
                perturbations.append(input_text[:-1] + "?")
            elif input_text.endswith("?"):
                perturbations.append(input_text[:-1] + ".")
            else:
                perturbations.append(input_text + ".")
        
        # Ensure we have the requested number of perturbations
        while len(perturbations) < self.num_perturbations + 1:
            perturbations.append(input_text)
        
        # Return only the requested number of perturbations
        return perturbations[:self.num_perturbations + 1]
    
    def estimate_uncertainty(
        self,
        input_text: str,
        **kwargs
    ) -> float:
        """
        Estimate uncertainty using perturbation-based method.

        Args:
            input_text: The input text context.
            **kwargs: Additional parameters.

        Returns:
            Perturbation-based uncertainty score.
        """
        try:
            # Generate perturbed versions of the input
            perturbations = self.generate_perturbations(input_text)

            # Generate outputs for all perturbations
            outputs = self.model.generate(perturbations, max_new_tokens=5)

            # Skip empty or failed outputs
            valid_outputs = [out for out in outputs if out and len(out) > 0]
            if len(valid_outputs) < 2:
                # If we don't have enough valid outputs, fall back to another method
                fallback = EntropyBasedUncertainty(self.model)
                return fallback.estimate_uncertainty(input_text, **kwargs)

            # Check consistency of outputs
            # Calculate similarity between outputs (simplistic approach)
            similarities = []
            for i in range(len(valid_outputs)):
                for j in range(i+1, len(valid_outputs)):
                    # Simple character-level overlap ratio
                    output_i = valid_outputs[i].lower()
                    output_j = valid_outputs[j].lower()

                    overlap = sum(1 for c in output_i if c in output_j)
                    max_len = max(len(output_i), len(output_j))
                    similarity = overlap / max_len if max_len > 0 else 1.0

                    similarities.append(similarity)

            # Calculate uncertainty as 1 - average similarity
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.5
            uncertainty = 1.0 - avg_similarity

            return float(uncertainty)

        except Exception as e:
            logger.warning(f"Error calculating SPUQ-inspired uncertainty: {e}")
            # Fall back to another method
            fallback = EntropyBasedUncertainty(self.model)
            return fallback.estimate_uncertainty(input_text, **kwargs)


class UncertaintyFactory:
    """Factory class for creating uncertainty estimation modules."""
    
    @staticmethod
    def create(
        method: str,
        model: Union[BaseModel, APIBasedModel],
        **kwargs
    ) -> UncertaintyEstimationModule:
        """
        Create an uncertainty estimation module.
        
        Args:
            method: The uncertainty estimation method to use.
            model: The base model to estimate uncertainty for.
            **kwargs: Additional parameters for the uncertainty estimator.
        
        Returns:
            An uncertainty estimation module.
        """
        if method == "entropy":
            return EntropyBasedUncertainty(model)
        elif method == "mc_dropout":
            return MCDropoutUncertainty(model, **kwargs)
        elif method == "token_confidence":
            return TokenConfidenceUncertainty(model)
        elif method == "spuq":
            return SPUQInspiredUncertainty(model, **kwargs)
        else:
            logger.warning(f"Unknown uncertainty method: {method}. Using entropy as default.")
            return EntropyBasedUncertainty(model)
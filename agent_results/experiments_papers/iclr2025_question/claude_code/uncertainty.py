"""
Uncertainty estimation methods for token-level uncertainty in language models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from typing import List, Dict, Any, Tuple, Optional, Union

class UncertaintyEstimator:
    """Base class for uncertainty estimation methods."""
    
    def __init__(self, model, tokenizer, device=None):
        """
        Initialize the uncertainty estimator.
        
        Args:
            model: The language model.
            tokenizer: The tokenizer for the language model.
            device: The device to run the model on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def estimate_uncertainty(self, 
                            input_ids: torch.Tensor, 
                            attention_mask: torch.Tensor, 
                            token_idx: int = -1) -> torch.Tensor:
        """
        Estimate the uncertainty for tokens at a specific position.
        
        Args:
            input_ids: The input token IDs.
            attention_mask: The attention mask.
            token_idx: The index of the token to estimate uncertainty for.
                      Default is -1 (the last token).
        
        Returns:
            A tensor containing the uncertainty estimates.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class EntropyEstimator(UncertaintyEstimator):
    """Uncertainty estimator based on predictive entropy."""
    
    def estimate_uncertainty(self, 
                            input_ids: torch.Tensor, 
                            attention_mask: torch.Tensor, 
                            token_idx: int = -1) -> torch.Tensor:
        """
        Estimate uncertainty using the entropy of the predicted probability distribution.
        
        Args:
            input_ids: The input token IDs.
            attention_mask: The attention mask.
            token_idx: The index of the token to estimate uncertainty for.
                      Default is -1 (the last token).
        
        Returns:
            A tensor containing the entropy values.
        """
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                return_dict=True)
            
            # Get logits for the specified token index
            logits = outputs.logits[:, token_idx, :]
            
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Calculate entropy
            log_probs = F.log_softmax(logits, dim=-1)
            entropy_values = -torch.sum(probs * log_probs, dim=-1)
            
        return entropy_values


class MCDropoutEstimator(UncertaintyEstimator):
    """Uncertainty estimator based on Monte Carlo dropout."""
    
    def __init__(self, model, tokenizer, device=None, num_samples=5):
        """
        Initialize the MC dropout estimator.
        
        Args:
            model: The language model.
            tokenizer: The tokenizer for the language model.
            device: The device to run the model on.
            num_samples: The number of dropout samples to use.
        """
        super().__init__(model, tokenizer, device)
        self.num_samples = num_samples
        
    def estimate_uncertainty(self, 
                            input_ids: torch.Tensor, 
                            attention_mask: torch.Tensor, 
                            token_idx: int = -1) -> torch.Tensor:
        """
        Estimate uncertainty using Monte Carlo dropout.
        
        Args:
            input_ids: The input token IDs.
            attention_mask: The attention mask.
            token_idx: The index of the token to estimate uncertainty for.
                      Default is -1 (the last token).
        
        Returns:
            A tensor containing the variance values.
        """
        # Enable dropout during inference
        self.model.train()
        
        # Run multiple forward passes with dropout
        all_probs = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                outputs = self.model(input_ids=input_ids, 
                                   attention_mask=attention_mask, 
                                   return_dict=True)
                
                # Get logits for the specified token index
                logits = outputs.logits[:, token_idx, :]
                
                # Convert logits to probabilities
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)
                
        # Stack probabilities from different samples
        all_probs = torch.stack(all_probs, dim=0)  # [num_samples, batch_size, vocab_size]
        
        # Calculate mean probabilities
        mean_probs = torch.mean(all_probs, dim=0)  # [batch_size, vocab_size]
        
        # Calculate variance (uncertainty)
        variance = torch.mean(torch.sum((all_probs - mean_probs.unsqueeze(0))**2, dim=-1), dim=0)
        
        # Set the model back to evaluation mode
        self.model.eval()
        
        return variance


class EnsembleEstimator(UncertaintyEstimator):
    """Uncertainty estimator based on ensemble disagreement."""
    
    def __init__(self, models, tokenizer, device=None):
        """
        Initialize the ensemble estimator.
        
        Args:
            models: A list of language models for the ensemble.
            tokenizer: The tokenizer for the language models.
            device: The device to run the models on.
        """
        super().__init__(None, tokenizer, device)
        self.models = models
        
    def estimate_uncertainty(self, 
                            input_ids: torch.Tensor, 
                            attention_mask: torch.Tensor, 
                            token_idx: int = -1) -> torch.Tensor:
        """
        Estimate uncertainty using ensemble disagreement.
        
        Args:
            input_ids: The input token IDs.
            attention_mask: The attention mask.
            token_idx: The index of the token to estimate uncertainty for.
                      Default is -1 (the last token).
        
        Returns:
            A tensor containing the disagreement values.
        """
        all_probs = []
        
        # Get predictions from all models in the ensemble
        with torch.no_grad():
            for model in self.models:
                model.to(self.device)
                outputs = model(input_ids=input_ids, 
                              attention_mask=attention_mask, 
                              return_dict=True)
                
                # Get logits for the specified token index
                logits = outputs.logits[:, token_idx, :]
                
                # Convert logits to probabilities
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)
                
        # Stack probabilities from different models
        all_probs = torch.stack(all_probs, dim=0)  # [num_models, batch_size, vocab_size]
        
        # Calculate mean probabilities
        mean_probs = torch.mean(all_probs, dim=0)  # [batch_size, vocab_size]
        
        # Calculate KL divergence between each model's predictions and the mean
        kl_divs = []
        for i in range(len(self.models)):
            kl_div = F.kl_div(all_probs[i].log(), mean_probs, reduction='none').sum(dim=-1)
            kl_divs.append(kl_div)
            
        # Average KL divergence across all models
        disagreement = torch.mean(torch.stack(kl_divs, dim=0), dim=0)
        
        return disagreement


def get_uncertainty_estimator(model, tokenizer, method="entropy", **kwargs):
    """
    Factory function to create an uncertainty estimator.
    
    Args:
        model: The language model or list of models.
        tokenizer: The tokenizer for the language model(s).
        method: The uncertainty estimation method to use.
            Options: "entropy", "mc_dropout", "ensemble".
        **kwargs: Additional arguments for the uncertainty estimator.
    
    Returns:
        An uncertainty estimator.
    """
    # Extract specific kwargs for each estimator type
    device = kwargs.get('device', None)
    
    if method == "entropy":
        return EntropyEstimator(model, tokenizer, device=device)
    elif method == "mc_dropout":
        num_samples = kwargs.get('dropout_samples', 5)
        return MCDropoutEstimator(model, tokenizer, device=device, num_samples=num_samples)
    elif method == "ensemble":
        assert isinstance(model, list), "Ensemble estimator requires a list of models"
        return EnsembleEstimator(model, tokenizer, device=device)
    else:
        raise ValueError(f"Unknown uncertainty estimation method: {method}")
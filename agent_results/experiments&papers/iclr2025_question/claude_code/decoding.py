"""
Decoding strategies for language models, including uncertainty-aware decoding.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from tqdm import tqdm

class BaseDecoder:
    """Base class for decoders."""
    
    def __init__(self, model, tokenizer, device=None, **kwargs):
        """
        Initialize the decoder.
        
        Args:
            model: The language model.
            tokenizer: The tokenizer for the language model.
            device: The device to run the model on.
            **kwargs: Additional parameters for decoding.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = kwargs
        
    def decode(self, input_ids, attention_mask=None, **kwargs):
        """
        Decode sequences.
        
        Args:
            input_ids: The input token IDs.
            attention_mask: The attention mask.
            **kwargs: Additional parameters for decoding.
        
        Returns:
            The decoded sequences and additional information.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class GreedyDecoder(BaseDecoder):
    """Greedy decoding strategy."""
    
    def decode(self, input_ids, attention_mask=None, max_length=100, **kwargs):
        """
        Decode sequences using greedy decoding.
        
        Args:
            input_ids: The input token IDs.
            attention_mask: The attention mask.
            max_length: The maximum length of the decoded sequences.
            **kwargs: Additional parameters for decoding.
        
        Returns:
            The decoded sequences and additional information.
        """
        # Move input tensors to the correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        # Set up for decoding
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]
        
        # To store the decoded sequences and log probabilities
        decoded_ids = input_ids.clone()
        log_probs = torch.zeros(batch_size, device=self.device)
        
        # Only decode if we have fewer tokens than max_length
        if current_length < max_length:
            # Loop until we reach max_length
            for _ in tqdm(range(current_length, max_length), desc="Greedy decoding", leave=False):
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(input_ids=decoded_ids, attention_mask=attention_mask, return_dict=True)
                    
                # Get logits for the last token
                next_token_logits = outputs.logits[:, -1, :]
                
                # Get the most likely token
                next_token_ids = torch.argmax(next_token_logits, dim=-1)
                log_probs += F.log_softmax(next_token_logits, dim=-1).gather(1, next_token_ids.unsqueeze(-1)).squeeze(-1)
                
                # Add the token to the sequence
                decoded_ids = torch.cat([decoded_ids, next_token_ids.unsqueeze(-1)], dim=-1)
                
                # Update attention mask if needed
                if attention_mask is not None:
                    attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=self.device)], dim=1)
                    
                # Check if all sequences have reached the end token
                if (next_token_ids == self.tokenizer.eos_token_id).all():
                    break
        
        # Decode the token IDs to strings
        decoded_texts = self.tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)
        
        return {
            "decoded_ids": decoded_ids,
            "decoded_texts": decoded_texts,
            "log_probs": log_probs,
        }


class BeamSearchDecoder(BaseDecoder):
    """Beam search decoding strategy."""
    
    def decode(self, input_ids, attention_mask=None, max_length=100, num_beams=5, **kwargs):
        """
        Decode sequences using beam search.
        
        Args:
            input_ids: The input token IDs.
            attention_mask: The attention mask.
            max_length: The maximum length of the decoded sequences.
            num_beams: The number of beams to use.
            **kwargs: Additional parameters for decoding.
        
        Returns:
            The decoded sequences and additional information.
        """
        # Use the model's generate method for beam search
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs
        )
        
        # Get the decoded sequences
        decoded_ids = outputs.sequences
        
        # Compute sequence scores
        if hasattr(outputs, "sequences_scores"):
            log_probs = outputs.sequences_scores
        else:
            # Approximate sequence scores as the mean of token scores
            log_probs = torch.mean(torch.stack(outputs.scores), dim=0).to(self.device)
        
        # Decode the token IDs to strings
        decoded_texts = self.tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)
        
        return {
            "decoded_ids": decoded_ids,
            "decoded_texts": decoded_texts,
            "log_probs": log_probs,
        }


class UncertaintyAwareDecoder(BaseDecoder):
    """Uncertainty-aware decoding strategy."""
    
    def __init__(self, model, tokenizer, uncertainty_estimator, device=None, **kwargs):
        """
        Initialize the uncertainty-aware decoder.
        
        Args:
            model: The language model.
            tokenizer: The tokenizer for the language model.
            uncertainty_estimator: The uncertainty estimator to use.
            device: The device to run the model on.
            **kwargs: Additional parameters for decoding.
        """
        super().__init__(model, tokenizer, device, **kwargs)
        self.uncertainty_estimator = uncertainty_estimator
        self.threshold = kwargs.get("threshold_init", 0.5)
        self.threshold_alpha = kwargs.get("threshold_alpha", 0.1)
        self.intervention_strategy = kwargs.get("intervention_strategy", "rerank")
        
    def _adjust_threshold(self, current_threshold, reward):
        """
        Dynamically adjust the uncertainty threshold.
        
        Args:
            current_threshold: The current threshold value.
            reward: The reward signal for adjusting the threshold.
        
        Returns:
            The updated threshold value.
        """
        new_threshold = current_threshold + self.threshold_alpha * reward
        
        # Constrain the threshold to be between 0 and 1
        new_threshold = max(0.0, min(1.0, new_threshold))
        
        return new_threshold
    
    def _intervene_rerank(self, logits, uncertainty, top_k=50):
        """
        Re-rank candidate tokens based on uncertainty.
        
        Args:
            logits: The logits for the next token.
            uncertainty: The token-level uncertainty estimates.
            top_k: The number of top candidates to consider.
        
        Returns:
            The re-ranked logits.
        """
        batch_size, vocab_size = logits.shape
        
        # Get top-k logits and indices
        top_k_logits, top_k_indices = torch.topk(logits, k=min(top_k, vocab_size), dim=-1)
        
        # Create a mask for tokens with high uncertainty
        high_uncertainty = uncertainty > self.threshold
        
        # Re-ranked logits initialized as the original logits
        reranked_logits = logits.clone()
        
        # For tokens with high uncertainty, re-rank based on a combination of logits and uncertainty
        if high_uncertainty.any():
            # Normalize uncertainty to [0, 1]
            normalized_uncertainty = uncertainty / (uncertainty.max() + 1e-10)
            
            # Calculate uncertainty penalty
            uncertainty_penalty = normalized_uncertainty.unsqueeze(1).expand_as(logits)
            
            # Apply the uncertainty penalty to the logits
            reranked_logits = logits - 5.0 * uncertainty_penalty
        
        return reranked_logits
    
    def _intervene_constrain(self, logits, uncertainty, evidence_tokens=None):
        """
        Constrain the sampling distribution based on evidence tokens.
        
        Args:
            logits: The logits for the next token.
            uncertainty: The token-level uncertainty estimates.
            evidence_tokens: The token IDs that are consistent with factual evidence.
        
        Returns:
            The constrained logits.
        """
        batch_size, vocab_size = logits.shape
        
        # Create a mask for tokens with high uncertainty
        high_uncertainty = uncertainty > self.threshold
        
        # Constrained logits initialized as the original logits
        constrained_logits = logits.clone()
        
        # For tokens with high uncertainty, constrain the sampling distribution
        if high_uncertainty.any() and evidence_tokens is not None:
            # Set a large negative value for tokens not in the evidence set
            mask = torch.zeros_like(constrained_logits, dtype=torch.bool)
            for i in range(batch_size):
                if high_uncertainty[i] and i < len(evidence_tokens):
                    mask[i, evidence_tokens[i]] = True
            
            # Apply the mask to the logits
            constrained_logits[~mask] = -1e10
        
        return constrained_logits
    
    def _intervene_special_token(self, logits, uncertainty):
        """
        Inject a special token indicating potential unreliability.
        
        Args:
            logits: The logits for the next token.
            uncertainty: The token-level uncertainty estimates.
        
        Returns:
            The modified logits.
        """
        batch_size, vocab_size = logits.shape
        
        # Create a mask for tokens with high uncertainty
        high_uncertainty = uncertainty > self.threshold
        
        # Modified logits initialized as the original logits
        modified_logits = logits.clone()
        
        # For tokens with high uncertainty, inject a special token
        if high_uncertainty.any():
            # Get the special token ID
            special_token_id = self.tokenizer.convert_tokens_to_ids("[UNCERTAIN]")
            
            # If the special token is not in the vocabulary, use the unknown token
            if special_token_id == self.tokenizer.unk_token_id:
                special_token_id = self.tokenizer.unk_token_id
                
            # Set a high probability for the special token
            for i in range(batch_size):
                if high_uncertainty[i]:
                    modified_logits[i] = -1e10  # Set all logits to a very low value
                    modified_logits[i, special_token_id] = 0  # Set the special token logit to 0
        
        return modified_logits
    
    def decode(self, input_ids, attention_mask=None, max_length=100, temperature=1.0, top_k=0, top_p=1.0, **kwargs):
        """
        Decode sequences using uncertainty-aware decoding.
        
        Args:
            input_ids: The input token IDs.
            attention_mask: The attention mask.
            max_length: The maximum length of the decoded sequences.
            temperature: The temperature for sampling.
            top_k: The number of top-k tokens to consider for sampling.
            top_p: The cumulative probability for nucleus sampling.
            **kwargs: Additional parameters for decoding.
        
        Returns:
            The decoded sequences and additional information.
        """
        # Move input tensors to the correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        # Set up for decoding
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]
        
        # To store the decoded sequences, log probabilities, and uncertainty
        decoded_ids = input_ids.clone()
        log_probs = torch.zeros(batch_size, device=self.device)
        uncertainties = []
        
        # Make sure we have at least one value in uncertainties
        # Add initial uncertainty measurement
        uncertainty = self.uncertainty_estimator.estimate_uncertainty(decoded_ids, attention_mask)
        uncertainties.append(uncertainty.cpu().numpy())
        
        # Only decode if we have fewer tokens than max_length
        if current_length < max_length:
            # Loop until we reach max_length
            for _ in tqdm(range(current_length, max_length), desc="Uncertainty-aware decoding", leave=False):
                # Estimate uncertainty for the current state
                uncertainty = self.uncertainty_estimator.estimate_uncertainty(decoded_ids, attention_mask)
                uncertainties.append(uncertainty.cpu().numpy())
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(input_ids=decoded_ids, attention_mask=attention_mask, return_dict=True)
                    
                # Get logits for the last token
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply uncertainty-aware intervention
                if self.intervention_strategy == "rerank":
                    next_token_logits = self._intervene_rerank(next_token_logits, uncertainty, top_k=top_k)
                elif self.intervention_strategy == "constrain":
                    # For simplicity, we're not implementing evidence retrieval here
                    # In a real implementation, you would retrieve evidence tokens based on the current context
                    evidence_tokens = None
                    next_token_logits = self._intervene_constrain(next_token_logits, uncertainty, evidence_tokens)
                elif self.intervention_strategy == "special_token":
                    next_token_logits = self._intervene_special_token(next_token_logits, uncertainty)
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    next_token_logits = self._top_k_filtering(next_token_logits, top_k)
                if top_p < 1.0:
                    next_token_logits = self._top_p_filtering(next_token_logits, top_p)
                
                # Get probabilities
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample from the probability distribution
                next_token_ids = torch.multinomial(next_token_probs, num_samples=1).squeeze(-1)
                log_probs += F.log_softmax(next_token_logits, dim=-1).gather(1, next_token_ids.unsqueeze(-1)).squeeze(-1)
                
                # Add the token to the sequence
                decoded_ids = torch.cat([decoded_ids, next_token_ids.unsqueeze(-1)], dim=-1)
                
                # Update attention mask if needed
                if attention_mask is not None:
                    attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=self.device)], dim=1)
                    
                # Update the threshold based on the uncertainty
                reward = -torch.mean(uncertainty).item()  # Use negative uncertainty as a reward
                self.threshold = self._adjust_threshold(self.threshold, reward)
                
                # Check if all sequences have reached the end token
                if (next_token_ids == self.tokenizer.eos_token_id).all():
                    break
        
        # Decode the token IDs to strings
        decoded_texts = self.tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)
        
        return {
            "decoded_ids": decoded_ids,
            "decoded_texts": decoded_texts,
            "log_probs": log_probs,
            "uncertainties": uncertainties,
            "final_threshold": self.threshold,
        }
    
    def _top_k_filtering(self, logits, top_k):
        """
        Filter logits using top-k filtering.
        
        Args:
            logits: The logits to filter.
            top_k: The number of top tokens to keep.
        
        Returns:
            The filtered logits.
        """
        top_k = min(top_k, logits.size(-1))
        
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = -float('Inf')
        
        return filtered_logits
    
    def _top_p_filtering(self, logits, top_p):
        """
        Filter logits using nucleus (top-p) filtering.
        
        Args:
            logits: The logits to filter.
            top_p: The cumulative probability threshold.
        
        Returns:
            The filtered logits.
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = -float('Inf')
        
        return filtered_logits


def get_decoder(model, tokenizer, config, uncertainty_estimator=None):
    """
    Factory function to create a decoder.
    
    Args:
        model: The language model or list of models.
        tokenizer: The tokenizer for the language model(s).
        config: The configuration for the decoder.
        uncertainty_estimator: The uncertainty estimator to use for uncertainty-aware decoding.
    
    Returns:
        A decoder.
    """
    decoding_method = config.get("decoding_method", "greedy")
    
    if decoding_method == "greedy":
        return GreedyDecoder(model, tokenizer, **config)
    elif decoding_method == "beam_search":
        return BeamSearchDecoder(model, tokenizer, **config)
    elif decoding_method == "uad":
        assert uncertainty_estimator is not None, "Uncertainty-aware decoding requires an uncertainty estimator"
        return UncertaintyAwareDecoder(model, tokenizer, uncertainty_estimator, **config)
    else:
        raise ValueError(f"Unknown decoding method: {decoding_method}")
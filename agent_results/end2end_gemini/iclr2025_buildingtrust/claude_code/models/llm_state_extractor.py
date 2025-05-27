"""
LLM State Extractor Module

This module is responsible for extracting internal states from LLMs
during text generation, including hidden states and attention weights.
"""

import os
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

logger = logging.getLogger(__name__)

class LLMStateExtractor:
    """
    Class for extracting internal states from Large Language Models.
    
    This class provides functionality for:
    1. Loading an LLM from HuggingFace
    2. Processing prompts and generating text
    3. Extracting hidden states and attention weights during generation
    4. Recording states at specific layers and timesteps
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        layers_to_extract: Optional[List[int]] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the LLM State Extractor.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            device: Device to load the model on ("cpu", "cuda", "cuda:0", etc.)
            layers_to_extract: List of layer indices to extract states from. If None, extracts from all layers.
            cache_dir: Directory to cache model and tokenizer files
        """
        logger.info(f"Initializing LLMStateExtractor with model: {model_name} on {device}")
        
        self.model_name = model_name
        self.device = device
        self.layers_to_extract = layers_to_extract
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            output_hidden_states=True,  # Enable hidden state output
            output_attentions=True,     # Enable attention weights output
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(device)
        
        # Get model info
        self.config = self.model.config
        self.num_layers = self.config.num_hidden_layers
        self.num_attention_heads = self.config.num_attention_heads
        
        if layers_to_extract is None:
            self.layers_to_extract = list(range(self.num_layers))
        
        logger.info(f"Model loaded successfully. Num layers: {self.num_layers}, Num heads: {self.num_attention_heads}")
    
    def _prepare_input(self, prompt: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize the input prompt and prepare it for the model.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Dictionary of model inputs
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return inputs
    
    def generate_with_states(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text from the input prompt and collect internal states.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling instead of greedy decoding
            top_p: Nucleus sampling probability
            
        Returns:
            Tuple of (generated_text, extracted_states)
            where extracted_states is a dictionary containing:
                - 'hidden_states': List of tensors for each layer at each time step
                - 'attention_weights': Attention weights for each layer and head
                - 'input_ids': Input token IDs
                - 'output_ids': Output token IDs
        """
        logger.info(f"Generating text with states for prompt: {prompt[:50]}...")
        
        # Tokenize and prepare inputs
        inputs = self._prepare_input(prompt)
        input_length = inputs.input_ids.shape[1]
        
        # Store states during generation
        extracted_states = {
            'hidden_states': {},      # Format: {layer_idx: [states_t0, states_t1, ...]}
            'attention_weights': {},  # Format: {layer_idx: {head_idx: [weights_t0, weights_t1, ...]}}
            'input_ids': inputs.input_ids[0].tolist(),
            'input_tokens': self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0]),
            'output_ids': [],
            'output_tokens': []
        }
        
        # Initialize containers for states for each layer
        for layer_idx in self.layers_to_extract:
            extracted_states['hidden_states'][layer_idx] = []
            extracted_states['attention_weights'][layer_idx] = {}
            for head_idx in range(self.num_attention_heads):
                extracted_states['attention_weights'][layer_idx][head_idx] = []
        
        # Generate tokens one by one to capture internal states
        with torch.no_grad():
            # First forward pass to get states for input sequence
            outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
            
            # Store hidden states for input sequence
            for layer_idx in self.layers_to_extract:
                # Hidden states are in the format: tuple(tensor(batch, seq_len, hidden_dim))
                layer_hidden_states = outputs.hidden_states[layer_idx + 1]  # +1 to skip embedding layer
                extracted_states['hidden_states'][layer_idx].append(layer_hidden_states.detach().cpu())
                
                # Attention weights are in the format: tuple(tensor(batch, num_heads, seq_len, seq_len))
                if outputs.attentions:
                    layer_attentions = outputs.attentions[layer_idx]
                    for head_idx in range(self.num_attention_heads):
                        head_attention = layer_attentions[0, head_idx, :, :]  # batch_idx=0, head=head_idx
                        extracted_states['attention_weights'][layer_idx][head_idx].append(head_attention.detach().cpu())
            
            # Autoregressive generation with state extraction at each step
            cur_input_ids = inputs.input_ids.clone()
            
            for _ in range(max_new_tokens):
                # Forward pass with current inputs
                outputs = self.model(
                    input_ids=cur_input_ids,
                    output_hidden_states=True,
                    output_attentions=True
                )
                
                # Get logits and sample next token
                next_token_logits = outputs.logits[:, -1, :]
                
                if do_sample:
                    # Apply temperature and top-p sampling
                    next_token_logits = next_token_logits / temperature
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    
                    # Top-p (nucleus) sampling
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Set logits of filtered tokens to -inf
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
                    
                    # Sample from filtered distribution
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append next token to input_ids
                cur_input_ids = torch.cat([cur_input_ids, next_token], dim=-1)
                new_token_id = next_token[0].item()
                extracted_states['output_ids'].append(new_token_id)
                extracted_states['output_tokens'].append(self.tokenizer.convert_ids_to_tokens(new_token_id))
                
                # Extract states for the new token
                for layer_idx in self.layers_to_extract:
                    # Hidden states for the new token (last position)
                    layer_hidden_states = outputs.hidden_states[layer_idx + 1][:, -1, :]  # +1 to skip embedding layer
                    extracted_states['hidden_states'][layer_idx].append(layer_hidden_states.detach().cpu())
                    
                    # Attention weights for the new token
                    if outputs.attentions:
                        layer_attentions = outputs.attentions[layer_idx]
                        for head_idx in range(self.num_attention_heads):
                            # Attention from last position to all other positions
                            head_attention = layer_attentions[0, head_idx, -1, :]  # batch_idx=0, head=head_idx, last position
                            extracted_states['attention_weights'][layer_idx][head_idx].append(head_attention.detach().cpu())
                
                # Check for end of sequence token
                if new_token_id == self.tokenizer.eos_token_id:
                    break
        
        # Decode the complete generated text
        generated_text = self.tokenizer.decode(cur_input_ids[0, input_length:], skip_special_tokens=True)
        
        logger.info(f"Generation complete. Output length: {len(extracted_states['output_ids'])} tokens")
        
        return generated_text, extracted_states
    
    def extract_states_for_sentence(
        self,
        sentence: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract states for a specific sentence within a context.
        Useful for analyzing specific reasoning steps.
        
        Args:
            sentence: Target sentence to extract states for
            context: Optional context to prepend to the sentence
            
        Returns:
            Dictionary of extracted states for the sentence
        """
        full_text = context + " " + sentence if context else sentence
        inputs = self._prepare_input(full_text)
        
        # Find the token span of the sentence
        sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
        full_tokens = inputs.input_ids[0].tolist()
        
        # Find the start position of sentence tokens in the full tokens
        start_idx = None
        for i in range(len(full_tokens) - len(sentence_tokens) + 1):
            if full_tokens[i:i+len(sentence_tokens)] == sentence_tokens:
                start_idx = i
                break
        
        if start_idx is None:
            logger.error(f"Could not locate sentence tokens in the full text")
            return {}
        
        end_idx = start_idx + len(sentence_tokens)
        
        # Forward pass to get all states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
        
        # Extract states only for the sentence tokens
        sentence_states = {
            'hidden_states': {},
            'attention_weights': {},
            'token_ids': full_tokens[start_idx:end_idx],
            'tokens': self.tokenizer.convert_ids_to_tokens(full_tokens[start_idx:end_idx])
        }
        
        for layer_idx in self.layers_to_extract:
            # Hidden states are in format (batch, seq_len, hidden_dim)
            layer_hidden_states = outputs.hidden_states[layer_idx + 1][0, start_idx:end_idx, :]  # +1 to skip embedding
            sentence_states['hidden_states'][layer_idx] = layer_hidden_states.detach().cpu()
            
            sentence_states['attention_weights'][layer_idx] = {}
            if outputs.attentions:
                layer_attentions = outputs.attentions[layer_idx]
                for head_idx in range(self.num_attention_heads):
                    # Extract attention patterns for sentence tokens (both source and destination)
                    head_attention = layer_attentions[0, head_idx, start_idx:end_idx, :]  # Attention from sentence to all
                    sentence_states['attention_weights'][layer_idx][head_idx] = head_attention.detach().cpu()
        
        return sentence_states
    
    def compute_token_importance(
        self,
        prompt: str,
        target_text: Optional[str] = None,
        method: str = "integrated_gradients",
        steps: int = 10
    ) -> Dict[str, Any]:
        """
        Compute importance scores for input tokens using attribution methods.
        
        Args:
            prompt: Input text prompt
            target_text: Optional target text to compute attribution for (for prompted generation)
            method: Attribution method ("integrated_gradients", "attention", "simple_gradient")
            steps: Number of steps for integrated gradients
            
        Returns:
            Dictionary containing token importance scores
        """
        logger.info(f"Computing token importance using {method}")
        
        # Implementation depends on the specific attribution method
        # For this prototype, we'll implement a simplified version based on attention
        
        if method == "attention":
            # Use attention weights as a proxy for importance
            inputs = self._prepare_input(prompt)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
            
            # Average attention weights across heads and layers
            if outputs.attentions:
                # Shape: [layers, batch, heads, seq_len, seq_len]
                all_attentions = torch.stack(outputs.attentions)
                # Average across heads and layers: [seq_len, seq_len]
                avg_attention = all_attentions.mean(dim=(0, 1, 2))
                
                # For each token, compute the total attention it receives
                # This is a simple proxy for importance
                token_importance = avg_attention.sum(dim=0)
                
                return {
                    'method': method,
                    'token_ids': inputs.input_ids[0].tolist(),
                    'tokens': self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0]),
                    'importance_scores': token_importance.detach().cpu().tolist()
                }
        
        # Placeholder for other methods (to be implemented)
        logger.warning(f"Method {method} not fully implemented. Returning attention-based importance.")
        return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_layers': self.num_layers,
            'num_attention_heads': self.num_attention_heads,
            'hidden_size': self.config.hidden_size,
            'vocab_size': self.config.vocab_size
        }
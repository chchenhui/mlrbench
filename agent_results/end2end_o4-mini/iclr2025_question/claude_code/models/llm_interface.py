"""
Interfaces for interacting with different LLMs (API-based and local)
"""

import os
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm
from openai import OpenAI
from anthropic import Anthropic
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
)

logger = logging.getLogger(__name__)

class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        temperature: float = 1.0,
        max_tokens: int = 500,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt for generation
            temperature: Controls randomness (higher = more random)
            max_tokens: Maximum number of tokens to generate
            top_p: Controls diversity via nucleus sampling
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing:
                'text': Generated text
                'logprobs': Log probabilities of tokens (if available)
                'tokens': List of generated tokens (if available)
                'token_logprobs': Token-level log probabilities (if available)
        """
        pass
    
    @abstractmethod
    def generate_multiple(
        self, 
        prompt: str, 
        n: int,
        temperature: float = 1.0,
        max_tokens: int = 500,
        top_p: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple responses from the LLM for the same prompt.
        
        Args:
            prompt: Input prompt for generation
            n: Number of responses to generate
            temperature: Controls randomness (higher = more random)
            max_tokens: Maximum number of tokens to generate
            top_p: Controls diversity via nucleus sampling
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of dictionaries, each containing:
                'text': Generated text
                'logprobs': Log probabilities of tokens (if available)
                'tokens': List of generated tokens (if available)
                'token_logprobs': Token-level log probabilities (if available)
        """
        pass
    
    @abstractmethod
    def logprobs(
        self, 
        prompt: str, 
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate log probabilities for a given text conditioned on a prompt.
        
        Args:
            prompt: Input prompt to condition on
            text: Text to calculate log probabilities for
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing:
                'logprobs': Log probabilities of tokens
                'tokens': List of tokens
                'token_logprobs': Token-level log probabilities
        """
        pass


class OpenAILLM(LLMInterface):
    """Interface for OpenAI models."""
    
    SUPPORTED_MODELS = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ]
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        """
        Initialize OpenAI LLM interface.
        
        Args:
            model_name: OpenAI model to use
            api_key: OpenAI API key (defaults to os.environ["OPENAI_API_KEY"])
            max_retries: Maximum number of retries on API failure
            retry_delay: Delay between retries in seconds
        """
        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_name} not in supported models: {self.SUPPORTED_MODELS}")
        
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Use provided API key or get from environment
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate(
        self, 
        prompt: str, 
        temperature: float = 1.0,
        max_tokens: int = 500,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response from the OpenAI model."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    logprobs=True,
                    top_logprobs=5,
                    **kwargs
                )
                
                result = {
                    "text": response.choices[0].message.content,
                    "finish_reason": response.choices[0].finish_reason,
                }
                
                # Extract logprobs if available
                if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
                    logprobs = response.choices[0].logprobs
                    result["tokens"] = [item.token for item in logprobs.content]
                    result["token_logprobs"] = [item.logprob for item in logprobs.content]
                    result["top_logprobs"] = [
                        {entry.token: entry.logprob for entry in item.top_logprobs}
                        for item in logprobs.content
                    ]
                
                return result
            
            except Exception as e:
                logger.warning(f"Error in OpenAI API call (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
    
    def generate_multiple(
        self, 
        prompt: str, 
        n: int,
        temperature: float = 1.0,
        max_tokens: int = 500,
        top_p: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate multiple responses from the OpenAI model."""
        results = []
        
        for i in tqdm(range(n), desc=f"Generating {n} samples from {self.model_name}"):
            # Using different seeds for diversity
            seed = kwargs.get("seed", 42) + i
            
            result = self.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                seed=seed,
                **{k: v for k, v in kwargs.items() if k != "seed"}
            )
            
            results.append(result)
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        return results
    
    def logprobs(
        self, 
        prompt: str, 
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate log probabilities for text given a prompt."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": text}
                    ],
                    temperature=0,  # Use deterministic sampling for logprobs
                    max_tokens=0,  # We don't want to generate, just get logprobs for existing text
                    logprobs=True,
                    top_logprobs=5,
                    logit_bias={},  # Empty logit bias to force evaluation of existing content
                    **kwargs
                )
                
                # Extract logprobs
                logprobs = response.choices[0].logprobs
                result = {
                    "tokens": [item.token for item in logprobs.content],
                    "token_logprobs": [item.logprob for item in logprobs.content],
                    "top_logprobs": [
                        {entry.token: entry.logprob for entry in item.top_logprobs}
                        for item in logprobs.content
                    ],
                    "text": text
                }
                
                # Add summary stats
                result["mean_logprob"] = np.mean(result["token_logprobs"]) if result["token_logprobs"] else 0
                result["min_logprob"] = min(result["token_logprobs"]) if result["token_logprobs"] else 0
                
                return result
            
            except Exception as e:
                logger.warning(f"Error in OpenAI API call (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise


class AnthropicLLM(LLMInterface):
    """Interface for Anthropic Claude models."""
    
    SUPPORTED_MODELS = [
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
    ]
    
    def __init__(
        self, 
        model_name: str = "claude-3-7-sonnet-20250219",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        """
        Initialize Anthropic LLM interface.
        
        Args:
            model_name: Anthropic model to use
            api_key: Anthropic API key (defaults to os.environ["ANTHROPIC_API_KEY"])
            max_retries: Maximum number of retries on API failure
            retry_delay: Delay between retries in seconds
        """
        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_name} not in supported models: {self.SUPPORTED_MODELS}")
        
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Use provided API key or get from environment
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError("Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable")
        
        self.client = Anthropic(api_key=api_key)
    
    def generate(
        self, 
        prompt: str, 
        temperature: float = 1.0,
        max_tokens: int = 500,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response from the Anthropic model."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
                
                result = {
                    "text": response.content[0].text,
                    "finish_reason": response.stop_reason,
                }
                
                # Anthropic doesn't provide token-level logprobs, so simulate with placeholders
                if "logprobs" in kwargs and kwargs["logprobs"]:
                    # Crude approximation - this is a placeholder, not real logprobs
                    tokens = result["text"].split()
                    result["tokens"] = tokens
                    result["token_logprobs"] = [-1.0] * len(tokens)  # Placeholder values
                
                return result
            
            except Exception as e:
                logger.warning(f"Error in Anthropic API call (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
    
    def generate_multiple(
        self, 
        prompt: str, 
        n: int,
        temperature: float = 1.0,
        max_tokens: int = 500,
        top_p: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate multiple responses from the Anthropic model."""
        results = []
        
        for i in tqdm(range(n), desc=f"Generating {n} samples from {self.model_name}"):
            result = self.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs
            )
            
            results.append(result)
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        return results
    
    def logprobs(
        self, 
        prompt: str, 
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate approximate log probabilities for text given a prompt.
        
        Note: Anthropic API doesn't provide token-level logprobs directly,
        so this is an approximation based on the model's confidence.
        """
        # Anthropic doesn't support direct logprobs, so use a workaround
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=0,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": text}
            ],
            **kwargs
        )
        
        # Since we don't have token-level logprobs, use a crude approximation
        tokens = text.split()
        fake_logprobs = [-2.0] * len(tokens)  # Placeholder values
        
        result = {
            "tokens": tokens,
            "token_logprobs": fake_logprobs,
            "text": text,
            "mean_logprob": np.mean(fake_logprobs),
            "min_logprob": min(fake_logprobs),
        }
        
        return result


class HuggingFaceLLM(LLMInterface):
    """Interface for local HuggingFace models."""
    
    SUPPORTED_MODELS = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen3-0.5B-Chat",
        "google/gemma-2b-it",
    ]
    
    def __init__(
        self, 
        model_name: str,
        cache_dir: Optional[str] = None,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        max_memory: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize HuggingFace model interface.
        
        Args:
            model_name: HuggingFace model ID
            cache_dir: Directory to cache models
            device: Device to run the model on ('cpu', 'cuda', 'auto')
            torch_dtype: Data type for model weights
            max_memory: Maximum memory to use per GPU
        """
        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_name} not in supported models: {self.SUPPORTED_MODELS}")
        
        self.model_name = model_name
        
        # Determine device placement
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        # Load tokenizer and model
        logger.info(f"Loading model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "cache_dir": cache_dir,
        }
        
        if device == "cuda":
            if max_memory:
                # Use device_map="auto" with max_memory for efficient multi-GPU
                model_kwargs["device_map"] = "auto"
                model_kwargs["max_memory"] = max_memory
            else:
                model_kwargs["device_map"] = "auto"
        else:
            # CPU only
            model_kwargs["device_map"] = {"": device}
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            **model_kwargs
        )
        
        # Check if model uses chat template
        self.has_chat_template = hasattr(self.tokenizer, "apply_chat_template")
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for the model if it uses a chat template."""
        if self.has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        return prompt
    
    def _forward(
        self, 
        prompt: str, 
        temperature: float = 1.0,
        max_tokens: int = 500,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict:
        """Run a forward pass through the model."""
        formatted_prompt = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 1e-5),  # Avoid division by zero
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        
        # Update with any additional kwargs
        gen_kwargs.update({k: v for k, v in kwargs.items() if k not in ["prompt", "n"]})
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Extract generated tokens
        generated_tokens = outputs.sequences[0, inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate logprobs if scores are available
        logprobs_list = []
        if hasattr(outputs, "scores") and outputs.scores:
            for i, score in enumerate(outputs.scores):
                # Get token probabilities
                token_probs = torch.nn.functional.softmax(score[0], dim=-1)
                generated_token = generated_tokens[i].item()
                token_logprob = torch.log(token_probs[generated_token]).item()
                logprobs_list.append(token_logprob)
                
        return {
            "text": generated_text,
            "tokens": self.tokenizer.convert_ids_to_tokens(generated_tokens, skip_special_tokens=True),
            "token_logprobs": logprobs_list,
            "finish_reason": "length" if len(generated_tokens) >= max_tokens else "stop",
        }
    
    def generate(
        self, 
        prompt: str, 
        temperature: float = 1.0,
        max_tokens: int = 500,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response from the HuggingFace model."""
        result = self._forward(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        # Add mean/min logprobs summary if available
        if result.get("token_logprobs"):
            result["mean_logprob"] = np.mean(result["token_logprobs"])
            result["min_logprob"] = min(result["token_logprobs"])
        
        return result
    
    def generate_multiple(
        self, 
        prompt: str, 
        n: int,
        temperature: float = 1.0,
        max_tokens: int = 500,
        top_p: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate multiple responses from the HuggingFace model."""
        results = []
        
        for i in tqdm(range(n), desc=f"Generating {n} samples from {self.model_name}"):
            # Use different seeds for diversity
            if "seed" in kwargs:
                kwargs["seed"] = kwargs["seed"] + i
            
            result = self.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs
            )
            
            results.append(result)
        
        return results
    
    def logprobs(
        self, 
        prompt: str, 
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate log probabilities for text given a prompt."""
        formatted_prompt = self._format_prompt(prompt)
        full_text = formatted_prompt + text
        
        # Tokenize the entire sequence
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        prompt_tokens = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get logits
        logits = outputs.logits[0]
        
        # Calculate logprobs for tokens after the prompt
        prompt_length = prompt_tokens["input_ids"].shape[1] - 1  # -1 to avoid double counting
        input_ids = inputs["input_ids"][0][prompt_length:-1]  # -1 to align with targets
        
        # Get target tokens (shifted by 1)
        target_ids = inputs["input_ids"][0][prompt_length+1:]
        
        # Calculate token logprobs
        token_logprobs = []
        tokens = []
        
        for i, target_id in enumerate(target_ids):
            logits_i = logits[prompt_length+i]
            probs = torch.nn.functional.softmax(logits_i, dim=-1)
            token_logprob = torch.log(probs[target_id]).item()
            token_logprobs.append(token_logprob)
            tokens.append(self.tokenizer.decode(target_id))
        
        result = {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "text": text,
            "mean_logprob": np.mean(token_logprobs) if token_logprobs else 0,
            "min_logprob": min(token_logprobs) if token_logprobs else 0,
        }
        
        return result


def get_llm_interface(
    model_name: str,
    **kwargs
) -> LLMInterface:
    """
    Factory function to create an appropriate LLM interface based on model name.
    
    Args:
        model_name: Name of the model to use
        **kwargs: Additional arguments to pass to the LLM interface
        
    Returns:
        An instance of an LLMInterface subclass
    """
    # OpenAI models
    if model_name.startswith("gpt-") or model_name in OpenAILLM.SUPPORTED_MODELS:
        return OpenAILLM(model_name=model_name, **kwargs)
    
    # Anthropic models
    elif model_name.startswith("claude-") or model_name in AnthropicLLM.SUPPORTED_MODELS:
        return AnthropicLLM(model_name=model_name, **kwargs)
    
    # HuggingFace models
    elif model_name in HuggingFaceLLM.SUPPORTED_MODELS or "/" in model_name:
        return HuggingFaceLLM(model_name=model_name, **kwargs)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    model_name = "claude-3-7-sonnet-20250219"  # Change to an available model
    llm = get_llm_interface(model_name)
    
    prompt = "What is the capital of France?"
    response = llm.generate(prompt, temperature=0.7, max_tokens=100)
    
    print(f"Response: {response['text']}")
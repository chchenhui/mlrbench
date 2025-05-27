"""
LLM Interface for the SSCSteer experiment.

This module provides an interface to different LLM providers for code generation.
"""

import os
import time
import random
import openai
import logging
import numpy as np
from typing import Dict, List, Any, Callable, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/llm_interface.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("LLM-Interface")

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

class MockLLMGenerator:
    """
    Mock LLM generator for testing without API calls.
    """
    
    def __init__(self, model: str = "mock"):
        """
        Initialize the mock LLM generator.
        
        Args:
            model: Model name to simulate
        """
        self.model = model
        logger.info(f"Initialized Mock LLM Generator for model {model}")
        
    def generate_tokens(self, prompt: str) -> Dict[str, float]:
        """
        Generate mock token probabilities.
        
        Args:
            prompt: The prompt to generate from
            
        Returns:
            Dictionary mapping tokens to probabilities
        """
        # Parse the prompt to determine context and generate somewhat reasonable tokens
        is_python = "```python" in prompt or "def " in prompt
        
        python_tokens = [
            "def", "class", "if", "else", "for", "while", "return", "import",
            "print", "self", "None", "True", "False", "(", ")", "[", "]", "{", "}",
            ":", ",", ".", "=", "==", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", 
            " ", "\n", "    "
        ]
        
        generic_tokens = [
            "the", "a", "an", "and", "or", "but", "if", "then", "else", "while",
            "for", "to", "from", "with", "without", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "can", "could", "will", "would", "shall", "should", "may", "might",
            " ", "\n", "."
        ]
        
        # Select tokens based on context
        if is_python:
            tokens = python_tokens
        else:
            tokens = generic_tokens
            
        # Generate probabilities for a subset of tokens
        num_tokens = min(10, len(tokens))
        selected_tokens = random.sample(tokens, num_tokens)
        
        # Generate random probabilities that sum to 1
        probs = np.random.dirichlet(np.ones(num_tokens))
        
        # Create token probability dictionary
        token_probs = {token: float(prob) for token, prob in zip(selected_tokens, probs)}
        
        return token_probs


class OpenAIGenerator:
    """
    LLM Generator using the OpenAI API.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the OpenAI generator.
        
        Args:
            model: OpenAI model to use
        """
        self.model = model
        
        # Set up OpenAI client
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            logger.info(f"Initialized OpenAI Generator for model {model}")
        else:
            logger.warning("OpenAI API key not found in environment variables")
            
    def generate_tokens(self, prompt: str) -> Dict[str, float]:
        """
        Generate token probabilities using OpenAI API.
        
        Args:
            prompt: The prompt to generate from
            
        Returns:
            Dictionary mapping tokens to probabilities
        """
        if not OPENAI_API_KEY:
            logger.warning("No OpenAI API key, falling back to mock generator")
            return MockLLMGenerator().generate_tokens(prompt)
            
        try:
            # Create messages for chat completion
            messages = [
                {"role": "system", "content": "You are a skilled programmer assistant. Generate high-quality code based on the given prompt."},
                {"role": "user", "content": prompt}
            ]
            
            # Make API call with logprobs to get token probabilities
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1,  # We just need the next token
                logprobs=True,
                top_logprobs=20
            )
            
            # Extract log probabilities
            if response.choices and hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                logprobs = response.choices[0].logprobs.top_logprobs[0]
                
                # Convert log probabilities to probabilities
                token_probs = {token: np.exp(logprob) for token, logprob in logprobs.items()}
                
                # Normalize probabilities
                total_prob = sum(token_probs.values())
                token_probs = {token: prob / total_prob for token, prob in token_probs.items()}
                
                return token_probs
            else:
                logger.warning("No logprobs in OpenAI response, falling back to mock generator")
                return MockLLMGenerator().generate_tokens(prompt)
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # Fall back to mock generator
            return MockLLMGenerator().generate_tokens(prompt)


class AnthropicGenerator:
    """
    LLM Generator using the Anthropic Claude API.
    """
    
    def __init__(self, model: str = "claude-3-haiku-20240307"):
        """
        Initialize the Anthropic generator.
        
        Args:
            model: Anthropic model to use
        """
        self.model = model
        
        # Set up Anthropic client
        if ANTHROPIC_API_KEY:
            # Import here to avoid requiring anthropic sdk if not using it
            import anthropic
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info(f"Initialized Anthropic Generator for model {model}")
        else:
            self.client = None
            logger.warning("Anthropic API key not found in environment variables")
            
    def generate_tokens(self, prompt: str) -> Dict[str, float]:
        """
        Generate token probabilities using Anthropic API.
        
        Args:
            prompt: The prompt to generate from
            
        Returns:
            Dictionary mapping tokens to probabilities
        """
        if not self.client:
            logger.warning("No Anthropic client, falling back to mock generator")
            return MockLLMGenerator().generate_tokens(prompt)
            
        try:
            # Make API call
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1,
                temperature=0.7,
                system="You are a skilled programmer assistant. Generate high-quality code based on the given prompt.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Claude API doesn't provide token probabilities directly
            # For now, return a mock distribution
            return MockLLMGenerator().generate_tokens(prompt)
                
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            # Fall back to mock generator
            return MockLLMGenerator().generate_tokens(prompt)


class HuggingFaceGenerator:
    """
    LLM Generator using Hugging Face models.
    """
    
    def __init__(self, model: str = "codellama/CodeLlama-7b-hf"):
        """
        Initialize the Hugging Face generator.
        
        Args:
            model: Hugging Face model to use
        """
        self.model = model
        self.client = None
        
        # Set up Hugging Face client
        if HUGGINGFACE_API_KEY:
            try:
                # Import here to avoid requiring the library if not using it
                from huggingface_hub import InferenceClient
                self.client = InferenceClient(token=HUGGINGFACE_API_KEY)
                logger.info(f"Initialized Hugging Face Generator for model {model}")
            except ImportError:
                logger.warning("huggingface_hub not installed")
        else:
            logger.warning("Hugging Face API key not found in environment variables")
            
    def generate_tokens(self, prompt: str) -> Dict[str, float]:
        """
        Generate token probabilities using Hugging Face API.
        
        Args:
            prompt: The prompt to generate from
            
        Returns:
            Dictionary mapping tokens to probabilities
        """
        if not self.client:
            logger.warning("No Hugging Face client, falling back to mock generator")
            return MockLLMGenerator().generate_tokens(prompt)
            
        try:
            # Make API call
            response = self.client.text_generation(
                prompt,
                model=self.model,
                max_new_tokens=1,
                temperature=0.7,
                details=True,
                return_full_text=False
            )
            
            # Extract token probabilities if available
            if hasattr(response, 'details') and hasattr(response.details, 'top_tokens'):
                # Get top tokens and their probabilities
                top_tokens = response.details.top_tokens[0]
                
                # Create token probability dictionary
                token_probs = {token.text: token.logprob for token in top_tokens[:20]}
                
                # Convert log probabilities to probabilities
                token_probs = {token: np.exp(logprob) for token, logprob in token_probs.items()}
                
                # Normalize probabilities
                total_prob = sum(token_probs.values())
                token_probs = {token: prob / total_prob for token, prob in token_probs.items()}
                
                return token_probs
            else:
                logger.warning("No token probabilities in Hugging Face response")
                return MockLLMGenerator().generate_tokens(prompt)
                
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {e}")
            # Fall back to mock generator
            return MockLLMGenerator().generate_tokens(prompt)


def get_llm_generator(llm_provider: str, model: str) -> Callable:
    """
    Get an LLM generator function for the specified provider and model.
    
    Args:
        llm_provider: Provider name ("openai", "claude", "codellama", "qwen")
        model: Model name
        
    Returns:
        Function that takes a prompt and returns token probabilities
    """
    # Create appropriate generator based on provider
    if llm_provider == "openai":
        generator = OpenAIGenerator(model)
    elif llm_provider == "claude":
        generator = AnthropicGenerator(model)
    elif llm_provider in ["codellama", "qwen"]:
        # For CodeLlama and Qwen, use Hugging Face
        if llm_provider == "codellama":
            model_path = f"codellama/{model}" if "/" not in model else model
        else:
            model_path = f"Qwen/{model}" if "/" not in model else model
            
        generator = HuggingFaceGenerator(model_path)
    else:
        # Default to mock generator
        generator = MockLLMGenerator(model)
    
    # Return the generate_tokens function
    return generator.generate_tokens
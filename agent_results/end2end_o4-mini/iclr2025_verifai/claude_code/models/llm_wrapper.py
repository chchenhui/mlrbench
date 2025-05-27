"""
LLM wrapper for ContractGPT.

This module implements the interface to large language models for code generation
based on formal specifications.
"""

import os
import time
import json
import requests
from typing import Dict, List, Tuple, Optional, Union, Any

# Check for API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

class LLMWrapper:
    """Wrapper for interacting with different LLMs."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize the LLM wrapper.
        
        Args:
            model_name: Name of the LLM to use.
            Available options:
            - "gpt-4o-mini": OpenAI's GPT-4o-mini
            - "claude-3-5-sonnet": Anthropic's Claude-3-5-sonnet
        """
        self.model_name = model_name
        
        # Check if required API keys are available
        if "gpt" in model_name and not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        if "claude" in model_name and not ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key not found in environment variables")
    
    def generate_code(self, prompt: str, temperature: float = 0.2, max_tokens: int = 2000) -> str:
        """
        Generate code using the specified LLM.
        
        Args:
            prompt: The prompt to send to the LLM.
            temperature: Controls randomness (0 = deterministic, 1 = creative).
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            Generated code as a string.
        """
        if "gpt" in self.model_name:
            return self._generate_with_openai(prompt, temperature, max_tokens)
        elif "claude" in self.model_name:
            return self._generate_with_anthropic(prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _generate_with_openai(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Generate code using OpenAI's API.
        
        Args:
            prompt: The prompt to send to the model.
            temperature: Controls randomness.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            Generated code as a string.
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                data=json.dumps(data)
            )
            
            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code}, {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except Exception as e:
            print(f"Error generating code with OpenAI: {e}")
            return ""
    
    def _generate_with_anthropic(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Generate code using Anthropic's API.
        
        Args:
            prompt: The prompt to send to the model.
            temperature: Controls randomness.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            Generated code as a string.
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                data=json.dumps(data)
            )
            
            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code}, {response.text}")
            
            result = response.json()
            return result["content"][0]["text"]
        
        except Exception as e:
            print(f"Error generating code with Anthropic: {e}")
            return ""


# Convenience function to generate code with default model
def generate_code(prompt: str, model_name: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
    """
    Generate code using the specified LLM.
    
    Args:
        prompt: The prompt to send to the LLM.
        model_name: Name of the LLM to use.
        temperature: Controls randomness (0 = deterministic, 1 = creative).
        
    Returns:
        Generated code as a string.
    """
    wrapper = LLMWrapper(model_name)
    return wrapper.generate_code(prompt, temperature)
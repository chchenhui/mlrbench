#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM utilities for the IETA framework.
This module provides functionality to interact with different LLM APIs and models.
"""

import os
import logging
import time
import json
import torch
from typing import List, Dict, Any, Optional
import openai
import anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

class LLMClient:
    """Base class for LLM API clients."""
    
    def __init__(self, model_name):
        """
        Initialize the LLM client.
        
        Args:
            model_name (str): Name of the model to use
        """
        self.model_name = model_name
    
    def generate(self, prompt, temperature=0.2, max_tokens=1024, n=1):
        """
        Generate completions for a prompt.
        
        Args:
            prompt (str): The prompt to generate completions for
            temperature (float): Sampling temperature
            max_tokens (int): Maximum number of tokens to generate
            n (int): Number of completions to generate
            
        Returns:
            list: List of generated completions
        """
        raise NotImplementedError("Subclasses must implement generate()")

class OpenAIClient(LLMClient):
    """Client for OpenAI's API."""
    
    def __init__(self, model_name="gpt-4o-mini"):
        """
        Initialize the OpenAI client.
        
        Args:
            model_name (str): Name of the model to use
        """
        super().__init__(model_name)
        
        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            api_key = "dummy_key_for_testing"
        
        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI client with model: {model_name}")
    
    def generate(self, prompt, temperature=0.2, max_tokens=1024, n=1):
        """
        Generate completions for a prompt using OpenAI API.
        
        Args:
            prompt (str): The prompt to generate completions for
            temperature (float): Sampling temperature
            max_tokens (int): Maximum number of tokens to generate
            n (int): Number of completions to generate
            
        Returns:
            list: List of generated completions
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                n=n
            )
            
            # Extract completions
            completions = [choice.message.content for choice in response.choices]
            return completions
        
        except Exception as e:
            logger.error(f"Error generating completions with OpenAI API: {e}")
            # For testing/demo, return a dummy response
            return [f"Dummy code completion for: {prompt[:50]}..."] * n

class AnthropicClient(LLMClient):
    """Client for Anthropic's API."""
    
    def __init__(self, model_name="claude-3-7-sonnet"):
        """
        Initialize the Anthropic client.
        
        Args:
            model_name (str): Name of the model to use
        """
        super().__init__(model_name)
        
        # Check for API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment variables")
            api_key = "dummy_key_for_testing"
        
        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Initialized Anthropic client with model: {model_name}")
    
    def generate(self, prompt, temperature=0.2, max_tokens=1024, n=1):
        """
        Generate completions for a prompt using Anthropic API.
        
        Args:
            prompt (str): The prompt to generate completions for
            temperature (float): Sampling temperature
            max_tokens (int): Maximum number of tokens to generate
            n (int): Number of completions to generate
            
        Returns:
            list: List of generated completions
        """
        completions = []
        
        try:
            # Generate n completions
            for _ in range(n):
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                completions.append(response.content[0].text)
        
        except Exception as e:
            logger.error(f"Error generating completions with Anthropic API: {e}")
            # For testing/demo, return a dummy response
            return [f"Dummy code completion for: {prompt[:50]}..."] * n
        
        return completions

class HuggingFaceClient(LLMClient):
    """Client for HuggingFace models."""
    
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        """
        Initialize the HuggingFace client.
        
        Args:
            model_name (str): Name or path of the model to use
        """
        super().__init__(model_name)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Check for GPU availability
            if torch.cuda.is_available():
                logger.info("Using GPU for inference")
                self.device = "cuda"
            else:
                logger.info("GPU not available, using CPU")
                self.device = "cpu"
            
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info(f"Initialized HuggingFace client with model: {model_name}")
        
        except Exception as e:
            logger.error(f"Error initializing HuggingFace model {model_name}: {e}")
            self.tokenizer = None
            self.model = None
            self.device = "cpu"
    
    def generate(self, prompt, temperature=0.2, max_tokens=1024, n=1):
        """
        Generate completions for a prompt using a HuggingFace model.
        
        Args:
            prompt (str): The prompt to generate completions for
            temperature (float): Sampling temperature
            max_tokens (int): Maximum number of tokens to generate
            n (int): Number of completions to generate
            
        Returns:
            list: List of generated completions
        """
        # Check if model is loaded
        if self.model is None or self.tokenizer is None:
            logger.warning("HuggingFace model not initialized, returning dummy response")
            return [f"Dummy code completion for: {prompt[:50]}..."] * n
        
        try:
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate completions
            completions = []
            for _ in range(n):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode the output
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Extract only the generated completion (remove the prompt)
                completion = output_text[len(prompt):]
                completions.append(completion)
            
            return completions
        
        except Exception as e:
            logger.error(f"Error generating completions with HuggingFace model: {e}")
            return [f"Dummy code completion for: {prompt[:50]}..."] * n

def get_llm_client(model_type, model_name):
    """
    Get an LLM client based on model type and name.
    
    Args:
        model_type (str): Type of model ("api" or "huggingface")
        model_name (str): Name of the model
        
    Returns:
        LLMClient: An initialized LLM client
    """
    if model_type == "api":
        # Determine which API client to use based on model name
        if "gpt" in model_name.lower() or "openai" in model_name.lower():
            return OpenAIClient(model_name)
        elif "claude" in model_name.lower() or "anthropic" in model_name.lower():
            return AnthropicClient(model_name)
        else:
            logger.warning(f"Unknown API model: {model_name}, defaulting to Anthropic")
            return AnthropicClient(model_name)
    
    elif model_type == "huggingface":
        return HuggingFaceClient(model_name)
    
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")

def prepare_code_prompt(task_description, function_signature=None, examples=None, instructions=None):
    """
    Prepare a prompt for code generation.
    
    Args:
        task_description (str): Description of the task
        function_signature (str, optional): Signature of the function to generate
        examples (list, optional): List of example input-output pairs
        instructions (str, optional): Additional instructions
        
    Returns:
        str: Formatted prompt for code generation
    """
    prompt_parts = []
    
    # Add task description
    prompt_parts.append(f"# Task: {task_description}\n")
    
    # Add function signature if provided
    if function_signature:
        prompt_parts.append(f"# Function Signature: {function_signature}\n")
    
    # Add examples if provided
    if examples:
        prompt_parts.append("# Examples:")
        for i, (input_ex, output_ex) in enumerate(examples):
            prompt_parts.append(f"# Input {i+1}: {input_ex}")
            prompt_parts.append(f"# Output {i+1}: {output_ex}")
        prompt_parts.append("")
    
    # Add instructions if provided
    if instructions:
        prompt_parts.append(f"# Instructions: {instructions}\n")
    
    # Add the request for code
    prompt_parts.append("# Please write Python code to solve this task:")
    
    if function_signature:
        # Extract function name from signature
        func_name = function_signature.split("(")[0].strip()
        prompt_parts.append(f"\n{function_signature}:")
        prompt_parts.append("    # Your implementation here")
    
    return "\n".join(prompt_parts)

def format_execution_trace_for_prompt(trace):
    """
    Format an execution trace for inclusion in a prompt.
    
    Args:
        trace (dict): Execution trace
        
    Returns:
        str: Formatted trace as a string
    """
    parts = []
    
    # Add error information if present
    if trace.get("error_type"):
        parts.append(f"Error Type: {trace['error_type']}")
        parts.append(f"Error Message: {trace['error_message']}")
        
        if trace.get("stack_trace"):
            parts.append("Stack Trace:")
            parts.append(trace['stack_trace'])
    
    # Add variable states
    if trace.get("variable_states"):
        parts.append("Variable States:")
        for var_key, var_value in trace["variable_states"].items():
            parts.append(f"  {var_key}: {var_value}")
    
    # Add stdout/stderr
    if trace.get("stdout"):
        parts.append("Standard Output:")
        parts.append(trace["stdout"])
    
    if trace.get("stderr"):
        parts.append("Standard Error:")
        parts.append(trace["stderr"])
    
    # Add execution time
    if "execution_time" in trace:
        parts.append(f"Execution Time: {trace['execution_time']:.4f} seconds")
    
    # Add timeout info
    if trace.get("timeout"):
        parts.append("Execution timed out.")
    
    return "\n".join(parts)
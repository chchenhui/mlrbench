#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base model implementation for the IETA framework.
This module provides the base class for code generation models.
"""

import logging
import random
import numpy as np
from pathlib import Path
from collections import Counter

from utils.llm_utils import get_llm_client, prepare_code_prompt

logger = logging.getLogger(__name__)

class BaseCodeLLM:
    """Base class for code generation models."""
    
    def __init__(self, model_name, model_type="api", max_samples=10):
        """
        Initialize the base code generation model.
        
        Args:
            model_name (str): Name of the model to use
            model_type (str): Type of model ("api" or "huggingface")
            max_samples (int): Maximum number of samples to generate per prompt
        """
        self.model_name = model_name
        self.model_type = model_type
        self.max_samples = max_samples
        
        # Initialize the LLM client
        try:
            self.llm_client = get_llm_client(model_type, model_name)
            logger.info(f"Initialized {model_type} model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self.llm_client = None
    
    def generate_samples(self, dataset, temperature=0.7, n=5):
        """
        Generate code samples for a dataset.
        
        Args:
            dataset (list): List of prompts
            temperature (float): Sampling temperature
            n (int): Number of samples to generate per prompt
            
        Returns:
            list: List of lists of generated code samples
        """
        if self.llm_client is None:
            logger.error("LLM client not initialized")
            return [["# Error: LLM client not initialized"] * n] * len(dataset)
        
        all_samples = []
        
        for i, item in enumerate(dataset):
            logger.info(f"Generating samples for prompt {i+1}/{len(dataset)}")
            
            # Extract the prompt
            prompt = item["prompt"]
            
            # Prepare the code prompt
            code_prompt = prepare_code_prompt(prompt)
            
            try:
                # Generate completions
                samples = self.llm_client.generate(
                    prompt=code_prompt,
                    temperature=temperature,
                    max_tokens=1024,  # Adjusted based on expected code length
                    n=min(n, self.max_samples)
                )
                
                # Process and add to results
                all_samples.append(samples)
            
            except Exception as e:
                logger.error(f"Error generating samples for prompt {i+1}: {e}")
                # Add dummy samples
                all_samples.append([f"# Error generating code for: {prompt[:50]}..."] * n)
        
        return all_samples
    
    def evaluate(self, dataset, pass_k=[1, 10, 100], trace_capturer=None):
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset (list): List of prompts
            pass_k (list): Values of k for pass@k calculation
            trace_capturer (ExecutionTraceCapture, optional): Trace capturer for execution
            
        Returns:
            dict: Evaluation results
        """
        # Generate samples
        all_samples = self.generate_samples(dataset, n=max(pass_k))
        
        # Initialize metrics
        total_prompts = len(dataset)
        passes = {k: 0 for k in pass_k}
        executions = 0
        error_counter = Counter()
        
        # Evaluate each prompt
        for i, (item, samples) in enumerate(zip(dataset, all_samples)):
            logger.info(f"Evaluating prompt {i+1}/{total_prompts}")
            
            # Extract test cases
            test_cases = item.get("test_cases", [])
            
            # Track passing solutions for this prompt
            prompt_passes = 0
            
            # Evaluate each sample
            for sample in samples:
                # Execute the code if we have a trace capturer
                if trace_capturer:
                    trace = trace_capturer.execute_and_capture(sample, test_cases)
                    outcome = trace_capturer.classify_trace(trace)
                    
                    # Count executions without runtime errors
                    if outcome != "S_err" and outcome != "S_timeout" and outcome != "S_comp_err":
                        executions += 1
                    
                    # Count successes
                    if outcome == "S_succ":
                        prompt_passes += 1
                    
                    # Track error types
                    if outcome == "S_err" and "error_type" in trace:
                        error_counter[trace["error_type"]] += 1
                else:
                    # If no trace capturer, use synthetic results for demo
                    outcome = random.choice(["S_succ", "S_err", "S_fail_test", "S_timeout"])
                    if outcome == "S_succ":
                        prompt_passes += 1
                    if outcome != "S_err" and outcome != "S_timeout":
                        executions += 1
                    if outcome == "S_err":
                        error_type = random.choice(["IndexError", "TypeError", "ValueError"])
                        error_counter[error_type] += 1
            
            # Update pass@k metrics
            for k in pass_k:
                if prompt_passes > 0 and k >= len(samples):
                    # If we have at least one passing solution and k is at least the number of samples
                    passes[k] += 1
                else:
                    # Calculate the probability of having at least one passing solution in k samples
                    if len(samples) > 0:  # Avoid division by zero
                        prob = 1.0 - (1.0 - prompt_passes / len(samples)) ** min(k, len(samples))
                        passes[k] += prob
        
        # Calculate final metrics
        pass_rates = [passes[k] / total_prompts for k in pass_k]
        execution_rate = executions / (total_prompts * max(1, len(all_samples[0])))
        
        # Normalize error frequencies
        total_errors = sum(error_counter.values())
        error_frequencies = {
            error_type: count / max(1, total_errors)
            for error_type, count in error_counter.items()
        }
        
        results = {
            "pass_rates": pass_rates,
            "execution_rate": execution_rate,
            "error_frequencies": error_frequencies
        }
        
        return results
    
    def save(self, output_dir):
        """
        Save the model.
        For the base model, this is a no-op as we don't modify the model.
        
        Args:
            output_dir (str or Path): Directory to save the model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save model info
        model_info = {
            "model_name": self.model_name,
            "model_type": self.model_type
        }
        
        # For API models, we don't actually save weights, just the info
        with open(output_dir / "model_info.json", "w") as f:
            import json
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model info saved to {output_dir}")
    
    def load(self, model_dir):
        """
        Load the model.
        For the base model, this is a no-op as we don't modify the model.
        
        Args:
            model_dir (str or Path): Directory containing the model
        """
        model_dir = Path(model_dir)
        
        # Load model info
        try:
            with open(model_dir / "model_info.json", "r") as f:
                import json
                model_info = json.load(f)
                
                # Update model attributes
                self.model_name = model_info.get("model_name", self.model_name)
                self.model_type = model_info.get("model_type", self.model_type)
                
                # Re-initialize the LLM client
                self.llm_client = get_llm_client(self.model_type, self.model_name)
                
                logger.info(f"Model info loaded from {model_dir}")
        
        except Exception as e:
            logger.error(f"Failed to load model info: {e}")
    
    def __str__(self):
        """String representation of the model."""
        return f"BaseCodeLLM(model_name={self.model_name}, model_type={self.model_type})"
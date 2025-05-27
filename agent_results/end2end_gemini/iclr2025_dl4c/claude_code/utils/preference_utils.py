#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preference utilities for the IETA framework.
This module provides functionality to generate preference pairs from execution traces.
"""

import logging
import random
from itertools import combinations
import numpy as np

logger = logging.getLogger(__name__)

def generate_preference_pairs(execution_results):
    """
    Generate preference pairs from execution results.
    
    Args:
        execution_results (list): List of execution results for each prompt
            Each entry is a list of results for different code samples for the same prompt
            
    Returns:
        list: List of preference pairs (prompt, chosen_code, rejected_code)
    """
    preference_pairs = []
    
    for prompt_results in execution_results:
        # Extract preference pairs from this prompt's results
        prompt_pairs = generate_prompt_preference_pairs(prompt_results)
        preference_pairs.extend(prompt_pairs)
    
    logger.info(f"Generated {len(preference_pairs)} preference pairs from {len(execution_results)} prompts")
    return preference_pairs

def generate_prompt_preference_pairs(prompt_results):
    """
    Generate preference pairs for a single prompt's results.
    
    Args:
        prompt_results (list): List of execution results for a single prompt
            
    Returns:
        list: List of preference pairs (prompt, chosen_code, rejected_code)
    """
    pairs = []
    prompt = ""  # Extract from results
    
    # Group results by outcome
    outcome_groups = {}
    for result in prompt_results:
        outcome = result["outcome"]
        if outcome not in outcome_groups:
            outcome_groups[outcome] = []
        outcome_groups[outcome].append(result)
        
        # Extract the prompt (should be the same for all results)
        if "prompt" in result:
            prompt = result["prompt"]
    
    # Define preference order (from best to worst)
    preference_order = ["S_succ", "S_fail_test", "S_err", "S_comp_err", "S_timeout"]
    
    # Generate pairs based on preference order
    for i in range(len(preference_order)):
        better_outcome = preference_order[i]
        if better_outcome not in outcome_groups:
            continue
            
        better_results = outcome_groups[better_outcome]
        
        # Compare with worse outcomes
        for j in range(i+1, len(preference_order)):
            worse_outcome = preference_order[j]
            if worse_outcome not in outcome_groups:
                continue
                
            worse_results = outcome_groups[worse_outcome]
            
            # Create preference pairs
            for better_result in better_results:
                for worse_result in worse_results:
                    pairs.append({
                        "prompt": prompt,
                        "chosen_code": better_result["code"],
                        "rejected_code": worse_result["code"],
                        "chosen_outcome": better_outcome,
                        "rejected_outcome": worse_outcome,
                        "chosen_trace": better_result.get("trace", {}),
                        "rejected_trace": worse_result.get("trace", {})
                    })
    
    # If we have multiple results with the same outcome, create finer-grained preferences
    for outcome, results in outcome_groups.items():
        if len(results) > 1:
            # For error outcomes, prefer code that gets further before error
            if outcome == "S_err":
                pairs.extend(generate_error_preference_pairs(results, prompt))
            
            # For successful outcomes, prefer code with better metrics (if available)
            elif outcome == "S_succ":
                pairs.extend(generate_success_preference_pairs(results, prompt))
    
    return pairs

def generate_error_preference_pairs(error_results, prompt):
    """
    Generate preference pairs for results with errors.
    Prefer:
    1. Code that executes more lines before failing
    2. Code with more detailed error traces
    3. Code with more common/fixable error types
    
    Args:
        error_results (list): List of results with errors
        prompt (str): The prompt for these results
        
    Returns:
        list: List of preference pairs
    """
    pairs = []
    
    # Compare all pairs of error results
    for result1, result2 in combinations(error_results, 2):
        # Skip if we don't have trace information
        if "trace" not in result1 or "trace" not in result2:
            continue
            
        trace1 = result1["trace"]
        trace2 = result2["trace"]
        
        # Check if we can determine a preference
        preference = None
        
        # Prefer more lines executed before error
        if "execution_line" in trace1 and "execution_line" in trace2:
            line1 = trace1["execution_line"]
            line2 = trace2["execution_line"]
            if line1 > line2:
                preference = (result1, result2)
            elif line2 > line1:
                preference = (result2, result1)
        
        # If still no preference, consider error type
        if preference is None and "error_type" in trace1 and "error_type" in trace2:
            # Define a rough hierarchy of error types (from more fixable to less)
            error_hierarchy = {
                "IndexError": 1,
                "TypeError": 2,
                "ValueError": 3,
                "AttributeError": 4,
                "NameError": 5,
                "SyntaxError": 6,
                "ZeroDivisionError": 7,
                "ImportError": 8,
                "KeyError": 9,
                "MemoryError": 10,
                "RecursionError": 11
            }
            
            error1 = trace1["error_type"]
            error2 = trace2["error_type"]
            
            score1 = error_hierarchy.get(error1, 100)  # Default high value for unknown errors
            score2 = error_hierarchy.get(error2, 100)
            
            if score1 < score2:
                preference = (result1, result2)
            elif score2 < score1:
                preference = (result2, result1)
        
        # If we determined a preference, add it to the pairs
        if preference:
            chosen, rejected = preference
            pairs.append({
                "prompt": prompt,
                "chosen_code": chosen["code"],
                "rejected_code": rejected["code"],
                "chosen_outcome": "S_err",
                "rejected_outcome": "S_err",
                "chosen_trace": chosen.get("trace", {}),
                "rejected_trace": rejected.get("trace", {})
            })
    
    return pairs

def generate_success_preference_pairs(success_results, prompt):
    """
    Generate preference pairs for successful results.
    Prefer:
    1. Code that passes more tests
    2. Code with better performance metrics
    3. Code that is more concise/elegant (if we have a metric for that)
    
    Args:
        success_results (list): List of successful results
        prompt (str): The prompt for these results
        
    Returns:
        list: List of preference pairs
    """
    pairs = []
    
    # Compare all pairs of successful results
    for result1, result2 in combinations(success_results, 2):
        # Skip if we don't have trace information
        if "trace" not in result1 or "trace" not in result2:
            continue
            
        trace1 = result1["trace"]
        trace2 = result2["trace"]
        
        # Check if we can determine a preference
        preference = None
        
        # Prefer code that passes more tests
        if "test_results" in trace1 and "test_results" in trace2:
            passed1 = sum(1 for test in trace1["test_results"] if test.get("passed", False))
            passed2 = sum(1 for test in trace2["test_results"] if test.get("passed", False))
            
            if passed1 > passed2:
                preference = (result1, result2)
            elif passed2 > passed1:
                preference = (result2, result1)
        
        # If still no preference, prefer code with better execution time
        if preference is None and "execution_time" in trace1 and "execution_time" in trace2:
            time1 = trace1["execution_time"]
            time2 = trace2["execution_time"]
            
            # Only prefer if the difference is significant (e.g., 20% faster)
            if time1 < time2 * 0.8:
                preference = (result1, result2)
            elif time2 < time1 * 0.8:
                preference = (result2, result1)
        
        # If we determined a preference, add it to the pairs
        if preference:
            chosen, rejected = preference
            pairs.append({
                "prompt": prompt,
                "chosen_code": chosen["code"],
                "rejected_code": rejected["code"],
                "chosen_outcome": "S_succ",
                "rejected_outcome": "S_succ",
                "chosen_trace": chosen.get("trace", {}),
                "rejected_trace": rejected.get("trace", {})
            })
    
    return pairs

def generate_synthetic_dataset(dataset, num_samples, error_types=None):
    """
    Generate a synthetic dataset of preference pairs for testing and demonstration.
    
    Args:
        dataset (list): Base dataset with prompts
        num_samples (int): Number of prompt-wise samples to generate
        error_types (list, optional): List of error types to include
        
    Returns:
        list: List of synthetic preference pairs
    """
    if error_types is None:
        error_types = ["IndexError", "TypeError", "ValueError", "ZeroDivisionError"]
    
    preference_pairs = []
    prompts = [item["prompt"] for item in dataset[:num_samples]]
    
    for prompt in prompts:
        # For each prompt, generate some synthetic pairs
        
        # 1. Success vs. Error pairs
        for error_type in error_types:
            # Successful implementation
            success_code = f"# Successful implementation for prompt: {prompt[:50]}..."
            
            # Implementation with error
            error_code = f"# Implementation with {error_type} for prompt: {prompt[:50]}..."
            
            # Add preference pair
            preference_pairs.append({
                "prompt": prompt,
                "chosen_code": success_code,
                "rejected_code": error_code,
                "chosen_outcome": "S_succ",
                "rejected_outcome": "S_err",
                "chosen_trace": {"outcome": "S_succ", "execution_time": 0.1},
                "rejected_trace": {"outcome": "S_err", "error_type": error_type}
            })
        
        # 2. Error vs. Timeout pairs
        for error_type in error_types[:2]:  # Use just a couple of error types
            # Implementation with error
            error_code = f"# Implementation with {error_type} for prompt: {prompt[:50]}..."
            
            # Implementation with timeout
            timeout_code = f"# Implementation with timeout for prompt: {prompt[:50]}..."
            
            # Add preference pair
            preference_pairs.append({
                "prompt": prompt,
                "chosen_code": error_code,
                "rejected_code": timeout_code,
                "chosen_outcome": "S_err",
                "rejected_outcome": "S_timeout",
                "chosen_trace": {"outcome": "S_err", "error_type": error_type},
                "rejected_trace": {"outcome": "S_timeout", "timeout": True}
            })
        
        # 3. Error vs. Error pairs (different severity)
        if len(error_types) >= 2:
            for i in range(len(error_types) - 1):
                # Less severe error
                less_severe_code = f"# Implementation with {error_types[i]} for prompt: {prompt[:50]}..."
                
                # More severe error
                more_severe_code = f"# Implementation with {error_types[i+1]} for prompt: {prompt[:50]}..."
                
                # Add preference pair
                preference_pairs.append({
                    "prompt": prompt,
                    "chosen_code": less_severe_code,
                    "rejected_code": more_severe_code,
                    "chosen_outcome": "S_err",
                    "rejected_outcome": "S_err",
                    "chosen_trace": {"outcome": "S_err", "error_type": error_types[i]},
                    "rejected_trace": {"outcome": "S_err", "error_type": error_types[i+1]}
                })
    
    return preference_pairs
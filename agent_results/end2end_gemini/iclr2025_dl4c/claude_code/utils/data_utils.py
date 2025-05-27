#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data utilities for the IETA framework.
This module provides functionality to load datasets, save results, and generate synthetic data.
"""

import json
import os
import logging
import random
from pathlib import Path
import numpy as np
from datasets import load_dataset as hf_load_dataset

logger = logging.getLogger(__name__)

def load_dataset(dataset_name, num_samples=None):
    """
    Load a code generation dataset.
    
    Args:
        dataset_name (str): Name of the dataset (humaneval, mbpp, apps)
        num_samples (int, optional): Number of samples to load (None = all)
        
    Returns:
        list: List of (prompt, test_cases) tuples
    """
    logger.info(f"Loading {dataset_name} dataset")
    
    if dataset_name == "humaneval":
        return load_humaneval_dataset(num_samples)
    elif dataset_name == "mbpp":
        return load_mbpp_dataset(num_samples)
    elif dataset_name == "apps":
        return load_apps_dataset(num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_humaneval_dataset(num_samples=None):
    """
    Load the HumanEval dataset.
    
    Args:
        num_samples (int, optional): Number of samples to load
        
    Returns:
        list: List of (prompt, test_cases) tuples
    """
    try:
        dataset = hf_load_dataset("openai_humaneval")
        
        # Process the dataset
        processed_data = []
        for item in dataset["test"]:
            prompt = item["prompt"]
            test_case = item["test"]
            entry = {"prompt": prompt, "test_cases": [test_case]}
            processed_data.append(entry)
        
        # Subsample if requested
        if num_samples is not None and num_samples < len(processed_data):
            processed_data = random.sample(processed_data, num_samples)
        
        logger.info(f"Loaded {len(processed_data)} samples from HumanEval dataset")
        return processed_data
    
    except Exception as e:
        logger.error(f"Error loading HumanEval dataset: {e}")
        # Return a small synthetic dataset for demonstration
        return generate_synthetic_humaneval(num_samples or 5)

def load_mbpp_dataset(num_samples=None):
    """
    Load the MBPP dataset.
    
    Args:
        num_samples (int, optional): Number of samples to load
        
    Returns:
        list: List of (prompt, test_cases) tuples
    """
    try:
        dataset = hf_load_dataset("mbpp")
        
        # Process the dataset
        processed_data = []
        for item in dataset["test"]:
            # Combine task_id and text to form the prompt
            prompt = f"# Task ID: {item['task_id']}\n# {item['text']}\n\n"
            test_cases = item["test_list"]
            entry = {"prompt": prompt, "test_cases": test_cases}
            processed_data.append(entry)
        
        # Subsample if requested
        if num_samples is not None and num_samples < len(processed_data):
            processed_data = random.sample(processed_data, num_samples)
        
        logger.info(f"Loaded {len(processed_data)} samples from MBPP dataset")
        return processed_data
    
    except Exception as e:
        logger.error(f"Error loading MBPP dataset: {e}")
        # Return a small synthetic dataset for demonstration
        return generate_synthetic_mbpp(num_samples or 5)

def load_apps_dataset(num_samples=None):
    """
    Load the APPS dataset.
    
    Args:
        num_samples (int, optional): Number of samples to load
        
    Returns:
        list: List of (prompt, test_cases) tuples
    """
    try:
        dataset = hf_load_dataset("codeparrot/apps", split="test")
        
        # Process the dataset
        processed_data = []
        for item in dataset:
            prompt = item["question"]
            test_cases = item["input_output"].get("inputs", [])
            expected_outputs = item["input_output"].get("outputs", [])
            
            # Create test cases from inputs and expected outputs
            formatted_tests = []
            for i, (test_in, test_out) in enumerate(zip(test_cases, expected_outputs)):
                test_case = f"# Test {i+1}\ninput_data = {repr(test_in)}\nexpected_output = {repr(test_out)}\n"
                test_case += f"assert solution(input_data) == expected_output, f'Test {i+1} failed'"
                formatted_tests.append(test_case)
            
            entry = {"prompt": prompt, "test_cases": formatted_tests}
            processed_data.append(entry)
        
        # Subsample if requested
        if num_samples is not None and num_samples < len(processed_data):
            processed_data = random.sample(processed_data, num_samples)
        
        logger.info(f"Loaded {len(processed_data)} samples from APPS dataset")
        return processed_data
    
    except Exception as e:
        logger.error(f"Error loading APPS dataset: {e}")
        # Return a small synthetic dataset for demonstration
        return generate_synthetic_apps(num_samples or 5)

def generate_synthetic_humaneval(num_samples=5):
    """Generate a synthetic HumanEval-like dataset for demonstration purposes."""
    logger.warning("Using synthetic HumanEval dataset")
    
    templates = [
        {
            "prompt": "def is_prime(n):\n    \"\"\"Return True if n is a prime number, False otherwise.\n    >>> is_prime(2)\n    True\n    >>> is_prime(8)\n    False\n    \"\"\"\n",
            "test_cases": ["assert is_prime(2) == True", "assert is_prime(3) == True", 
                          "assert is_prime(4) == False", "assert is_prime(11) == True"]
        },
        {
            "prompt": "def factorial(n):\n    \"\"\"Return the factorial of n.\n    >>> factorial(0)\n    1\n    >>> factorial(5)\n    120\n    \"\"\"\n",
            "test_cases": ["assert factorial(0) == 1", "assert factorial(1) == 1", 
                          "assert factorial(5) == 120", "assert factorial(10) == 3628800"]
        },
        {
            "prompt": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\n    >>> fibonacci(0)\n    0\n    >>> fibonacci(1)\n    1\n    >>> fibonacci(10)\n    55\n    \"\"\"\n",
            "test_cases": ["assert fibonacci(0) == 0", "assert fibonacci(1) == 1", 
                          "assert fibonacci(2) == 1", "assert fibonacci(10) == 55"]
        },
        {
            "prompt": "def longest_common_substring(s1, s2):\n    \"\"\"Return the longest common substring of s1 and s2.\n    >>> longest_common_substring(\"abcdef\", \"bcd\")\n    \"bcd\"\n    >>> longest_common_substring(\"abc\", \"xyz\")\n    \"\"\n    \"\"\"\n",
            "test_cases": ["assert longest_common_substring(\"abcdef\", \"bcd\") == \"bcd\"", 
                          "assert longest_common_substring(\"abc\", \"xyz\") == \"\"",
                          "assert longest_common_substring(\"abcdef\", \"defg\") == \"def\""]
        },
        {
            "prompt": "def gcd(a, b):\n    \"\"\"Return the greatest common divisor of a and b.\n    >>> gcd(12, 8)\n    4\n    >>> gcd(15, 0)\n    15\n    \"\"\"\n",
            "test_cases": ["assert gcd(12, 8) == 4", "assert gcd(15, 0) == 15", 
                          "assert gcd(0, 5) == 5", "assert gcd(48, 18) == 6"]
        },
        {
            "prompt": "def remove_duplicates(lst):\n    \"\"\"Remove duplicates from lst and return the result.\n    >>> remove_duplicates([1, 2, 3, 1, 2])\n    [1, 2, 3]\n    >>> remove_duplicates([1, 1, 1])\n    [1]\n    \"\"\"\n",
            "test_cases": ["assert remove_duplicates([1, 2, 3, 1, 2]) == [1, 2, 3]", 
                          "assert remove_duplicates([1, 1, 1]) == [1]",
                          "assert remove_duplicates([]) == []"]
        },
        {
            "prompt": "def sort_dict_by_value(d):\n    \"\"\"Sort a dictionary by value in descending order and return a list of keys.\n    >>> sort_dict_by_value({\"a\": 3, \"b\": 1, \"c\": 2})\n    ['a', 'c', 'b']\n    >>> sort_dict_by_value({\"a\": 1, \"b\": 1})\n    ['a', 'b']\n    \"\"\"\n",
            "test_cases": ["assert sort_dict_by_value({\"a\": 3, \"b\": 1, \"c\": 2}) == ['a', 'c', 'b']", 
                          "assert sorted(sort_dict_by_value({\"a\": 1, \"b\": 1})) == ['a', 'b']",
                          "assert sort_dict_by_value({}) == []"]
        },
        {
            "prompt": "def count_vowels(s):\n    \"\"\"Count the number of vowels in s.\n    >>> count_vowels(\"hello\")\n    2\n    >>> count_vowels(\"world\")\n    1\n    \"\"\"\n",
            "test_cases": ["assert count_vowels(\"hello\") == 2", "assert count_vowels(\"world\") == 1", 
                          "assert count_vowels(\"\") == 0", "assert count_vowels(\"aeiou\") == 5"]
        },
        {
            "prompt": "def is_palindrome(s):\n    \"\"\"Return True if s is a palindrome, False otherwise.\n    >>> is_palindrome(\"racecar\")\n    True\n    >>> is_palindrome(\"hello\")\n    False\n    \"\"\"\n",
            "test_cases": ["assert is_palindrome(\"racecar\") == True", "assert is_palindrome(\"hello\") == False", 
                          "assert is_palindrome(\"\") == True", "assert is_palindrome(\"a\") == True"]
        },
        {
            "prompt": "def invert_dict(d):\n    \"\"\"Invert a dictionary, mapping values to keys.\n    >>> invert_dict({\"a\": 1, \"b\": 2})\n    {1: \"a\", 2: \"b\"}\n    >>> invert_dict({\"a\": 1, \"b\": 1})\n    {1: \"b\"}\n    \"\"\"\n",
            "test_cases": ["assert invert_dict({\"a\": 1, \"b\": 2}) == {1: \"a\", 2: \"b\"}", 
                          "assert invert_dict({\"a\": 1, \"b\": 1}) == {1: \"b\"}",
                          "assert invert_dict({}) == {}"]
        }
    ]
    
    # Ensure we have enough templates
    while len(templates) < num_samples:
        templates.extend(templates[:num_samples - len(templates)])
    
    # Sample from templates
    selected_templates = random.sample(templates, num_samples)
    return selected_templates

def generate_synthetic_mbpp(num_samples=5):
    """Generate a synthetic MBPP-like dataset for demonstration purposes."""
    logger.warning("Using synthetic MBPP dataset")
    
    templates = [
        {
            "prompt": "# Task ID: 101\n# Write a function to check if a given string is a palindrome.\n\n",
            "test_cases": ["assert is_palindrome(\"racecar\") == True", 
                           "assert is_palindrome(\"hello\") == False",
                           "assert is_palindrome(\"\") == True"]
        },
        {
            "prompt": "# Task ID: 102\n# Write a function to count the occurrences of each element in a list.\n\n",
            "test_cases": ["assert count_elements([1, 2, 3, 1, 2, 1]) == {1: 3, 2: 2, 3: 1}", 
                           "assert count_elements([]) == {}",
                           "assert count_elements([5, 5, 5]) == {5: 3}"]
        },
        {
            "prompt": "# Task ID: 103\n# Write a function to find the sum of digits of a number.\n\n",
            "test_cases": ["assert sum_of_digits(123) == 6", 
                           "assert sum_of_digits(0) == 0",
                           "assert sum_of_digits(999) == 27"]
        },
        {
            "prompt": "# Task ID: 104\n# Write a function to find all permutations of a string.\n\n",
            "test_cases": ["assert sorted(find_permutations(\"abc\")) == sorted([\"abc\", \"acb\", \"bac\", \"bca\", \"cab\", \"cba\"])", 
                           "assert find_permutations(\"\") == [\"\"]",
                           "assert sorted(find_permutations(\"a\")) == [\"a\"]"]
        },
        {
            "prompt": "# Task ID: 105\n# Write a function to calculate the depth of a binary tree.\n\n",
            "test_cases": ["class Node:\n    def __init__(self, val=None, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right\n\nroot = Node(1, Node(2, Node(4), Node(5)), Node(3))\nassert tree_depth(root) == 3",
                           "assert tree_depth(None) == 0",
                           "assert tree_depth(Node(1)) == 1"]
        },
        {
            "prompt": "# Task ID: 106\n# Write a function to check if two strings are anagrams.\n\n",
            "test_cases": ["assert is_anagram(\"listen\", \"silent\") == True", 
                           "assert is_anagram(\"hello\", \"world\") == False",
                           "assert is_anagram(\"\", \"\") == True"]
        },
        {
            "prompt": "# Task ID: 107\n# Write a function to find the longest common prefix of a list of strings.\n\n",
            "test_cases": ["assert longest_common_prefix([\"flower\", \"flow\", \"flight\"]) == \"fl\"", 
                           "assert longest_common_prefix([\"dog\", \"car\", \"race\"]) == \"\"",
                           "assert longest_common_prefix([]) == \"\""]
        },
        {
            "prompt": "# Task ID: 108\n# Write a function to implement a binary search algorithm.\n\n",
            "test_cases": ["assert binary_search([1, 2, 3, 4, 5], 3) == 2", 
                           "assert binary_search([1, 2, 3, 4, 5], 6) == -1",
                           "assert binary_search([], 1) == -1"]
        },
        {
            "prompt": "# Task ID: 109\n# Write a function to reverse a list without using the built-in reverse function.\n\n",
            "test_cases": ["assert reverse_list([1, 2, 3, 4, 5]) == [5, 4, 3, 2, 1]", 
                           "assert reverse_list([]) == []",
                           "assert reverse_list([1]) == [1]"]
        },
        {
            "prompt": "# Task ID: 110\n# Write a function to find the first non-repeating character in a string.\n\n",
            "test_cases": ["assert first_non_repeating(\"apple\") == \"a\"", 
                           "assert first_non_repeating(\"aabb\") == None",
                           "assert first_non_repeating(\"\") == None"]
        }
    ]
    
    # Ensure we have enough templates
    while len(templates) < num_samples:
        templates.extend(templates[:num_samples - len(templates)])
    
    # Sample from templates
    selected_templates = random.sample(templates, num_samples)
    return selected_templates

def generate_synthetic_apps(num_samples=5):
    """Generate a synthetic APPS-like dataset for demonstration purposes."""
    logger.warning("Using synthetic APPS dataset")
    
    templates = [
        {
            "prompt": "You are given an array of integers. Write a function 'solution' that returns the sum of all the positive integers in the array.",
            "test_cases": ["input_data = [1, -2, 3, 4, -5]\nexpected_output = 8\nassert solution(input_data) == expected_output",
                          "input_data = [-1, -2, -3]\nexpected_output = 0\nassert solution(input_data) == expected_output"]
        },
        {
            "prompt": "Given two strings s and t, write a function 'solution' that determines if s is a subsequence of t. A subsequence is a string that can be derived from another string by deleting some or no characters without changing the order of the remaining characters.",
            "test_cases": ["input_data = {\"s\": \"abc\", \"t\": \"ahbgdc\"}\nexpected_output = True\nassert solution(input_data[\"s\"], input_data[\"t\"]) == expected_output",
                          "input_data = {\"s\": \"axc\", \"t\": \"ahbgdc\"}\nexpected_output = False\nassert solution(input_data[\"s\"], input_data[\"t\"]) == expected_output"]
        },
        {
            "prompt": "You are given a string containing only the characters '(', ')', '{', '}', '[' and ']'. Write a function 'solution' that determines if the input string is valid. An input string is valid if: Open brackets must be closed by the same type of brackets. Open brackets must be closed in the correct order.",
            "test_cases": ["input_data = \"()\"\nexpected_output = True\nassert solution(input_data) == expected_output",
                          "input_data = \"(){[]}([)])\"\nexpected_output = False\nassert solution(input_data) == expected_output"]
        },
        {
            "prompt": "You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. Write a function 'solution' that returns in how many distinct ways can you climb to the top.",
            "test_cases": ["input_data = 2\nexpected_output = 2\nassert solution(input_data) == expected_output",
                          "input_data = 5\nexpected_output = 8\nassert solution(input_data) == expected_output"]
        },
        {
            "prompt": "Given an array of integers, write a function 'solution' that finds the contiguous subarray (containing at least one number) which has the largest sum and return its sum.",
            "test_cases": ["input_data = [-2, 1, -3, 4, -1, 2, 1, -5, 4]\nexpected_output = 6\nassert solution(input_data) == expected_output",
                          "input_data = [1]\nexpected_output = 1\nassert solution(input_data) == expected_output"]
        },
        {
            "prompt": "Given an integer array nums, write a function 'solution' that returns all unique triplets in the array which gives the sum of zero.",
            "test_cases": ["input_data = [-1, 0, 1, 2, -1, -4]\nexpected_output = [[-1, -1, 2], [-1, 0, 1]]\nassert sorted(sorted(x) for x in solution(input_data)) == sorted(sorted(x) for x in expected_output)",
                          "input_data = [0, 0, 0]\nexpected_output = [[0, 0, 0]]\nassert sorted(sorted(x) for x in solution(input_data)) == sorted(sorted(x) for x in expected_output)"]
        },
        {
            "prompt": "Given an array of integers and an integer k, write a function 'solution' that finds the total number of continuous subarrays whose sum equals to k.",
            "test_cases": ["input_data = {\"nums\": [1, 1, 1], \"k\": 2}\nexpected_output = 2\nassert solution(input_data[\"nums\"], input_data[\"k\"]) == expected_output",
                          "input_data = {\"nums\": [1, 2, 3], \"k\": 3}\nexpected_output = 2\nassert solution(input_data[\"nums\"], input_data[\"k\"]) == expected_output"]
        },
        {
            "prompt": "Given a string s, write a function 'solution' that finds the length of the longest substring without repeating characters.",
            "test_cases": ["input_data = \"abcabcbb\"\nexpected_output = 3\nassert solution(input_data) == expected_output",
                          "input_data = \"bbbbb\"\nexpected_output = 1\nassert solution(input_data) == expected_output"]
        },
        {
            "prompt": "You are given an array of prices where prices[i] is the price of a given stock on the ith day. Write a function 'solution' that maximizes your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.",
            "test_cases": ["input_data = [7, 1, 5, 3, 6, 4]\nexpected_output = 5\nassert solution(input_data) == expected_output",
                          "input_data = [7, 6, 4, 3, 1]\nexpected_output = 0\nassert solution(input_data) == expected_output"]
        },
        {
            "prompt": "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', write a function 'solution' that returns the minimum number of moves required to make s valid. A move consists of swapping any two adjacent characters.",
            "test_cases": ["input_data = \"([])\"\nexpected_output = 0\nassert solution(input_data) == expected_output",
                          "input_data = \"([)]\"\nexpected_output = 1\nassert solution(input_data) == expected_output"]
        }
    ]
    
    # Ensure we have enough templates
    while len(templates) < num_samples:
        templates.extend(templates[:num_samples - len(templates)])
    
    # Sample from templates
    selected_templates = random.sample(templates, num_samples)
    return selected_templates

def save_results(results, output_path):
    """
    Save experiment results to a JSON file.
    
    Args:
        results (dict): Results to save
        output_path (str or Path): Path to save the results
    """
    output_path = Path(output_path)
    
    # Ensure the directory exists
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    # Convert results
    serializable_results = convert_to_serializable(results)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")

def generate_synthetic_dataset(dataset, num_samples, error_types=None):
    """
    Generate a synthetic dataset for testing and demonstration.
    
    Args:
        dataset (list): Base dataset with prompts
        num_samples (int): Number of samples to generate
        error_types (list, optional): List of error types to include
        
    Returns:
        list: List of synthetic samples
    """
    if error_types is None:
        error_types = ["IndexError", "TypeError", "ValueError", "ZeroDivisionError"]
    
    # Select a subset of the dataset
    if num_samples > len(dataset):
        num_samples = len(dataset)
    
    synthetic_dataset = []
    
    for i in range(num_samples):
        # Select a prompt from the dataset
        prompt = dataset[i % len(dataset)]["prompt"]
        
        # Generate synthetic implementations with different error types
        implementations = []
        
        # Good implementation
        implementations.append({
            "code": f"# Good implementation for: {prompt}",
            "outcome": "S_succ"
        })
        
        # Generate implementations with different errors
        for error_type in error_types:
            implementations.append({
                "code": f"# Bad implementation with {error_type} for: {prompt}",
                "outcome": "S_err",
                "error_type": error_type
            })
        
        # Implementation with timeout
        implementations.append({
            "code": f"# Implementation with timeout for: {prompt}",
            "outcome": "S_timeout"
        })
        
        # Implementation with failed tests
        implementations.append({
            "code": f"# Implementation with incorrect output for: {prompt}",
            "outcome": "S_fail_test"
        })
        
        synthetic_dataset.append({
            "prompt": prompt,
            "implementations": implementations
        })
    
    return synthetic_dataset
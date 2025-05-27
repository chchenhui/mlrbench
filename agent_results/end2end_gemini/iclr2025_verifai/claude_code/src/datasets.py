"""
Datasets for evaluating the SSCSteer framework.

This module provides functions to create and load datasets for evaluation,
including a subset of HumanEval and custom semantic tasks.
"""

import json
import os
from typing import List, Dict, Any, Tuple, Optional
from datasets import load_dataset

def get_humaneval_subset(num_problems: int = 20) -> List[Dict[str, Any]]:
    """
    Get a subset of the HumanEval dataset.
    
    Args:
        num_problems: Number of problems to include in the subset
        
    Returns:
        List of problem dictionaries
    """
    # Load the HumanEval dataset from HuggingFace
    try:
        dataset = load_dataset("openai_humaneval")
        problems = dataset["test"]
        
        # Format the problems
        formatted_problems = []
        
        for i, problem in enumerate(problems):
            if i >= num_problems:
                break
                
            # Extract test cases from the test string
            test_str = problem['test']
            test_cases = []
            
            # Parse the test string to extract test cases
            for line in test_str.split('\n'):
                if 'assert' in line:
                    # Extract function call from assert statement
                    assert_parts = line.split('assert')
                    if len(assert_parts) > 1:
                        assertion = assert_parts[1].strip()
                        
                        # Extract function call and expected output
                        if '==' in assertion:
                            call, expected = assertion.split('==', 1)
                            call = call.strip()
                            expected = expected.strip()
                            
                            # Clean up expected value
                            try:
                                expected = eval(expected)
                            except:
                                # If we can't evaluate, use as is
                                expected = expected
                                
                            test_cases.append({
                                'input': call,
                                'expected': expected
                            })
            
            # Create the prompt
            prompt = f"""
# {problem['prompt']}

Please implement the function as specified above:

```python
{problem['prompt']}
"""
            
            # Add problem to dataset
            formatted_problems.append({
                'id': f"humaneval_{problem['task_id']}",
                'prompt': prompt,
                'test_cases': test_cases,
                'reference_solution': problem['canonical_solution']
            })
            
        return formatted_problems
        
    except Exception as e:
        print(f"Error loading HumanEval dataset: {e}")
        
        # Return a minimal built-in dataset if loading fails
        return get_fallback_humaneval_subset(num_problems)


def get_fallback_humaneval_subset(num_problems: int = 20) -> List[Dict[str, Any]]:
    """
    Get a fallback subset of HumanEval-like problems when dataset loading fails.
    
    Args:
        num_problems: Number of problems to include
        
    Returns:
        List of problem dictionaries
    """
    problems = [
        {
            'id': 'humaneval_0',
            'prompt': """
# Write a function that takes a list of integers and returns the sum of the even elements.
# Example:
# >>> sum_even([1, 2, 3, 4, 5])
# 6

def sum_even(lst):
    \"\"\"
    Takes a list of integers and returns the sum of the even elements.
    >>> sum_even([1, 2, 3, 4, 5])
    6
    \"\"\"
""",
            'test_cases': [
                {'input': 'sum_even([1, 2, 3, 4, 5])', 'expected': 6},
                {'input': 'sum_even([2, 4, 6, 8])', 'expected': 20},
                {'input': 'sum_even([1, 3, 5])', 'expected': 0},
                {'input': 'sum_even([])', 'expected': 0}
            ],
            'reference_solution': 'def sum_even(lst):\n    return sum(x for x in lst if x % 2 == 0)\n'
        },
        {
            'id': 'humaneval_1',
            'prompt': """
# Write a function that takes a string and returns the number of vowels it contains.
# Example:
# >>> count_vowels("hello")
# 2

def count_vowels(s):
    \"\"\"
    Takes a string and returns the number of vowels it contains.
    >>> count_vowels("hello")
    2
    \"\"\"
""",
            'test_cases': [
                {'input': 'count_vowels("hello")', 'expected': 2},
                {'input': 'count_vowels("world")', 'expected': 1},
                {'input': 'count_vowels("aeiou")', 'expected': 5},
                {'input': 'count_vowels("xyz")', 'expected': 0},
                {'input': 'count_vowels("")', 'expected': 0}
            ],
            'reference_solution': 'def count_vowels(s):\n    return sum(1 for c in s.lower() if c in "aeiou")\n'
        },
        {
            'id': 'humaneval_2',
            'prompt': """
# Write a function that takes a list of integers and returns a new list containing only the prime numbers.
# Example:
# >>> get_primes([1, 2, 3, 4, 5, 6])
# [2, 3, 5]

def get_primes(lst):
    \"\"\"
    Takes a list of integers and returns a new list containing only the prime numbers.
    >>> get_primes([1, 2, 3, 4, 5, 6])
    [2, 3, 5]
    \"\"\"
""",
            'test_cases': [
                {'input': 'get_primes([1, 2, 3, 4, 5, 6])', 'expected': [2, 3, 5]},
                {'input': 'get_primes([2, 3, 5, 7, 11, 13])', 'expected': [2, 3, 5, 7, 11, 13]},
                {'input': 'get_primes([4, 6, 8, 10])', 'expected': []},
                {'input': 'get_primes([])', 'expected': []}
            ],
            'reference_solution': '''def get_primes(lst):
    def is_prime(n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    return [n for n in lst if is_prime(n)]
'''
        },
        {
            'id': 'humaneval_3',
            'prompt': """
# Write a function that takes a string and returns True if it is a palindrome, False otherwise.
# A palindrome is a string that reads the same backward as forward.
# Example:
# >>> is_palindrome("radar")
# True
# >>> is_palindrome("hello")
# False

def is_palindrome(s):
    \"\"\"
    Takes a string and returns True if it is a palindrome, False otherwise.
    >>> is_palindrome("radar")
    True
    >>> is_palindrome("hello")
    False
    \"\"\"
""",
            'test_cases': [
                {'input': 'is_palindrome("radar")', 'expected': True},
                {'input': 'is_palindrome("hello")', 'expected': False},
                {'input': 'is_palindrome("A man a plan a canal Panama")', 'expected': False},
                {'input': 'is_palindrome("")', 'expected': True},
                {'input': 'is_palindrome("a")', 'expected': True}
            ],
            'reference_solution': 'def is_palindrome(s):\n    return s == s[::-1]\n'
        },
        {
            'id': 'humaneval_4',
            'prompt': """
# Write a function that takes two sorted lists and merges them into a single sorted list.
# Example:
# >>> merge_sorted([1, 3, 5], [2, 4, 6])
# [1, 2, 3, 4, 5, 6]

def merge_sorted(lst1, lst2):
    \"\"\"
    Takes two sorted lists and merges them into a single sorted list.
    >>> merge_sorted([1, 3, 5], [2, 4, 6])
    [1, 2, 3, 4, 5, 6]
    \"\"\"
""",
            'test_cases': [
                {'input': 'merge_sorted([1, 3, 5], [2, 4, 6])', 'expected': [1, 2, 3, 4, 5, 6]},
                {'input': 'merge_sorted([1, 2, 3], [4, 5, 6])', 'expected': [1, 2, 3, 4, 5, 6]},
                {'input': 'merge_sorted([], [1, 2, 3])', 'expected': [1, 2, 3]},
                {'input': 'merge_sorted([1, 2, 3], [])', 'expected': [1, 2, 3]},
                {'input': 'merge_sorted([], [])', 'expected': []}
            ],
            'reference_solution': '''def merge_sorted(lst1, lst2):
    result = []
    i, j = 0, 0
    while i < len(lst1) and j < len(lst2):
        if lst1[i] <= lst2[j]:
            result.append(lst1[i])
            i += 1
        else:
            result.append(lst2[j])
            j += 1
    result.extend(lst1[i:])
    result.extend(lst2[j:])
    return result
'''
        }
    ]
    
    # Return up to the requested number of problems
    return problems[:num_problems]


def create_semantic_tasks() -> List[Dict[str, Any]]:
    """
    Create custom tasks focused on semantic correctness.
    
    Returns:
        List of problem dictionaries
    """
    semantic_tasks = [
        {
            'id': 'semantic_null_check',
            'prompt': """
# Write a function that takes a potentially None input and safely returns its length.
# If the input is None, return 0.
# Example:
# >>> safe_len(None)
# 0
# >>> safe_len("hello")
# 5

def safe_len(obj):
    \"\"\"
    Takes a potentially None input and safely returns its length.
    If the input is None, return 0.
    >>> safe_len(None)
    0
    >>> safe_len("hello")
    5
    \"\"\"
""",
            'test_cases': [
                {'input': 'safe_len(None)', 'expected': 0},
                {'input': 'safe_len("hello")', 'expected': 5},
                {'input': 'safe_len([1, 2, 3])', 'expected': 3},
                {'input': 'safe_len([])', 'expected': 0}
            ],
            'formal_specs': [
                "null_check(obj)"
            ]
        },
        {
            'id': 'semantic_array_bounds',
            'prompt': """
# Write a function that safely gets an item at the given index from a list.
# If the index is out of bounds, return None.
# Example:
# >>> safe_get([1, 2, 3], 1)
# 2
# >>> safe_get([1, 2, 3], 5)
# None

def safe_get(lst, idx):
    \"\"\"
    Safely gets an item at the given index from a list.
    If the index is out of bounds, return None.
    >>> safe_get([1, 2, 3], 1)
    2
    >>> safe_get([1, 2, 3], 5)
    None
    \"\"\"
""",
            'test_cases': [
                {'input': 'safe_get([1, 2, 3], 1)', 'expected': 2},
                {'input': 'safe_get([1, 2, 3], 5)', 'expected': None},
                {'input': 'safe_get([1, 2, 3], -1)', 'expected': 3},
                {'input': 'safe_get([1, 2, 3], -5)', 'expected': None},
                {'input': 'safe_get([], 0)', 'expected': None}
            ],
            'formal_specs': [
                "bounds_check(lst, idx)"
            ]
        },
        {
            'id': 'semantic_division',
            'prompt': """
# Write a function that safely divides two numbers.
# If the divisor is zero, return None.
# Example:
# >>> safe_divide(10, 2)
# 5.0
# >>> safe_divide(10, 0)
# None

def safe_divide(a, b):
    \"\"\"
    Safely divides two numbers.
    If the divisor is zero, return None.
    >>> safe_divide(10, 2)
    5.0
    >>> safe_divide(10, 0)
    None
    \"\"\"
""",
            'test_cases': [
                {'input': 'safe_divide(10, 2)', 'expected': 5.0},
                {'input': 'safe_divide(10, 0)', 'expected': None},
                {'input': 'safe_divide(0, 10)', 'expected': 0.0},
                {'input': 'safe_divide(-10, 2)', 'expected': -5.0}
            ],
            'formal_specs': [
                "division_safety(b)"
            ]
        },
        {
            'id': 'semantic_resource_leak',
            'prompt': """
# Write a function that reads a file and returns its contents.
# Make sure to properly close the file even if an error occurs.
# Example:
# >>> read_file("example.txt")
# "File contents..."

def read_file(filename):
    \"\"\"
    Reads a file and returns its contents.
    Make sure to properly close the file even if an error occurs.
    >>> read_file("example.txt")
    "File contents..."
    \"\"\"
""",
            'test_cases': [
                {'input': 'type(read_file("nonexistent_file.txt"))', 'expected': type(None)}
            ],
            'formal_specs': [
                "resource_management(file)"
            ]
        },
        {
            'id': 'semantic_type_check',
            'prompt': """
# Write a function that adds two numbers, but first checks that both inputs are numbers.
# If either input is not a number, return None.
# Example:
# >>> type_safe_add(5, 3)
# 8
# >>> type_safe_add("5", 3)
# None

def type_safe_add(a, b):
    \"\"\"
    Adds two numbers, but first checks that both inputs are numbers.
    If either input is not a number, return None.
    >>> type_safe_add(5, 3)
    8
    >>> type_safe_add("5", 3)
    None
    \"\"\"
""",
            'test_cases': [
                {'input': 'type_safe_add(5, 3)', 'expected': 8},
                {'input': 'type_safe_add("5", 3)', 'expected': None},
                {'input': 'type_safe_add(5, "3")', 'expected': None},
                {'input': 'type_safe_add(5.5, 3)', 'expected': 8.5}
            ],
            'formal_specs': [
                "type_safety(a, b)"
            ]
        }
    ]
    
    return semantic_tasks


def save_datasets(output_dir: str = '../data') -> None:
    """
    Save the evaluation datasets to disk.
    
    Args:
        output_dir: Directory to save the datasets
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the datasets
    humaneval = get_humaneval_subset(20)
    semantic_tasks = create_semantic_tasks()
    
    # Save the datasets
    with open(os.path.join(output_dir, 'humaneval_subset.json'), 'w') as f:
        json.dump(humaneval, f, indent=2)
        
    with open(os.path.join(output_dir, 'semantic_tasks.json'), 'w') as f:
        json.dump(semantic_tasks, f, indent=2)
        
    print(f"Saved datasets to {output_dir}")


def load_datasets(data_dir: str = '../data') -> Dict[str, List[Dict[str, Any]]]:
    """
    Load the evaluation datasets from disk.
    
    Args:
        data_dir: Directory containing the datasets
        
    Returns:
        Dictionary mapping dataset names to lists of problem dictionaries
    """
    # Check if the datasets exist
    humaneval_path = os.path.join(data_dir, 'humaneval_subset.json')
    semantic_path = os.path.join(data_dir, 'semantic_tasks.json')
    
    # Initialize dictionary to hold datasets
    datasets = {}
    
    # Load HumanEval subset
    if os.path.exists(humaneval_path):
        with open(humaneval_path, 'r') as f:
            datasets['humaneval'] = json.load(f)
    else:
        # Generate and save the dataset
        datasets['humaneval'] = get_humaneval_subset(20)
        
    # Load semantic tasks
    if os.path.exists(semantic_path):
        with open(semantic_path, 'r') as f:
            datasets['semantic'] = json.load(f)
    else:
        # Generate and save the dataset
        datasets['semantic'] = create_semantic_tasks()
        
    return datasets


# Execute this when the module is run as a script
if __name__ == "__main__":
    # Save the datasets to disk
    save_datasets(os.path.join(os.path.dirname(__file__), '..', 'data'))
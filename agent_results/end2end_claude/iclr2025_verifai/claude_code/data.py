"""
Data loading and processing module for the VERIL experiment.
"""

import os
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass
import random
import json

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

from config import DATA_DIR, DATASET_NAME, DATASET_SIZE
from utils import logger, save_json, load_json

@dataclass
class CodeProblem:
    """Class representing a code problem."""
    id: str
    prompt: str
    reference_solution: str
    test_cases: List[str]
    difficulty: Optional[str] = None
    tags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "reference_solution": self.reference_solution,
            "test_cases": self.test_cases,
            "difficulty": self.difficulty,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeProblem':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            reference_solution=data["reference_solution"],
            test_cases=data["test_cases"],
            difficulty=data.get("difficulty"),
            tags=data.get("tags"),
        )


class CodeGenerationDataset(Dataset):
    """Dataset for code generation tasks."""
    
    def __init__(self, problems: List[CodeProblem], tokenizer):
        """Initialize the dataset."""
        self.problems = problems
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        """Get the number of problems."""
        return len(self.problems)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a problem by index."""
        problem = self.problems[idx]
        
        # Prepare prompt template
        prompt_text = f"""Write a Python function to solve the following problem:

{problem.prompt}

Your solution should be complete and correct.
```python
"""
        
        # Tokenize inputs and outputs
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return {
            "id": problem.id,
            "inputs": inputs,
            "prompt": prompt_text,
            "reference_solution": problem.reference_solution,
            "test_cases": problem.test_cases,
        }


def load_humaneval_dataset(max_problems: Optional[int] = None) -> List[CodeProblem]:
    """
    Load problems from the HumanEval dataset.
    
    Args:
        max_problems: Maximum number of problems to load (optional)
        
    Returns:
        List of CodeProblem objects
    """
    logger.info("Loading HumanEval dataset...")
    
    cache_file = DATA_DIR / "humaneval_dataset.json"
    
    # Check if cached dataset exists
    if os.path.exists(cache_file):
        logger.info(f"Loading cached dataset from {cache_file}")
        data = load_json(cache_file)
        problems = [CodeProblem.from_dict(item) for item in data]
        if max_problems is not None:
            problems = problems[:max_problems]
        return problems
    
    # Load from Hugging Face
    dataset = load_dataset("openai_humaneval")["test"]
    
    problems = []
    for i, item in enumerate(tqdm(dataset, desc="Processing HumanEval problems")):
        if max_problems is not None and i >= max_problems:
            break
            
        # Extract problem details
        task_id = item["task_id"]
        prompt = item["prompt"]
        canonical_solution = item["canonical_solution"]
        test_cases = [item["test"]]
        
        # Create CodeProblem object
        problem = CodeProblem(
            id=task_id,
            prompt=prompt,
            reference_solution=canonical_solution,
            test_cases=test_cases,
        )
        problems.append(problem)
    
    # Cache the dataset
    logger.info(f"Caching dataset to {cache_file}")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    data = [problem.to_dict() for problem in problems]
    save_json(data, cache_file)
    
    return problems


def load_apps_dataset(split: str = "test", max_problems: Optional[int] = None) -> List[CodeProblem]:
    """
    Load problems from the APPS dataset.
    
    Args:
        split: Dataset split to use ("train", "valid", or "test")
        max_problems: Maximum number of problems to load (optional)
        
    Returns:
        List of CodeProblem objects
    """
    logger.info(f"Loading APPS dataset ({split} split)...")
    
    cache_file = DATA_DIR / f"apps_{split}_dataset.json"
    
    # Check if cached dataset exists
    if os.path.exists(cache_file):
        logger.info(f"Loading cached dataset from {cache_file}")
        data = load_json(cache_file)
        problems = [CodeProblem.from_dict(item) for item in data]
        if max_problems is not None:
            problems = problems[:max_problems]
        return problems
    
    # Load from Hugging Face
    dataset = load_dataset("codeparrot/apps")[split]
    
    difficulties = ["introductory", "interview", "competition"]
    problems = []
    
    for i, item in enumerate(tqdm(dataset, desc=f"Processing APPS {split} problems")):
        if max_problems is not None and i >= max_problems:
            break
            
        # Extract problem details
        problem_id = f"apps_{split}_{i}"
        prompt = item["question"]
        solutions = item["solutions"]
        test_cases = item["input_output"].get("inputs", [])
        difficulty = difficulties[item["difficulty"]]
        
        # Use the first solution as reference
        reference_solution = solutions[0] if solutions else ""
        
        # Create CodeProblem object
        problem = CodeProblem(
            id=problem_id,
            prompt=prompt,
            reference_solution=reference_solution,
            test_cases=test_cases,
            difficulty=difficulty,
        )
        problems.append(problem)
    
    # Cache the dataset
    logger.info(f"Caching dataset to {cache_file}")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    data = [problem.to_dict() for problem in problems]
    save_json(data, cache_file)
    
    return problems


def create_custom_dataset(max_problems: Optional[int] = None) -> List[CodeProblem]:
    """
    Create a small custom dataset of programming problems.
    
    Args:
        max_problems: Maximum number of problems to create (optional)
        
    Returns:
        List of CodeProblem objects
    """
    logger.info("Creating custom dataset...")
    
    custom_problems = [
        {
            "id": "custom_1",
            "prompt": "Write a function to check if a string is a palindrome. A palindrome is a string that reads the same backward as forward.",
            "reference_solution": """def is_palindrome(s):
    s = s.lower()
    # Remove all non-alphanumeric chars
    s = ''.join(c for c in s if c.isalnum())
    return s == s[::-1]""",
            "test_cases": [
                """
assert is_palindrome("racecar") == True
assert is_palindrome("A man a plan a canal Panama") == True
assert is_palindrome("hello") == False
assert is_palindrome("") == True
assert is_palindrome("a") == True
assert is_palindrome("Ab1ba") == True
"""
            ],
            "difficulty": "easy",
            "tags": ["strings", "algorithms"],
        },
        {
            "id": "custom_2",
            "prompt": "Implement a function to find the nth Fibonacci number. The Fibonacci sequence is defined as F(0) = 0, F(1) = 1, and F(n) = F(n-1) + F(n-2) for n > 1.",
            "reference_solution": """def fibonacci(n):
    if n < 0:
        raise ValueError("Input must be non-negative")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b""",
            "test_cases": [
                """
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(2) == 1
assert fibonacci(5) == 5
assert fibonacci(10) == 55
"""
            ],
            "difficulty": "easy",
            "tags": ["recursion", "dynamic programming"],
        },
        {
            "id": "custom_3",
            "prompt": "Write a function to check if two strings are anagrams. An anagram is a word formed by rearranging the letters of another word, using all original letters exactly once.",
            "reference_solution": """def are_anagrams(s1, s2):
    # Remove spaces and convert to lowercase
    s1 = s1.lower().replace(" ", "")
    s2 = s2.lower().replace(" ", "")
    
    # Check if the sorted strings are equal
    return sorted(s1) == sorted(s2)""",
            "test_cases": [
                """
assert are_anagrams("listen", "silent") == True
assert are_anagrams("triangle", "integral") == True
assert are_anagrams("hello", "world") == False
assert are_anagrams("a gentleman", "elegant man") == True
assert are_anagrams("", "") == True
"""
            ],
            "difficulty": "easy",
            "tags": ["strings", "algorithms"],
        },
        {
            "id": "custom_4",
            "prompt": "Implement a function to reverse a linked list. The function should take the head of a linked list and return the head of the reversed list. The Node class is defined as: class Node: def __init__(self, val=0, next=None): self.val = val; self.next = next",
            "reference_solution": """def reverse_linked_list(head):
    prev = None
    current = head
    
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
        
    return prev""",
            "test_cases": [
                """
# Helper function to create a linked list from a list
def create_linked_list(lst):
    if not lst:
        return None
    head = Node(lst[0])
    current = head
    for val in lst[1:]:
        current.next = Node(val)
        current = current.next
    return head

# Helper function to convert a linked list to a list
def linked_list_to_list(head):
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result

# Test case 1: Simple list
head1 = create_linked_list([1, 2, 3, 4, 5])
reversed_head1 = reverse_linked_list(head1)
assert linked_list_to_list(reversed_head1) == [5, 4, 3, 2, 1]

# Test case 2: Single element
head2 = create_linked_list([1])
reversed_head2 = reverse_linked_list(head2)
assert linked_list_to_list(reversed_head2) == [1]

# Test case 3: Empty list
head3 = create_linked_list([])
reversed_head3 = reverse_linked_list(head3)
assert linked_list_to_list(reversed_head3) == []
"""
            ],
            "difficulty": "medium",
            "tags": ["linked lists", "data structures"],
        },
        {
            "id": "custom_5",
            "prompt": "Implement a function to find all prime numbers less than or equal to n using the Sieve of Eratosthenes algorithm.",
            "reference_solution": """def sieve_of_eratosthenes(n):
    if n < 2:
        return []
    
    # Initialize the sieve
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    
    # Mark non-primes
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    
    # Collect primes
    return [i for i in range(2, n + 1) if sieve[i]]""",
            "test_cases": [
                """
assert sieve_of_eratosthenes(10) == [2, 3, 5, 7]
assert sieve_of_eratosthenes(20) == [2, 3, 5, 7, 11, 13, 17, 19]
assert sieve_of_eratosthenes(1) == []
assert sieve_of_eratosthenes(2) == [2]
"""
            ],
            "difficulty": "medium",
            "tags": ["algorithms", "mathematics"],
        },
    ]
    
    problems = []
    for i, item in enumerate(custom_problems):
        if max_problems is not None and i >= max_problems:
            break
            
        problem = CodeProblem(
            id=item["id"],
            prompt=item["prompt"],
            reference_solution=item["reference_solution"],
            test_cases=item["test_cases"],
            difficulty=item.get("difficulty"),
            tags=item.get("tags"),
        )
        problems.append(problem)
    
    return problems


def load_dataset_by_name(name: str, max_problems: Optional[int] = None) -> List[CodeProblem]:
    """
    Load dataset by name.
    
    Args:
        name: Dataset name ("HumanEval", "APPS", or "custom")
        max_problems: Maximum number of problems to load (optional)
        
    Returns:
        List of CodeProblem objects
    """
    if name.lower() == "humaneval":
        return load_humaneval_dataset(max_problems)
    elif name.lower() == "apps":
        return load_apps_dataset(max_problems=max_problems)
    elif name.lower() == "custom":
        return create_custom_dataset(max_problems)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_dataloader(
    problems: List[CodeProblem], 
    tokenizer, 
    batch_size: int = 8, 
    shuffle: bool = True
) -> DataLoader:
    """
    Create a data loader for code generation problems.
    
    Args:
        problems: List of code problems
        tokenizer: Tokenizer for the model
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader for the dataset
    """
    dataset = CodeGenerationDataset(problems, tokenizer)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=lambda batch: {
            "id": [item["id"] for item in batch],
            "inputs": {
                k: torch.stack([item["inputs"][k] for item in batch]) 
                for k in batch[0]["inputs"]
            },
            "prompt": [item["prompt"] for item in batch],
            "reference_solution": [item["reference_solution"] for item in batch],
            "test_cases": [item["test_cases"] for item in batch],
        }
    )
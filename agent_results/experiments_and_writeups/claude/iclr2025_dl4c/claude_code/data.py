"""
Data preparation module for the adaptive code assistant experiment.
Includes developer profile simulation and code task dataset preparation.
"""

import os
import json
import random
import numpy as np
import torch
from datasets import load_dataset
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from utils import ensure_dir, save_json, load_json

logger = logging.getLogger("adaptive_code_assistant.data")

class DeveloperProfile:
    """
    Class to represent a developer profile with specific preferences and coding habits.
    These profiles will be used to simulate different developers interacting with the code assistant.
    """
    
    def __init__(
        self,
        dev_id: str,
        language_preferences: Dict[str, float] = None,
        formatting_preferences: Dict[str, Any] = None,
        interaction_style: Dict[str, float] = None,
        expertise_level: float = 0.5,
        feedback_frequency: float = 0.5
    ):
        """
        Initialize a developer profile.
        
        Args:
            dev_id: Unique identifier for the developer
            language_preferences: Preferences for programming languages (0.0 to 1.0)
            formatting_preferences: Code formatting preferences
            interaction_style: How the developer interacts with the assistant
            expertise_level: Developer's expertise level (0.0 to 1.0)
            feedback_frequency: How often the developer provides feedback (0.0 to 1.0)
        """
        self.dev_id = dev_id
        
        # Default language preferences if none provided
        self.language_preferences = language_preferences or {
            "python": 0.8,
            "javascript": 0.5,
            "java": 0.3,
            "c++": 0.2,
            "rust": 0.1
        }
        
        # Default formatting preferences if none provided
        self.formatting_preferences = formatting_preferences or {
            "indentation": {
                "style": random.choice(["spaces", "tabs"]),
                "width": random.choice([2, 4, 8])
            },
            "line_length": random.randint(79, 120),
            "docstring_style": random.choice(["google", "numpy", "sphinx"]),
            "variable_naming": random.choice(["snake_case", "camelCase"]),
            "brace_style": random.choice(["same_line", "new_line"])
        }
        
        # Default interaction style if none provided
        self.interaction_style = interaction_style or {
            "verbosity": random.uniform(0.1, 0.9),
            "patience": random.uniform(0.1, 0.9),
            "detail_orientation": random.uniform(0.1, 0.9),
            "innovation_preference": random.uniform(0.1, 0.9)
        }
        
        self.expertise_level = expertise_level
        self.feedback_frequency = feedback_frequency
        
        logger.info(f"Created developer profile: {dev_id}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "dev_id": self.dev_id,
            "language_preferences": self.language_preferences,
            "formatting_preferences": self.formatting_preferences,
            "interaction_style": self.interaction_style,
            "expertise_level": self.expertise_level,
            "feedback_frequency": self.feedback_frequency
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeveloperProfile':
        """Create profile from dictionary."""
        return cls(
            dev_id=data["dev_id"],
            language_preferences=data["language_preferences"],
            formatting_preferences=data["formatting_preferences"],
            interaction_style=data["interaction_style"],
            expertise_level=data["expertise_level"],
            feedback_frequency=data["feedback_frequency"]
        )
    
    def generate_feedback(self, code_assistant_response: str) -> Dict[str, Any]:
        """
        Generate simulated feedback based on the developer profile.
        
        Args:
            code_assistant_response: Response from the code assistant
            
        Returns:
            Dictionary containing feedback information
        """
        # Decide if the developer would provide feedback
        if random.random() > self.feedback_frequency:
            return {"provided": False}
        
        # Check if the response aligns with formatting preferences
        formatting_alignment = self._check_formatting_alignment(code_assistant_response)
        
        # Simulate satisfaction score
        satisfaction = self._calculate_satisfaction(code_assistant_response, formatting_alignment)
        
        # Generate textual feedback
        feedback_text = self._generate_feedback_text(satisfaction, formatting_alignment)
        
        return {
            "provided": True,
            "satisfaction": satisfaction,
            "feedback_text": feedback_text,
            "formatting_alignment": formatting_alignment
        }
    
    def _check_formatting_alignment(self, code: str) -> float:
        """Simulate checking if code matches formatting preferences."""
        # In a real implementation, this would analyze the code structure
        # For simulation, we'll use a random value skewed by the developer's preferences
        return random.uniform(0.3, 0.9)
    
    def _calculate_satisfaction(self, code: str, formatting_alignment: float) -> float:
        """Calculate simulated satisfaction based on code and formatting alignment."""
        # Combine multiple factors with different weights
        return 0.5 * formatting_alignment + 0.3 * random.random() + 0.2 * self.expertise_level
    
    def _generate_feedback_text(self, satisfaction: float, formatting_alignment: float) -> str:
        """Generate simulated textual feedback."""
        feedback_options = [
            "Please use my preferred indentation style.",
            "I prefer shorter variable names.",
            "Please include more comments.",
            "Use more descriptive function names.",
            "Organize imports differently.",
            "Use more or fewer blank lines.",
            "Use different parameter ordering.",
            "Include type hints.",
            "Make the code more concise.",
            "Add error handling."
        ]
        
        # Select 1-3 feedback items based on satisfaction level
        num_items = max(1, min(3, int((1 - satisfaction) * 5)))
        selected_feedback = random.sample(feedback_options, k=num_items)
        
        return " ".join(selected_feedback)

def generate_developer_profiles(num_profiles: int, save_path: str = None) -> List[DeveloperProfile]:
    """
    Generate multiple developer profiles for simulation.
    
    Args:
        num_profiles: Number of profiles to generate
        save_path: Optional path to save the profiles
        
    Returns:
        List of DeveloperProfile objects
    """
    profiles = []
    
    for i in range(num_profiles):
        dev_id = f"developer_{i+1}"
        expertise = random.uniform(0.1, 0.9)
        feedback_freq = random.uniform(0.2, 0.8)
        
        # Create random language preferences
        languages = ["python", "javascript", "java", "c++", "rust", "go", "csharp"]
        lang_prefs = {lang: random.uniform(0.1, 0.9) for lang in random.sample(languages, 4)}
        
        profile = DeveloperProfile(
            dev_id=dev_id,
            language_preferences=lang_prefs,
            expertise_level=expertise,
            feedback_frequency=feedback_freq
        )
        
        profiles.append(profile)
    
    # Save profiles if path provided
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        save_json(
            [profile.to_dict() for profile in profiles],
            save_path
        )
    
    logger.info(f"Generated {num_profiles} developer profiles")
    return profiles

class CodeTaskDataset:
    """
    Class to handle code task datasets for evaluation.
    Uses the HumanEval dataset from OpenAI or similar datasets.
    """
    
    def __init__(self, dataset_name: str = "openai/humaneval"):
        """
        Initialize the code task dataset.
        
        Args:
            dataset_name: Name of the dataset to load
        """
        try:
            self.dataset = load_dataset(dataset_name)
            logger.info(f"Loaded dataset: {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            # Create a mini version for experimentation
            self.dataset = self._create_mini_dataset()
            logger.info("Created mini dataset as fallback")
    
    def _create_mini_dataset(self) -> Dict:
        """Create a minimal dataset for experimentation."""
        mini_dataset = {
            "train": {
                "task_id": [f"mini_task_{i}" for i in range(10)],
                "prompt": [f"Write a function to {task}" for task in [
                    "calculate the factorial of a number",
                    "find the greatest common divisor of two numbers",
                    "check if a string is a palindrome",
                    "find the nth Fibonacci number",
                    "reverse a linked list",
                    "count words in a string",
                    "find the largest number in a list",
                    "check if a number is prime",
                    "convert decimal to binary",
                    "sort a list using bubble sort"
                ]],
                "canonical_solution": [
                    "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)",
                    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
                    "def is_palindrome(s):\n    return s == s[::-1]",
                    "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)",
                    "def reverse_linked_list(head):\n    prev = None\n    current = head\n    while current:\n        next_node = current.next\n        current.next = prev\n        prev = current\n        current = next_node\n    return prev",
                    "def count_words(s):\n    return len(s.split())",
                    "def find_largest(nums):\n    return max(nums)",
                    "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
                    "def decimal_to_binary(n):\n    return bin(n)[2:]",
                    "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"
                ],
                "test_cases": [
                    "assert factorial(5) == 120",
                    "assert gcd(48, 18) == 6",
                    "assert is_palindrome('racecar') == True",
                    "assert fibonacci(7) == 13",
                    "# Test cases for linked list omitted as they require a class definition",
                    "assert count_words('Hello world from Python') == 4",
                    "assert find_largest([3, 7, 2, 9, 1]) == 9",
                    "assert is_prime(17) == True",
                    "assert decimal_to_binary(10) == '1010'",
                    "assert bubble_sort([64, 34, 25, 12, 22, 11, 90]) == [11, 12, 22, 25, 34, 64, 90]"
                ]
            }
        }
        return mini_dataset
    
    def get_task(self, task_idx: int) -> Dict[str, str]:
        """
        Get a specific task from the dataset.
        
        Args:
            task_idx: Index of the task to retrieve
            
        Returns:
            Dictionary containing task information
        """
        if isinstance(self.dataset, dict):
            # Mini dataset
            if task_idx < len(self.dataset["train"]["task_id"]):
                return {
                    "task_id": self.dataset["train"]["task_id"][task_idx],
                    "prompt": self.dataset["train"]["prompt"][task_idx],
                    "canonical_solution": self.dataset["train"]["canonical_solution"][task_idx],
                    "test_cases": self.dataset["train"]["test_cases"][task_idx]
                }
            else:
                # Return a random task if index is out of bounds
                rand_idx = random.randint(0, len(self.dataset["train"]["task_id"]) - 1)
                return {
                    "task_id": self.dataset["train"]["task_id"][rand_idx],
                    "prompt": self.dataset["train"]["prompt"][rand_idx],
                    "canonical_solution": self.dataset["train"]["canonical_solution"][rand_idx],
                    "test_cases": self.dataset["train"]["test_cases"][rand_idx]
                }
        else:
            # HumanEval dataset
            if task_idx < len(self.dataset["train"]):
                item = self.dataset["train"][task_idx]
                return {
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "canonical_solution": item["canonical_solution"],
                    "test_cases": item["test"]  # Key name might differ based on dataset version
                }
            else:
                # Return a random task if index is out of bounds
                rand_idx = random.randint(0, len(self.dataset["train"]) - 1)
                item = self.dataset["train"][rand_idx]
                return {
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "canonical_solution": item["canonical_solution"],
                    "test_cases": item["test"]  # Key name might differ based on dataset version
                }
    
    def get_random_task(self) -> Dict[str, str]:
        """Get a random task from the dataset."""
        if isinstance(self.dataset, dict):
            idx = random.randint(0, len(self.dataset["train"]["task_id"]) - 1)
            return {
                "task_id": self.dataset["train"]["task_id"][idx],
                "prompt": self.dataset["train"]["prompt"][idx],
                "canonical_solution": self.dataset["train"]["canonical_solution"][idx],
                "test_cases": self.dataset["train"]["test_cases"][idx]
            }
        else:
            idx = random.randint(0, len(self.dataset["train"]) - 1)
            item = self.dataset["train"][idx]
            return {
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "canonical_solution": item["canonical_solution"],
                "test_cases": item["test"]  # Key name might differ based on dataset version
            }
    
    def get_n_tasks(self, n: int) -> List[Dict[str, str]]:
        """Get n random tasks from the dataset."""
        tasks = []
        for _ in range(n):
            tasks.append(self.get_random_task())
        return tasks
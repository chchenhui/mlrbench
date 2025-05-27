import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

class CodingTask:
    """Represents a single coding task with context and solution."""
    
    def __init__(self, task_id: str, context: str, solution: str, description: str = None, tags: List[str] = None):
        self.task_id = task_id
        self.context = context
        self.solution = solution
        self.description = description or ""
        self.tags = tags or []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'context': self.context,
            'solution': self.solution,
            'description': self.description,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodingTask':
        return cls(
            task_id=data['task_id'],
            context=data['context'],
            solution=data['solution'],
            description=data.get('description', ''),
            tags=data.get('tags', [])
        )

class DeveloperProfile:
    """Simulates a developer's preferences and coding style."""
    
    def __init__(self, profile_id: str, embedding_dim: int = 64):
        self.profile_id = profile_id
        self.embedding = np.zeros(embedding_dim)  # Initialize with zeros
        self.beta = 0.9  # Exponential moving average factor
        self.interaction_history = []
        
        # Simulate style preferences
        self.preferences = {
            'indent_style': np.random.choice(['spaces', 'tabs']),
            'indent_size': np.random.choice([2, 4]),
            'use_semicolons': np.random.choice([True, False]),
            'bracket_style': np.random.choice(['same_line', 'new_line']),
            'camel_case': np.random.choice([True, False]),
            'snake_case': np.random.choice([True, False]),
            'comment_style': np.random.choice(['inline', 'block', 'docstring']),
            'verbosity': np.random.uniform(0.2, 0.8)  # 0: terse, 1: verbose
        }
    
    def update_embedding(self, context, action, reward):
        """Update developer profile embedding based on interaction."""
        # Project the interaction to embedding space (simplified)
        interaction_embedding = np.random.randn(len(self.embedding))
        interaction_embedding = interaction_embedding / np.linalg.norm(interaction_embedding)
        
        # Apply exponential moving average update
        self.embedding = self.beta * self.embedding + (1 - self.beta) * interaction_embedding
        
        # Normalize embedding
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            self.embedding = self.embedding / norm
        
        # Record interaction
        self.interaction_history.append({
            'context': context[:50] + "..." if len(context) > 50 else context,  # Truncate for storage
            'action': action[:50] + "..." if len(action) > 50 else action,
            'reward': reward
        })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'profile_id': self.profile_id,
            'embedding': self.embedding.tolist(),
            'preferences': self.preferences
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeveloperProfile':
        profile = cls(profile_id=data['profile_id'])
        profile.embedding = np.array(data['embedding'])
        profile.preferences = data['preferences']
        return profile

def load_dataset(data_dir: str, dataset_name: str = None) -> Dataset:
    """
    Load dataset from local directory or Hugging Face.
    
    For our simulation, we'll use a sample of code from GitHub or load
    a pre-existing coding dataset like HumanEval or MBPP.
    """
    # Check if we have a local dataset
    local_data_path = os.path.join(data_dir, 'coding_tasks.json')
    
    if os.path.exists(local_data_path):
        # Load from local file
        with open(local_data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Convert to Hugging Face Dataset
        data_dict = {
            'task_id': [],
            'context': [],
            'solution': [],
            'description': [],
            'tags': []
        }
        
        for item in raw_data:
            task = CodingTask.from_dict(item)
            data_dict['task_id'].append(task.task_id)
            data_dict['context'].append(task.context)
            data_dict['solution'].append(task.solution)
            data_dict['description'].append(task.description)
            data_dict['tags'].append(task.tags)
        
        return Dataset.from_dict(data_dict)
    
    # If no local dataset, try to download from Hugging Face
    if dataset_name is None:
        # Default to HumanEval for code completion tasks
        dataset_name = "openai/humaneval"
    
    try:
        dataset = load_dataset(dataset_name)
        # For HumanEval, extract context (prompt) and solution (canonical_solution)
        if dataset_name == "openai/humaneval":
            data_dict = {
                'task_id': [f"task_{i}" for i in range(len(dataset['test']))],
                'context': dataset['test']['prompt'],
                'solution': dataset['test']['canonical_solution'],
                'description': dataset['test']['task_id'],
                'tags': [['python']] * len(dataset['test'])
            }
            return Dataset.from_dict(data_dict)
        return dataset
    except Exception as e:
        print(f"Failed to load dataset {dataset_name}: {e}")
        # Generate synthetic data if everything else fails
        return _generate_synthetic_dataset(100)

def _generate_synthetic_dataset(num_samples: int) -> Dataset:
    """Generate synthetic coding tasks for simulation purposes."""
    data_dict = {
        'task_id': [],
        'context': [],
        'solution': [],
        'description': [],
        'tags': []
    }
    
    # Common function templates
    function_templates = [
        ("Calculate factorial", 
         "def factorial(n):\n    # Calculate the factorial of n\n    ",
         "def factorial(n):\n    # Calculate the factorial of n\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)\n"),
        
        ("Find maximum in list", 
         "def find_max(numbers):\n    # Find the maximum value in the list\n    ",
         "def find_max(numbers):\n    # Find the maximum value in the list\n    if not numbers:\n        return None\n    max_val = numbers[0]\n    for num in numbers[1:]:\n        if num > max_val:\n            max_val = num\n    return max_val\n"),
        
        ("Check if string is palindrome", 
         "def is_palindrome(text):\n    # Check if the given text is a palindrome\n    ",
         "def is_palindrome(text):\n    # Check if the given text is a palindrome\n    text = text.lower()\n    cleaned_text = ''.join(c for c in text if c.isalnum())\n    return cleaned_text == cleaned_text[::-1]\n"),
        
        ("Count word frequency", 
         "def word_frequency(text):\n    # Count frequency of each word in the text\n    ",
         "def word_frequency(text):\n    # Count frequency of each word in the text\n    words = text.lower().split()\n    frequency = {}\n    for word in words:\n        if word in frequency:\n            frequency[word] += 1\n        else:\n            frequency[word] = 1\n    return frequency\n"),
        
        ("Find prime numbers up to n",
         "def find_primes(n):\n    # Find all prime numbers up to n\n    ",
         "def find_primes(n):\n    # Find all prime numbers up to n\n    primes = []\n    for i in range(2, n+1):\n        is_prime = True\n        for j in range(2, int(i**0.5) + 1):\n            if i % j == 0:\n                is_prime = False\n                break\n        if is_prime:\n            primes.append(i)\n    return primes\n")
    ]
    
    # Generate samples
    for i in range(num_samples):
        template_idx = i % len(function_templates)
        description, context, solution = function_templates[template_idx]
        
        data_dict['task_id'].append(f"synthetic_task_{i}")
        data_dict['context'].append(context)
        data_dict['solution'].append(solution)
        data_dict['description'].append(description)
        data_dict['tags'].append(['python', 'synthetic'])
    
    return Dataset.from_dict(data_dict)

def prepare_datasets(dataset: Dataset, seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train, validation, and test sets."""
    # Shuffle and split the dataset
    dataset = dataset.shuffle(seed=seed)
    
    # Calculate split sizes: 70% train, 15% validation, 15% test
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    
    # Create the splits
    train_dataset = dataset.select(range(train_size))
    valid_dataset = dataset.select(range(train_size, train_size + valid_size))
    test_dataset = dataset.select(range(train_size + valid_size, len(dataset)))
    
    return train_dataset, valid_dataset, test_dataset

def tokenize_code(code_text: str, tokenizer: AutoTokenizer, max_length: int = 512) -> Dict[str, torch.Tensor]:
    """Tokenize code text using the provided tokenizer."""
    return tokenizer(
        code_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

def simulate_developer_feedback(
    suggestion: str, 
    developer_profile: DeveloperProfile, 
    ground_truth: str = None
) -> Tuple[float, Dict[str, float]]:
    """
    Simulate developer feedback on code suggestions.
    
    Returns:
        Tuple of (total_reward, feedback_signals)
    """
    feedback = {}
    
    # 1. Acceptance signal (binary)
    # Higher probability of acceptance if suggestion matches developer preferences
    accept_prob = 0.7  # Base probability
    
    # Adjust based on style preferences
    if developer_profile.preferences['indent_style'] == 'spaces' and '    ' in suggestion:
        accept_prob += 0.1
    if developer_profile.preferences['indent_style'] == 'tabs' and '\t' in suggestion:
        accept_prob += 0.1
    
    if developer_profile.preferences['use_semicolons'] and ';' in suggestion:
        accept_prob += 0.05
    
    if developer_profile.preferences['bracket_style'] == 'same_line' and ') {' in suggestion:
        accept_prob += 0.05
    if developer_profile.preferences['bracket_style'] == 'new_line' and ')\n{' in suggestion:
        accept_prob += 0.05
    
    # Cap probability
    accept_prob = min(max(accept_prob, 0.1), 0.95)
    
    # Determine acceptance
    acceptance = np.random.random() < accept_prob
    feedback['accept'] = 1.0 if acceptance else 0.0
    
    # 2. Edit distance (normalized)
    if ground_truth is not None:
        # Calculate Levenshtein distance
        from Levenshtein import distance
        edit_dist = distance(suggestion, ground_truth)
        max_len = max(len(suggestion), len(ground_truth))
        normalized_edit_dist = 1.0 - (edit_dist / max_len if max_len > 0 else 0)
        feedback['edit_distance'] = normalized_edit_dist
    else:
        # If no ground truth, use a random value weighted by acceptance
        edit_dist_mean = 0.8 if acceptance else 0.3
        feedback['edit_distance'] = np.random.beta(edit_dist_mean * 10, (1-edit_dist_mean) * 10)
    
    # 3. Dwell time (normalized)
    # Simulate dwell time - longer for more complex code or bad suggestions
    suggestion_complexity = len(suggestion) / 100  # Simple complexity measure
    dwell_base = 0.3 + suggestion_complexity * 0.2
    dwell_random = np.random.beta(2, 5)  # Right-skewed distribution
    dwell_time = dwell_base + dwell_random
    
    # Normalize to [0, 1], where higher is better (less dwell time)
    dwell_norm = 1.0 - min(dwell_time, 1.0)
    feedback['dwell_time'] = dwell_norm
    
    # 4. Comment changes
    # Simulate likelihood of comment modification
    comment_change = 0.0
    if '# ' in suggestion or '"""' in suggestion:
        # If suggestion has comments, developer might modify them
        if np.random.random() < 0.4:
            comment_change = np.random.uniform(0.3, 0.7)
    feedback['comment_change'] = comment_change
    
    # Compute total reward as weighted sum of signals
    alpha1, alpha2, alpha3, alpha4 = 0.4, 0.3, 0.2, 0.1  # Weights for each signal
    total_reward = (
        alpha1 * feedback['accept'] + 
        alpha2 * feedback['edit_distance'] + 
        alpha3 * feedback['dwell_time'] + 
        alpha4 * feedback['comment_change']
    )
    
    return total_reward, feedback

def simulate_developer_interactions(
    model, 
    developer_profile: DeveloperProfile,
    tasks: List[CodingTask],
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Simulate interactions between a developer and the code assistant model.
    
    Args:
        model: The code suggestion model
        developer_profile: The simulated developer profile
        tasks: List of coding tasks to complete
        device: Device to run inference on
    
    Returns:
        Dict containing interaction metrics
    """
    metrics = {
        'acceptance_rate': 0.0,
        'avg_edit_distance': 0.0,
        'avg_reward': 0.0,
        'task_completion_times': [],
        'code_quality_scores': []
    }
    
    total_acceptance = 0
    total_edit_distance = 0.0
    total_reward = 0.0
    
    for task in tasks:
        # Generate suggestion for the task context
        suggestion = model.generate_suggestion(task.context, developer_profile, device)
        
        # Simulate developer feedback
        reward, feedback = simulate_developer_feedback(
            suggestion=suggestion,
            developer_profile=developer_profile,
            ground_truth=task.solution
        )
        
        # Update metrics
        total_acceptance += feedback['accept']
        total_edit_distance += feedback['edit_distance']
        total_reward += reward
        
        # Simulate task completion time (in seconds)
        # Baseline time + penalty for poor suggestions
        base_time = np.random.normal(300, 50)  # Mean 5 minutes with some variation
        suggestion_quality_factor = 0.5 + reward * 0.5  # Scale from 0.5 to 1.0
        completion_time = base_time / suggestion_quality_factor
        metrics['task_completion_times'].append(completion_time)
        
        # Simulate code quality (0-100 scale)
        base_quality = np.random.normal(70, 10)  # Base quality around 70/100
        quality_boost = reward * 20  # Up to 20 point boost for good suggestions
        code_quality = min(max(base_quality + quality_boost, 0), 100)
        metrics['code_quality_scores'].append(code_quality)
        
        # Update developer profile based on interaction
        developer_profile.update_embedding(task.context, suggestion, reward)
    
    # Compute average metrics
    num_tasks = len(tasks)
    if num_tasks > 0:
        metrics['acceptance_rate'] = total_acceptance / num_tasks
        metrics['avg_edit_distance'] = total_edit_distance / num_tasks
        metrics['avg_reward'] = total_reward / num_tasks
    
    return metrics
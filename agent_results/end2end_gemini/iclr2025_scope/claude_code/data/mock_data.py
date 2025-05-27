"""
Mock data for quick experiments and testing.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import random
import numpy as np
from transformers import AutoTokenizer


class MockTextClassificationDataset(Dataset):
    """
    Mock dataset for text classification tasks.
    Generates random token IDs and labels.
    """
    def __init__(
        self,
        tokenizer_name="distilbert-base-uncased",
        num_samples=100,
        seq_length=128,
        num_classes=2,
        seed=42
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_classes = num_classes
        
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Generate data
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate random data for the dataset."""
        data = []
        vocab_size = self.tokenizer.vocab_size
        
        for i in range(self.num_samples):
            # Generate random token IDs (with some structure to be more realistic)
            # Use a fixed short sequence to avoid OOM issues
            short_seq_len = min(16, self.seq_length - 2)

            # Start with CLS token
            input_ids = [self.tokenizer.cls_token_id]

            # Add random tokens for the sequence (from common token IDs)
            input_ids.extend(
                np.random.randint(100, min(1000, vocab_size - 100), size=short_seq_len).tolist()
            )

            # End with SEP token
            input_ids.append(self.tokenizer.sep_token_id)
            
            # Pad if needed (for consistency)
            if len(input_ids) < self.seq_length:
                input_ids.extend([self.tokenizer.pad_token_id] * (self.seq_length - len(input_ids)))
            
            # Create attention mask (1 for tokens, 0 for padding)
            attention_mask = [1 if id != self.tokenizer.pad_token_id else 0 for id in input_ids]
            
            # Random label
            label = random.randint(0, self.num_classes - 1)
            
            # Convert to tensors
            sample = {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(label)
            }
            
            data.append(sample)
        
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


class MockTaskGenerator:
    """
    Mock task generator for meta-learning and continual learning experiments.
    Creates random tasks without requiring network access.
    """
    def __init__(
        self,
        tokenizer_name="distilbert-base-uncased",
        num_classes=2,
        seq_length=128,
        seed=42
    ):
        self.tokenizer_name = tokenizer_name
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.base_seed = seed
    
    def create_meta_learning_tasks(
        self,
        n_tasks=10,
        n_examples_per_task=16,
        n_query_examples=16,
        seed=None
    ):
        """
        Create a set of tasks for meta-learning.
        Each task has support and query sets.
        """
        if seed is None:
            seed = self.base_seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        tasks = []
        
        for task_id in range(n_tasks):
            # Create a task-specific seed
            task_seed = seed + task_id
            
            # Create dataset for this task
            dataset = MockTextClassificationDataset(
                tokenizer_name=self.tokenizer_name,
                num_samples=n_examples_per_task + n_query_examples,
                seq_length=self.seq_length,
                num_classes=self.num_classes,
                seed=task_seed
            )
            
            # Split into support and query sets
            all_indices = list(range(len(dataset)))
            support_indices = all_indices[:n_examples_per_task]
            query_indices = all_indices[n_examples_per_task:]
            
            support_set = Subset(dataset, support_indices)
            query_set = Subset(dataset, query_indices)
            
            tasks.append({
                "support_set": support_set,
                "query_set": query_set,
                "dataset_name": f"mock_task_{task_id}",
                "metadata": {
                    "num_classes": self.num_classes,
                    "dataset_name": f"mock_task_{task_id}",
                    "text_field": "text",
                    "label_field": "label"
                }
            })
        
        return tasks
    
    def create_continual_learning_sequence(
        self,
        n_tasks=5,
        n_examples_per_task=100,
        validation_ratio=0.2,
        seed=None
    ):
        """
        Create a sequence of tasks for continual learning.
        Each task has a training and validation set.
        """
        if seed is None:
            seed = self.base_seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        task_sequence = []
        
        for task_id in range(n_tasks):
            # Create a task-specific seed
            task_seed = seed + task_id
            
            # Create dataset for this task
            dataset = MockTextClassificationDataset(
                tokenizer_name=self.tokenizer_name,
                num_samples=n_examples_per_task,
                seq_length=self.seq_length,
                num_classes=self.num_classes,
                seed=task_seed
            )
            
            # Split into train and validation sets
            all_indices = list(range(len(dataset)))
            n_val = int(n_examples_per_task * validation_ratio)
            val_indices = all_indices[:n_val]
            train_indices = all_indices[n_val:]
            
            train_set = Subset(dataset, train_indices)
            val_set = Subset(dataset, val_indices)
            
            task_sequence.append({
                "train_set": train_set,
                "val_set": val_set,
                "dataset_name": f"mock_task_{task_id}",
                "task_id": task_id,
                "metadata": {
                    "num_classes": self.num_classes,
                    "dataset_name": f"mock_task_{task_id}",
                    "text_field": "text",
                    "label_field": "label"
                }
            })
        
        return task_sequence
    
    def create_personalized_user_streams(
        self,
        n_users=5,
        n_tasks_per_user=3,
        n_examples_per_task=50,
        validation_ratio=0.2,
        seed=None
    ):
        """
        Create data streams for multiple simulated users.
        """
        if seed is None:
            seed = self.base_seed
        
        user_streams = {}
        
        for user_id in range(n_users):
            # Set a different seed for each user
            user_seed = seed + user_id * 100
            
            # Create a task sequence for this user
            task_sequence = self.create_continual_learning_sequence(
                n_tasks=n_tasks_per_user,
                n_examples_per_task=n_examples_per_task,
                validation_ratio=validation_ratio,
                seed=user_seed
            )
            
            user_streams[f"user_{user_id}"] = task_sequence
        
        return user_streams
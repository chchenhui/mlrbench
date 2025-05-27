"""
Datasets for task simulation in continual learning and meta-learning scenarios.
"""

import os
import torch
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


class TextClassificationTaskGenerator:
    """
    Generator for text classification tasks from various datasets.
    Simulates user/task streams for continual learning and meta-learning.
    """
    def __init__(
        self,
        tokenizer_name: str = "distilbert-base-uncased",
        max_seq_length: int = 128,
        cache_dir: str = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.cache_dir = cache_dir
        self._datasets = {}
    
    def _load_and_preprocess_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load and preprocess a dataset.
        """
        if dataset_name == "glue/sst2":
            dataset = load_dataset("glue", "sst2", cache_dir=self.cache_dir)
            text_field = "sentence"
            label_field = "label"
        
        elif dataset_name == "glue/mnli":
            dataset = load_dataset("glue", "mnli", cache_dir=self.cache_dir)
            text_field = "premise"  # Will combine with hypothesis
            label_field = "label"
        
        elif dataset_name == "ag_news":
            dataset = load_dataset("ag_news", cache_dir=self.cache_dir)
            text_field = "text"
            label_field = "label"
        
        elif dataset_name == "amazon_reviews":
            dataset = load_dataset("amazon_reviews_multi", "en", cache_dir=self.cache_dir)
            text_field = "review_body"
            label_field = "stars"  # 1-5 stars as classes
        
        elif dataset_name == "imdb":
            dataset = load_dataset("imdb", cache_dir=self.cache_dir)
            text_field = "text"
            label_field = "label"
        
        elif dataset_name == "tweet_eval":
            dataset = load_dataset("tweet_eval", "sentiment", cache_dir=self.cache_dir)
            text_field = "text"
            label_field = "label"
        
        elif dataset_name == "yelp_review_full":
            dataset = load_dataset("yelp_review_full", cache_dir=self.cache_dir)
            text_field = "text"
            label_field = "label"
        
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        def preprocess_function(examples):
            # Special case for MNLI with premise and hypothesis
            if dataset_name == "glue/mnli":
                texts = [
                    f"{examples['premise'][i]} [SEP] {examples['hypothesis'][i]}"
                    for i in range(len(examples[label_field]))
                ]
            else:
                texts = examples[text_field]
            
            encoding = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            
            # Convert to dict of tensors
            result = {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": torch.tensor(examples[label_field])
            }
            
            return result
        
        # Apply preprocessing
        tokenized_datasets = {}
        for split in dataset.keys():
            # Process in batches to avoid OOM
            tokenized_split = dataset[split].map(
                preprocess_function,
                batched=True,
                batch_size=1000,
                remove_columns=dataset[split].column_names
            )
            tokenized_datasets[split] = tokenized_split
        
        # Store dataset metadata
        metadata = {
            "num_classes": len(set(dataset["train"][label_field])),
            "dataset_name": dataset_name,
            "text_field": text_field,
            "label_field": label_field
        }
        
        return {
            "datasets": tokenized_datasets,
            "metadata": metadata
        }
    
    def get_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get a preprocessed dataset by name.
        """
        if dataset_name not in self._datasets:
            self._datasets[dataset_name] = self._load_and_preprocess_dataset(dataset_name)
        
        return self._datasets[dataset_name]
    
    def create_meta_learning_tasks(
        self,
        dataset_names: List[str],
        n_tasks: int = 10,
        n_examples_per_task: int = 16,
        n_query_examples: int = 16,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Create a set of tasks for meta-learning.
        Each task has support and query sets.
        
        Args:
            dataset_names: List of dataset names to sample from
            n_tasks: Number of tasks to create
            n_examples_per_task: Number of examples in the support set per task
            n_query_examples: Number of examples in the query set per task
            seed: Random seed for reproducibility
            
        Returns:
            List of tasks, each with support and query sets
        """
        random.seed(seed)
        np.random.seed(seed)
        
        tasks = []
        
        for _ in range(n_tasks):
            # Randomly select a dataset
            dataset_name = random.choice(dataset_names)
            data = self.get_dataset(dataset_name)
            dataset = data["datasets"]["train"]
            metadata = data["metadata"]
            
            # Randomly sample examples for support and query sets
            all_indices = list(range(len(dataset)))
            selected_indices = np.random.choice(
                all_indices,
                n_examples_per_task + n_query_examples,
                replace=False
            ).tolist()
            
            support_indices = selected_indices[:n_examples_per_task]
            query_indices = selected_indices[n_examples_per_task:]
            
            # Create support and query sets
            support_set = torch.utils.data.Subset(dataset, support_indices)
            query_set = torch.utils.data.Subset(dataset, query_indices)
            
            tasks.append({
                "support_set": support_set,
                "query_set": query_set,
                "dataset_name": dataset_name,
                "metadata": metadata
            })
        
        return tasks
    
    def create_continual_learning_sequence(
        self,
        dataset_names: List[str],
        n_tasks: int = 5,
        n_examples_per_task: int = 100,
        validation_ratio: float = 0.2,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Create a sequence of tasks for continual learning.
        Each task has a training and validation set.
        
        Args:
            dataset_names: List of dataset names to sample from
            n_tasks: Number of tasks in the sequence
            n_examples_per_task: Number of examples per task
            validation_ratio: Ratio of examples to use for validation
            seed: Random seed for reproducibility
            
        Returns:
            List of tasks in sequence, each with train and validation sets
        """
        random.seed(seed)
        np.random.seed(seed)
        
        task_sequence = []
        
        for task_id in range(n_tasks):
            # For more diverse task sequences, use a different dataset for each task
            if task_id < len(dataset_names):
                dataset_name = dataset_names[task_id]
            else:
                dataset_name = random.choice(dataset_names)
            
            data = self.get_dataset(dataset_name)
            dataset = data["datasets"]["train"]
            metadata = data["metadata"]
            
            # Randomly sample examples for this task
            all_indices = list(range(len(dataset)))
            selected_indices = np.random.choice(
                all_indices,
                n_examples_per_task,
                replace=False
            ).tolist()
            
            # Split into train and validation
            n_val = int(n_examples_per_task * validation_ratio)
            val_indices = selected_indices[:n_val]
            train_indices = selected_indices[n_val:]
            
            # Create train and validation sets
            train_set = torch.utils.data.Subset(dataset, train_indices)
            val_set = torch.utils.data.Subset(dataset, val_indices)
            
            task_sequence.append({
                "train_set": train_set,
                "val_set": val_set,
                "dataset_name": dataset_name,
                "task_id": task_id,
                "metadata": metadata
            })
        
        return task_sequence
    
    def create_personalized_user_streams(
        self,
        dataset_names: List[str],
        n_users: int = 5,
        n_tasks_per_user: int = 3,
        n_examples_per_task: int = 50,
        validation_ratio: float = 0.2,
        seed: int = 42
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create data streams for multiple simulated users.
        Each user has a sequence of tasks representing their personalized data stream.
        
        Args:
            dataset_names: List of dataset names to sample from
            n_users: Number of users to simulate
            n_tasks_per_user: Number of tasks in each user's stream
            n_examples_per_task: Number of examples per task
            validation_ratio: Ratio of examples to use for validation
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary mapping user IDs to their task sequences
        """
        user_streams = {}
        
        for user_id in range(n_users):
            # Set a different seed for each user to ensure diversity
            user_seed = seed + user_id
            
            # Create a task sequence for this user
            task_sequence = self.create_continual_learning_sequence(
                dataset_names=dataset_names,
                n_tasks=n_tasks_per_user,
                n_examples_per_task=n_examples_per_task,
                validation_ratio=validation_ratio,
                seed=user_seed
            )
            
            user_streams[f"user_{user_id}"] = task_sequence
        
        return user_streams


class MetaLearningDataset(Dataset):
    """
    Dataset wrapper for meta-learning tasks.
    """
    def __init__(self, tasks):
        self.tasks = tasks
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        task = self.tasks[idx]
        return {
            "support_set": task["support_set"],
            "query_set": task["query_set"],
            "metadata": task["metadata"]
        }


def create_meta_batch_collator(tokenizer_pad_token_id=0):
    """
    Create a collator function for meta-learning batches.
    """
    def collate_meta_batch(batch):
        """
        Collate function for meta-learning batches.
        Each batch contains a single task with support and query sets.
        """
        support_set = batch[0]["support_set"]
        query_set = batch[0]["query_set"]
        
        # Collate support set
        support_input_ids = torch.stack([item["input_ids"] for item in support_set])
        support_attention_mask = torch.stack([item["attention_mask"] for item in support_set])
        support_labels = torch.stack([item["labels"] for item in support_set])
        
        # Collate query set
        query_input_ids = torch.stack([item["input_ids"] for item in query_set])
        query_attention_mask = torch.stack([item["attention_mask"] for item in query_set])
        query_labels = torch.stack([item["labels"] for item in query_set])
        
        return {
            "support": {
                "input_ids": support_input_ids,
                "attention_mask": support_attention_mask,
                "labels": support_labels
            },
            "query": {
                "input_ids": query_input_ids,
                "attention_mask": query_attention_mask,
                "labels": query_labels
            },
            "metadata": batch[0]["metadata"]
        }
    
    return collate_meta_batch
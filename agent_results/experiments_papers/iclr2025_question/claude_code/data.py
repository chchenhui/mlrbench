"""
Data utilities for loading and preprocessing datasets.
"""

import torch
from datasets import load_dataset
from typing import Dict, List, Any, Optional, Union, Tuple
from tqdm import tqdm

class DataProcessor:
    """Data processor for loading and preprocessing datasets."""
    
    def __init__(self, tokenizer, max_length=512, dataset_config=None):
        """
        Initialize the data processor.
        
        Args:
            tokenizer: The tokenizer to use for preprocessing.
            max_length: The maximum sequence length.
            dataset_config: The configuration for the dataset.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_config = dataset_config or {}
        
    def load_dataset(self, dataset_name, split="validation", cache_dir=None):
        """
        Load a dataset from the Hugging Face Datasets library.
        
        Args:
            dataset_name: The name of the dataset to load.
            split: The split of the dataset to load.
            cache_dir: The directory to cache the dataset.
        
        Returns:
            The loaded dataset.
        """
        return load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    
    def preprocess_squad(self, dataset, max_samples=None):
        """
        Preprocess a SQuAD dataset for question answering.
        
        Args:
            dataset: The SQuAD dataset to preprocess.
            max_samples: The maximum number of samples to process.
        
        Returns:
            A preprocessed dataset with questions and contexts.
        """
        questions = []
        contexts = []
        answers = []
        
        for i, example in enumerate(tqdm(dataset, desc="Preprocessing SQuAD")):
            if max_samples is not None and i >= max_samples:
                break
                
            # Get the question and context
            question = example["question"]
            context = example["context"]
            
            # Get the answers
            if "answers" in example:
                answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
            else:
                answer = ""
                
            questions.append(question)
            contexts.append(context)
            answers.append(answer)
            
        # Tokenize the questions and contexts
        encoded_questions = self.tokenizer(
            questions,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize the answers
        encoded_answers = self.tokenizer(
            answers,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "questions": questions,
            "contexts": contexts,
            "answers": answers,
            "encoded_questions": encoded_questions,
            "encoded_answers": encoded_answers,
        }
    
    def preprocess_xsum(self, dataset, max_samples=None):
        """
        Preprocess an XSum dataset for summarization.
        
        Args:
            dataset: The XSum dataset to preprocess.
            max_samples: The maximum number of samples to process.
        
        Returns:
            A preprocessed dataset with documents and summaries.
        """
        documents = []
        summaries = []
        
        for i, example in enumerate(tqdm(dataset, desc="Preprocessing XSum")):
            if max_samples is not None and i >= max_samples:
                break
                
            # Get the document and summary
            document = example["document"]
            summary = example["summary"]
                
            documents.append(document)
            summaries.append(summary)
            
        # Tokenize the documents
        encoded_documents = self.tokenizer(
            documents,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize the summaries
        encoded_summaries = self.tokenizer(
            summaries,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "documents": documents,
            "summaries": summaries,
            "encoded_documents": encoded_documents,
            "encoded_summaries": encoded_summaries,
        }
    
    def preprocess_dataset(self, dataset_name, split="validation", max_samples=None, cache_dir=None):
        """
        Preprocess a dataset.
        
        Args:
            dataset_name: The name of the dataset to preprocess.
            split: The split of the dataset to preprocess.
            max_samples: The maximum number of samples to process.
            cache_dir: The directory to cache the dataset.
        
        Returns:
            A preprocessed dataset.
        """
        # Load the dataset
        dataset = self.load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        
        # Preprocess the dataset based on its type
        if "squad" in dataset_name:
            return self.preprocess_squad(dataset, max_samples=max_samples)
        elif "xsum" in dataset_name:
            return self.preprocess_xsum(dataset, max_samples=max_samples)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
            
    def prepare_batch(self, batch, input_key, target_key=None):
        """
        Prepare a batch for training or evaluation.
        
        Args:
            batch: The batch to prepare.
            input_key: The key for the input data.
            target_key: The key for the target data.
        
        Returns:
            A dictionary with the prepared batch.
        """
        inputs = batch[input_key]
        targets = batch[target_key] if target_key else None
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "target_ids": targets["input_ids"] if targets else None,
            "target_attention_mask": targets["attention_mask"] if targets else None,
        }
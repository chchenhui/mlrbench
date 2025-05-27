#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data processing module for the LLM-TAC experiment.
This module handles loading, processing, and splitting the Coq proof dataset.
"""

import os
import logging
import random
import json
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from sklearn.model_selection import train_test_split

from utils import parse_tactic_sequence

logger = logging.getLogger(__name__)

class CoqDataProcessor:
    """Class for processing Coq proof data."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the CoqDataProcessor.
        
        Args:
            data_dir: Directory containing the Coq proof data
        """
        self.data_dir = data_dir
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load the Coq proof data.
        
        For this experiment, we'll use a synthetic dataset that represents
        Coq proof states and tactics.
        
        Returns:
            List of raw data examples
        """
        logger.info("Loading Coq proof data")
        
        # For the experiment, we'll create a synthetic dataset
        # In a real-world scenario, this would load from actual Coq files
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        synthetic_data_path = os.path.join(self.data_dir, "synthetic_coq_data.json")
        
        # If synthetic data already exists, load it
        if os.path.exists(synthetic_data_path):
            logger.info(f"Loading existing synthetic data from {synthetic_data_path}")
            with open(synthetic_data_path, 'r') as f:
                self.raw_data = json.load(f)
            return self.raw_data
        
        # Otherwise, create synthetic data
        logger.info("Creating synthetic Coq proof data")
        self.raw_data = self._create_synthetic_data(n_examples=500)
        
        # Save synthetic data
        with open(synthetic_data_path, 'w') as f:
            json.dump(self.raw_data, f, indent=2)
        
        return self.raw_data
    
    def _create_synthetic_data(self, n_examples: int) -> List[Dict[str, Any]]:
        """
        Create synthetic data for Coq proofs.
        
        Args:
            n_examples: Number of examples to create
            
        Returns:
            List of synthetic data examples
        """
        # Make sure we have enough examples for splitting
        n_examples = max(n_examples, 50)
        
        # Templates for generating synthetic Coq proof data
        goal_templates = [
            "forall n m : nat, n + m = m + n",
            "forall n : nat, n + 0 = n",
            "forall (A B : Prop), A /\\ B -> B /\\ A",
            "forall (A B C : Prop), (A -> B -> C) -> (A /\\ B -> C)",
            "forall (A B : Prop), A \\/ B -> B \\/ A",
            "forall (A : Type) (x y : A), x = y -> y = x",
            "forall (A : Type) (x y z : A), x = y -> y = z -> x = z",
            "forall n m : nat, n <= m -> m >= n",
            "forall n : nat, n * 0 = 0",
            "forall n m : nat, n * (m + 1) = n * m + n"
        ]
        
        hypothesis_templates = [
            "H: n = m",
            "H: forall x : nat, x + 0 = x",
            "H: a /\\ b",
            "H: a \\/ b",
            "H: a -> b",
            "H: n <= m",
            "H: exists x, P x",
            "H: ~ a",
            "H: a <-> b",
            "H: f x = y"
        ]
        
        tactic_templates = [
            "intros",
            "apply",
            "simpl",
            "rewrite",
            "assert",
            "destruct",
            "induction",
            "unfold",
            "split",
            "exists",
            "reflexivity",
            "auto",
            "tauto"
        ]
        
        library_templates = [
            "Require Import Coq.Arith.Arith.",
            "Require Import Coq.Bool.Bool.",
            "Require Import Coq.Lists.List.",
            "Require Import Coq.Logic.Classical.",
            "Require Import Coq.Relations.Relation_Definitions.",
            "Require Import Coq.Sets.Ensembles.",
            "Require Import Coq.Strings.String.",
            "Require Import Coq.ZArith.ZArith."
        ]
        
        synthetic_data = []
        
        for i in range(n_examples):
            # Randomly choose a goal
            goal = random.choice(goal_templates)
            
            # Randomly choose 0-3 hypotheses
            n_hypotheses = random.randint(0, 3)
            hypotheses = random.sample(hypothesis_templates, n_hypotheses)
            
            # Randomly choose 3-7 tactics
            n_tactics = random.randint(3, 7)
            tactics = []
            for _ in range(n_tactics):
                tactic = random.choice(tactic_templates)
                # Add some arguments to make it more realistic
                if tactic == "apply":
                    tactic += " H"
                elif tactic == "destruct":
                    tactic += " H as [H1 H2]"
                elif tactic == "induction":
                    tactic += " n"
                elif tactic == "rewrite":
                    tactic += " H"
                
                tactics.append(tactic)
            
            # Randomly choose 1-3 library imports
            n_libraries = random.randint(1, 3)
            libraries = random.sample(library_templates, n_libraries)
            
            example = {
                "id": f"theorem_{i}",
                "goal": goal,
                "hypotheses": hypotheses,
                "tactics": tactics,
                "libraries": libraries,
                "difficulty": random.choice(["easy", "medium", "hard"]),
                "domain": random.choice(["arithmetic", "logic", "lists", "relations"])
            }
            
            synthetic_data.append(example)
        
        return synthetic_data
    
    def process_data(self) -> List[Dict[str, Any]]:
        """
        Process the raw data into a format suitable for model training.
        
        Returns:
            List of processed data examples
        """
        logger.info("Processing Coq proof data")
        
        if self.raw_data is None:
            self.raw_data = self.load_data()
        
        processed_data = []
        
        for example in self.raw_data:
            # For each tactic in the sequence, create an example with the current state and next tactic
            tactics = example["tactics"]
            
            for i, tactic in enumerate(tactics):
                # The goal state and hypotheses remain the same throughout the proof
                # In a real scenario, these would change after each tactic application
                goal_state = example["goal"]
                hypotheses = example["hypotheses"]
                
                # Previous tactics (context)
                prev_tactics = tactics[:i] if i > 0 else []
                
                # Next tactic (target)
                next_tactic = tactic
                
                # Remaining tactics
                remaining_tactics = tactics[i+1:] if i < len(tactics) - 1 else []
                
                processed_example = {
                    "id": f"{example['id']}_{i}",
                    "goal_state": goal_state,
                    "hypotheses": hypotheses,
                    "libraries": example["libraries"],
                    "prev_tactics": prev_tactics,
                    "next_tactic": next_tactic,
                    "remaining_tactics": remaining_tactics,
                    "difficulty": example["difficulty"],
                    "domain": example["domain"]
                }
                
                processed_data.append(processed_example)
        
        self.processed_data = processed_data
        return processed_data
    
    def process_and_split_data(self, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process and split the data into training, validation, and test sets.
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of remaining data for validation
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if self.processed_data is None:
            self.processed_data = self.process_data()
        
        # Extract unique theorem IDs (before the underscore)
        theorem_ids = list(set([ex["id"].split("_")[0] for ex in self.processed_data]))
        
        # Handle small datasets by ensuring minimum sizes for each split
        if len(theorem_ids) < 10:
            # Manual split for very small datasets
            test_size = min(0.2, 1/len(theorem_ids))
            val_size = min(0.1, 1/len(theorem_ids))
        
        # Split at the theorem level to avoid data leakage
        test_count = max(1, int(len(theorem_ids) * test_size))
        val_count = max(1, int(len(theorem_ids) * val_size))
        train_count = len(theorem_ids) - test_count - val_count
        
        # Ensure we have at least one theorem in each split
        if train_count < 1:
            # Redistribute if needed
            if test_count > 1:
                test_count -= 1
                train_count += 1
            elif val_count > 1:
                val_count -= 1
                train_count += 1
        
        # Shuffle the theorem IDs
        np.random.seed(42)
        np.random.shuffle(theorem_ids)
        
        # Assign to splits
        test_theorem_ids = theorem_ids[:test_count]
        val_theorem_ids = theorem_ids[test_count:test_count+val_count]
        train_theorem_ids = theorem_ids[test_count+val_count:]
        
        # Assign examples to splits based on theorem ID
        train_data = [ex for ex in self.processed_data if ex["id"].split("_")[0] in train_theorem_ids]
        val_data = [ex for ex in self.processed_data if ex["id"].split("_")[0] in val_theorem_ids]
        test_data = [ex for ex in self.processed_data if ex["id"].split("_")[0] in test_theorem_ids]
        
        logger.info(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test examples")
        
        return train_data, val_data, test_data
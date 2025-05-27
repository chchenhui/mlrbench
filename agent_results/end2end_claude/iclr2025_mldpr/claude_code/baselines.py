"""
Baseline Methods for Contextual Dataset Deprecation Framework

This module implements baseline methods for dataset deprecation to compare against
the proposed Contextual Dataset Deprecation Framework:

1. Control (Traditional): Simple removal of datasets without structured deprecation
2. Basic Framework: Implementation with only warning labels and basic notifications

These baselines represent current practices in dataset repositories.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import time
import random
from typing import List, Dict, Any, Tuple, Optional, Set, Union

from experimental_design import WarningLevel, DeprecationRecord, SyntheticDataset, DatasetVersion, DeprecationStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log.txt')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("baseline_methods")

class TraditionalDeprecation:
    """
    Implementation of traditional dataset deprecation (control condition).
    
    This baseline represents the current practice of simply removing datasets
    without structured deprecation processes or alternative recommendations.
    """
    
    def __init__(
        self,
        datasets: Dict[str, SyntheticDataset] = None,
        data_dir: str = None
    ):
        self.strategy = DeprecationStrategy.CONTROL
        self.datasets = datasets or {}
        self.removed_datasets = {}
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'baseline_data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Tracking for evaluation
        self.removal_history = []
        self.access_attempts = []
        
        logger.info("Initialized TraditionalDeprecation baseline")
    
    def remove_dataset(self, dataset_id: str, reason: str = None) -> bool:
        """
        Remove a dataset completely.
        
        Args:
            dataset_id: ID of the dataset to remove
            reason: Optional reason for removal
            
        Returns:
            True if dataset was removed, False otherwise
        """
        if dataset_id not in self.datasets:
            logger.warning(f"Dataset {dataset_id} not found")
            return False
        
        # Store removed dataset
        self.removed_datasets[dataset_id] = self.datasets[dataset_id]
        
        # Record removal
        removal_record = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "reason": reason or "Unspecified reason"
        }
        self.removal_history.append(removal_record)
        
        # Remove from active datasets
        del self.datasets[dataset_id]
        
        logger.info(f"Removed dataset {dataset_id}")
        return True
    
    def access_dataset(self, dataset_id: str) -> Tuple[bool, Optional[SyntheticDataset]]:
        """
        Attempt to access a dataset.
        
        Args:
            dataset_id: ID of the dataset to access
            
        Returns:
            Tuple of (success, dataset)
        """
        # Check if dataset exists
        if dataset_id in self.datasets:
            self.access_attempts.append({
                "dataset_id": dataset_id,
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "removed": False
            })
            logger.debug(f"Successful access to dataset {dataset_id}")
            return True, self.datasets[dataset_id]
        
        # Check if dataset was removed
        if dataset_id in self.removed_datasets:
            self.access_attempts.append({
                "dataset_id": dataset_id,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "removed": True
            })
            logger.debug(f"Failed access to removed dataset {dataset_id}")
            return False, None
        
        # Dataset doesn't exist
        self.access_attempts.append({
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "removed": False
        })
        logger.debug(f"Failed access to non-existent dataset {dataset_id}")
        return False, None
    
    def find_alternative(self, dataset_id: str) -> List[str]:
        """
        Find alternative datasets for a removed dataset.
        
        In the traditional approach, this usually requires manual search by users
        without structured recommendations.
        
        Args:
            dataset_id: ID of the removed dataset
            
        Returns:
            List of potential alternative dataset IDs (empty for this baseline)
        """
        # Traditional approach doesn't provide alternatives
        return []
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the traditional approach.
        
        Returns:
            Dictionary of evaluation metrics
        """
        results = {
            "removed_count": len(self.removed_datasets),
            "access_attempts": len(self.access_attempts),
            "unsuccessful_access_rate": 0.0,
            "removed_access_attempts": 0,
            "alternatives_provided": 0
        }
        
        # Calculate unsuccessful access rate
        if self.access_attempts:
            unsuccessful = sum(1 for attempt in self.access_attempts if not attempt["success"])
            results["unsuccessful_access_rate"] = unsuccessful / len(self.access_attempts)
        
        # Count attempts to access removed datasets
        results["removed_access_attempts"] = sum(
            1 for attempt in self.access_attempts if attempt["removed"]
        )
        
        return results
    
    def save_evaluation_data(self, output_dir: str = None) -> str:
        """
        Save evaluation data to disk.
        
        Args:
            output_dir: Directory to save data to. If None, uses the data_dir.
            
        Returns:
            Path to the saved data directory
        """
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, f"traditional_{int(time.time())}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save removal history
        with open(os.path.join(output_dir, "removal_history.json"), "w") as f:
            json.dump(self.removal_history, f, indent=2)
        
        # Save access attempts
        with open(os.path.join(output_dir, "access_attempts.json"), "w") as f:
            json.dump(self.access_attempts, f, indent=2)
        
        # Save evaluation results
        evaluation = self.evaluate()
        with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
            json.dump(evaluation, f, indent=2)
        
        logger.info(f"Saved evaluation data to {output_dir}")
        return output_dir


class BasicDeprecation:
    """
    Implementation of a basic deprecation framework (baseline condition).
    
    This baseline represents a simple improvement over the traditional approach,
    with basic warning labels and notifications but minimal structure.
    """
    
    def __init__(
        self,
        datasets: Dict[str, SyntheticDataset] = None,
        data_dir: str = None
    ):
        self.strategy = DeprecationStrategy.BASIC
        self.datasets = datasets or {}
        self.warning_labels = {}  # dataset_id -> warning_level
        self.removed_datasets = {}
        self.alternatives = {}  # dataset_id -> list of alternative dataset_ids
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'baseline_data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Tracking for evaluation
        self.warning_history = []
        self.removal_history = []
        self.access_attempts = []
        self.alternative_views = []
        
        logger.info("Initialized BasicDeprecation baseline")
    
    def apply_warning(
        self, 
        dataset_id: str, 
        warning_level: WarningLevel,
        reason: str = None
    ) -> bool:
        """
        Apply a warning label to a dataset.
        
        Args:
            dataset_id: ID of the dataset
            warning_level: Warning level to apply
            reason: Optional reason for the warning
            
        Returns:
            True if warning was applied, False otherwise
        """
        if dataset_id not in self.datasets:
            logger.warning(f"Dataset {dataset_id} not found")
            return False
        
        # Record warning
        warning_record = {
            "dataset_id": dataset_id,
            "warning_level": warning_level.name,
            "timestamp": datetime.now().isoformat(),
            "reason": reason or "Unspecified reason"
        }
        self.warning_history.append(warning_record)
        
        # Update warning label
        self.warning_labels[dataset_id] = warning_level
        
        logger.info(f"Applied warning level {warning_level.name} to dataset {dataset_id}")
        return True
    
    def remove_dataset(
        self, 
        dataset_id: str, 
        reason: str = None,
        alternatives: List[str] = None
    ) -> bool:
        """
        Remove a dataset with basic alternatives.
        
        Args:
            dataset_id: ID of the dataset to remove
            reason: Optional reason for removal
            alternatives: Optional list of alternative dataset IDs
            
        Returns:
            True if dataset was removed, False otherwise
        """
        if dataset_id not in self.datasets:
            logger.warning(f"Dataset {dataset_id} not found")
            return False
        
        # Store removed dataset
        self.removed_datasets[dataset_id] = self.datasets[dataset_id]
        
        # Store alternatives
        if alternatives:
            self.alternatives[dataset_id] = alternatives
        
        # Record removal
        removal_record = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "reason": reason or "Unspecified reason",
            "alternatives": alternatives or []
        }
        self.removal_history.append(removal_record)
        
        # Remove from active datasets
        del self.datasets[dataset_id]
        
        # Remove warning label if it exists
        if dataset_id in self.warning_labels:
            del self.warning_labels[dataset_id]
        
        logger.info(f"Removed dataset {dataset_id}")
        return True
    
    def access_dataset(self, dataset_id: str) -> Tuple[bool, Optional[SyntheticDataset], Optional[WarningLevel]]:
        """
        Attempt to access a dataset.
        
        Args:
            dataset_id: ID of the dataset to access
            
        Returns:
            Tuple of (success, dataset, warning_level)
        """
        # Check if dataset exists
        if dataset_id in self.datasets:
            # Get warning level if it exists
            warning_level = self.warning_labels.get(dataset_id)
            
            self.access_attempts.append({
                "dataset_id": dataset_id,
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "removed": False,
                "warning_level": warning_level.name if warning_level else "NONE"
            })
            
            logger.debug(f"Successful access to dataset {dataset_id} with warning level {warning_level.name if warning_level else 'NONE'}")
            return True, self.datasets[dataset_id], warning_level
        
        # Check if dataset was removed
        if dataset_id in self.removed_datasets:
            self.access_attempts.append({
                "dataset_id": dataset_id,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "removed": True,
                "warning_level": "REMOVED"
            })
            logger.debug(f"Failed access to removed dataset {dataset_id}")
            return False, None, None
        
        # Dataset doesn't exist
        self.access_attempts.append({
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "removed": False,
            "warning_level": None
        })
        logger.debug(f"Failed access to non-existent dataset {dataset_id}")
        return False, None, None
    
    def get_alternatives(self, dataset_id: str) -> List[str]:
        """
        Get alternative datasets for a deprecated or removed dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            List of alternative dataset IDs
        """
        alternatives = self.alternatives.get(dataset_id, [])
        
        if alternatives:
            self.alternative_views.append({
                "dataset_id": dataset_id,
                "timestamp": datetime.now().isoformat(),
                "alternatives_count": len(alternatives)
            })
        
        return alternatives
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the basic approach.
        
        Returns:
            Dictionary of evaluation metrics
        """
        results = {
            "warned_count": len(self.warning_labels),
            "removed_count": len(self.removed_datasets),
            "access_attempts": len(self.access_attempts),
            "unsuccessful_access_rate": 0.0,
            "warned_access_count": 0,
            "alternatives_provided": sum(len(alts) for alts in self.alternatives.values()),
            "alternative_views": len(self.alternative_views)
        }
        
        # Calculate unsuccessful access rate
        if self.access_attempts:
            unsuccessful = sum(1 for attempt in self.access_attempts if not attempt["success"])
            results["unsuccessful_access_rate"] = unsuccessful / len(self.access_attempts)
        
        # Count accesses to warned datasets
        results["warned_access_count"] = sum(
            1 for attempt in self.access_attempts 
            if attempt["success"] and attempt.get("warning_level") and attempt["warning_level"] != "NONE"
        )
        
        return results
    
    def save_evaluation_data(self, output_dir: str = None) -> str:
        """
        Save evaluation data to disk.
        
        Args:
            output_dir: Directory to save data to. If None, uses the data_dir.
            
        Returns:
            Path to the saved data directory
        """
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, f"basic_{int(time.time())}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save warning history
        with open(os.path.join(output_dir, "warning_history.json"), "w") as f:
            json.dump(self.warning_history, f, indent=2)
        
        # Save removal history
        with open(os.path.join(output_dir, "removal_history.json"), "w") as f:
            json.dump(self.removal_history, f, indent=2)
        
        # Save access attempts
        with open(os.path.join(output_dir, "access_attempts.json"), "w") as f:
            json.dump(self.access_attempts, f, indent=2)
        
        # Save alternative views
        with open(os.path.join(output_dir, "alternative_views.json"), "w") as f:
            json.dump(self.alternative_views, f, indent=2)
        
        # Save evaluation results
        evaluation = self.evaluate()
        with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
            json.dump(evaluation, f, indent=2)
        
        logger.info(f"Saved evaluation data to {output_dir}")
        return output_dir

def run_traditional_simulation(
    datasets: Dict[str, SyntheticDataset],
    deprecation_records: Dict[str, DeprecationRecord],
    n_accesses: int = 100
) -> TraditionalDeprecation:
    """
    Run a simulation of the traditional dataset deprecation approach.
    
    Args:
        datasets: Dictionary of synthetic datasets
        deprecation_records: Dictionary of deprecation records
        n_accesses: Number of simulated access attempts
        
    Returns:
        The simulation instance
    """
    logger.info("Starting traditional deprecation simulation")
    
    # Create a copy of datasets to avoid modifying the original
    datasets_copy = {k: v for k, v in datasets.items()}
    
    # Initialize traditional deprecation
    traditional = TraditionalDeprecation(datasets=datasets_copy)
    
    # Remove datasets that have deprecation records with high warning levels
    for dataset_id, record in deprecation_records.items():
        if record.warning_level in [WarningLevel.LIMITED_USE, WarningLevel.DEPRECATED]:
            if dataset_id in datasets_copy:
                traditional.remove_dataset(
                    dataset_id=dataset_id,
                    reason=record.issue_description
                )
    
    # Simulate random access attempts
    all_dataset_ids = list(datasets.keys())  # Use original datasets to include removed ones
    for _ in range(n_accesses):
        dataset_id = random.choice(all_dataset_ids)
        traditional.access_dataset(dataset_id)
    
    # Save evaluation data
    traditional.save_evaluation_data()
    
    logger.info("Completed traditional deprecation simulation")
    return traditional

def run_basic_simulation(
    datasets: Dict[str, SyntheticDataset],
    deprecation_records: Dict[str, DeprecationRecord],
    n_accesses: int = 100
) -> BasicDeprecation:
    """
    Run a simulation of the basic dataset deprecation approach.
    
    Args:
        datasets: Dictionary of synthetic datasets
        deprecation_records: Dictionary of deprecation records
        n_accesses: Number of simulated access attempts
        
    Returns:
        The simulation instance
    """
    logger.info("Starting basic deprecation simulation")
    
    # Create a copy of datasets to avoid modifying the original
    datasets_copy = {k: v for k, v in datasets.items()}
    
    # Initialize basic deprecation
    basic = BasicDeprecation(datasets=datasets_copy)
    
    # Apply warning labels based on deprecation records
    for dataset_id, record in deprecation_records.items():
        if dataset_id in datasets_copy:
            # Apply warning for Caution and Limited Use
            if record.warning_level in [WarningLevel.CAUTION, WarningLevel.LIMITED_USE]:
                basic.apply_warning(
                    dataset_id=dataset_id,
                    warning_level=record.warning_level,
                    reason=record.issue_description
                )
            # Remove datasets with Deprecated level
            elif record.warning_level == WarningLevel.DEPRECATED:
                basic.remove_dataset(
                    dataset_id=dataset_id,
                    reason=record.issue_description,
                    alternatives=record.recommended_alternatives
                )
    
    # Simulate random access attempts
    all_dataset_ids = list(datasets.keys())  # Use original datasets to include removed ones
    for _ in range(n_accesses):
        dataset_id = random.choice(all_dataset_ids)
        success, _, _ = basic.access_dataset(dataset_id)
        
        # If access failed and it was a removed dataset, occasionally check for alternatives
        if not success and dataset_id in basic.removed_datasets:
            if random.random() < 0.5:  # 50% chance
                basic.get_alternatives(dataset_id)
    
    # Save evaluation data
    basic.save_evaluation_data()
    
    logger.info("Completed basic deprecation simulation")
    return basic

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from claude_code.dataset_generator import load_datasets, load_deprecation_records
    
    # Load datasets and deprecation records
    datasets = load_datasets()
    deprecation_records = load_deprecation_records()
    
    # Run traditional simulation
    traditional = run_traditional_simulation(datasets, deprecation_records)
    
    # Run basic simulation
    basic = run_basic_simulation(datasets, deprecation_records)
"""
Dataset Generator for Contextual Dataset Deprecation Framework

This module provides functionality to create and manage synthetic datasets
with controlled properties for use in evaluating the Contextual Dataset 
Deprecation Framework.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum

from experimental_design import WarningLevel, DeprecationRecord, SyntheticDataset, DatasetVersion

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log.txt')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dataset_generator")

# Constants
DATASETS_DIR = os.path.join(os.path.dirname(__file__), 'datasets')
if not os.path.exists(DATASETS_DIR):
    os.makedirs(DATASETS_DIR)

def generate_dataset_collection(save_to_disk: bool = True) -> Dict[str, SyntheticDataset]:
    """
    Generate a collection of synthetic datasets with different characteristics
    for experimental evaluation.
    
    Args:
        save_to_disk: Whether to save the generated datasets to disk
        
    Returns:
        Dictionary of dataset_id -> SyntheticDataset
    """
    datasets = {}
    
    # Dataset 1: A clean classification dataset (no issues)
    datasets["clean_dataset"] = SyntheticDataset(
        dataset_id="clean_dataset",
        task_type="classification",
        n_samples=1000,
        n_features=10,
        bias_level=0.0,
        version="1.0.0",
        ethical_issues=[]
    )
    
    # Dataset 2: A classification dataset with minor bias issues
    datasets["minor_bias_dataset"] = SyntheticDataset(
        dataset_id="minor_bias_dataset",
        task_type="classification",
        n_samples=1000,
        n_features=10,
        bias_level=0.3,
        version="1.0.0",
        ethical_issues=["Slight demographic imbalance"]
    )
    
    # Dataset 3: A classification dataset with significant bias issues
    datasets["major_bias_dataset"] = SyntheticDataset(
        dataset_id="major_bias_dataset",
        task_type="classification",
        n_samples=1000,
        n_features=10,
        bias_level=0.7,
        version="1.0.0",
        ethical_issues=["Severe demographic bias", "Performance disparities across groups"]
    )
    
    # Dataset 4: A regression dataset (no issues)
    datasets["clean_regression"] = SyntheticDataset(
        dataset_id="clean_regression",
        task_type="regression",
        n_samples=1000,
        n_features=15,
        bias_level=0.0,
        version="1.0.0",
        ethical_issues=[]
    )
    
    # Dataset 5: A regression dataset with moderate bias
    datasets["biased_regression"] = SyntheticDataset(
        dataset_id="biased_regression",
        task_type="regression",
        n_samples=1000,
        n_features=15,
        bias_level=0.5,
        version="1.0.0",
        ethical_issues=["Moderate demographic bias"]
    )
    
    if save_to_disk:
        save_datasets(datasets)
    
    logger.info(f"Generated {len(datasets)} synthetic datasets")
    return datasets

def save_datasets(datasets: Dict[str, SyntheticDataset]) -> None:
    """
    Save synthetic datasets to disk.
    
    Args:
        datasets: Dictionary of dataset_id -> SyntheticDataset
    """
    for dataset_id, dataset in datasets.items():
        # Create a subdirectory for this dataset
        dataset_dir = os.path.join(DATASETS_DIR, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save dataset metadata
        metadata_file = os.path.join(dataset_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            metadata = {
                "dataset_id": dataset.dataset_id,
                "task_type": dataset.task_type,
                "n_samples": dataset.n_samples,
                "n_features": dataset.n_features,
                "bias_level": dataset.bias_level,
                "version": dataset.version,
                "ethical_issues": dataset.ethical_issues,
                "content_hash": dataset.content_hash,
                "created_at": datetime.now().isoformat()
            }
            json.dump(metadata, f, indent=2)
        
        # Save data as CSV for convenience
        X_df, y_series = dataset.to_pandas()
        data_df = pd.concat([X_df, y_series], axis=1)
        csv_file = os.path.join(dataset_dir, 'data.csv')
        data_df.to_csv(csv_file, index=False)
        
        # Also save as numpy arrays for faster loading
        np_file = os.path.join(dataset_dir, 'data.npz')
        np.savez(
            np_file,
            X=dataset.data["X"],
            y=dataset.data["y"],
            metadata=np.array([json.dumps(metadata)])
        )
        
        logger.info(f"Saved dataset {dataset_id} to {dataset_dir}")

def load_datasets() -> Dict[str, SyntheticDataset]:
    """
    Load synthetic datasets from disk.
    
    Returns:
        Dictionary of dataset_id -> SyntheticDataset
    """
    datasets = {}
    
    # Check if datasets directory exists
    if not os.path.exists(DATASETS_DIR):
        logger.warning(f"Datasets directory {DATASETS_DIR} does not exist. Generating new datasets.")
        return generate_dataset_collection(save_to_disk=True)
    
    # Iterate through subdirectories
    for dataset_id in os.listdir(DATASETS_DIR):
        dataset_dir = os.path.join(DATASETS_DIR, dataset_id)
        
        # Skip if not a directory
        if not os.path.isdir(dataset_dir):
            continue
        
        # Load metadata
        metadata_file = os.path.join(dataset_dir, 'metadata.json')
        if not os.path.exists(metadata_file):
            logger.warning(f"Metadata file not found for dataset {dataset_id}. Skipping.")
            continue
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load data
        np_file = os.path.join(dataset_dir, 'data.npz')
        if os.path.exists(np_file):
            with np.load(np_file) as data:
                X = data['X']
                y = data['y']
                
                # Create dataset
                dataset = SyntheticDataset(
                    dataset_id=metadata["dataset_id"],
                    task_type=metadata["task_type"],
                    n_samples=metadata["n_samples"],
                    n_features=metadata["n_features"],
                    bias_level=metadata["bias_level"],
                    version=metadata["version"],
                    ethical_issues=metadata["ethical_issues"]
                )
                
                # Replace generated data with loaded data
                dataset.data["X"] = X
                dataset.data["y"] = y
                dataset.content_hash = metadata["content_hash"]
                
                datasets[dataset_id] = dataset
                logger.info(f"Loaded dataset {dataset_id} from {dataset_dir}")
        else:
            # Try loading from CSV
            csv_file = os.path.join(dataset_dir, 'data.csv')
            if os.path.exists(csv_file):
                data_df = pd.read_csv(csv_file)
                
                # Extract X and y
                feature_cols = [col for col in data_df.columns if col != 'target']
                X = data_df[feature_cols].values
                y = data_df['target'].values
                
                # Create dataset
                dataset = SyntheticDataset(
                    dataset_id=metadata["dataset_id"],
                    task_type=metadata["task_type"],
                    n_samples=metadata["n_samples"],
                    n_features=metadata["n_features"],
                    bias_level=metadata["bias_level"],
                    version=metadata["version"],
                    ethical_issues=metadata["ethical_issues"]
                )
                
                # Replace generated data with loaded data
                dataset.data["X"] = X
                dataset.data["y"] = y
                dataset.content_hash = metadata["content_hash"]
                
                datasets[dataset_id] = dataset
                logger.info(f"Loaded dataset {dataset_id} from CSV in {dataset_dir}")
            else:
                logger.warning(f"Data file not found for dataset {dataset_id}. Skipping.")
    
    # If no datasets were loaded, generate new ones
    if not datasets:
        logger.warning("No datasets found. Generating new datasets.")
        return generate_dataset_collection(save_to_disk=True)
    
    return datasets

def generate_deprecation_records(datasets: Dict[str, SyntheticDataset]) -> Dict[str, DeprecationRecord]:
    """
    Generate deprecation records for the synthetic datasets.
    
    Args:
        datasets: Dictionary of dataset_id -> SyntheticDataset
        
    Returns:
        Dictionary of dataset_id -> DeprecationRecord
    """
    records = {}
    
    # No record for clean_dataset (it has no issues)
    
    # Record for minor_bias_dataset (Caution level)
    if "minor_bias_dataset" in datasets:
        records["minor_bias_dataset"] = DeprecationRecord(
            dataset_id="minor_bias_dataset",
            warning_level=WarningLevel.CAUTION,
            issue_description="This dataset shows a slight demographic imbalance that may lead to performance disparities.",
            evidence_links=["https://example.com/bias_analysis/minor_bias_dataset"],
            affected_groups=["Demographic group A"],
            recommended_alternatives=["clean_dataset"]
        )
    
    # Record for major_bias_dataset (Limited Use level)
    if "major_bias_dataset" in datasets:
        records["major_bias_dataset"] = DeprecationRecord(
            dataset_id="major_bias_dataset",
            warning_level=WarningLevel.LIMITED_USE,
            issue_description="This dataset contains severe demographic bias resulting in significant performance disparities across groups.",
            evidence_links=["https://example.com/bias_analysis/major_bias_dataset", "https://example.com/ethical_review/major_bias_dataset"],
            affected_groups=["Demographic group A", "Demographic group B"],
            recommended_alternatives=["clean_dataset", "minor_bias_dataset"]
        )
    
    # No record for clean_regression (it has no issues)
    
    # Record for biased_regression (Caution level)
    if "biased_regression" in datasets:
        records["biased_regression"] = DeprecationRecord(
            dataset_id="biased_regression",
            warning_level=WarningLevel.CAUTION,
            issue_description="This dataset exhibits moderate demographic bias that may affect prediction fairness.",
            evidence_links=["https://example.com/bias_analysis/biased_regression"],
            affected_groups=["Demographic group C"],
            recommended_alternatives=["clean_regression"]
        )
    
    logger.info(f"Generated {len(records)} deprecation records")
    return records

def save_deprecation_records(records: Dict[str, DeprecationRecord]) -> None:
    """
    Save deprecation records to disk.
    
    Args:
        records: Dictionary of dataset_id -> DeprecationRecord
    """
    records_dir = os.path.join(os.path.dirname(__file__), 'deprecation_records')
    os.makedirs(records_dir, exist_ok=True)
    
    records_file = os.path.join(records_dir, 'deprecation_records.json')
    
    serialized_records = {}
    for dataset_id, record in records.items():
        serialized_records[dataset_id] = record.to_dict()
    
    with open(records_file, 'w') as f:
        json.dump(serialized_records, f, indent=2)
    
    logger.info(f"Saved {len(records)} deprecation records to {records_file}")

def load_deprecation_records() -> Dict[str, DeprecationRecord]:
    """
    Load deprecation records from disk.
    
    Returns:
        Dictionary of dataset_id -> DeprecationRecord
    """
    records = {}
    
    records_dir = os.path.join(os.path.dirname(__file__), 'deprecation_records')
    records_file = os.path.join(records_dir, 'deprecation_records.json')
    
    if not os.path.exists(records_file):
        logger.warning(f"Deprecation records file {records_file} not found. Loading datasets to generate new records.")
        datasets = load_datasets()
        return generate_deprecation_records(datasets)
    
    with open(records_file, 'r') as f:
        serialized_records = json.load(f)
    
    for dataset_id, record_dict in serialized_records.items():
        records[dataset_id] = DeprecationRecord.from_dict(record_dict)
    
    logger.info(f"Loaded {len(records)} deprecation records from {records_file}")
    return records

if __name__ == "__main__":
    # Generate and save synthetic datasets
    datasets = generate_dataset_collection(save_to_disk=True)
    
    # Generate and save deprecation records
    records = generate_deprecation_records(datasets)
    save_deprecation_records(records)
    
    logger.info("Dataset generation complete")
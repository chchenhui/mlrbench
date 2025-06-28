"""
Data loading module for the model zoo retrieval experiment.
This module provides data loading utilities for the experiment.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
import logging
import numpy as np
from tqdm import tqdm

# Local imports
from config import TRAIN_CONFIG, LOG_CONFIG

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["log_level"]),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dataloader")

class ModelGraphDataset(Dataset):
    """
    Dataset for model graph representations.
    """
    
    def __init__(self, model_ids, model_graphs, task_labels, model_performance):
        """
        Initialize the dataset.
        
        Args:
            model_ids: List of model IDs
            model_graphs: List of model graph lists (one list per model)
            task_labels: List of task labels for each model
            model_performance: List of performance values for each model
        """
        self.model_ids = model_ids
        self.model_graphs = model_graphs
        self.task_labels = task_labels
        self.model_performance = torch.tensor(model_performance, dtype=torch.float32)
        
        assert len(model_ids) == len(model_graphs) == len(task_labels) == len(model_performance), \
            "Mismatch in dataset sizes"
    
    def __len__(self):
        return len(self.model_ids)
    
    def __getitem__(self, idx):
        return (
            self.model_graphs[idx],
            self.model_performance[idx],
            self.model_ids[idx]
        )

def create_dataloaders(dataset, train_ratio=0.8, batch_size=TRAIN_CONFIG["batch_size"]):
    """
    Create training and validation dataloaders.
    
    Args:
        dataset: ModelGraphDataset object
        train_ratio: Ratio of data to use for training
        batch_size: Batch size for dataloaders
        
    Returns:
        (train_loader, val_loader) tuple
    """
    # Split dataset
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size
    
    # Get unique task labels
    unique_tasks = {}
    for i, label in enumerate(dataset.task_labels):
        if label not in unique_tasks:
            unique_tasks[label] = []
        unique_tasks[label].append(i)
    
    # Stratified split by task
    train_indices = []
    val_indices = []
    
    for task, indices in unique_tasks.items():
        random.shuffle(indices)
        train_count = int(len(indices) * train_ratio)
        train_indices.extend(indices[:train_count])
        val_indices.extend(indices[train_count:])
    
    # Create subdatasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: ([item[0] for item in x], 
                              torch.stack([item[1] for item in x]),
                              [item[2] for item in x])
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: ([item[0] for item in x], 
                              torch.stack([item[1] for item in x]),
                              [item[2] for item in x])
    )
    
    logger.info(f"Created dataloaders: train={len(train_dataset)} samples, val={len(val_dataset)} samples")
    
    return train_loader, val_loader

def create_cross_validation_folds(dataset, num_folds=5):
    """
    Create cross-validation folds with stratification by task.
    
    Args:
        dataset: ModelGraphDataset object
        num_folds: Number of CV folds
        
    Returns:
        List of (train_indices, val_indices) tuples, one for each fold
    """
    # Get unique task labels
    unique_tasks = {}
    for i, label in enumerate(dataset.task_labels):
        if label not in unique_tasks:
            unique_tasks[label] = []
        unique_tasks[label].append(i)
    
    # Create folds
    folds = []
    
    for fold in range(num_folds):
        train_indices = []
        val_indices = []
        
        for task, indices in unique_tasks.items():
            # Shuffle indices
            shuffled_indices = indices.copy()
            random.shuffle(shuffled_indices)
            
            # Calculate fold size
            fold_size = len(shuffled_indices) // num_folds
            
            # Get start and end indices
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < num_folds - 1 else len(shuffled_indices)
            
            # Split indices
            fold_val_indices = shuffled_indices[start_idx:end_idx]
            fold_train_indices = [idx for idx in shuffled_indices if idx not in fold_val_indices]
            
            # Add to folds
            train_indices.extend(fold_train_indices)
            val_indices.extend(fold_val_indices)
        
        folds.append((train_indices, val_indices))
    
    logger.info(f"Created {num_folds} cross-validation folds with stratification")
    
    return folds

def create_cv_dataloaders(dataset, folds, batch_size=TRAIN_CONFIG["batch_size"]):
    """
    Create dataloaders for cross-validation.
    
    Args:
        dataset: ModelGraphDataset object
        folds: List of (train_indices, val_indices) tuples
        batch_size: Batch size for dataloaders
        
    Returns:
        List of (train_loader, val_loader) tuples, one for each fold
    """
    dataloaders = []
    
    for fold, (train_indices, val_indices) in enumerate(folds):
        # Create subdatasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: ([item[0] for item in x], 
                                  torch.stack([item[1] for item in x]),
                                  [item[2] for item in x])
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: ([item[0] for item in x], 
                                  torch.stack([item[1] for item in x]),
                                  [item[2] for item in x])
        )
        
        dataloaders.append((train_loader, val_loader))
    
    logger.info(f"Created dataloaders for {len(folds)} cross-validation folds")
    
    return dataloaders


# Test code
if __name__ == "__main__":
    # Create a simple test dataset
    from torch_geometric.data import Data
    
    # Create dummy data
    num_models = 100
    
    # Create random graphs
    random_graphs = []
    for i in range(num_models):
        model_graphs = []
        for j in range(3):  # 3 layers per model
            # Create random graph
            num_nodes = 10
            edge_index = torch.randint(0, num_nodes, (2, 20))
            x = torch.randn(num_nodes, 16)
            edge_attr = torch.randn(20, 8)
            
            # Create graph
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            graph.n_in = num_nodes // 2
            graph.n_out = num_nodes - graph.n_in
            graph.layer_name = f"layer_{j}"
            graph.total_nodes = num_nodes
            
            model_graphs.append(graph)
        
        random_graphs.append(model_graphs)
    
    # Create random task labels
    task_labels = []
    for i in range(num_models):
        if i < 30:
            task_labels.append("task_1")
        elif i < 70:
            task_labels.append("task_2")
        else:
            task_labels.append("task_3")
    
    # Create random model IDs and performance values
    model_ids = [f"model_{i}" for i in range(num_models)]
    performance = np.random.uniform(0.5, 0.95, size=num_models)
    
    # Create dataset
    dataset = ModelGraphDataset(model_ids, random_graphs, task_labels, performance)
    
    # Test dataloaders
    train_loader, val_loader = create_dataloaders(dataset)
    
    # Test batch
    for batch_idx, (model_graphs, batch_perf, batch_ids) in enumerate(train_loader):
        print(f"Batch {batch_idx}: {len(model_graphs)} models, {batch_perf.shape} performance values")
        print(f"IDs: {batch_ids[:3]}")
        
        # Check first model
        print(f"First model: {len(model_graphs[0])} layers")
        print(f"First layer: {model_graphs[0][0]}")
        
        # Only check first batch
        break
    
    # Test cross-validation
    folds = create_cross_validation_folds(dataset, num_folds=5)
    cv_dataloaders = create_cv_dataloaders(dataset, folds)
    
    # Check fold sizes
    for fold, (train_loader, val_loader) in enumerate(cv_dataloaders):
        print(f"Fold {fold}: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples")
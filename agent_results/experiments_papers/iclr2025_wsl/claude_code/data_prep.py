"""
Data preparation module for model zoo retrieval experiment.
This module handles downloading, processing, and managing the model weights.
"""

import os
import json
import random
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from torchvision import models as tv_models
from transformers import AutoModel

# Local imports
from config import DATA_DIR, DATASET_CONFIG, LOG_CONFIG

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["log_level"]),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_prep")

class ModelZooDataset:
    """Dataset class for managing a collection of neural network models."""
    
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)
        self.metadata_file = self.data_dir / "model_metadata.json"
        self.weights_dir = self.data_dir / "model_weights"
        self.models_metadata = defaultdict(dict)
        self.task_to_models = defaultdict(list)
        
        # Create directories if they don't exist
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Device for model loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def create_synthetic_model_zoo(self):
        """
        Create a synthetic model zoo with varied architectures and tasks.
        This generates mock models for the experiment.
        """
        logger.info("Creating synthetic model zoo...")
        
        # Create vision models
        self._create_vision_models()
        
        # Create NLP models
        self._create_nlp_models()
        
        # Create scientific/physics models
        self._create_scientific_models()
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Created synthetic model zoo with {len(self.models_metadata)} models")
        return self.models_metadata
    
    def _create_vision_models(self):
        """Create vision models for the model zoo."""
        config = DATASET_CONFIG["vision_models"]
        num_models = config["num_models"]
        
        logger.info(f"Creating {num_models} vision models...")
        
        for i in tqdm(range(num_models), desc="Vision Models"):
            # Select random architecture, task, and dataset
            arch = random.choice(config["architectures"])
            task = random.choice(config["tasks"])
            dataset = random.choice(config["datasets"])
            
            # Generate a unique model ID
            model_id = f"vision_{arch}_{task}_{dataset}_{i}"
            
            # Get the model
            try:
                if arch == "resnet18":
                    model = tv_models.resnet18(pretrained=False)
                elif arch == "resnet34":
                    model = tv_models.resnet34(pretrained=False)
                elif arch == "vgg16":
                    model = tv_models.vgg16(pretrained=False)
                elif arch == "mobilenet_v2":
                    model = tv_models.mobilenet_v2(pretrained=False)
                elif arch == "efficientnet_b0":
                    model = tv_models.efficientnet_b0(pretrained=False)
                else:
                    logger.warning(f"Unknown architecture: {arch}, skipping")
                    continue
                
                # Apply some random initialization to simulate different training runs
                for param in model.parameters():
                    param.data = torch.randn_like(param.data) * random.uniform(0.01, 0.1)
                
                # For classification, we need to modify the final layer based on dataset
                if task == "classification":
                    if dataset == "imagenet":
                        num_classes = 1000
                    elif dataset == "cifar10":
                        num_classes = 10
                    else:
                        num_classes = 100
                    
                    # Replace final layer based on architecture
                    if arch.startswith("resnet"):
                        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                    elif arch.startswith("vgg"):
                        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
                    elif arch.startswith("mobilenet"):
                        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
                    elif arch.startswith("efficientnet"):
                        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
                
                # Save the model weights
                save_path = self.weights_dir / f"{model_id}.pt"
                torch.save(model.state_dict(), save_path)
                
                # Add metadata
                performance = random.uniform(0.3, 0.95)  # Simulate varying performance
                self.models_metadata[model_id] = {
                    "id": model_id,
                    "type": "vision",
                    "architecture": arch,
                    "task": task,
                    "dataset": dataset,
                    "performance": performance,
                    "path": str(save_path),
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                }
                
                # Add to task map
                task_key = f"{task}_{dataset}"
                self.task_to_models[task_key].append(model_id)
            
            except Exception as e:
                logger.error(f"Error creating vision model {model_id}: {e}")
    
    def _create_nlp_models(self):
        """Create NLP models for the model zoo."""
        config = DATASET_CONFIG["nlp_models"]
        num_models = config["num_models"]
        
        logger.info(f"Creating {num_models} NLP models...")
        
        for i in tqdm(range(num_models), desc="NLP Models"):
            # Select random architecture, task, and dataset
            arch = random.choice(config["architectures"])
            task = random.choice(config["tasks"])
            dataset = random.choice(config["datasets"])
            
            # Generate a unique model ID
            model_id = f"nlp_{arch}_{task}_{dataset}_{i}"
            
            try:
                # Load a pre-trained model with random initialization
                model = AutoModel.from_pretrained(arch, return_dict=True)
                
                # Apply some random initialization to simulate different training runs
                for param in model.parameters():
                    param.data = torch.randn_like(param.data) * random.uniform(0.01, 0.1)
                
                # Save the model weights
                save_path = self.weights_dir / f"{model_id}.pt"
                torch.save(model.state_dict(), save_path)
                
                # Add metadata
                performance = random.uniform(0.3, 0.95)  # Simulate varying performance
                self.models_metadata[model_id] = {
                    "id": model_id,
                    "type": "nlp",
                    "architecture": arch,
                    "task": task,
                    "dataset": dataset,
                    "performance": performance,
                    "path": str(save_path),
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                }
                
                # Add to task map
                task_key = f"{task}_{dataset}"
                self.task_to_models[task_key].append(model_id)
            
            except Exception as e:
                logger.error(f"Error creating NLP model {model_id}: {e}")
    
    def _create_scientific_models(self):
        """Create scientific models for the model zoo."""
        config = DATASET_CONFIG["scientific_models"]
        num_models = config["num_models"]
        
        logger.info(f"Creating {num_models} scientific models...")
        
        for i in tqdm(range(num_models), desc="Scientific Models"):
            # Select random architecture, task, and dataset
            arch = random.choice(config["architectures"])
            task = random.choice(config["tasks"])
            dataset = random.choice(config["datasets"])
            
            # Generate a unique model ID
            model_id = f"sci_{arch}_{task}_{dataset}_{i}"
            
            try:
                # Create a simple model for scientific tasks
                if arch == "mlp":
                    input_dim = random.choice([2, 3, 4])
                    hidden_dims = [random.choice([32, 64, 128]) for _ in range(random.randint(2, 4))]
                    output_dim = random.choice([1, 2, 3])
                    
                    layers = []
                    prev_dim = input_dim
                    for hidden_dim in hidden_dims:
                        layers.append(torch.nn.Linear(prev_dim, hidden_dim))
                        layers.append(torch.nn.ReLU())
                        prev_dim = hidden_dim
                    layers.append(torch.nn.Linear(prev_dim, output_dim))
                    
                    model = torch.nn.Sequential(*layers)
                
                elif arch == "cnn":
                    input_channels = random.choice([1, 3])
                    output_dim = random.choice([1, 2, 3])
                    
                    model = torch.nn.Sequential(
                        torch.nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(2),
                        torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(2),
                        torch.nn.Flatten(),
                        torch.nn.Linear(32 * 8 * 8, 128),
                        torch.nn.ReLU(),
                        torch.nn.Linear(128, output_dim)
                    )
                
                else:
                    logger.warning(f"Unknown architecture: {arch}, skipping")
                    continue
                
                # Save the model weights
                save_path = self.weights_dir / f"{model_id}.pt"
                torch.save(model.state_dict(), save_path)
                
                # Add metadata
                performance = random.uniform(0.3, 0.95)  # Simulate varying performance
                self.models_metadata[model_id] = {
                    "id": model_id,
                    "type": "scientific",
                    "architecture": arch,
                    "task": task,
                    "dataset": dataset,
                    "performance": performance,
                    "path": str(save_path),
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                }
                
                # Add to task map
                task_key = f"{task}_{dataset}"
                self.task_to_models[task_key].append(model_id)
            
            except Exception as e:
                logger.error(f"Error creating scientific model {model_id}: {e}")
    
    def _save_metadata(self):
        """Save metadata to disk."""
        # Save model metadata
        with open(self.metadata_file, 'w') as f:
            json.dump({
                "models": self.models_metadata,
                "task_to_models": self.task_to_models
            }, f, indent=2)
        
        logger.info(f"Saved metadata to {self.metadata_file}")
    
    def load_metadata(self):
        """Load model metadata from disk."""
        if not os.path.exists(self.metadata_file):
            logger.error(f"Metadata file not found: {self.metadata_file}")
            return False
        
        with open(self.metadata_file, 'r') as f:
            data = json.load(f)
            self.models_metadata = data["models"]
            self.task_to_models = data["task_to_models"]
        
        logger.info(f"Loaded metadata for {len(self.models_metadata)} models")
        return True
    
    def get_model_weights(self, model_id):
        """
        Load a model's weights from disk.
        
        Args:
            model_id: The ID of the model to load.
            
        Returns:
            A dictionary of model weights.
        """
        if model_id not in self.models_metadata:
            logger.error(f"Model not found: {model_id}")
            return None
        
        path = self.models_metadata[model_id]["path"]
        try:
            weights = torch.load(path, map_location=self.device)
            return weights
        except Exception as e:
            logger.error(f"Error loading model weights for {model_id}: {e}")
            return None
    
    def get_models_by_task(self, task, dataset):
        """
        Get all models for a specific task and dataset.
        
        Args:
            task: The task (e.g., 'classification', 'detection').
            dataset: The dataset (e.g., 'imagenet', 'cifar10').
            
        Returns:
            A list of model IDs.
        """
        task_key = f"{task}_{dataset}"
        return self.task_to_models.get(task_key, [])
    
    def get_model_metadata(self, model_id):
        """
        Get metadata for a specific model.
        
        Args:
            model_id: The ID of the model.
            
        Returns:
            A dictionary of model metadata.
        """
        return self.models_metadata.get(model_id, {})
    
    def get_all_models(self):
        """
        Get all model IDs.
        
        Returns:
            A list of all model IDs.
        """
        return list(self.models_metadata.keys())
    
    def generate_functional_pairs(self, num_pairs=100, same_ratio=0.5):
        """
        Generate functionally similar and dissimilar model pairs for contrastive learning.
        
        Args:
            num_pairs: Number of pairs to generate.
            same_ratio: Ratio of functionally similar pairs.
            
        Returns:
            A list of tuples (model_id1, model_id2, is_similar).
        """
        pairs = []
        all_task_keys = list(self.task_to_models.keys())
        
        # Calculate number of similar pairs
        num_similar = int(num_pairs * same_ratio)
        num_dissimilar = num_pairs - num_similar
        
        # Generate similar pairs (same task and dataset)
        for _ in range(num_similar):
            # Randomly select a task that has at least 2 models
            valid_task_keys = [k for k in all_task_keys if len(self.task_to_models[k]) >= 2]
            if not valid_task_keys:
                logger.warning("Not enough models for similar pairs")
                break
                
            task_key = random.choice(valid_task_keys)
            models = self.task_to_models[task_key]
            model1, model2 = random.sample(models, 2)
            pairs.append((model1, model2, 1))  # 1 indicates similar
        
        # Generate dissimilar pairs (different tasks or datasets)
        for _ in range(num_dissimilar):
            if len(all_task_keys) < 2:
                logger.warning("Not enough task types for dissimilar pairs")
                break
                
            # Select two different task keys
            task_key1, task_key2 = random.sample(all_task_keys, 2)
            
            # Select a model from each task
            if not self.task_to_models[task_key1] or not self.task_to_models[task_key2]:
                continue
                
            model1 = random.choice(self.task_to_models[task_key1])
            model2 = random.choice(self.task_to_models[task_key2])
            pairs.append((model1, model2, 0))  # 0 indicates dissimilar
        
        logger.info(f"Generated {len(pairs)} model pairs: {num_similar} similar, {len(pairs) - num_similar} dissimilar")
        return pairs
    
    def apply_symmetry_transforms(self, model_id, transform_type="permutation"):
        """
        Apply symmetry-preserving transforms to a model's weights.
        
        Args:
            model_id: The ID of the model to transform.
            transform_type: Type of transform to apply ('permutation', 'scaling', 'dropout').
            
        Returns:
            Transformed weights and the original weights.
        """
        weights = self.get_model_weights(model_id)
        if weights is None:
            return None, None
        
        transformed_weights = {}
        
        for key, weight in weights.items():
            # Skip non-tensor weights
            if not isinstance(weight, torch.Tensor):
                transformed_weights[key] = weight
                continue
                
            # Skip weights that aren't 2D matrices (e.g., biases)
            if len(weight.shape) != 2:
                transformed_weights[key] = weight
                continue
            
            # Apply transform based on type
            if transform_type == "permutation" and random.random() < 0.15:
                # Apply permutation (for output neurons)
                n_out = weight.shape[0]
                perm = torch.randperm(n_out)
                transformed_weights[key] = weight[perm]
                
            elif transform_type == "scaling":
                # Apply scaling
                scale = torch.rand(weight.shape[0]) * 1.5 + 0.5  # Random scaling in [0.5, 2.0]
                transformed_weights[key] = weight * scale.unsqueeze(1)
                
            elif transform_type == "dropout" and random.random() < 0.05:
                # Apply dropout (zero out random weights)
                mask = torch.rand_like(weight) > 0.05
                transformed_weights[key] = weight * mask
                
            else:
                transformed_weights[key] = weight
        
        return transformed_weights, weights
    
    def stats_summary(self):
        """Generate summary statistics for the model zoo."""
        if not self.models_metadata:
            logger.warning("No models in metadata. Call load_metadata() or create_synthetic_model_zoo() first.")
            return {}
        
        stats = {
            "total_models": len(self.models_metadata),
            "models_by_type": defaultdict(int),
            "models_by_task": defaultdict(int),
            "models_by_dataset": defaultdict(int),
            "models_by_architecture": defaultdict(int),
            "parameter_count_stats": {
                "min": float('inf'),
                "max": 0,
                "mean": 0,
                "median": 0
            }
        }
        
        param_counts = []
        
        for model_id, metadata in self.models_metadata.items():
            model_type = metadata["type"]
            task = metadata["task"]
            dataset = metadata["dataset"]
            arch = metadata["architecture"]
            num_params = metadata["num_parameters"]
            
            stats["models_by_type"][model_type] += 1
            stats["models_by_task"][task] += 1
            stats["models_by_dataset"][dataset] += 1
            stats["models_by_architecture"][arch] += 1
            
            param_counts.append(num_params)
            
            if num_params < stats["parameter_count_stats"]["min"]:
                stats["parameter_count_stats"]["min"] = num_params
            if num_params > stats["parameter_count_stats"]["max"]:
                stats["parameter_count_stats"]["max"] = num_params
        
        if param_counts:
            stats["parameter_count_stats"]["mean"] = sum(param_counts) / len(param_counts)
            stats["parameter_count_stats"]["median"] = sorted(param_counts)[len(param_counts) // 2]
        
        return stats


if __name__ == "__main__":
    # Create and initialize the dataset
    dataset = ModelZooDataset()
    dataset.create_synthetic_model_zoo()
    
    # Print some statistics
    stats = dataset.stats_summary()
    print(json.dumps(stats, indent=2))
    
    # Generate some pairs for testing
    pairs = dataset.generate_functional_pairs(num_pairs=10)
    for model_id1, model_id2, is_similar in pairs:
        meta1 = dataset.get_model_metadata(model_id1)
        meta2 = dataset.get_model_metadata(model_id2)
        print(f"Pair: {model_id1} - {model_id2}, Similar: {is_similar}")
        print(f"  Model 1: {meta1['type']}, {meta1['architecture']}, {meta1['task']}, {meta1['dataset']}")
        print(f"  Model 2: {meta2['type']}, {meta2['architecture']}, {meta2['task']}, {meta2['dataset']}")
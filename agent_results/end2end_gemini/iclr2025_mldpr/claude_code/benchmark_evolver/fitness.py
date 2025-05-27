"""
Fitness functions for the Benchmark Evolver.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.metrics import pairwise_distances
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import evaluate_model

logger = logging.getLogger(__name__)

class FitnessEvaluator:
    """Class for evaluating the fitness of benchmark individuals."""
    
    def __init__(self, target_models, dataset, device, batch_size=32,
                 weights=(0.6, 0.2, 0.2)):
        """
        Initialize the fitness evaluator.
        
        Args:
            target_models: Dictionary of {model_name: model} pairs to evaluate against
            dataset: The dataset to use for generating adversarial examples
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
            weights: Tuple of weights for (challenge, diversity, novelty)
        """
        self.target_models = target_models
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.weights = weights
        
        # Keep track of previously seen transformations for novelty
        self.seen_individuals = []
        
        # Sample indices for evaluation (for consistency)
        self.eval_indices = np.random.choice(
            len(dataset), min(1000, len(dataset)), replace=False
        )
        
        logger.info(f"FitnessEvaluator initialized with {len(target_models)} target models")
        logger.info(f"Evaluation set size: {len(self.eval_indices)}")
    
    def evaluate(self, individual):
        """
        Evaluate the fitness of an individual.
        
        Args:
            individual: The individual to evaluate
        
        Returns:
            Dictionary of fitness components
        """
        # Calculate fitness components
        challenge_score = self._evaluate_challenge(individual)
        diversity_score = self._evaluate_diversity(individual)
        novelty_score = self._evaluate_novelty(individual)
        complexity_penalty = self._evaluate_complexity(individual)
        
        # Compute weighted sum
        w_challenge, w_diversity, w_novelty = self.weights
        overall_fitness = (
            w_challenge * challenge_score +
            w_diversity * diversity_score +
            w_novelty * novelty_score -
            0.1 * complexity_penalty  # Small penalty for complexity
        )
        
        # Add individual to seen individuals
        self.seen_individuals.append(individual)
        
        # Keep only a window of recent individuals to compare against
        max_history = 50
        if len(self.seen_individuals) > max_history:
            self.seen_individuals = self.seen_individuals[-max_history:]
        
        # Return all components
        return {
            'overall': overall_fitness,
            'challenge': challenge_score,
            'diversity': diversity_score,
            'novelty': novelty_score,
            'complexity_penalty': complexity_penalty
        }
    
    def _evaluate_challenge(self, individual):
        """
        Evaluate how challenging the transformed data is for target models.
        
        Args:
            individual: The individual to evaluate
        
        Returns:
            Challenge score
        """
        # Generate adversarial batch
        from .transformations import apply_transformation_sequence
        import torchvision.transforms as transforms
        
        # Sample a subset of eval indices for efficiency
        sample_indices = np.random.choice(
            self.eval_indices, min(100, len(self.eval_indices)), replace=False
        )
        
        # Get original images and labels
        original_images = []
        labels = []
        
        for idx in sample_indices:
            image, label = self.dataset[idx]
            original_images.append(image)
            labels.append(label)
        
        # Convert to tensors
        if not isinstance(original_images[0], torch.Tensor):
            original_images = [transforms.ToTensor()(img) if not isinstance(img, torch.Tensor) else img 
                             for img in original_images]
        
        original_images = torch.stack(original_images)
        labels = torch.tensor(labels)
        
        # Apply transformations to create adversarial examples
        transformed_images = []
        for img in original_images:
            transformed_img = apply_transformation_sequence(img, individual.transformations)
            transformed_images.append(transformed_img)
        
        transformed_images = torch.stack(transformed_images)
        
        # Create tensor datasets and dataloaders
        original_dataset = TensorDataset(original_images, labels)
        transformed_dataset = TensorDataset(transformed_images, labels)
        
        original_loader = DataLoader(original_dataset, batch_size=self.batch_size)
        transformed_loader = DataLoader(transformed_dataset, batch_size=self.batch_size)
        
        # Evaluate models on original and transformed data
        performance_degradation = []
        
        for model_name, model in self.target_models.items():
            model.eval()
            
            # Error rate on original data
            correct_original = 0
            total_original = 0
            
            with torch.no_grad():
                for images, targets in original_loader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total_original += targets.size(0)
                    correct_original += (predicted == targets).sum().item()
            
            original_error_rate = 1.0 - (correct_original / total_original)
            
            # Error rate on transformed data
            correct_transformed = 0
            total_transformed = 0
            
            with torch.no_grad():
                for images, targets in transformed_loader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total_transformed += targets.size(0)
                    correct_transformed += (predicted == targets).sum().item()
            
            transformed_error_rate = 1.0 - (correct_transformed / total_transformed)
            
            # Calculate degradation
            degradation = transformed_error_rate - original_error_rate
            performance_degradation.append(degradation)
        
        # Average degradation across all models
        avg_degradation = np.mean(performance_degradation)
        
        # Map to a [0, 1] score, capping at reasonable values
        # 0.3 degradation (30% worse) is considered highly challenging
        challenge_score = min(avg_degradation / 0.3, 1.0)
        
        return challenge_score
    
    def _evaluate_diversity(self, individual):
        """
        Evaluate diversity of the individual's failure patterns.
        
        Args:
            individual: The individual to evaluate
        
        Returns:
            Diversity score
        """
        # If this is the first individual, assign a default diversity
        if not self.seen_individuals:
            return 0.5
        
        # Extract transformation types and parameters
        def get_transformation_signature(ind):
            sig = []
            for transform in ind.transformations:
                # Get the type and parameters
                t_type = type(transform).__name__
                t_params = transform.get_params()
                sig.append((t_type, tuple(sorted(t_params.items()))))
            return tuple(sorted(sig))
        
        # Get transformation signatures for the current and previous individuals
        current_sig = get_transformation_signature(individual)
        previous_sigs = [get_transformation_signature(ind) for ind in self.seen_individuals]
        
        # Count exact matches
        exact_matches = sum(1 for sig in previous_sigs if sig == current_sig)
        
        if exact_matches > 0:
            # Penalize exact duplicates
            return 0.0
        
        # For non-exact matches, calculate diversity of transformation types
        current_types = set(type(t).__name__ for t in individual.transformations)
        
        diversity_scores = []
        for ind in self.seen_individuals:
            other_types = set(type(t).__name__ for t in ind.transformations)
            
            # Jaccard similarity
            intersection = len(current_types.intersection(other_types))
            union = len(current_types.union(other_types))
            
            if union > 0:
                similarity = intersection / union
                diversity = 1.0 - similarity
                diversity_scores.append(diversity)
        
        # Average diversity
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.5
        
        return avg_diversity
    
    def _evaluate_novelty(self, individual):
        """
        Evaluate novelty of the transformations.
        
        Args:
            individual: The individual to evaluate
        
        Returns:
            Novelty score
        """
        # For simplicity, use diversity as a proxy for novelty in this implementation
        # In a more complex implementation, we would compare with original data properties
        return self._evaluate_diversity(individual)
    
    def _evaluate_complexity(self, individual):
        """
        Evaluate complexity of the transformation sequence.
        
        Args:
            individual: The individual to evaluate
        
        Returns:
            Complexity penalty
        """
        # Count the number of transformations
        num_transforms = len(individual.transformations)
        
        # Calculate complexity based on number of transformations
        # Normalize to [0, 1] based on max allowed transformations
        complexity = num_transforms / 5  # Assuming max_transformations = 5
        
        return complexity
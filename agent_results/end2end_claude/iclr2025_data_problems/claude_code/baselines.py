#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of baseline attribution methods for comparison with AGT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

class InfluenceFunctionEstimator:
    """
    Estimator for influence functions, based on the method from
    "Understanding Black-box Predictions via Influence Functions" (Koh and Liang, 2017)
    and adapted based on "Influence Functions for Scalable Data Attribution in Diffusion Models" 
    (Mlodozeniec et al., 2024).
    
    This is a simplified implementation for demonstration purposes.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        damping: float = 0.01,
        scale: float = 25.0,
        recursion_depth: int = 100,
        r_averaging: int = 10
    ):
        """
        Initialize the influence function estimator.
        
        Args:
            model: Model to analyze
            train_loader: DataLoader with training data
            damping: Damping factor for Hessian approximation
            scale: Scaling factor for influence calculation
            recursion_depth: Depth of recursion for Hessian-vector product
            r_averaging: Number of random vectors for stochastic approximation
        """
        self.model = model
        self.train_loader = train_loader
        self.damping = damping
        self.scale = scale
        self.recursion_depth = recursion_depth
        self.r_averaging = r_averaging
        
        logger.info("Initializing InfluenceFunctionEstimator")
    
    def _parameter_grad(self, batch, device):
        """
        Compute gradient of loss on batch with respect to parameters.
        
        Args:
            batch: Batch of data
            device: Device to use for computation
            
        Returns:
            Gradient vector (flattened)
        """
        self.model.zero_grad()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        source_idx = batch["source_idx"].to(device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            source_idx=source_idx
        )
        
        loss = outputs["loss"]
        loss.backward()
        
        # Get gradients
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        
        # Concatenate all gradients
        grad_vec = torch.cat(grads)
        
        return grad_vec
    
    def _hvp(self, batch, vector, device):
        """
        Compute Hessian-vector product using R-operator.
        
        Args:
            batch: Batch of data
            vector: Vector to compute product with
            device: Device to use for computation
            
        Returns:
            Hessian-vector product
        """
        # Compute gradient
        grad_vec = self._parameter_grad(batch, device)
        
        # Initialize starting point for HVP calculation
        hvp = torch.zeros_like(vector)
        
        # Use stochastic approximation with multiple random vectors
        for _ in range(self.r_averaging):
            # Generate random vector for approximation
            r = torch.randn_like(vector)
            r = r / torch.norm(r)
            
            # Compute gradient-vector product
            self.model.zero_grad()
            grad_r = torch.sum(grad_vec * r)
            grad_r.backward()
            
            # Get gradients from second backward pass
            hessian_r = []
            for param in self.model.parameters():
                if param.grad is not None:
                    hessian_r.append(param.grad.view(-1))
            
            hessian_r_vec = torch.cat(hessian_r)
            
            # Update HVP
            hvp += hessian_r_vec / self.r_averaging
        
        return hvp
    
    def _s_test(self, test_batch, device):
        """
        Compute s_test (inverse Hessian-vector product).
        
        Args:
            test_batch: Test example batch
            device: Device to use for computation
            
        Returns:
            Inverse Hessian-vector product
        """
        # Compute gradient at test point
        test_grad = self._parameter_grad(test_batch, device)
        
        # Initialize s_test
        s_test = test_grad.clone()
        
        # Compute inverse HVP using conjugate gradient with damping
        for _ in range(self.recursion_depth):
            # Compute HVP
            train_batch = next(iter(self.train_loader))
            hvp = self._hvp(train_batch, s_test, device)
            
            # Update s_test
            s_test = test_grad + (1 - self.damping) * s_test - hvp / self.scale
        
        return s_test
    
    def compute_influence(self, test_batch, device):
        """
        Compute influence scores for each training example on the test example.
        
        Args:
            test_batch: Test example batch
            device: Device to use for computation
            
        Returns:
            List of influence scores for each training example
        """
        # Compute s_test
        s_test = self._s_test(test_batch, device)
        
        # Compute influence for each training example
        influences = []
        
        for train_batch in self.train_loader:
            # Compute gradient for training example
            train_grad = self._parameter_grad(train_batch, device)
            
            # Compute influence
            influence = torch.dot(s_test, train_grad) / len(train_batch["input_ids"])
            influences.append(influence.item())
        
        return influences

class ContentMatchingAttributor:
    """
    Simple baseline that attributes based on content similarity.
    It compares the hidden representations of the test example with training examples.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize the content matching attributor.
        
        Args:
            model: Model to analyze
            train_loader: DataLoader with training data
            similarity_threshold: Threshold for similarity matching
        """
        self.model = model
        self.train_loader = train_loader
        self.similarity_threshold = similarity_threshold
        
        # Cache for training examples' representations
        self.train_representations = {}
        self.train_sources = {}
        
        logger.info("Initializing ContentMatchingAttributor")
    
    def _get_representation(self, batch, device):
        """
        Get model representation for a batch.
        
        Args:
            batch: Batch of data
            device: Device to use for computation
            
        Returns:
            Model representation
        """
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Forward pass (no gradient needed)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Use hidden states from the last layer, CLS token
        if "hidden_states" in outputs:
            hidden_states = outputs["hidden_states"][-1][:, 0, :]  # [batch_size, hidden_size]
        else:
            # If hidden_states not available, use the output directly
            hidden_states = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        return hidden_states
    
    def _compute_similarity(self, repr1, repr2):
        """
        Compute cosine similarity between representations.
        
        Args:
            repr1: First representation tensor
            repr2: Second representation tensor
            
        Returns:
            Cosine similarity
        """
        # Normalize
        repr1_norm = repr1 / repr1.norm(dim=1, keepdim=True)
        repr2_norm = repr2 / repr2.norm(dim=1, keepdim=True)
        
        # Compute similarity
        similarity = torch.matmul(repr1_norm, repr2_norm.transpose(0, 1))
        
        return similarity
    
    def build_index(self, device):
        """
        Build index of training examples for fast lookup.
        
        Args:
            device: Device to use for computation
        """
        logger.info("Building content matching index")
        
        batch_idx = 0
        representations = []
        sources = []
        
        # Collect representations and sources
        for batch in self.train_loader:
            batch_repr = self._get_representation(batch, device)
            batch_sources = batch["source_idx"].cpu().numpy()
            
            for i, (repr_i, source_i) in enumerate(zip(batch_repr, batch_sources)):
                # Store with unique identifier
                idx = batch_idx * self.train_loader.batch_size + i
                self.train_representations[idx] = repr_i.cpu()
                self.train_sources[idx] = source_i
            
            batch_idx += 1
        
        logger.info(f"Built index with {len(self.train_representations)} examples")
    
    def attribute(self, test_batch, device, top_k=5):
        """
        Attribute test example to training examples based on content similarity.
        
        Args:
            test_batch: Test example batch
            device: Device to use for computation
            top_k: Number of top matches to return
            
        Returns:
            Dictionary with source attributions and scores
        """
        # Get representation for test example
        test_repr = self._get_representation(test_batch, device)
        
        # If index not built, build it
        if not self.train_representations:
            self.build_index(device)
        
        # Compute similarity with all training examples
        all_similarities = {}
        for idx, train_repr in self.train_representations.items():
            similarity = self._compute_similarity(test_repr, train_repr.unsqueeze(0).to(device))
            all_similarities[idx] = similarity.item()
        
        # Get top-k matches
        top_indices = sorted(all_similarities.keys(), 
                            key=lambda x: all_similarities[x], 
                            reverse=True)[:top_k]
        
        # Get sources and scores
        source_scores = {}
        for idx in top_indices:
            source = int(self.train_sources[idx])
            score = all_similarities[idx]
            
            if source in source_scores:
                source_scores[source] = max(source_scores[source], score)
            else:
                source_scores[source] = score
        
        # Filter by threshold
        filtered_sources = {s: score for s, score in source_scores.items() 
                           if score >= self.similarity_threshold}
        
        # Sort by score
        sorted_sources = sorted(filtered_sources.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "sources": [s for s, _ in sorted_sources],
            "scores": [s for _, s in sorted_sources]
        }

class RandomAttributor:
    """
    Random baseline that assigns random attributions.
    Useful for establishing a lower bound on performance.
    """
    
    def __init__(
        self,
        num_sources: int,
        num_attributions: int = 1
    ):
        """
        Initialize the random attributor.
        
        Args:
            num_sources: Number of possible sources
            num_attributions: Number of attributions to make
        """
        self.num_sources = num_sources
        self.num_attributions = num_attributions
        
        logger.info(f"Initializing RandomAttributor with {num_sources} sources")
    
    def attribute(self, test_batch=None, device=None):
        """
        Generate random attributions.
        
        Args:
            test_batch: Ignored (for API compatibility)
            device: Ignored (for API compatibility)
            
        Returns:
            Dictionary with random source attributions and scores
        """
        # Generate random sources
        sources = np.random.choice(self.num_sources, size=self.num_attributions, replace=False)
        
        # Generate random scores
        scores = np.random.uniform(size=self.num_attributions)
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        
        return {
            "sources": sources[sorted_indices].tolist(),
            "scores": scores[sorted_indices].tolist()
        }

def evaluate_attributor(
    attributor,
    test_loader,
    device,
    top_k=5
):
    """
    Evaluate an attribution method.
    
    Args:
        attributor: Attribution method to evaluate
        test_loader: DataLoader with test data
        device: Device to use for computation
        top_k: Number of top attributions to consider
        
    Returns:
        Dictionary with evaluation metrics
    """
    correct = 0
    total = 0
    
    for batch in test_loader:
        # Get true sources
        true_sources = batch["source_idx"].cpu().numpy()
        
        # Get attributions
        attributions = attributor.attribute(batch, device, top_k=top_k)
        pred_sources = attributions["sources"]
        
        # Count correct attributions
        for i, true_source in enumerate(true_sources):
            # Check if true source is in top-k predictions
            if i < len(pred_sources) and true_source in pred_sources[:top_k]:
                correct += 1
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }

if __name__ == "__main__":
    # Simple test for the baselines
    class MockModel(nn.Module):
        def __init__(self, input_dim=768, output_dim=10):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
        
        def forward(self, input_ids=None, attention_mask=None, source_idx=None):
            # Mock hidden states
            batch_size = input_ids.shape[0]
            hidden_states = torch.randn(batch_size, input_ids.shape[1], 768)
            
            # Mock output
            output = self.linear(hidden_states[:, 0, :])
            
            result = {
                "hidden_states": [hidden_states],
                "source_logits": output
            }
            
            if source_idx is not None:
                loss = F.cross_entropy(output, source_idx)
                result["loss"] = loss
            
            return result
    
    # Create mock model and data
    model = MockModel()
    
    # Create mock dataloader
    batch_size = 4
    num_batches = 3
    
    class MockDataLoader:
        def __init__(self, batch_size=4, num_batches=3):
            self.batch_size = batch_size
            self.num_batches = num_batches
        
        def __iter__(self):
            for _ in range(self.num_batches):
                yield {
                    "input_ids": torch.randint(0, 1000, (self.batch_size, 10)),
                    "attention_mask": torch.ones(self.batch_size, 10),
                    "source_idx": torch.randint(0, 10, (self.batch_size,))
                }
        
        def __len__(self):
            return self.num_batches
    
    train_loader = MockDataLoader()
    test_loader = MockDataLoader(num_batches=1)
    
    # Test InfluenceFunctionEstimator
    print("Testing InfluenceFunctionEstimator...")
    
    influence_estimator = InfluenceFunctionEstimator(
        model=model,
        train_loader=train_loader,
        recursion_depth=2,  # Small for testing
        r_averaging=2  # Small for testing
    )
    
    test_batch = next(iter(test_loader))
    influences = influence_estimator.compute_influence(test_batch, torch.device("cpu"))
    print(f"Influences: {influences}")
    
    # Test ContentMatchingAttributor
    print("Testing ContentMatchingAttributor...")
    
    content_matcher = ContentMatchingAttributor(
        model=model,
        train_loader=train_loader
    )
    
    content_matcher.build_index(torch.device("cpu"))
    
    test_batch = next(iter(test_loader))
    attributions = content_matcher.attribute(test_batch, torch.device("cpu"))
    print(f"Content attributions: {attributions}")
    
    # Test RandomAttributor
    print("Testing RandomAttributor...")
    
    random_attributor = RandomAttributor(num_sources=10, num_attributions=3)
    
    attributions = random_attributor.attribute()
    print(f"Random attributions: {attributions}")
    
    print("All baseline tests completed!")
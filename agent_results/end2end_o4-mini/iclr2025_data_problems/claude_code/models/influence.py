"""
Influence function approximation using LiSSA for the Gradient-Informed Fingerprinting (GIF) method.

This module implements fast approximation of influence functions to refine candidate
attributions based on training dynamics. It uses Linear-time Stochastic Second-order
Algorithm (LiSSA) to efficiently estimate the inverse Hessian-vector products.
"""

import os
import json
import pickle
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InfluenceEstimator:
    """
    Estimate influence scores using Fast Approximate Influence Functions with LiSSA.
    
    Implementation based on "Understanding Black-box Predictions via Influence Functions"
    (Koh & Liang, 2017) and "Fast Approximation of Influence Functions in Large Neural Networks"
    (Grey et al., 2023).
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        train_dataloader: DataLoader,
        device: str = None,
        damping: float = 0.01,
        scale: float = 1.0,
        lissa_iterations: int = 10,
        lissa_samples: int = 10,
        lissa_depth: int = 10000,
        use_expectation: bool = True,
        matrix_free: bool = True
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # LiSSA parameters
        self.damping = damping
        self.scale = scale
        self.lissa_iterations = lissa_iterations
        self.lissa_samples = lissa_samples  # Number of random vectors for expectation
        self.lissa_depth = lissa_depth  # Number of iterations for each recursive approximation
        self.use_expectation = use_expectation  # Whether to use expectations for trace estimation
        self.matrix_free = matrix_free  # Whether to use matrix-free HVP
        
        # Move model to device
        self.model.to(self.device)
        
        # Cache for HVP results
        self.hvp_cache = {}
        
        # Create parameter vector for model
        self.param_names = []
        self.param_shapes = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_names.append(name)
                self.param_shapes[name] = param.shape
    
    def _params_to_vector(self) -> torch.Tensor:
        """Convert model parameters to a single vector."""
        return torch.cat([param.detach().view(-1) for param in self.model.parameters() if param.requires_grad])
    
    def _gradients_to_vector(self, gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert gradients to a single vector."""
        return torch.cat([gradients[name].view(-1) for name in self.param_names])
    
    def _vector_to_gradients(self, vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert a vector back to a gradient dictionary."""
        gradients = {}
        
        offset = 0
        for name in self.param_names:
            size = np.prod(self.param_shapes[name])
            gradients[name] = vector[offset:offset + size].view(self.param_shapes[name])
            offset += size
        
        return gradients
    
    def compute_loss_and_gradient(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss and gradient for a single input-target pair."""
        # Forward pass
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Extract gradients
        grads = {name: param.grad.clone() for name, param in self.model.named_parameters() 
                if param.requires_grad and param.grad is not None}
        
        return loss.detach(), self._gradients_to_vector(grads)
    
    def compute_batch_gradients(
        self, 
        dataloader: DataLoader
    ) -> Dict[str, Tuple[str, torch.Tensor]]:
        """Compute and store gradients for all samples in a dataset."""
        gradients = {}
        losses = {}
        
        self.model.eval()  # Ensure model is in eval mode for consistent gradients
        
        for batch in tqdm(dataloader, desc="Computing gradients for all samples"):
            if len(batch) == 2:  # Simple dataset with inputs and targets
                inputs, targets = batch
                ids = [str(i) for i in range(len(inputs))]
            else:  # Custom dataset with sample IDs
                ids = batch["id"]
                inputs = batch["input"]
                targets = batch["target"]
            
            # Convert to tensors if needed
            inputs = inputs.to(self.device) if isinstance(inputs, torch.Tensor) else torch.tensor(inputs, device=self.device)
            targets = targets.to(self.device) if isinstance(targets, torch.Tensor) else torch.tensor(targets, device=self.device)
            
            # Process each sample individually
            for i, id_ in enumerate(ids):
                loss, grad = self.compute_loss_and_gradient(inputs[i:i+1], targets[i:i+1])
                gradients[id_] = grad
                losses[id_] = loss.item()
        
        return {"gradients": gradients, "losses": losses}
    
    def hvp(
        self, 
        vec: torch.Tensor, 
        dataloader: Optional[DataLoader] = None
    ) -> torch.Tensor:
        """
        Compute Hessian-vector product (HVP): H * vec.
        
        For matrix-free method, we use the identity:
            H * v = (∇(∇f(θ)ᵀv)) = (d/dα)∇f(θ + α*v)|α=0
        """
        if dataloader is None:
            dataloader = self.train_dataloader
        
        if self.matrix_free:
            return self._hvp_matrix_free(vec, dataloader)
        else:
            return self._hvp_exact(vec, dataloader)
    
    def _hvp_matrix_free(
        self, 
        vec: torch.Tensor, 
        dataloader: DataLoader
    ) -> torch.Tensor:
        """Compute HVP using the matrix-free method."""
        # Cache key based on vector hash (for repeated calls with same vector)
        vec_hash = hash(vec.cpu().numpy().tobytes())
        cache_key = f"hvp_matrix_free_{vec_hash}"
        
        if cache_key in self.hvp_cache:
            return self.hvp_cache[cache_key]
        
        self.model.eval()
        vector = vec.clone().to(self.device)
        
        # Compute original gradient
        grad_sum = torch.zeros_like(vector)
        num_samples = 0
        
        # Process in batches
        for batch in dataloader:
            if len(batch) == 2:  # Simple dataset with inputs and targets
                inputs, targets = batch
            else:  # Custom dataset with sample IDs
                inputs = batch["input"]
                targets = batch["target"]
            
            # Convert to tensors if needed
            inputs = inputs.to(self.device) if isinstance(inputs, torch.Tensor) else torch.tensor(inputs, device=self.device)
            targets = targets.to(self.device) if isinstance(targets, torch.Tensor) else torch.tensor(targets, device=self.device)
            
            batch_size = inputs.shape[0]
            num_samples += batch_size
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward(create_graph=True)  # Need to create_graph for second derivative
            
            # Get current parameter gradients
            grads = {name: param.grad.clone() for name, param in self.model.named_parameters() 
                    if param.requires_grad and param.grad is not None}
            
            # Compute grad-vec dot product
            dot_prod = torch.sum(self._gradients_to_vector(grads) * vector)
            
            # Compute second derivative
            self.model.zero_grad()
            dot_prod.backward()
            
            # Accumulate second derivatives
            hessian_vec_prod = torch.cat([
                param.grad.detach().view(-1) for param in self.model.named_parameters() 
                if param.requires_grad and param.grad is not None
            ])
            
            grad_sum += hessian_vec_prod
        
        # Normalize by number of samples
        hvp_result = grad_sum / num_samples
        
        # Cache result
        self.hvp_cache[cache_key] = hvp_result
        
        return hvp_result
    
    def _hvp_exact(
        self, 
        vec: torch.Tensor, 
        dataloader: DataLoader
    ) -> torch.Tensor:
        """Compute HVP by explicitly forming the Hessian matrix."""
        # Not implemented for large models due to memory constraints
        raise NotImplementedError("Exact HVP computation is not implemented for large models")
    
    def lissa(
        self, 
        vec: torch.Tensor, 
        dataloader: Optional[DataLoader] = None,
        damping: Optional[float] = None,
        scale: Optional[float] = None,
        iterations: Optional[int] = None,
        samples: Optional[int] = None,
        depth: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute inverse-Hessian-vector product (IHVP): H⁻¹ * vec using LiSSA.
        
        LiSSA uses a recursive approximation:
            H⁻¹v ≈ v + (I - H/μ)H⁻¹v = v + (I - H/μ)v + (I - H/μ)²v + ... + (I - H/μ)ᵏv
        
        where μ is the damping parameter (like a regularization term).
        """
        if dataloader is None:
            dataloader = self.train_dataloader
        
        # Use provided parameters or defaults
        damping = damping or self.damping
        scale = scale or self.scale
        iterations = iterations or self.lissa_iterations
        samples = samples or self.lissa_samples
        depth = depth or self.lissa_depth
        
        # Cache key for repeated calls
        vec_hash = hash(vec.cpu().numpy().tobytes())
        cache_key = f"lissa_{vec_hash}_{damping}_{scale}_{iterations}_{samples}_{depth}"
        
        if cache_key in self.hvp_cache:
            return self.hvp_cache[cache_key]
        
        vector = vec.clone().to(self.device)
        ihvp_estimates = []
        
        # Run multiple trials with different random seeds for variance reduction
        for sample in range(samples):
            # Set a seed for reproducibility
            torch.manual_seed(sample)
            
            # Initialize estimate with scaled vector
            estimate = vector.clone() * scale
            
            # LiSSA recursive approximation
            for j in range(depth):
                # Update estimate using (I - H/μ)ᵏv
                hvp = self.hvp(estimate, dataloader)
                estimate = vector + estimate - hvp / damping
                
                # Early stopping if converged
                if j > 0 and torch.norm(hvp / damping) < 1e-4:
                    break
            
            ihvp_estimates.append(estimate)
        
        # Average across samples
        ihvp_result = torch.stack(ihvp_estimates).mean(dim=0)
        
        # Cache result
        self.hvp_cache[cache_key] = ihvp_result
        
        return ihvp_result
    
    def compute_influence_score(
        self, 
        test_inputs: torch.Tensor, 
        test_targets: torch.Tensor, 
        train_inputs: torch.Tensor, 
        train_targets: torch.Tensor
    ) -> float:
        """
        Compute the influence score of a training sample on a test sample.
        
        The influence score measures how much the model's loss on the test sample would
        change if the training sample were removed from the training set:
            I(z_test, z_train) = -∇L(z_test)ᵀ H⁻¹ ∇L(z_train)
        """
        # Compute test gradient
        _, test_grad = self.compute_loss_and_gradient(test_inputs, test_targets)
        
        # Compute inverse-Hessian-vector product
        ihvp = self.lissa(test_grad)
        
        # Compute training gradient
        _, train_grad = self.compute_loss_and_gradient(train_inputs, train_targets)
        
        # Compute influence score: -∇L(test)ᵀ H⁻¹ ∇L(train)
        influence = -torch.dot(ihvp, train_grad).item()
        
        return influence
    
    def compute_batch_influences(
        self, 
        test_batch: Dict[str, torch.Tensor], 
        train_gradients: Dict[str, torch.Tensor],
        batch_size: int = 1
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute influence scores for a batch of test samples on all training samples.
        
        Returns a nested dictionary mapping test sample IDs to dictionaries of
        training sample IDs to influence scores.
        """
        influence_scores = {}
        
        # Extract test samples
        test_ids = test_batch["id"]
        test_inputs = test_batch["input"].to(self.device)
        test_targets = test_batch["target"].to(self.device)
        
        # Process test samples in small batches
        for i in range(0, len(test_ids), batch_size):
            batch_end = min(i + batch_size, len(test_ids))
            batch_test_ids = test_ids[i:batch_end]
            batch_test_inputs = test_inputs[i:batch_end]
            batch_test_targets = test_targets[i:batch_end]
            
            # Process each test sample individually
            for j, test_id in enumerate(batch_test_ids):
                test_input = batch_test_inputs[j:j+1]
                test_target = batch_test_targets[j:j+1]
                
                # Compute test gradient
                _, test_grad = self.compute_loss_and_gradient(test_input, test_target)
                
                # Compute inverse-Hessian-vector product
                ihvp = self.lissa(test_grad)
                
                # Compute influence scores for all training samples
                scores = {}
                for train_id, train_grad in train_gradients.items():
                    # Compute influence score: -∇L(test)ᵀ H⁻¹ ∇L(train)
                    influence = -torch.dot(ihvp, train_grad).item()
                    scores[train_id] = influence
                
                influence_scores[test_id] = scores
        
        return influence_scores
    
    def save(self, path: str) -> None:
        """Save the influence estimator state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create state dictionary without the model and dataloader
        state = {
            "damping": self.damping,
            "scale": self.scale,
            "lissa_iterations": self.lissa_iterations,
            "lissa_samples": self.lissa_samples,
            "lissa_depth": self.lissa_depth,
            "use_expectation": self.use_expectation,
            "matrix_free": self.matrix_free,
            "param_names": self.param_names,
            "param_shapes": self.param_shapes
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved influence estimator state to {path}")
    
    def load(self, path: str) -> None:
        """Load the influence estimator state from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        # Restore state
        self.damping = state["damping"]
        self.scale = state["scale"]
        self.lissa_iterations = state["lissa_iterations"]
        self.lissa_samples = state["lissa_samples"]
        self.lissa_depth = state["lissa_depth"]
        self.use_expectation = state["use_expectation"]
        self.matrix_free = state["matrix_free"]
        self.param_names = state["param_names"]
        self.param_shapes = state["param_shapes"]
        
        logger.info(f"Loaded influence estimator state from {path}")


class AttributionRefiner:
    """
    Refine candidate attributions using influence functions.
    
    Given a set of candidate training samples identified by the ANN search,
    this class refines the ranking using influence scores to improve attribution accuracy.
    """
    
    def __init__(
        self,
        influence_estimator: InfluenceEstimator,
        top_k: int = 10
    ):
        self.influence_estimator = influence_estimator
        self.top_k = top_k
    
    def refine_candidates(
        self,
        test_sample: Dict[str, torch.Tensor],
        candidate_ids: List[str],
        train_dataset: Dataset,
        id_to_index: Dict[str, int]
    ) -> List[Tuple[str, float]]:
        """
        Refine a list of candidate attributions for a test sample.
        
        Args:
            test_sample: Test sample dictionary with 'id', 'input', 'target' keys
            candidate_ids: List of candidate training sample IDs from ANN search
            train_dataset: Training dataset containing the candidate samples
            id_to_index: Mapping from sample IDs to indices in the training dataset
        
        Returns:
            List of (candidate_id, influence_score) tuples, sorted by influence score
        """
        # Prepare test sample input and target
        test_input = test_sample["input"].to(self.influence_estimator.device)
        test_target = test_sample["target"].to(self.influence_estimator.device)
        
        # Compute influence scores for each candidate
        influence_scores = []
        
        for candidate_id in candidate_ids:
            if candidate_id in id_to_index:
                # Get candidate from training dataset
                idx = id_to_index[candidate_id]
                candidate = train_dataset[idx]
                
                # Extract input and target
                candidate_input = candidate["input"].to(self.influence_estimator.device)
                candidate_target = candidate["target"].to(self.influence_estimator.device)
                
                # Compute influence score
                score = self.influence_estimator.compute_influence_score(
                    test_input, test_target, candidate_input, candidate_target
                )
                
                influence_scores.append((candidate_id, score))
        
        # Sort by influence score (higher is more influential)
        influence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return influence_scores[:self.top_k]
    
    def batch_refine_candidates(
        self,
        test_samples: List[Dict[str, torch.Tensor]],
        candidate_ids_list: List[List[str]],
        train_dataset: Dataset,
        id_to_index: Dict[str, int]
    ) -> List[List[Tuple[str, float]]]:
        """
        Refine candidate attributions for a batch of test samples.
        
        Args:
            test_samples: List of test sample dictionaries
            candidate_ids_list: List of lists of candidate IDs (one per test sample)
            train_dataset: Training dataset containing the candidate samples
            id_to_index: Mapping from sample IDs to indices in the training dataset
        
        Returns:
            List of lists of (candidate_id, influence_score) tuples, sorted by influence score
        """
        # Process each test sample individually
        refined_candidates = []
        
        for i, test_sample in enumerate(test_samples):
            candidate_ids = candidate_ids_list[i]
            refined = self.refine_candidates(test_sample, candidate_ids, train_dataset, id_to_index)
            refined_candidates.append(refined)
        
        return refined_candidates


# Command-line interface for testing
if __name__ == "__main__":
    import sys
    import argparse
    import matplotlib.pyplot as plt
    import torch.optim as optim
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.probe import ProbeNetwork
    
    parser = argparse.ArgumentParser(description="Test influence function approximation")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden layer dimension")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters (output dimension)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples for testing")
    parser.add_argument("--lissa_samples", type=int, default=5, help="Number of LiSSA trials")
    parser.add_argument("--lissa_depth", type=int, default=100, help="Depth of LiSSA recursion")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for dataloader")
    parser.add_argument("--damping", type=float, default=0.01, help="Damping parameter")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create synthetic data for testing
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    n_clusters = args.n_clusters
    num_samples = args.num_samples
    
    # Create random embeddings and labels
    np.random.seed(42)
    torch.manual_seed(42)
    
    embeddings = torch.randn(num_samples, embedding_dim)
    labels = torch.randint(0, n_clusters, (num_samples,))
    
    # Create dataset
    dataset = TensorDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create probe network
    probe = ProbeNetwork(
        input_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=n_clusters,
        num_layers=2,
        dropout=0.1
    )
    
    # Train the probe network briefly
    optimizer = optim.Adam(probe.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        for batch_embeddings, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = probe(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Create influence estimator
    influence_estimator = InfluenceEstimator(
        model=probe,
        loss_fn=criterion,
        train_dataloader=dataloader,
        damping=args.damping,
        lissa_samples=args.lissa_samples,
        lissa_depth=args.lissa_depth
    )
    
    # Test a simple influence calculation
    test_idx = np.random.randint(0, num_samples)
    test_embedding = embeddings[test_idx:test_idx+1]
    test_label = labels[test_idx:test_idx+1]
    
    train_idx = np.random.randint(0, num_samples)
    train_embedding = embeddings[train_idx:train_idx+1]
    train_label = labels[train_idx:train_idx+1]
    
    # Compute influence
    influence = influence_estimator.compute_influence_score(
        test_embedding, test_label, train_embedding, train_label
    )
    
    print(f"Influence of training sample {train_idx} on test sample {test_idx}: {influence:.6f}")
    
    # Test batch influence computation with a few samples
    num_test = 10
    test_indices = np.random.choice(num_samples, num_test, replace=False)
    train_indices = np.random.choice(num_samples, 20, replace=False)
    
    # Compute test gradients
    test_gradients = {}
    for i in test_indices:
        test_embedding = embeddings[i:i+1]
        test_label = labels[i:i+1]
        _, test_grad = influence_estimator.compute_loss_and_gradient(test_embedding, test_label)
        test_gradients[str(i)] = test_grad
    
    # Compute train gradients
    train_gradients = {}
    for i in train_indices:
        train_embedding = embeddings[i:i+1]
        train_label = labels[i:i+1]
        _, train_grad = influence_estimator.compute_loss_and_gradient(train_embedding, train_label)
        train_gradients[str(i)] = train_grad
    
    # Compute influence scores for all test-train pairs
    influence_matrix = np.zeros((len(test_indices), len(train_indices)))
    
    for test_idx, test_i in enumerate(test_indices):
        test_grad = test_gradients[str(test_i)]
        ihvp = influence_estimator.lissa(test_grad)
        
        for train_idx, train_i in enumerate(train_indices):
            train_grad = train_gradients[str(train_i)]
            influence = -torch.dot(ihvp, train_grad).item()
            influence_matrix[test_idx, train_idx] = influence
    
    # Plot influence heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(influence_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Influence score')
    plt.xlabel('Training sample index')
    plt.ylabel('Test sample index')
    plt.title('Influence scores between test and training samples')
    plt.savefig(os.path.join(args.output_dir, 'influence_heatmap.png'))
    
    # Summary
    print(f"Influence matrix shape: {influence_matrix.shape}")
    print(f"Mean influence: {np.mean(influence_matrix):.6f}")
    print(f"Std influence: {np.std(influence_matrix):.6f}")
    print(f"Min influence: {np.min(influence_matrix):.6f}")
    print(f"Max influence: {np.max(influence_matrix):.6f}")
    
    # Save results
    results = {
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "n_clusters": n_clusters,
        "num_samples": num_samples,
        "lissa_samples": args.lissa_samples,
        "lissa_depth": args.lissa_depth,
        "damping": args.damping,
        "test_indices": test_indices.tolist(),
        "train_indices": train_indices.tolist(),
        "influence_matrix": influence_matrix.tolist(),
        "mean_influence": float(np.mean(influence_matrix)),
        "std_influence": float(np.std(influence_matrix)),
        "min_influence": float(np.min(influence_matrix)),
        "max_influence": float(np.max(influence_matrix))
    }
    
    with open(os.path.join(args.output_dir, 'influence_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output_dir}/influence_results.json")
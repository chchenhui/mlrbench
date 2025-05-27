"""
Implementation of the targeted low-rank gradient surgery component for the 
Cluster-Driven Certified Unlearning method.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad


class TargetedLowRankGradientSurgery:
    """
    Class for performing targeted low-rank gradient surgery within affected subspaces
    to erase memorized data without retraining the full model.
    """
    
    def __init__(self, model, learning_rates=None, adapter_rank=4):
        """
        Initialize the gradient surgery module.
        
        Args:
            model (torch.nn.Module): The LLM model
            learning_rates (dict, optional): Dictionary mapping cluster indices to learning rates
            adapter_rank (int): Rank of the low-rank adapters for sequential deletions
        """
        self.model = model
        self.learning_rates = learning_rates or {}
        self.adapter_rank = adapter_rank
        self.adapters = nn.ModuleDict()  # For storing adapters for sequential deletions
        
    def compute_deletion_gradient(self, deletion_set):
        """
        Compute the aggregate gradient on the deletion set.
        
        Args:
            deletion_set (list): List of (inputs, targets) tuples for data to be deleted
            
        Returns:
            grad_list (list): List of gradients for each parameter
        """
        self.model.eval()  # Set model to evaluation mode
        
        # Initialize gradients with zeros
        grad_list = [torch.zeros_like(p) for p in self.model.parameters()]
        
        # Accumulate gradients for each example in deletion set
        for inputs, targets in deletion_set:
            # Forward pass
            outputs = self.model(**inputs)
            
            # Compute loss
            if hasattr(self.model, "compute_loss"):
                loss = self.model.compute_loss(outputs, targets)
            else:
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                
                # Reshape if needed
                if logits.shape[:-1] != targets.shape:
                    targets = targets.view(-1)
                    logits = logits.view(-1, logits.size(-1))
                    
                loss = loss_fct(logits, targets)
            
            # Compute gradients
            example_grads = grad(loss, self.model.parameters())
            
            # Accumulate gradients
            for i, g in enumerate(example_grads):
                grad_list[i] += g
                
        # Average gradients
        n_examples = len(deletion_set)
        for i in range(len(grad_list)):
            grad_list[i] /= n_examples
            
        return grad_list
        
    def project_gradient(self, grad_list, cluster_basis):
        """
        Project gradient onto cluster subspace.
        
        Args:
            grad_list (list): List of gradients for each parameter
            cluster_basis (torch.Tensor): Orthonormal basis for the cluster
            
        Returns:
            projected_grad (list): Projected gradients
            orthogonal_grad (list): Orthogonal components of gradients
        """
        # Convert basis to tensor if it's numpy
        if isinstance(cluster_basis, np.ndarray):
            device = grad_list[0].device
            cluster_basis = torch.tensor(cluster_basis, device=device)
            
        # Initialize projected and orthogonal gradients
        projected_grad = [torch.zeros_like(g) for g in grad_list]
        orthogonal_grad = [g.clone() for g in grad_list]
        
        # Project each parameter's gradient
        for i, g in enumerate(grad_list):
            # Flatten gradient
            flat_g = g.view(-1)
            
            # If flat_g is smaller than basis, pad it
            if flat_g.size(0) < cluster_basis.size(0):
                padding = torch.zeros(cluster_basis.size(0) - flat_g.size(0), device=flat_g.device)
                flat_g = torch.cat([flat_g, padding])
            
            # If flat_g is larger than basis, truncate basis
            if flat_g.size(0) > cluster_basis.size(0):
                cluster_basis = torch.nn.functional.pad(
                    cluster_basis, 
                    (0, 0, 0, flat_g.size(0) - cluster_basis.size(0))
                )
                
            # Project gradient onto subspace: g_k = U_k U_k^T g
            projected = cluster_basis @ (cluster_basis.T @ flat_g)
            
            # Reshape projected gradient back to original shape
            projected_reshaped = projected[:g.numel()].view(g.shape)
            
            # Store projected gradient
            projected_grad[i] = projected_reshaped
            
            # Compute orthogonal component: g_perp_k = g - g_k
            orthogonal_grad[i] = g - projected_reshaped
            
        return projected_grad, orthogonal_grad
        
    def apply_gradient_update(self, cluster_idx, projected_grad, learning_rate=None):
        """
        Apply the targeted gradient update to erase information in the cluster.
        
        Args:
            cluster_idx (int): Cluster index
            projected_grad (list): Projected gradients for the cluster
            learning_rate (float, optional): Learning rate for the update
            
        Returns:
            updated_params (list): Updated parameters
        """
        # Use provided learning rate or look up in dictionary
        if learning_rate is None:
            learning_rate = self.learning_rates.get(cluster_idx, 0.01)
            
        # Get current parameters
        current_params = [p.clone() for p in self.model.parameters()]
        
        # Apply targeted gradient update: θ' = θ - η * g_k
        updated_params = []
        for p, g_k in zip(current_params, projected_grad):
            updated_params.append(p - learning_rate * g_k)
            
        return updated_params
        
    def create_low_rank_adapter(self, cluster_idx, deletion_set, cluster_basis):
        """
        Create a low-rank adapter for the cluster to handle sequential deletions.
        
        Args:
            cluster_idx (int): Cluster index
            deletion_set (list): List of (inputs, targets) tuples for data to be deleted
            cluster_basis (torch.Tensor): Orthonormal basis for the cluster
            
        Returns:
            adapter (nn.Module): Low-rank adapter for the cluster
        """
        # Create a unique name for the adapter
        adapter_name = f"adapter_{cluster_idx}"
        
        # Compute deletion gradient
        deletion_grad = self.compute_deletion_gradient(deletion_set)
        
        # Project gradient onto cluster subspace
        projected_grad, _ = self.project_gradient(deletion_grad, cluster_basis)
        
        # Create adapter module
        class LowRankAdapter(nn.Module):
            def __init__(self, original_params, projected_gradients, rank, learning_rate):
                super().__init__()
                self.original_params = [p.detach().clone() for p in original_params]
                self.rank = rank
                self.learning_rate = learning_rate
                
                # Create low-rank matrices for each parameter
                self.weight_As = nn.ParameterList()
                self.weight_Bs = nn.ParameterList()
                
                for i, (param, grad) in enumerate(zip(original_params, projected_gradients)):
                    # Skip parameters with no gradient
                    if grad.norm() < 1e-10:
                        self.weight_As.append(nn.Parameter(torch.zeros(1, 1)))
                        self.weight_Bs.append(nn.Parameter(torch.zeros(1, 1)))
                        continue
                        
                    # If the parameter is 1D (e.g., bias), reshape it
                    if len(param.shape) == 1:
                        param_shape = (1, param.shape[0])
                        reshaped_grad = grad.view(param_shape)
                    else:
                        param_shape = param.shape
                        reshaped_grad = grad
                    
                    # Perform SVD on the gradient
                    try:
                        U, S, V = torch.svd(reshaped_grad)
                    
                        # Keep only top-k singular values/vectors
                        k = min(rank, min(U.shape[1], V.shape[1]))
                        
                        # Scale singular vectors by sqrt(S) and learning_rate
                        scale = torch.sqrt(S[:k]) * torch.sqrt(learning_rate)
                        
                        # Create low-rank matrices
                        A = U[:, :k] * scale.view(1, -1)
                        B = V[:, :k] * scale.view(1, -1)
                        
                        # Store as parameters
                        self.weight_As.append(nn.Parameter(A))
                        self.weight_Bs.append(nn.Parameter(B))
                    except:
                        # Fallback if SVD fails
                        self.weight_As.append(nn.Parameter(torch.zeros(1, 1)))
                        self.weight_Bs.append(nn.Parameter(torch.zeros(1, 1)))
            
            def forward(self, x):
                """Apply the adapter (not actually used, just for compatibility)"""
                return x
                
            def get_adapted_parameters(self):
                """Get the parameters after applying the adapter"""
                adapted_params = []
                
                for i, orig_param in enumerate(self.original_params):
                    # Skip parameters with dummy adapters
                    if self.weight_As[i].size(0) == 1 and self.weight_As[i].size(1) == 1:
                        adapted_params.append(orig_param.clone())
                        continue
                        
                    # If the parameter is 1D, reshape the adapter matrices
                    if len(orig_param.shape) == 1:
                        # For bias terms: Compute A @ B^T and reshape back to 1D
                        delta = (self.weight_As[i] @ self.weight_Bs[i].t()).view(-1)
                        adapted_params.append(orig_param - delta)
                    else:
                        # For weight matrices: Compute orig - A @ B^T
                        adapted_params.append(
                            orig_param - self.weight_As[i] @ self.weight_Bs[i].t()
                        )
                        
                return adapted_params
        
        # Create and store the adapter
        adapter = LowRankAdapter(
            self.model.parameters(),
            projected_grad,
            self.adapter_rank,
            self.learning_rates.get(cluster_idx, 0.01)
        )
        
        self.adapters[adapter_name] = adapter
        
        return adapter
        
    def update_model_with_adapters(self):
        """
        Update the model parameters using all accumulated adapters.
        
        Returns:
            None: Updates parameters in-place
        """
        if not self.adapters:
            return
            
        # Get original parameters
        orig_params = list(self.model.parameters())
        
        # Initialize adapted parameters with original values
        adapted_params = [p.detach().clone() for p in orig_params]
        
        # Apply each adapter sequentially
        for adapter_name, adapter in self.adapters.items():
            adapter_params = adapter.get_adapted_parameters()
            
            # Add adapter's contribution
            for i, (adapted, adapter_p) in enumerate(zip(adapted_params, adapter_params)):
                adapted_params[i] = adapter_p
                
        # Update model parameters
        with torch.no_grad():
            for i, (param, adapted) in enumerate(zip(orig_params, adapted_params)):
                param.copy_(adapted)
"""
Implementation of the influence score approximation component for identifying clusters 
that encode information about the deletion set.
"""

import numpy as np
import torch
from torch.autograd import grad


class InfluenceScoreApproximation:
    """
    Class for approximating influence scores to identify clusters encoding information
    about the deletion set.
    """
    
    def __init__(self, model, threshold=0.1, ridge_lambda=0.01, neumann_steps=10):
        """
        Initialize the influence score approximation module.
        
        Args:
            model (torch.nn.Module): The LLM model
            threshold (float): Threshold for influence scores to mark clusters for intervention
            ridge_lambda (float): Regularization parameter for Hessian stabilization
            neumann_steps (int): Number of steps for Neumann series approximation
        """
        self.model = model
        self.threshold = threshold
        self.ridge_lambda = ridge_lambda
        self.neumann_steps = neumann_steps
        
    def compute_loss_gradient(self, inputs, targets, model_output=None):
        """
        Compute the gradient of the loss with respect to model parameters.
        
        Args:
            inputs (dict): Input tensors (ids, attention_mask, etc.)
            targets (torch.Tensor): Target tokens
            model_output (torch.Tensor, optional): Pre-computed model output to save computation
            
        Returns:
            loss_grad (list): List of gradients for each parameter
        """
        # Forward pass if model_output not provided
        if model_output is None:
            model_output = self.model(**inputs)
        
        # Compute loss (typically cross-entropy for language models)
        if hasattr(self.model, "compute_loss"):
            # Some models have their own loss computation
            loss = self.model.compute_loss(model_output, targets)
        else:
            # Default to cross-entropy loss for language modeling
            logits = model_output.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            
            # Reshape if needed
            if logits.shape[:-1] != targets.shape:
                targets = targets.view(-1)
                logits = logits.view(-1, logits.size(-1))
                
            loss = loss_fct(logits, targets)
        
        # Compute gradients
        loss_grad = grad(loss, self.model.parameters(), create_graph=True)
        
        return list(loss_grad)
        
    def compute_hvp_neumann(self, v, inputs, targets):
        """
        Compute Hessian-vector product (HVP) using Neumann series approximation.
        This approximates H^(-1)v where H is the Hessian of the loss.
        
        Args:
            v (list): List of gradients or vectors matching parameter shapes
            inputs (dict): Input tensors
            targets (torch.Tensor): Target tokens
            
        Returns:
            result (list): Approximation of H^(-1)v
        """
        # Compute initial gradients for R(θ)
        model_output = self.model(**inputs)
        
        # Add small ridge regularization
        if hasattr(self.model, "compute_loss"):
            reg_loss = self.model.compute_loss(model_output, targets)
        else:
            logits = model_output.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            
            # Reshape if needed
            if logits.shape[:-1] != targets.shape:
                targets = targets.view(-1)
                logits = logits.view(-1, logits.size(-1))
                
            reg_loss = loss_fct(logits, targets)
        
        # Add ridge regularization
        for param in self.model.parameters():
            reg_loss += self.ridge_lambda * (param ** 2).sum() / 2
        
        # Compute gradients with respect to regularized loss
        grads = grad(reg_loss, self.model.parameters(), create_graph=True)
        
        # Compute gradient-vector products iteratively for Neumann approximation
        result = [torch.zeros_like(p) for p in self.model.parameters()]
        curr_v = [v_i.clone() for v_i in v]
        
        for i in range(self.neumann_steps):
            # Compute gradient-vector product
            gvp = self.compute_gvp(curr_v, grads)
            
            # Update result with series term
            for j, (res_j, v_j) in enumerate(zip(result, curr_v)):
                result[j] = res_j + v_j
                
            # Update v for next iteration (v ← v - gvp + λv)
            for j, (v_j, gvp_j) in enumerate(zip(curr_v, gvp)):
                curr_v[j] = v_j - gvp_j + self.ridge_lambda * v_j
                
        return result
        
    def compute_gvp(self, v, grads):
        """
        Compute gradient-vector product.
        
        Args:
            v (list): List of vectors
            grads (list): List of gradients
            
        Returns:
            gvp (list): Gradient-vector product
        """
        # Compute dot product between gradients and vectors
        dot_product = 0
        for g_i, v_i in zip(grads, v):
            dot_product += torch.sum(g_i * v_i)
            
        # Compute gradient of dot product with respect to parameters
        gvp = grad(dot_product, self.model.parameters(), create_graph=True)
        
        return list(gvp)
        
    def compute_cluster_influence(self, deletion_set, cluster_bases, validation_data=None):
        """
        Compute influence scores for each cluster with respect to the deletion set.
        
        Args:
            deletion_set (list): List of (inputs, targets) tuples for data to be deleted
            cluster_bases (dict): Dictionary mapping cluster indices to orthonormal bases
            validation_data (list, optional): Validation data for computing R(θ)
            
        Returns:
            influence_scores (dict): Dictionary mapping cluster indices to influence scores
        """
        if validation_data is None:
            # If no validation data provided, use a subset of the deletion set
            validation_data = deletion_set[:min(10, len(deletion_set))]
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Compute gradients for R(θ) on validation data
        r_grads = []
        for inputs, targets in validation_data:
            grad_i = self.compute_loss_gradient(inputs, targets)
            r_grads.append(grad_i)
            
        # Average gradients
        avg_r_grad = []
        for i in range(len(r_grads[0])):
            avg_r_grad.append(torch.mean(torch.stack([g[i] for g in r_grads]), dim=0))
        
        # Compute Hessian inverse approximation using validation data
        h_inv_r_grad = self.compute_hvp_neumann(avg_r_grad, validation_data[0][0], validation_data[0][1])
        
        # Compute gradients for each example in deletion set
        deletion_grads = []
        for inputs, targets in deletion_set:
            grad_i = self.compute_loss_gradient(inputs, targets)
            deletion_grads.append(grad_i)
            
        # Average gradients
        avg_deletion_grad = []
        for i in range(len(deletion_grads[0])):
            avg_deletion_grad.append(torch.mean(torch.stack([g[i] for g in deletion_grads]), dim=0))
        
        # Compute influence scores for each cluster
        influence_scores = {}
        
        for cluster_idx, basis in cluster_bases.items():
            # Convert basis to tensor if it's numpy
            if isinstance(basis, np.ndarray):
                basis = torch.tensor(basis, device=avg_deletion_grad[0].device)
                
            # Initialize cluster influence score
            I_k = 0
            
            # Project gradients onto cluster subspace and compute dot product
            for i, (del_grad_i, r_grad_i) in enumerate(zip(avg_deletion_grad, h_inv_r_grad)):
                # Flatten gradients for the projection
                flat_del_grad = del_grad_i.view(-1)
                flat_r_grad = r_grad_i.view(-1)
                
                # If basis is too large, we'll use a smaller random projection
                if basis.shape[1] > 1000:  # Arbitrary threshold
                    random_indices = np.random.choice(basis.shape[1], 1000, replace=False)
                    reduced_basis = basis[:, random_indices]
                    
                    # Project onto smaller basis
                    projection = reduced_basis @ (reduced_basis.T @ flat_del_grad)
                else:
                    # Project onto full basis
                    projection = basis @ (basis.T @ flat_del_grad)
                    
                # Compute dot product with R(θ) gradient
                I_k += torch.dot(projection, flat_r_grad)
                
            # Store influence score
            influence_scores[cluster_idx] = I_k.item()
            
        return influence_scores
        
    def identify_affected_clusters(self, influence_scores):
        """
        Identify clusters that should be targeted for unlearning based on influence scores.
        
        Args:
            influence_scores (dict): Dictionary mapping cluster indices to influence scores
            
        Returns:
            affected_clusters (list): List of cluster indices to target for unlearning
        """
        # Sort clusters by absolute influence score
        sorted_clusters = sorted(influence_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Select clusters above threshold
        affected_clusters = [cluster_idx for cluster_idx, score in sorted_clusters 
                            if abs(score) > self.threshold]
        
        return affected_clusters
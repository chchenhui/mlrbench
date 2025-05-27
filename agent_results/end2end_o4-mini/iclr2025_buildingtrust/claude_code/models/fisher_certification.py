"""
Implementation of the Fisher information certification component for the 
Cluster-Driven Certified Unlearning method.
"""

import math
import numpy as np
import torch
from torch.autograd import grad


class FisherInformationCertification:
    """
    Class for quantifying the divergence between the original and unlearned models
    via a second-order approximation using the Fisher Information Matrix.
    """
    
    def __init__(self, original_model, unlearned_model, epsilon=0.1, n_samples=100):
        """
        Initialize the certification module.
        
        Args:
            original_model (torch.nn.Module): Original LLM model
            unlearned_model (torch.nn.Module): Unlearned LLM model
            epsilon (float): KL divergence threshold for certification
            n_samples (int): Number of samples to use for estimating the Fisher Information Matrix
        """
        self.original_model = original_model
        self.unlearned_model = unlearned_model
        self.epsilon = epsilon
        self.n_samples = n_samples
        
    def compute_parameter_difference(self):
        """
        Compute the difference between original and unlearned model parameters.
        
        Returns:
            delta_theta (list): List of parameter differences
        """
        original_params = list(self.original_model.parameters())
        unlearned_params = list(self.unlearned_model.parameters())
        
        delta_theta = []
        for orig_p, unl_p in zip(original_params, unlearned_params):
            delta_theta.append(unl_p - orig_p)
            
        return delta_theta
        
    def estimate_fisher_matrix(self, data_loader):
        """
        Estimate the Fisher Information Matrix using samples from the data distribution.
        
        Args:
            data_loader: DataLoader providing samples from the data distribution
            
        Returns:
            fisher_diag (list): Diagonal of the Fisher Information Matrix
        """
        # Set model to evaluation mode
        self.original_model.eval()
        
        # Initialize diagonal Fisher matrix with zeros
        fisher_diag = [torch.zeros_like(p) for p in self.original_model.parameters()]
        
        # Accumulate Fisher information
        sample_count = 0
        for batch in data_loader:
            # Break if we've processed enough samples
            if sample_count >= self.n_samples:
                break
                
            # Get inputs and compute model output
            inputs = batch[0]  # Assuming batch is (inputs, targets)
            
            # Forward pass
            outputs = self.original_model(**inputs)
            
            # Sample from the model's predicted distribution
            logits = outputs.logits
            probs = torch.softmax(logits.view(-1, logits.size(-1)), dim=-1)
            
            # Sample indices from the probability distribution
            sampled_indices = torch.multinomial(probs, 1).view(-1)
            
            # Construct one-hot targets from sampled indices
            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(1, sampled_indices.unsqueeze(1), 1)
            
            # Compute log-likelihood
            log_probs = torch.log_softmax(logits.view(-1, logits.size(-1)), dim=-1)
            log_likelihood = torch.sum(one_hot * log_probs)
            
            # Compute gradients of log-likelihood
            log_likelihood_grads = grad(log_likelihood, self.original_model.parameters(), 
                                        create_graph=False, retain_graph=False)
            
            # Accumulate outer product (diagonal only for efficiency)
            for i, g in enumerate(log_likelihood_grads):
                fisher_diag[i] += g * g
                
            # Update sample count
            sample_count += inputs['input_ids'].size(0)
            
            # Break if enough samples processed
            if sample_count >= self.n_samples:
                break
                
        # Normalize by number of samples
        for i in range(len(fisher_diag)):
            fisher_diag[i] /= sample_count
            
        return fisher_diag
        
    def compute_kl_divergence(self, delta_theta, fisher_diag):
        """
        Compute the KL divergence between the original and unlearned models
        using the parameter difference and Fisher Information Matrix.
        
        Args:
            delta_theta (list): Difference between model parameters
            fisher_diag (list): Diagonal of the Fisher Information Matrix
            
        Returns:
            kl_div (float): Estimated KL divergence
        """
        # Initialize KL divergence
        kl_div = 0.0
        
        # Compute KL divergence as 0.5 * delta_theta^T * F * delta_theta
        for delta, fisher in zip(delta_theta, fisher_diag):
            # Multiply element-wise and sum
            delta_fisher_delta = torch.sum(delta * fisher * delta)
            kl_div += 0.5 * delta_fisher_delta.item()
            
        return kl_div
        
    def certify(self, data_loader, refine_callback=None):
        """
        Certify that the unlearning operation meets the specified KL divergence bound.
        
        Args:
            data_loader: DataLoader providing samples from the data distribution
            refine_callback (callable, optional): Callback function for refining the unlearned model
                                                 if certification fails
            
        Returns:
            is_certified (bool): Whether the unlearning is certified
            kl_div (float): Computed KL divergence
            certificate (dict): Certificate details
        """
        # Compute parameter difference
        delta_theta = self.compute_parameter_difference()
        
        # Estimate Fisher Information Matrix
        fisher_diag = self.estimate_fisher_matrix(data_loader)
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence(delta_theta, fisher_diag)
        
        # Check if KL divergence is within the specified bound
        is_certified = kl_div <= self.epsilon
        
        # Create certificate
        certificate = {
            'is_certified': is_certified,
            'kl_divergence': kl_div,
            'threshold_epsilon': self.epsilon,
            'confidence_level': 1 - math.exp(-self.epsilon),  # Approximate confidence level
            'samples_used': self.n_samples
        }
        
        # If not certified and a refine callback is provided, attempt refinement
        if not is_certified and refine_callback is not None:
            # Call the refinement callback
            self.unlearned_model = refine_callback(self.unlearned_model, kl_div, self.epsilon)
            
            # Recompute certification with refined model
            return self.certify(data_loader)
            
        return is_certified, kl_div, certificate
        
    def generate_certificate(self, certificate, output_file=None):
        """
        Generate a human-readable certificate document.
        
        Args:
            certificate (dict): Certificate details
            output_file (str, optional): Path to save the certificate
            
        Returns:
            certificate_text (str): Text representation of the certificate
        """
        # Create certificate text
        certificate_text = "MACHINE UNLEARNING CERTIFICATE\n"
        certificate_text += "==============================\n\n"
        
        certificate_text += f"Certification Status: {'✓ CERTIFIED' if certificate['is_certified'] else '✗ NOT CERTIFIED'}\n"
        certificate_text += f"KL Divergence: {certificate['kl_divergence']:.6f}\n"
        certificate_text += f"Threshold Epsilon: {certificate['threshold_epsilon']:.6f}\n"
        certificate_text += f"Statistical Confidence Level: {certificate['confidence_level']*100:.2f}%\n"
        certificate_text += f"Samples Used for Estimation: {certificate['samples_used']}\n\n"
        
        certificate_text += "This certificate confirms that the unlearned model "
        
        if certificate['is_certified']:
            certificate_text += "successfully removed the requested information "
            certificate_text += "without introducing significant distributional shifts. "
            certificate_text += f"With {certificate['confidence_level']*100:.2f}% confidence, "
            certificate_text += "the KL divergence between the original and unlearned models "
            certificate_text += f"is at most {certificate['threshold_epsilon']:.6f}."
        else:
            certificate_text += "DID NOT successfully remove the requested information "
            certificate_text += "without introducing significant distributional shifts. "
            certificate_text += f"The KL divergence of {certificate['kl_divergence']:.6f} "
            certificate_text += f"exceeds the specified threshold of {certificate['threshold_epsilon']:.6f}."
            
        # Add timestamp
        import datetime
        certificate_text += f"\n\nCertificate generated on: {datetime.datetime.now()}\n"
        
        # Save to file if output_file is provided
        if output_file:
            with open(output_file, 'w') as f:
                f.write(certificate_text)
                
        return certificate_text
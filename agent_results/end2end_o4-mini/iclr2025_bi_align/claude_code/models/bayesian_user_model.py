#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bayesian User Model implementation for UDRA.

This module handles the Bayesian inference of user preferences based on feedback.
"""

import numpy as np
from scipy.stats import multivariate_normal

class BayesianUserModel:
    """
    Bayesian inference model for user preferences.
    
    This model maintains a posterior distribution over user preference vectors
    and updates it based on observed user corrections.
    """
    
    def __init__(self, feature_dim, prior_mean=None, prior_cov=None):
        """
        Initialize the Bayesian user model.
        
        Args:
            feature_dim (int): Dimension of the feature vector
            prior_mean (np.ndarray, optional): Prior mean of preference vector
            prior_cov (np.ndarray, optional): Prior covariance matrix of preference vector
        """
        self.feature_dim = feature_dim
        
        # Set default prior if not provided
        if prior_mean is None:
            # Start with uniform preference weights
            self.mean = np.zeros(feature_dim)
        else:
            self.mean = prior_mean
            
        if prior_cov is None:
            # Start with independent dimensions with unit variance
            self.cov = np.eye(feature_dim)
        else:
            self.cov = prior_cov
            
        # Keep track of all feedback points for debugging/visualization
        self.feedback_history = []
    
    def update(self, state, action_chosen, action_available, features_map, learning_rate=0.1):
        """
        Update the preference model based on user feedback.
        
        Args:
            state (np.ndarray): Current state
            action_chosen (int/np.ndarray): Action chosen by the user
            action_available (list): List of available actions (including chosen)
            features_map (callable): Function that maps (state, action) to features
            learning_rate (float): Learning rate for update (controls update speed)
            
        Returns:
            updated_mean (np.ndarray): Updated posterior mean
            updated_cov (np.ndarray): Updated posterior covariance
        """
        # Store feedback for later analysis
        self.feedback_history.append((state, action_chosen))
        
        # Compute features for chosen action and alternatives
        chosen_features = features_map(state, action_chosen)
        
        # Check action type to determine how to process alternatives
        if isinstance(action_chosen, (int, np.integer)):
            # Discrete action case
            alternative_features = [features_map(state, a) for a in action_available
                                   if a != action_chosen]
        else:
            # Continuous action case (just sample some alternatives)
            # Here we sample random alternatives from a Gaussian around the current mean
            alternatives = []
            for _ in range(5):  # Generate 5 alternative actions
                alt = np.random.normal(0, 1, len(action_chosen))
                alt = alt / (np.linalg.norm(alt) + 1e-8)  # Normalize
                alternatives.append(alt)
            
            alternative_features = [features_map(state, a) for a in alternatives]
        
        # Apply Bayesian preference update using Laplace approximation
        # We use a simplified approach for computational efficiency
        
        # 1. Compute utilities for chosen and alternative actions
        u_chosen = np.dot(self.mean, chosen_features)
        u_alternatives = [np.dot(self.mean, feat) for feat in alternative_features]
        
        # 2. Compute softmax probabilities (likelihood of user choosing each action)
        max_u = max(u_chosen, *u_alternatives)
        exp_u_chosen = np.exp(u_chosen - max_u)
        exp_u_alternatives = [np.exp(u - max_u) for u in u_alternatives]
        
        # Normalize to get probabilities
        total_exp = exp_u_chosen + sum(exp_u_alternatives)
        p_chosen = exp_u_chosen / total_exp
        
        # 3. Compute gradient and Hessian of log likelihood
        # Gradient: difference between chosen features and expected features
        expected_features = p_chosen * chosen_features
        for i, alt_feat in enumerate(alternative_features):
            p_alt = exp_u_alternatives[i] / total_exp
            expected_features += p_alt * alt_feat
        
        gradient = chosen_features - expected_features
        
        # 4. Update mean using gradient
        # Scale by learning rate for more stable updates
        self.mean = self.mean + learning_rate * np.dot(self.cov, gradient)
        
        # 5. Update covariance using a simplified approach
        # Instead of full Laplace, we just shrink the covariance slightly
        # This prevents collapse and maintains exploration
        shrink_factor = 1.0 - 0.1 * learning_rate
        self.cov = shrink_factor * self.cov
        
        # Return the updated parameters
        return self.mean, self.cov
    
    def predict_preference(self, features):
        """
        Predict user preference (utility) for a given feature vector.
        
        Args:
            features (np.ndarray): Feature vector
            
        Returns:
            utility (float): Predicted utility
            uncertainty (float): Uncertainty in the prediction
        """
        # Compute utility using current estimate of preference vector
        utility = np.dot(self.mean, features)
        
        # Compute uncertainty in this prediction from covariance
        # Higher values indicate greater uncertainty
        uncertainty = np.sqrt(np.dot(features.T, np.dot(self.cov, features)))
        
        return utility, uncertainty
    
    def reset(self):
        """Reset the model to initial state."""
        self.mean = np.zeros(self.feature_dim)
        self.cov = np.eye(self.feature_dim)
        self.feedback_history = []
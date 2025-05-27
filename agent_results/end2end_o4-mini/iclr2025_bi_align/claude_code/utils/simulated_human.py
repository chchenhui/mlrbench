#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulated Human module for UDRA experiments.

This module implements a simulated human that provides feedback to the agent
based on its true preferences and the agent's actions and uncertainty.
"""

import numpy as np

class SimulatedHuman:
    """
    Simulated human for providing feedback in experiments.
    
    The simulated human has a true preference vector and provides
    corrections when the agent's actions deviate too much from its preferences.
    It also responds to uncertainty signals from the agent.
    """
    
    def __init__(self, true_preference, feature_dim, correction_threshold=0.2, 
                 noise_level=0.05, uncertainty_sensitivity=0.3):
        """
        Initialize the simulated human.
        
        Args:
            true_preference (np.ndarray): True preference vector of the simulated human
            feature_dim (int): Dimension of the feature space
            correction_threshold (float): Threshold for providing corrections
            noise_level (float): Level of noise in human feedback
            uncertainty_sensitivity (float): How sensitive the human is to agent uncertainty
        """
        self.true_preference = true_preference
        self.feature_dim = feature_dim
        self.correction_threshold = correction_threshold
        self.noise_level = noise_level
        self.uncertainty_sensitivity = uncertainty_sensitivity
        
        # Create a feature projection for mapping state-action pairs to features
        np.random.seed(42)  # For reproducibility
        self.feature_projection = np.random.randn(feature_dim, 20)  # Assuming state+action <= 20 dim
    
    def compute_features(self, state, action):
        """
        Compute feature representation φ(s,a) for state-action pair.
        
        Args:
            state: State vector
            action: Action (discrete or continuous)
            
        Returns:
            features: Feature representation of the state-action pair
        """
        # Process discrete vs continuous actions
        if isinstance(action, (int, np.integer)):
            # One-hot encode discrete actions
            action_vec = np.zeros(10)  # Assume at most 10 actions
            if action < 10:
                action_vec[action] = 1.0
        else:
            # Use continuous action vector directly
            action_vec = np.array(action)
        
        # Combine state and action
        combined = np.concatenate([state[:10], action_vec[:10]])  # Limit dimensions
        
        # Project to feature space
        features = self.feature_projection @ combined[:20]
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def compute_utility(self, state, action):
        """
        Compute utility of state-action pair based on true preferences.
        
        Args:
            state: State vector
            action: Action (discrete or continuous)
            
        Returns:
            utility: Utility value
        """
        features = self.compute_features(state, action)
        utility = np.dot(self.true_preference, features)
        return utility
    
    def select_best_action(self, state, available_actions):
        """
        Select best action based on true preferences.
        
        Args:
            state: State vector
            available_actions: List of available actions
            
        Returns:
            best_action: Action with highest utility
        """
        # Calculate utility for each action
        utilities = [self.compute_utility(state, action) for action in available_actions]
        
        # Find action with highest utility
        best_idx = np.argmax(utilities)
        return available_actions[best_idx]
    
    def provide_feedback(self, state, agent_action, q_values, uncertainty):
        """
        Decide whether to provide feedback based on action quality and uncertainty.
        
        Args:
            state: Current state
            agent_action: Action selected by agent
            q_values: Q-values estimated by agent
            uncertainty: Uncertainty in agent's action
            
        Returns:
            feedback: Corrected action if feedback provided, None otherwise
        """
        # Compute utility of agent's action using true preference
        agent_utility = self.compute_utility(state, agent_action)
        
        # Generate a set of possible actions
        if isinstance(agent_action, (int, np.integer)):
            # For discrete actions, consider all possible actions
            all_actions = list(range(len(q_values)))
            
            # Find best action according to true preference
            best_action = self.select_best_action(state, all_actions)
            best_utility = self.compute_utility(state, best_action)
            
        else:
            # For continuous actions, generate alternatives
            all_actions = [agent_action]
            for _ in range(5):  # Generate 5 random alternatives
                random_action = np.random.normal(0, 1, len(agent_action))
                random_action = random_action / np.linalg.norm(random_action)
                all_actions.append(random_action)
            
            # Find best action according to true preference
            best_action = self.select_best_action(state, all_actions)
            best_utility = self.compute_utility(state, best_action)
        
        # Compute utility difference
        utility_diff = best_utility - agent_utility
        
        # Adjust correction threshold based on agent's expressed uncertainty
        # Higher agent uncertainty -> lower threshold (more likely to correct)
        adjusted_threshold = self.correction_threshold * (1.0 - self.uncertainty_sensitivity * uncertainty)
        
        # Decide whether to provide feedback
        if utility_diff > adjusted_threshold:
            # Add noise to feedback to simulate human error
            if np.random.random() < self.noise_level and isinstance(best_action, (int, np.integer)):
                # With small probability, give a random action different from agent's action
                possible_actions = [a for a in all_actions if a != agent_action]
                if possible_actions:
                    return np.random.choice(possible_actions)
            
            # Otherwise provide best action as feedback
            return best_action
        
        # No feedback
        return None
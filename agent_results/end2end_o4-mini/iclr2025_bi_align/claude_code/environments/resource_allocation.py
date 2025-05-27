#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Resource Allocation Environment for UDRA experiments.

This environment simulates a resource allocation task where an agent must decide
how to distribute limited resources across different locations or tasks.
"""

import numpy as np
import gym
from gym import spaces

class ResourceAllocationEnv(gym.Env):
    """
    A simulated environment for resource allocation tasks.
    
    The agent must decide how to allocate resources among different options.
    States represent the current resource availability and demands.
    Actions represent the allocation decisions.
    """
    
    def __init__(self, state_dim=10, action_dim=5, feature_dim=8, max_steps=20):
        """
        Initialize the resource allocation environment.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            feature_dim (int): Dimension of the feature map φ(s,a)
            max_steps (int): Maximum number of steps per episode
        """
        super(ResourceAllocationEnv, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.max_steps = max_steps
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        # Initialize internal state
        self.current_step = 0
        self.state = None
        
        # Generate random feature map matrices for state-action to feature conversion
        # This maps state-action pairs to the feature space used for preference modeling
        self.state_feature_map = np.random.randn(feature_dim, state_dim)
        self.action_feature_map = np.random.randn(feature_dim, action_dim)
        
        # Initialize resource demands and capacities
        self.reset()
    
    def compute_features(self, state, action):
        """
        Compute feature representation φ(s,a) for state-action pair.

        This is used for user preference modeling and alignment computation.

        Args:
            state: Current state vector
            action: Action vector or integer

        Returns:
            feature_vector: Feature representation of the state-action pair
        """
        # Handle discrete vs. continuous actions
        if isinstance(action, (int, np.integer)):
            # Convert discrete action to one-hot vector
            action_vec = np.zeros(self.action_dim)
            action_vec[action] = 1.0
            action = action_vec

        # Ensure action is a numpy array
        action = np.array(action, dtype=np.float32)

        # Compute features by combining state and action through feature maps
        state_features = self.state_feature_map @ state
        action_features = self.action_feature_map @ action

        # Combine features using various operations (element-wise product, sum, etc.)
        features = np.zeros(self.feature_dim)
        features[:self.feature_dim//2] = state_features[:self.feature_dim//2] * action_features[:self.feature_dim//2]
        features[self.feature_dim//2:] = state_features[self.feature_dim//2:] + action_features[self.feature_dim//2:]

        # Normalize the feature vector
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features
    
    def step(self, action):
        """
        Take a step in the environment with the given action.

        Args:
            action: Action vector or integer representing resource allocation

        Returns:
            next_state: New state after taking action
            reward: Reward obtained from taking action
            done: Whether the episode is finished
            info: Additional information dictionary
        """
        self.current_step += 1

        # Handle discrete vs. continuous actions
        if isinstance(action, (int, np.integer)):
            # Convert discrete action to allocation vector
            # For discrete actions, we allocate 70% to the chosen resource and
            # distribute the remaining 30% evenly among others
            action_vec = np.ones(self.action_dim) * 0.3 / (self.action_dim - 1)
            action_vec[action] = 0.7
            action = action_vec

        # Ensure action is in correct format and within bounds
        action = np.clip(np.array(action, dtype=np.float32), 0.0, 1.0)

        # Normalize action to sum to 1 (representing allocation proportions)
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        
        # Extract current resource demands and capacities
        resource_demands = self.state[:self.action_dim]
        resource_capacities = self.state[self.action_dim:2*self.action_dim]
        
        # Compute allocation amounts based on action proportions
        allocations = action * np.sum(resource_capacities)
        
        # Calculate satisfaction ratio (how well demands are met, clipped at 1.0)
        satisfaction_ratio = np.minimum(allocations / np.maximum(resource_demands, 1e-6), 1.0)
        
        # Base reward is average satisfaction ratio
        base_reward = np.mean(satisfaction_ratio)
        
        # Penalty for oversupplying resources (wasted resources)
        oversupply = np.maximum(allocations - resource_demands, 0.0)
        oversupply_penalty = -0.5 * np.sum(oversupply) / np.sum(resource_capacities)
        
        # Penalty for very uneven distributions
        distribution_variance = np.var(satisfaction_ratio)
        variance_penalty = -0.3 * distribution_variance
        
        # Total reward
        reward = base_reward + oversupply_penalty + variance_penalty
        
        # Update state
        # - Update resource capacities (some restoration)
        new_capacities = resource_capacities * 0.8 + np.random.uniform(0.1, 0.2, self.action_dim)
        
        # - Update resource demands (some randomness)
        new_demands = resource_demands * 0.7 + np.random.uniform(0.1, 0.3, self.action_dim)
        
        # - Create other state components (environmental factors, etc.)
        other_factors = np.random.normal(0, 0.1, self.state_dim - 2*self.action_dim)
        
        # Construct new state
        self.state = np.concatenate([new_demands, new_capacities, other_factors])
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Compute state-action features for info
        features = self.compute_features(self.state, action)
        
        info = {
            'features': features,
            'satisfaction_ratio': satisfaction_ratio,
            'oversupply': oversupply,
            'distribution_variance': distribution_variance
        }
        
        return self.state, reward, done, info
    
    def reset(self):
        """
        Reset the environment for a new episode.
        
        Returns:
            state: Initial state for the new episode
        """
        self.current_step = 0
        
        # Initialize resource demands
        resource_demands = np.random.uniform(0.5, 1.5, self.action_dim)
        
        # Initialize resource capacities
        resource_capacities = np.random.uniform(0.8, 1.2, self.action_dim) * np.sum(resource_demands)
        
        # Initialize other state components (environmental factors, etc.)
        other_factors = np.random.normal(0, 0.1, self.state_dim - 2*self.action_dim)
        
        # Construct initial state
        self.state = np.concatenate([resource_demands, resource_capacities, other_factors])
        
        return self.state
    
    def render(self, mode='human'):
        """Render the environment (not implemented)."""
        pass
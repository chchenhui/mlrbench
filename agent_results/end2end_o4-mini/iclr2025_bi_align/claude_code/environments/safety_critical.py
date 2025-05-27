#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Safety-Critical Environment for UDRA experiments.

This environment simulates safety-critical decision-making scenarios,
such as autonomous driving at intersections or medical triage decisions.
"""

import numpy as np
import gym
from gym import spaces

class SafetyCriticalEnv(gym.Env):
    """
    A simulated environment for safety-critical decision-making tasks.
    
    The agent must make decisions with safety implications while balancing
    efficiency and risk. States represent the current situation, and
    actions represent different decision options with varying risk profiles.
    """
    
    def __init__(self, state_dim=12, action_dim=4, feature_dim=10, max_steps=20):
        """
        Initialize the safety-critical environment.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            feature_dim (int): Dimension of the feature map φ(s,a)
            max_steps (int): Maximum number of steps per episode
        """
        super(SafetyCriticalEnv, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.max_steps = max_steps
        
        # Define action types
        self.action_types = ["cautious", "balanced", "efficient", "aggressive"]
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(action_dim)
        
        # Initialize internal state
        self.current_step = 0
        self.state = None
        
        # Generate random feature map matrices for state-action to feature conversion
        # This maps state-action pairs to the feature space used for preference modeling
        self.state_feature_map = np.random.randn(feature_dim, state_dim)
        self.action_features = np.array([
            # Safety features (first half)
            [0.9, 0.7, 0.3, 0.1],  # Safety level
            [0.8, 0.6, 0.4, 0.2],  # Caution measure
            [0.95, 0.7, 0.5, 0.3],  # Risk avoidance
            [0.9, 0.6, 0.4, 0.2],  # Predictability
            [0.8, 0.5, 0.3, 0.1],  # Defensive posture
            # Efficiency features (second half)
            [0.3, 0.5, 0.8, 0.9],  # Speed/efficiency
            [0.2, 0.4, 0.7, 0.9],  # Resource utilization
            [0.1, 0.3, 0.7, 0.9],  # Time optimization
            [0.3, 0.5, 0.7, 0.8],  # Energy efficiency
            [0.2, 0.4, 0.6, 0.8],  # Task completion rate
        ])
        
        # Properties of different scenarios
        self.scenario_risk_factors = np.linspace(0.2, 0.8, 5)  # Different risk levels
        
        # Initialize the environment
        self.reset()
    
    def compute_features(self, state, action):
        """
        Compute feature representation φ(s,a) for state-action pair.
        
        This is used for user preference modeling and alignment computation.
        
        Args:
            state: Current state vector
            action: Action index (integer)
            
        Returns:
            feature_vector: Feature representation of the state-action pair
        """
        # Process state through feature map
        state_features = self.state_feature_map @ state
        
        # Get action features
        action_features = self.action_features[:, action]
        
        # Combine state and action features
        # First half: more safety-focused
        # Second half: more efficiency-focused
        features = np.zeros(self.feature_dim)
        half_dim = self.feature_dim // 2
        
        # Combine with element-wise product and sum
        features[:half_dim] = state_features[:half_dim] * action_features[:half_dim]
        features[half_dim:] = state_features[half_dim:] * action_features[half_dim:]
        
        # Normalize the feature vector
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features
    
    def step(self, action):
        """
        Take a step in the environment with the given action.
        
        Args:
            action: Action index (integer) representing decision
            
        Returns:
            next_state: New state after taking action
            reward: Reward obtained from taking action
            done: Whether the episode is finished
            info: Additional information dictionary
        """
        self.current_step += 1
        
        # Ensure action is an integer within range
        if isinstance(action, np.ndarray):
            action = np.argmax(action)  # Convert one-hot to index
        action = int(action) % self.action_dim
        
        # Extract current scenario info from state
        scenario_type = int(np.argmax(self.state[:5]))  # First 5 elements encode scenario type
        risk_factor = self.scenario_risk_factors[scenario_type]
        
        # Get action characteristics
        action_safety = self.action_features[0, action]  # First row represents safety
        action_efficiency = self.action_features[6, action]  # Row 6 represents efficiency
        
        # Compute base reward based on action efficiency
        base_reward = action_efficiency
        
        # Compute risk of accident/failure based on risk factor and action safety
        risk_probability = risk_factor * (1 - action_safety)
        
        # Determine if a failure/accident occurs
        accident_occurs = np.random.random() < risk_probability
        
        # Apply reward/penalty
        if accident_occurs:
            # Significant penalty for safety failures
            safety_penalty = -2.0 * (1 + risk_factor)  # Higher penalty in riskier scenarios
            reward = base_reward + safety_penalty
            failure = True
        else:
            # Reward for successful navigation with no penalty
            success_reward = 0.5 * (1 + action_efficiency) 
            reward = base_reward + success_reward
            failure = False
        
        # Update state
        # - Generate new scenario type (sometimes stays same, sometimes changes)
        if np.random.random() < 0.3:  # 30% chance to change scenario
            new_scenario = np.random.randint(0, 5)
        else:
            new_scenario = scenario_type
            
        # - Encode scenario type as one-hot
        scenario_encoding = np.zeros(5)
        scenario_encoding[new_scenario] = 1.0
        
        # - Generate other state components (situational factors)
        # These represent environment conditions, nearby objects, system status, etc.
        situation_factors = np.random.normal(0, 0.2, self.state_dim - 5)
        
        # - Construct new state
        self.state = np.concatenate([scenario_encoding, situation_factors])
        
        # Check if episode is done
        done = self.current_step >= self.max_steps or failure
        
        # Compute state-action features for info
        features = self.compute_features(self.state, action)
        
        info = {
            'features': features,
            'risk_factor': risk_factor,
            'action_safety': action_safety,
            'action_efficiency': action_efficiency,
            'accident_occurs': accident_occurs,
            'scenario_type': new_scenario
        }
        
        return self.state, reward, done, info
    
    def reset(self):
        """
        Reset the environment for a new episode.
        
        Returns:
            state: Initial state for the new episode
        """
        self.current_step = 0
        
        # Initialize scenario type (0-4)
        scenario_type = np.random.randint(0, 5)
        
        # Encode scenario type as one-hot
        scenario_encoding = np.zeros(5)
        scenario_encoding[scenario_type] = 1.0
        
        # Initialize other state components (situational factors)
        situation_factors = np.random.normal(0, 0.2, self.state_dim - 5)
        
        # Construct initial state
        self.state = np.concatenate([scenario_encoding, situation_factors])
        
        return self.state
    
    def render(self, mode='human'):
        """Render the environment (not implemented)."""
        pass
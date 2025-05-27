#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Uncertainty-Driven Reciprocal Alignment (UDRA) implementation.

This module implements the UDRA algorithm with Bayesian user modeling,
uncertainty estimation, and bidirectional feedback mechanisms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .bayesian_user_model import BayesianUserModel

class EnsembleQNetwork(nn.Module):
    """
    Ensemble of Q-networks for uncertainty estimation.
    
    This uses multiple Q-networks to estimate both the expected Q-value
    and the uncertainty in that estimate.
    """
    
    def __init__(self, state_dim, action_dim, ensemble_size=5, hidden_size=64):
        """
        Initialize the ensemble Q-network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            ensemble_size (int): Number of networks in the ensemble
            hidden_size (int): Size of hidden layers
        """
        super(EnsembleQNetwork, self).__init__()
        
        self.ensemble_size = ensemble_size
        
        # Create ensemble of networks
        self.q_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim)
            ) for _ in range(ensemble_size)
        ])
    
    def forward(self, state):
        """
        Forward pass through the ensemble network.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            mean_q (torch.Tensor): Mean Q-values across ensemble
            std_q (torch.Tensor): Standard deviation of Q-values (uncertainty)
        """
        # Get predictions from each network in the ensemble
        ensemble_qs = torch.stack([network(state) for network in self.q_networks])
        
        # Compute mean and standard deviation across ensemble
        mean_q = ensemble_qs.mean(dim=0)
        std_q = ensemble_qs.std(dim=0)
        
        return mean_q, std_q

class UDRA:
    """
    Uncertainty-Driven Reciprocal Alignment (UDRA) agent.
    
    This agent combines ensemble Q-learning for uncertainty estimation with
    Bayesian preference modeling for adaptive alignment to human preferences.
    """
    
    def __init__(self, state_dim, action_dim, feature_dim, learning_rate=0.001, gamma=0.99, 
                 lambda_val=0.5, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update_freq=10,
                 ensemble_size=5):
        """
        Initialize the UDRA agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            feature_dim (int): Dimension of the feature space
            learning_rate (float): Learning rate for optimization
            gamma (float): Discount factor
            lambda_val (float): Weight for alignment loss
            epsilon_start (float): Initial exploration rate
            epsilon_end (float): Final exploration rate
            epsilon_decay (float): Rate of exploration decay
            buffer_size (int): Size of replay buffer
            batch_size (int): Batch size for training
            target_update_freq (int): Frequency of target network updates
            ensemble_size (int): Number of networks in the ensemble
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.lambda_val = lambda_val
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize ensemble Q-networks
        self.q_ensemble = EnsembleQNetwork(state_dim, action_dim, ensemble_size)
        self.target_ensemble = EnsembleQNetwork(state_dim, action_dim, ensemble_size)
        
        # Copy parameters from Q-ensemble to target ensemble
        self.target_ensemble.load_state_dict(self.q_ensemble.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_ensemble.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = []
        self.buffer_size = buffer_size
        
        # Initialize Bayesian user model
        self.user_model = BayesianUserModel(feature_dim)
        
        # Store feature function (will be set by environment)
        self.feature_function = None
        
        # Training parameters
        self.train_step_counter = 0
    
    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            action (int): Selected action
        """
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Random action
            return np.random.randint(0, self.action_dim)
        else:
            # Greedy action from Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values, _ = self.q_ensemble(state_tensor)
                return np.argmax(q_values.numpy()[0])
    
    def select_action_with_uncertainty(self, state):
        """
        Select an action and estimate uncertainty.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            action (int): Selected action
            q_values (np.ndarray): Mean Q-values for all actions
            uncertainty (np.ndarray): Uncertainty estimates for all actions
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get Q-values and uncertainties from ensemble
        with torch.no_grad():
            mean_q, std_q = self.q_ensemble(state_tensor)
            mean_q = mean_q.numpy()[0]
            std_q = std_q.numpy()[0]
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Random action
            action = np.random.randint(0, self.action_dim)
        else:
            # Greedy action
            action = np.argmax(mean_q)
        
        # Return the selected action, Q-values, and uncertainties
        return action, mean_q, std_q[action]
    
    def update_from_feedback(self, state, corrected_action):
        """
        Update the agent based on human feedback using Bayesian preference updates.
        
        Args:
            state (np.ndarray): State where feedback was given
            corrected_action (int): Action suggested by human
        """
        # Use a feature function from the environment to compute features
        # This is a placeholder - in a real system, the environment would provide this
        def compute_features(s, a):
            # Basic features based on one-hot action encoding
            action_features = np.zeros(self.action_dim)
            if isinstance(a, (int, np.integer)):
                action_features[a] = 1.0
            else:
                action_features = a
                
            # Combine with state features
            combined = np.concatenate([s, action_features])
            
            # Project to feature dimension using random projection (environment would do better)
            if not hasattr(self, 'feature_projection'):
                np.random.seed(42)  # For reproducibility
                self.feature_projection = np.random.randn(self.feature_dim, len(combined))
                
            features = self.feature_projection @ combined
            features = features / (np.linalg.norm(features) + 1e-8)  # Normalize
            
            return features
        
        # Get available actions (in a real system, this would come from the environment)
        available_actions = list(range(self.action_dim))
        
        # Update the user model with the human feedback
        updated_mean, _ = self.user_model.update(
            state=state,
            action_chosen=corrected_action,
            action_available=available_actions,
            features_map=compute_features
        )
        
        # Add the experience to the replay buffer with additional alignment reward
        # The human-corrected action gets a bonus based on the preference model
        # Compute feature vector
        corrected_features = compute_features(state, corrected_action)
        
        # Compute alignment reward based on updated preference weights
        alignment_reward = np.dot(updated_mean, corrected_features)
        
        # Add to replay buffer with combined reward
        self._add_to_replay_buffer(
            state=state,
            action=corrected_action,
            reward=alignment_reward,  # Use alignment reward
            next_state=state,  # Placeholder, will be overwritten by next environment step
            done=False,
        )
        
        # Decrease epsilon (exploration) after feedback
        self.epsilon = max(self.epsilon_end, self.epsilon * 0.9)
    
    def update_from_environment(self, state, action, reward, next_state, done):
        """
        Update the agent based on environment feedback.
        
        This combines the environmental reward with the alignment reward
        based on the current user preference model.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received from environment
            next_state (np.ndarray): Next state
            done (bool): Whether the episode is done
        """
        # Compute feature vector for the state-action pair
        def compute_features(s, a):
            # Basic features based on one-hot action encoding
            action_features = np.zeros(self.action_dim)
            if isinstance(a, (int, np.integer)):
                action_features[a] = 1.0
            else:
                action_features = a
                
            # Combine with state features
            combined = np.concatenate([s, action_features])
            
            # Project to feature dimension using random projection (environment would do better)
            if not hasattr(self, 'feature_projection'):
                np.random.seed(42)  # For reproducibility
                self.feature_projection = np.random.randn(self.feature_dim, len(combined))
                
            features = self.feature_projection @ combined
            features = features / (np.linalg.norm(features) + 1e-8)  # Normalize
            
            return features
        
        # Compute feature vector
        features = compute_features(state, action)
        
        # Compute alignment reward based on current preference weights
        alignment_reward = np.dot(self.user_model.mean, features)
        
        # Combine environmental and alignment rewards
        combined_reward = reward + self.lambda_val * alignment_reward
        
        # Add experience to replay buffer
        self._add_to_replay_buffer(state, action, combined_reward, next_state, done)
        
        # Train the agent
        self._train()
        
        # Update target network if needed
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_ensemble.load_state_dict(self.q_ensemble.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _add_to_replay_buffer(self, state, action, reward, next_state, done):
        """
        Add experience to replay buffer.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether the episode is done
        """
        # Add experience to buffer
        self.replay_buffer.append((state, action, reward, next_state, done))
        
        # If buffer is full, remove oldest experience
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def _train(self):
        """Train the agent using sampled experiences from replay buffer."""
        # Skip if not enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch_indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        # Different loss for each network in the ensemble
        total_loss = 0
        
        # Train each network separately with different subsets of the data
        # This helps create diversity in the ensemble
        for i, q_network in enumerate(self.q_ensemble.q_networks):
            # Randomly sample subset of batch for this network (with replacement)
            subset_indices = torch.randint(0, self.batch_size, (self.batch_size // 2,))
            
            # Get states, actions, rewards for this subset
            subnet_states = states[subset_indices]
            subnet_actions = actions[subset_indices]
            subnet_rewards = rewards[subset_indices]
            subnet_next_states = next_states[subset_indices]
            subnet_dones = dones[subset_indices]
            
            # Get Q-values for current states and actions
            q_values = q_network(subnet_states).gather(1, subnet_actions.unsqueeze(1)).squeeze(1)
            
            # Get target Q-values
            with torch.no_grad():
                # Get Q-values for next states from target network
                target_q_values = self.target_ensemble.q_networks[i](subnet_next_states).max(1)[0]
                
                # Compute target values using Bellman equation
                targets = subnet_rewards + self.gamma * target_q_values * (1 - subnet_dones)
            
            # Compute loss for this network
            loss = F.mse_loss(q_values, targets)
            total_loss += loss
        
        # Update all networks
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Increment train step counter
        self.train_step_counter += 1
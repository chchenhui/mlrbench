#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseline RLHF (Reinforcement Learning with Human Feedback) implementation.

This module implements a standard reinforcement learning approach with static 
alignment to human preferences. It does not explicitly model uncertainty
or maintain a Bayesian user model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    """
    Neural network for Q-value estimation.
    """
    
    def __init__(self, state_dim, action_dim, hidden_size=64):
        """
        Initialize the Q-network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_size (int): Size of hidden layers
        """
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            q_values (torch.Tensor): Q-values for each action
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class BaselineRLHF:
    """
    Baseline implementation of Reinforcement Learning with Human Feedback.
    
    This agent uses DQN with experience replay and a separate target network.
    It incorporates human feedback as additional reward signals but uses a
    static alignment approach rather than a Bayesian user model.
    """
    
    def __init__(self, state_dim, action_dim, feature_dim, learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update_freq=10):
        """
        Initialize the baseline RLHF agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            feature_dim (int): Dimension of the feature space
            learning_rate (float): Learning rate for optimization
            gamma (float): Discount factor
            epsilon_start (float): Initial exploration rate
            epsilon_end (float): Final exploration rate
            epsilon_decay (float): Rate of exploration decay
            buffer_size (int): Size of replay buffer
            batch_size (int): Batch size for training
            target_update_freq (int): Frequency of target network updates
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        
        # Copy parameters from Q-network to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = []
        self.buffer_size = buffer_size
        
        # Initialize static human preference model
        # For the baseline, we use a simple linear model without Bayesian updates
        self.preference_weights = np.zeros(feature_dim)
        
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
                q_values = self.q_network(state_tensor).numpy()[0]
                return np.argmax(q_values)
    
    def select_action_with_uncertainty(self, state):
        """
        Select an action and estimate uncertainty.
        
        For the baseline, the uncertainty estimate is a simple heuristic.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            action (int): Selected action
            q_values (np.ndarray): Q-values for all actions
            uncertainty (float): Estimated uncertainty (heuristic)
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_tensor).numpy()[0]
        
        # Basic uncertainty heuristic: small random noise inversely proportional to training steps
        # (more training = less uncertainty)
        base_uncertainty = 1.0 / (1.0 + 0.01 * self.train_step_counter)
        uncertainty = base_uncertainty * (1.0 + 0.2 * np.random.random())
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Random action
            action = np.random.randint(0, self.action_dim)
        else:
            # Greedy action
            action = np.argmax(q_values)
        
        return action, q_values, uncertainty
    
    def update_from_feedback(self, state, corrected_action):
        """
        Update the agent based on human feedback.
        
        For the baseline, we update the preference weights using a simple rule.
        
        Args:
            state (np.ndarray): State where feedback was given
            corrected_action (int): Action suggested by human
        """
        # For the baseline, we use a simple update rule for preference weights
        # This is much simpler than the Bayesian approach in UDRA
        
        # Get the current action the policy would take
        current_action = self.select_action(state)
        
        # Only update if the corrected action is different
        if current_action != corrected_action:
            # Add the experience to the replay buffer with additional reward
            # The human-corrected action gets a bonus reward
            self._add_to_replay_buffer(state, corrected_action, 1.0, state, False)
            
            # Decrease epsilon (exploration) slightly after feedback
            self.epsilon = max(self.epsilon_end, self.epsilon * 0.95)
    
    def update_from_environment(self, state, action, reward, next_state, done):
        """
        Update the agent based on environment feedback.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether the episode is done
        """
        # Add experience to replay buffer
        self._add_to_replay_buffer(state, action, reward, next_state, done)
        
        # Train the agent
        self._train()
        
        # Update target network if needed
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
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
        
        # Get Q-values for current states and actions
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)
        
        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Increment train step counter
        self.train_step_counter += 1
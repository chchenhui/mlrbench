"""
Models module for the Dynamic Human-AI Co-Adaptation experiment.
Includes implementation of various agent types.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

class QNetwork(nn.Module):
    """
    Neural network to approximate the Q-function.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class ImitationNetwork(nn.Module):
    """
    Neural network for imitation learning component.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ImitationNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

class ReplayBuffer:
    """
    Replay buffer for storing experience tuples.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done, explanation=None):
        if explanation is None:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer.append((state, action, reward, next_state, done, explanation))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class DynamicAlignmentAgent:
    """
    Agent implementing the proposed Dynamic Human-AI Co-Adaptation framework.
    Uses a hybrid RL-imitation learning architecture with explanation generation.
    """
    def __init__(self, n_features, learning_rate=0.001, discount_factor=0.95, 
                imitation_weight=0.3, buffer_size=10000, device="cpu"):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.imitation_weight = imitation_weight
        self.device = device
        
        # Q-Network (RL component)
        self.q_network = QNetwork(n_features, n_features).to(device)
        self.target_q_network = QNetwork(n_features, n_features).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Imitation Network
        self.imitation_network = ImitationNetwork(n_features, n_features).to(device)
        self.imitation_optimizer = optim.Adam(self.imitation_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        # Demonstration buffer for imitation learning
        self.demonstration_buffer = []
        
        # Internal representation of user preferences
        self.estimated_preferences = None
        
    def select_action(self, state, deterministic=False):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy (for evaluation)
            
        Returns:
            int: Selected action (item index)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        # Use deterministic policy for evaluation
        if deterministic:
            with torch.no_grad():
                # Combine Q-values and imitation policy
                q_values = self.q_network(state)
                imitation_logits = self.imitation_network(state)
                
                # Weighted combination
                combined = (1 - self.imitation_weight) * q_values + self.imitation_weight * imitation_logits
                
                # Select action with highest combined value
                return torch.argmax(combined).item()
        
        # Epsilon-greedy for training
        if random.random() < self.epsilon:
            return random.randint(0, self.n_features - 1)
        else:
            with torch.no_grad():
                # Combine Q-values and imitation policy
                q_values = self.q_network(state)
                imitation_logits = self.imitation_network(state)
                
                # Weighted combination
                combined = (1 - self.imitation_weight) * q_values + self.imitation_weight * imitation_logits
                
                # Select action with highest combined value
                return torch.argmax(combined).item()
    
    def update(self, state, action, reward, next_state, done, explanation=None):
        """
        Update the agent's policy based on experience.
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Whether the episode is done
            explanation: Explanation for the action (if available)
        """
        # Convert to tensors if necessary
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).to(self.device)
        
        # Add experience to replay buffer
        if explanation is not None:
            self.replay_buffer.add(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done, explanation)
        else:
            self.replay_buffer.add(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
        
        # If the reward is high, add to demonstration buffer for imitation learning
        if reward > 0.7:  # Threshold for "good" experiences
            self.demonstration_buffer.append((state.cpu().numpy(), action))
        
        # Update networks if enough samples
        if len(self.replay_buffer) >= 64:
            self._update_networks()
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _update_networks(self, batch_size=64):
        """Update Q-network and imitation network from replay buffer samples."""
        # Sample from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        
        # Check if explanations are included
        has_explanations = len(batch[0]) > 5
        
        if has_explanations:
            states, actions, rewards, next_states, dones, explanations = zip(*batch)
        else:
            states, actions, rewards, next_states, dones = zip(*batch)
            explanations = None
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Update Q-network
        # Compute Q-values for current states and actions
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            targets = rewards + (1 - dones) * self.discount_factor * max_next_q_values
        
        # Compute loss and update Q-network
        q_loss = F.mse_loss(q_values, targets)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Update imitation network if there are demonstrations
        if len(self.demonstration_buffer) > 0:
            # Sample from demonstration buffer
            demo_indices = np.random.choice(len(self.demonstration_buffer), 
                                          min(batch_size, len(self.demonstration_buffer)), 
                                          replace=False)
            demo_states, demo_actions = zip(*[self.demonstration_buffer[i] for i in demo_indices])
            
            # Convert to tensors
            demo_states = torch.FloatTensor(demo_states).to(self.device)
            demo_actions = torch.LongTensor(demo_actions).to(self.device)
            
            # Compute logits and loss
            logits = self.imitation_network(demo_states)
            imitation_loss = F.cross_entropy(logits, demo_actions)
            
            # If explanations are available, incorporate them
            if explanations is not None:
                # Process explanations (assuming they are feature importance values)
                explanation_tensors = []
                for exp in explanations:
                    if exp is not None:
                        explanation_tensors.append(torch.FloatTensor(exp).to(self.device))
                    else:
                        # If no explanation, use uniform weights
                        explanation_tensors.append(torch.ones(self.n_features).to(self.device) / self.n_features)
                
                # Stack tensors
                if explanation_tensors:
                    explanations_tensor = torch.stack(explanation_tensors)
                    
                    # Apply explanations as attention weights to network features
                    # This is a simplified way to incorporate explanations
                    attention_loss = F.mse_loss(
                        F.softmax(self.imitation_network.fc2(states), dim=1),
                        explanations_tensor
                    )
                    
                    # Add attention loss to imitation loss
                    imitation_loss += 0.1 * attention_loss
            
            # Update imitation network
            self.imitation_optimizer.zero_grad()
            imitation_loss.backward()
            self.imitation_optimizer.step()
        
        # Periodically update target network
        if random.random() < 0.1:  # 10% chance to update
            self.target_q_network.load_state_dict(self.q_network.state_dict())

class StaticRLHFAgent:
    """
    Baseline agent implementing static Reinforcement Learning from Human Feedback.
    Does not adapt to changing preferences and does not provide explanations.
    """
    def __init__(self, n_features, learning_rate=0.001, discount_factor=0.95, 
                buffer_size=10000, device="cpu"):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.device = device
        
        # Q-Network
        self.q_network = QNetwork(n_features, n_features).to(device)
        self.target_q_network = QNetwork(n_features, n_features).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
    def select_action(self, state, deterministic=False):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy (for evaluation)
            
        Returns:
            int: Selected action (item index)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        # Use deterministic policy for evaluation
        if deterministic:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()
        
        # Epsilon-greedy for training
        if random.random() < self.epsilon:
            return random.randint(0, self.n_features - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()
    
    def update(self, state, action, reward, next_state, done, explanation=None):
        """
        Update the agent's policy based on experience.
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Whether the episode is done
            explanation: Ignored in this agent
        """
        # Convert to tensors if necessary
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).to(self.device)
        
        # Add experience to replay buffer
        self.replay_buffer.add(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
        
        # Update networks if enough samples
        if len(self.replay_buffer) >= 64:
            self._update_networks()
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _update_networks(self, batch_size=64):
        """Update Q-network from replay buffer samples."""
        # Sample from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q-values for current states and actions
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            targets = rewards + (1 - dones) * self.discount_factor * max_next_q_values
        
        # Compute loss and update Q-network
        q_loss = F.mse_loss(q_values, targets)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Periodically update target network
        if random.random() < 0.1:  # 10% chance to update
            self.target_q_network.load_state_dict(self.q_network.state_dict())

class DirectRLAIFAgent:
    """
    Baseline agent implementing direct Reinforcement Learning from AI Feedback.
    Uses a simulated AI feedback model and updates less frequently.
    """
    def __init__(self, n_features, learning_rate=0.001, discount_factor=0.95, 
                buffer_size=10000, device="cpu"):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.device = device
        
        # Q-Network
        self.q_network = QNetwork(n_features, n_features).to(device)
        self.target_q_network = QNetwork(n_features, n_features).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # AI feedback model (simplified)
        self.feedback_model = QNetwork(n_features * 2, 1).to(device)  # Takes state and action features
        self.feedback_optimizer = optim.Adam(self.feedback_model.parameters(), lr=learning_rate)
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        # LLM simulation parameters
        self.update_frequency = 5  # How often to update from AI feedback
        self.step_counter = 0
        
    def select_action(self, state, deterministic=False):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy (for evaluation)
            
        Returns:
            int: Selected action (item index)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        # Use deterministic policy for evaluation
        if deterministic:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()
        
        # Epsilon-greedy for training
        if random.random() < self.epsilon:
            return random.randint(0, self.n_features - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()
    
    def update(self, state, action, reward, next_state, done, explanation=None):
        """
        Update the agent's policy based on experience.
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Whether the episode is done
            explanation: Ignored in this agent
        """
        # Convert to tensors if necessary
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).to(self.device)
        
        # Add experience to replay buffer
        self.replay_buffer.add(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
        
        # Increment step counter
        self.step_counter += 1
        
        # Update networks if enough samples and it's time to update
        if len(self.replay_buffer) >= 64 and self.step_counter % self.update_frequency == 0:
            self._update_networks()
        
        # Update AI feedback model
        if len(self.replay_buffer) >= 32:
            self._update_feedback_model()
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _update_networks(self, batch_size=64):
        """Update Q-network from replay buffer samples with AI feedback."""
        # Sample from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Generate AI feedback for the experiences
        with torch.no_grad():
            ai_feedback = self._generate_ai_feedback(states, actions)
        
        # Combine original rewards with AI feedback
        combined_rewards = 0.7 * rewards + 0.3 * ai_feedback
        
        # Compute Q-values for current states and actions
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            targets = combined_rewards + (1 - dones) * self.discount_factor * max_next_q_values
        
        # Compute loss and update Q-network
        q_loss = F.mse_loss(q_values, targets)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Periodically update target network
        if random.random() < 0.1:  # 10% chance to update
            self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def _update_feedback_model(self, batch_size=32):
        """Update the AI feedback model."""
        # Sample from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, _, _ = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        
        # Create inputs for feedback model
        action_features = torch.zeros(batch_size, self.n_features).to(self.device)
        for i, action in enumerate(actions):
            action_features[i, action] = 1.0
        
        inputs = torch.cat([states, action_features], dim=1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        
        # Update feedback model to predict rewards
        predictions = self.feedback_model(inputs)
        loss = F.mse_loss(predictions, rewards)
        
        self.feedback_optimizer.zero_grad()
        loss.backward()
        self.feedback_optimizer.step()
    
    def _generate_ai_feedback(self, states, actions):
        """
        Generate AI feedback for state-action pairs.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            torch.Tensor: Feedback scores
        """
        batch_size = states.shape[0]
        
        # Create inputs for feedback model
        action_features = torch.zeros(batch_size, self.n_features).to(self.device)
        for i, action in enumerate(actions):
            action_features[i, action] = 1.0
        
        inputs = torch.cat([states, action_features], dim=1)
        
        # Generate feedback
        feedback = self.feedback_model(inputs).squeeze(1)
        
        # Normalize feedback to [0, 1] range
        feedback = torch.sigmoid(feedback)
        
        return feedback
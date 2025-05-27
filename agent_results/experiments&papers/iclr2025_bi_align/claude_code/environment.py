"""
Environment module for the recommendation system simulation.
"""

import numpy as np
import torch
from scipy.spatial.distance import cosine

class RecommendationEnvironment:
    """
    A simulated environment for recommendation systems with dynamic user preferences.
    
    This environment simulates users interacting with a recommendation system.
    User preferences can shift over time, requiring the agent to adapt.
    """
    
    def __init__(self, n_users, n_items, n_features, user_preferences, 
                 preference_shift_interval=20, preference_shift_magnitude=0.3):
        """
        Initialize the recommendation environment.
        
        Args:
            n_users (int): Number of users in the system
            n_items (int): Number of items available for recommendation
            n_features (int): Dimensionality of item and preference features
            user_preferences (numpy.ndarray): Initial user preferences as vectors
            preference_shift_interval (int): Number of episodes between preference shifts
            preference_shift_magnitude (float): Magnitude of preference shifts
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_features = n_features
        self.initial_user_preferences = user_preferences.copy()
        self.user_preferences = user_preferences.copy()
        self.preference_shift_interval = preference_shift_interval
        self.preference_shift_magnitude = preference_shift_magnitude
        
        # Generate item features (fixed throughout the experiment)
        self.item_features = np.random.normal(0, 1, (n_items, n_features))
        self.item_features = self.item_features / np.linalg.norm(self.item_features, axis=1, keepdims=True)
        
        # Current user being served
        self.current_user = None
        
        # Historical interactions to track user behavior
        self.interaction_history = []
        
        # Track preference shifts for evaluation
        self.preference_shifts = []
        
    def reset(self):
        """
        Reset the environment for a new episode.
        
        Returns:
            numpy.ndarray: The initial state (randomly selected user and their preferences)
        """
        # Select a random user
        self.current_user = np.random.randint(0, self.n_users)
        self.interaction_history = []
        
        # Initial state is the user's preference vector
        state = self.user_preferences[self.current_user].copy()
        
        return state
    
    def step(self, action):
        """
        Take a step in the environment by recommending an item (action) to the current user.
        
        Args:
            action (int): Index of the item to recommend
            
        Returns:
            tuple: (next_state, reward, done, info)
                - next_state (numpy.ndarray): The new state after taking the action
                - reward (float): The reward for the action
                - done (bool): Whether the episode is finished
                - info (dict): Additional information
        """
        # Validate action
        if not 0 <= action < self.n_items:
            raise ValueError(f"Invalid action: {action}. Must be between 0 and {self.n_items-1}")
        
        # Get the recommended item's features
        item_features = self.item_features[action]
        
        # Calculate reward based on similarity between user preferences and item features
        # Higher similarity = higher reward
        user_pref = self.user_preferences[self.current_user]
        similarity = np.dot(user_pref, item_features)
        reward = max(0, similarity)  # Non-negative reward
        
        # Record interaction
        self.interaction_history.append({
            "user": self.current_user,
            "item": action,
            "reward": reward
        })
        
        # Next state is current user preferences (could be enhanced with interaction history)
        next_state = self.user_preferences[self.current_user].copy()
        
        # Episode ends after each recommendation for simplicity
        done = True
        
        # Additional info
        info = {
            "user_preference": user_pref,
            "item_features": item_features,
            "similarity": similarity
        }
        
        return next_state, reward, done, info
    
    def shift_preferences(self):
        """
        Shift user preferences to simulate evolving user interests over time.
        """
        # Generate random shift vectors
        shift_vectors = np.random.normal(0, 1, (self.n_users, self.n_features))
        shift_vectors = shift_vectors / np.linalg.norm(shift_vectors, axis=1, keepdims=True)
        
        # Apply shifts to user preferences
        self.user_preferences = self.user_preferences + self.preference_shift_magnitude * shift_vectors
        
        # Normalize preferences
        self.user_preferences = self.user_preferences / np.linalg.norm(self.user_preferences, axis=1, keepdims=True)
        
        # Record this shift for evaluation
        self.preference_shifts.append({
            "magnitude": self.preference_shift_magnitude,
            "shifted_preferences": self.user_preferences.copy()
        })
    
    def evaluate_alignment(self, agent):
        """
        Evaluate how well the agent's recommendations align with current user preferences.
        
        Args:
            agent: The agent to evaluate
            
        Returns:
            float: Alignment score between 0 and 1
        """
        alignment_scores = []
        
        for user_idx in range(self.n_users):
            # Get user preference
            user_pref = self.user_preferences[user_idx]
            
            # Get agent's recommendation for this user
            state = user_pref.copy()
            action = agent.select_action(state, deterministic=True)
            
            # Calculate alignment as similarity between recommended item and user preference
            item_features = self.item_features[action]
            similarity = np.dot(user_pref, item_features)
            alignment_scores.append(similarity)
        
        # Return average alignment score
        return np.mean(alignment_scores)
    
    def evaluate_trust(self, agent):
        """
        Evaluate user trust based on consistency of recommendations.
        
        Args:
            agent: The agent to evaluate
            
        Returns:
            float: Trust score between 0 and 1
        """
        trust_scores = []
        
        for user_idx in range(self.n_users):
            # Get user preference
            user_pref = self.user_preferences[user_idx]
            
            # Simulate multiple recommendations to check consistency
            recommendations = []
            for _ in range(5):  # Sample 5 recommendations
                state = user_pref.copy()
                action = agent.select_action(state, deterministic=False)  # Stochastic mode
                recommendations.append(action)
            
            # Calculate trust as consistency of recommendations
            # Higher consistency = higher trust
            unique_items = len(set(recommendations))
            consistency = 1 - (unique_items - 1) / 4  # Normalize to [0, 1]
            trust_scores.append(consistency)
        
        # Return average trust score
        return np.mean(trust_scores)
    
    def evaluate_adaptability(self, agent, current_episode):
        """
        Evaluate how well the agent adapts to preference shifts.
        
        Args:
            agent: The agent to evaluate
            current_episode: Current episode number
            
        Returns:
            float: Adaptability score between 0 and 1
        """
        # If no shifts have occurred yet, return a default score
        if len(self.preference_shifts) == 0:
            return 0.5
        
        adaptability_scores = []
        
        for user_idx in range(self.n_users):
            # Get current user preference
            current_pref = self.user_preferences[user_idx]
            
            # Get most recent previous preference (before the last shift)
            if len(self.preference_shifts) == 1:
                previous_pref = self.initial_user_preferences[user_idx]
            else:
                previous_pref = self.preference_shifts[-2]["shifted_preferences"][user_idx]
            
            # Get agent's recommendation for current preference
            state = current_pref.copy()
            current_action = agent.select_action(state, deterministic=True)
            
            # Calculate preference change magnitude
            pref_change = 1 - np.dot(current_pref, previous_pref)
            
            # Get features of recommended item
            item_features = self.item_features[current_action]
            
            # Calculate alignment with current preferences
            current_alignment = np.dot(current_pref, item_features)
            
            # Calculate adaptability score
            # Higher adaptability = better alignment despite preference changes
            adaptability = current_alignment / (pref_change + 0.1)  # Add small constant to avoid division by zero
            adaptability = min(1.0, adaptability)  # Cap at 1.0
            adaptability_scores.append(adaptability)
        
        # Return average adaptability score
        return np.mean(adaptability_scores)
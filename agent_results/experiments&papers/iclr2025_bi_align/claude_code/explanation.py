"""
Explanation module for the Dynamic Human-AI Co-Adaptation framework.
Implements mechanisms to generate human-centric explanations of AI decisions.
"""

import numpy as np
import torch
import torch.nn.functional as F

class ExplanationGenerator:
    """
    Generates explanations for AI decisions using causal reasoning approaches.
    
    Explanations help users understand how their feedback influences AI decisions,
    fostering user awareness and control.
    """
    
    def __init__(self, threshold=0.1):
        """
        Initialize the explanation generator.
        
        Args:
            threshold (float): Threshold for determining significant feature contributions
        """
        self.threshold = threshold
        
    def generate(self, state, action, agent):
        """
        Generate explanation for an agent's action.
        
        Implements the causal reasoning approach described in the proposal:
        E(s, a) = sum_{c ∈ C} π(a|s, c) · I(c|s, a)
        
        Args:
            state: The current state
            action: The action taken by the agent
            agent: The agent that took the action
            
        Returns:
            dict: Explanation containing feature contributions and other relevant information
        """
        # Ensure state is a tensor
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(agent.device)
            
        # Get feature importances through gradient analysis
        feature_importances = self._compute_feature_importances(state, action, agent)
        
        # Identify key causal factors (features with high importance)
        causal_factors = {}
        for i, importance in enumerate(feature_importances):
            if importance > self.threshold:
                causal_factors[f"feature_{i}"] = float(importance)
        
        # Generate counterfactual explanations
        counterfactuals = self._generate_counterfactuals(state, action, agent)
        
        # Combine into final explanation
        explanation = {
            "feature_importances": feature_importances,
            "causal_factors": causal_factors,
            "counterfactuals": counterfactuals
        }
        
        return explanation
    
    def _compute_feature_importances(self, state, action, agent):
        """
        Compute feature importances using gradient-based attribution.
        
        Args:
            state: The current state
            action: The action taken by the agent
            agent: The agent that took the action
            
        Returns:
            numpy.ndarray: Importance scores for each feature
        """
        # Ensure we can compute gradients
        state.requires_grad_(True)
        
        # Forward pass through Q-network
        q_values = agent.q_network(state)
        q_value = q_values[action]
        
        # Backward pass to compute gradients
        q_value.backward()
        
        # Get gradients w.r.t. input features
        gradients = state.grad.cpu().detach().numpy()
        
        # Compute feature importances as absolute gradient values
        importances = np.abs(gradients)
        
        # Normalize to sum to 1
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        return importances
    
    def _generate_counterfactuals(self, state, action, agent):
        """
        Generate counterfactual explanations showing how changes to features affect actions.
        
        Args:
            state: The current state
            action: The action taken by the agent
            agent: The agent that took the action
            
        Returns:
            dict: Counterfactual explanations
        """
        counterfactuals = {}
        
        # Make a copy of the state
        state_np = state.cpu().detach().numpy()
        
        # Identify top 3 important features
        importances = self._compute_feature_importances(state, action, agent)
        top_features = np.argsort(-importances)[:3]
        
        # For each top feature, create counterfactual by altering it
        for feature_idx in top_features:
            # Increase feature value
            modified_state = state_np.copy()
            modified_state[feature_idx] += 0.5  # Increase by a fixed amount
            
            # Get new action
            modified_state_tensor = torch.FloatTensor(modified_state).to(agent.device)
            with torch.no_grad():
                modified_q_values = agent.q_network(modified_state_tensor)
                new_action = torch.argmax(modified_q_values).item()
            
            # If action changes, add to counterfactuals
            if new_action != action:
                counterfactuals[f"increase_feature_{feature_idx}"] = {
                    "original_action": int(action),
                    "new_action": int(new_action),
                    "change": "increase"
                }
            
            # Decrease feature value
            modified_state = state_np.copy()
            modified_state[feature_idx] -= 0.5  # Decrease by a fixed amount
            
            # Get new action
            modified_state_tensor = torch.FloatTensor(modified_state).to(agent.device)
            with torch.no_grad():
                modified_q_values = agent.q_network(modified_state_tensor)
                new_action = torch.argmax(modified_q_values).item()
            
            # If action changes, add to counterfactuals
            if new_action != action:
                counterfactuals[f"decrease_feature_{feature_idx}"] = {
                    "original_action": int(action),
                    "new_action": int(new_action),
                    "change": "decrease"
                }
        
        return counterfactuals
    
    def generate_natural_language_explanation(self, explanation):
        """
        Convert the structured explanation to natural language.
        
        Args:
            explanation (dict): Structured explanation from generate()
            
        Returns:
            str: Natural language explanation
        """
        # Extract components from structured explanation
        feature_importances = explanation["feature_importances"]
        causal_factors = explanation["causal_factors"]
        counterfactuals = explanation["counterfactuals"]
        
        # Start with main explanation
        text = "I recommended this item because "
        
        # Add information about top features
        if causal_factors:
            top_features = sorted(causal_factors.items(), key=lambda x: x[1], reverse=True)
            feature_texts = []
            for feature, importance in top_features:
                feature_idx = int(feature.split("_")[1])
                if importance > 0.5:
                    strength = "strongly"
                elif importance > 0.3:
                    strength = "moderately"
                else:
                    strength = "somewhat"
                feature_texts.append(f"{feature} {strength} influenced this recommendation")
            
            text += ", ".join(feature_texts)
        else:
            text += "it seemed to match your preferences"
        
        # Add counterfactual information
        if counterfactuals:
            text += ". "
            cf_texts = []
            for cf_name, cf_details in list(counterfactuals.items())[:2]:  # Limit to 2 counterfactuals
                feature_idx = int(cf_name.split("_")[-1])
                change = cf_details["change"]
                cf_texts.append(f"If feature {feature_idx} was {change}d, I would have recommended a different item")
            
            text += " ".join(cf_texts)
        
        return text
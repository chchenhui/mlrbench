"""
Alignment models for the CEVA framework.

This module implements various AI alignment models, including:
1. Static alignment model (traditional approach)
2. Adaptive alignment model (simple uniform adaptation)
3. CEVA model (multi-level value representation with differential adaptation rates)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from value_evolution import ValueSystem

class BaseAlignmentModel:
    """Base class for alignment models."""
    def __init__(
        self,
        value_dimensions: List[str],
        name: str = "base_model",
        description: str = "Base alignment model",
        random_seed: Optional[int] = None
    ):
        """
        Initialize a base alignment model.
        
        Args:
            value_dimensions: List of value dimension names
            name: Model name
            description: Model description
            random_seed: Random seed for reproducibility
        """
        self.value_dimensions = value_dimensions
        self.name = name
        self.description = description
        
        if random_seed is not None:
            self.random_state = np.random.RandomState(random_seed)
        else:
            self.random_state = np.random.RandomState()
            
        # Initialize AI value system with uniform values
        initial_values = np.ones(len(value_dimensions)) / len(value_dimensions)
        self.value_system = ValueSystem(value_dimensions, initial_values, self.random_state)
        
        # Store alignment history
        self.value_history = [self.value_system.values.copy()]
        self.alignment_scores = []
        
    def update(self, human_values: ValueSystem) -> None:
        """
        Update the model's value system based on human values.
        
        Args:
            human_values: Human value system to align with
        """
        raise NotImplementedError("Subclasses must implement update method")
    
    def generate_response(self, query: str, human_values: ValueSystem) -> Dict:
        """
        Generate a response to a user query based on the current value system.
        
        Args:
            query: User query
            human_values: Human value system to consider
            
        Returns:
            Dictionary with response text and metadata
        """
        raise NotImplementedError("Subclasses must implement generate_response method")
    
    def calculate_alignment_score(self, human_values: ValueSystem) -> float:
        """
        Calculate alignment score between model's values and human values.
        
        Args:
            human_values: Human value system to compare with
            
        Returns:
            Alignment score (higher is better)
        """
        # Calculate distance between value systems
        distance = self.value_system.distance(human_values)
        
        # Convert to alignment score (1 - normalized distance)
        # Maximum distance in n-dimensional space with values summing to 1 is sqrt(2)
        max_distance = np.sqrt(2)
        alignment_score = 1 - (distance / max_distance)
        
        return alignment_score
    
    def get_value_history_df(self) -> pd.DataFrame:
        """
        Get the model's value history as a DataFrame.
        
        Returns:
            DataFrame with columns for each value dimension and rows for each time step
        """
        history_array = np.array(self.value_history)
        return pd.DataFrame(
            history_array,
            columns=self.value_dimensions
        )
    
    def get_alignment_scores_df(self) -> pd.DataFrame:
        """
        Get the model's alignment scores as a DataFrame.
        
        Returns:
            DataFrame with alignment scores over time
        """
        return pd.DataFrame({
            'time': list(range(len(self.alignment_scores))),
            'alignment_score': self.alignment_scores
        })


class StaticAlignmentModel(BaseAlignmentModel):
    """
    Traditional static alignment model that does not adapt to changes in human values.
    Once initial alignment is established, the model's values remain fixed.
    """
    def __init__(
        self,
        value_dimensions: List[str],
        name: str = "static_alignment",
        description: str = "Traditional static alignment model - no adaptation",
        random_seed: Optional[int] = None
    ):
        """Initialize a static alignment model."""
        super().__init__(value_dimensions, name, description, random_seed)
        self.initialized = False
    
    def update(self, human_values: ValueSystem) -> None:
        """
        Update the model's value system based on human values.
        For static alignment, this only happens once at initialization.
        
        Args:
            human_values: Human value system to align with
        """
        # Only align once at the beginning
        if not self.initialized:
            # Copy human values exactly
            for dim in self.value_dimensions:
                self.value_system.set_value(dim, human_values.get_value(dim))
            self.initialized = True
            
        # Record current values in history
        self.value_history.append(self.value_system.values.copy())
        
        # Calculate and record alignment score
        alignment_score = self.calculate_alignment_score(human_values)
        self.alignment_scores.append(alignment_score)
        
    def generate_response(self, query: str, human_values: ValueSystem) -> Dict:
        """
        Generate a response to a user query based on the current value system.
        
        Args:
            query: User query
            human_values: Human value system to consider
            
        Returns:
            Dictionary with response text and metadata
        """
        # Calculate alignment score
        alignment_score = self.calculate_alignment_score(human_values)
        
        # Simplified response generation:
        # Response quality depends on alignment score
        # In a real implementation, this would involve more sophisticated language generation
        response_quality = alignment_score
        
        # Return response with metadata
        return {
            "response": f"Static model response (alignment: {alignment_score:.2f})",
            "alignment_score": alignment_score,
            "satisfaction_score": response_quality,
            "adaptation": 0.0  # No adaptation happens
        }


class AdaptiveAlignmentModel(BaseAlignmentModel):
    """
    Simple adaptive alignment model that uniformly adapts to changes in human values.
    """
    def __init__(
        self,
        value_dimensions: List[str],
        update_rate: float = 0.5,
        name: str = "adaptive_alignment",
        description: str = "Simple adaptive alignment model - uniform adaptation",
        random_seed: Optional[int] = None
    ):
        """
        Initialize an adaptive alignment model.
        
        Args:
            value_dimensions: List of value dimension names
            update_rate: Rate at which the model adapts to human values (0-1)
            name: Model name
            description: Model description
            random_seed: Random seed for reproducibility
        """
        super().__init__(value_dimensions, name, description, random_seed)
        self.update_rate = update_rate
    
    def update(self, human_values: ValueSystem) -> None:
        """
        Update the model's value system based on human values.
        
        Args:
            human_values: Human value system to align with
        """
        # Update each value dimension
        for dim in self.value_dimensions:
            current_value = self.value_system.get_value(dim)
            human_value = human_values.get_value(dim)
            
            # Linear interpolation between current and target values
            new_value = (1 - self.update_rate) * current_value + self.update_rate * human_value
            self.value_system.set_value(dim, new_value)
            
        # Record current values in history
        self.value_history.append(self.value_system.values.copy())
        
        # Calculate and record alignment score
        alignment_score = self.calculate_alignment_score(human_values)
        self.alignment_scores.append(alignment_score)
        
    def generate_response(self, query: str, human_values: ValueSystem) -> Dict:
        """
        Generate a response to a user query based on the current value system.
        
        Args:
            query: User query
            human_values: Human value system to consider
            
        Returns:
            Dictionary with response text and metadata
        """
        # Calculate alignment score
        alignment_score = self.calculate_alignment_score(human_values)
        
        # Simplified response generation:
        # Response quality depends on alignment score
        # In a real implementation, this would involve more sophisticated language generation
        response_quality = alignment_score
        
        # Return response with metadata
        return {
            "response": f"Adaptive model response (alignment: {alignment_score:.2f})",
            "alignment_score": alignment_score,
            "satisfaction_score": response_quality,
            "adaptation": self.update_rate  # Uniform adaptation rate
        }


class CEVAModel(BaseAlignmentModel):
    """
    Co-Evolutionary Value Alignment (CEVA) model with multi-level value representation.
    
    This model maintains a three-level representation of human values:
    1. Core safety values: Fundamental ethical principles that remain relatively stable
    2. Cultural values: Shared societal values that evolve slowly
    3. Personal preferences: Individual-specific values that may change more rapidly
    
    Each level has a different adaptation rate, allowing for more nuanced alignment.
    """
    def __init__(
        self,
        value_dimensions: List[str],
        value_levels: Dict[str, List[str]],
        adaptation_rates: Dict[str, float],
        bidirectional: bool = False,
        feedback_mechanism: str = "none",
        name: str = "ceva_model",
        description: str = "CEVA model with multi-level value adaptation",
        random_seed: Optional[int] = None
    ):
        """
        Initialize a CEVA model.
        
        Args:
            value_dimensions: List of value dimension names
            value_levels: Dictionary mapping level names to lists of dimensions
            adaptation_rates: Dictionary mapping level names to adaptation rates
            bidirectional: Whether to enable bidirectional feedback
            feedback_mechanism: Type of feedback mechanism to use
            name: Model name
            description: Model description
            random_seed: Random seed for reproducibility
        """
        super().__init__(value_dimensions, name, description, random_seed)
        
        # Validate that all dimensions are assigned to a level
        all_assigned_dims = []
        for level, dims in value_levels.items():
            all_assigned_dims.extend(dims)
        
        if set(all_assigned_dims) != set(value_dimensions):
            raise ValueError("All value dimensions must be assigned to exactly one level")
            
        self.value_levels = value_levels
        self.adaptation_rates = adaptation_rates
        self.bidirectional = bidirectional
        self.feedback_mechanism = feedback_mechanism
        
        # Initialize confidence in value estimates
        self.value_confidence = {dim: 0.5 for dim in value_dimensions}
        
        # Store feedback history
        self.feedback_history = []
        
        # For bidirectional alignment, we need to track detected value shifts
        self.detected_shifts = []
    
    def _get_adaptation_rate(self, dimension: str) -> float:
        """Get the adaptation rate for a specific dimension based on its level."""
        for level, dims in self.value_levels.items():
            if dimension in dims:
                return self.adaptation_rates[level]
        
        # Default to the slowest adaptation rate if level not found
        return min(self.adaptation_rates.values())
    
    def update(self, human_values: ValueSystem) -> None:
        """
        Update the model's value system based on human values.
        
        Args:
            human_values: Human value system to align with
        """
        # Update each value dimension with level-specific adaptation rates
        for dim in self.value_dimensions:
            current_value = self.value_system.get_value(dim)
            human_value = human_values.get_value(dim)
            adaptation_rate = self._get_adaptation_rate(dim)
            
            # Higher confidence leads to faster adaptation
            effective_rate = adaptation_rate * self.value_confidence[dim]
            
            # Linear interpolation between current and target values
            new_value = (1 - effective_rate) * current_value + effective_rate * human_value
            self.value_system.set_value(dim, new_value)
            
            # Update confidence based on consistency of observations
            # If human values are stable, confidence increases; if they change a lot, confidence decreases
            value_diff = abs(human_value - current_value)
            conf_update = 0.01 * (1 - value_diff)  # Small updates to confidence
            self.value_confidence[dim] = max(0.1, min(1.0, self.value_confidence[dim] + conf_update))
            
        # Record current values in history
        self.value_history.append(self.value_system.values.copy())
        
        # Calculate and record alignment score
        alignment_score = self.calculate_alignment_score(human_values)
        self.alignment_scores.append(alignment_score)
        
    def detect_value_shifts(self, human_values: ValueSystem, threshold: float = 0.1) -> List[str]:
        """
        Detect significant shifts in human values.
        
        Args:
            human_values: Current human value system
            threshold: Threshold for considering a shift significant
            
        Returns:
            List of dimensions where significant shifts were detected
        """
        # If we don't have enough history, return empty list
        if len(self.value_history) < 5:
            return []
            
        shifted_dimensions = []
        
        # Compare current human values with the average of last few AI model values
        recent_values = np.array(self.value_history[-5:])
        average_recent_values = np.mean(recent_values, axis=0)
        
        for dim in self.value_dimensions:
            dim_idx = self.value_system.dim_to_idx[dim]
            recent_avg = average_recent_values[dim_idx]
            current_human = human_values.get_value(dim)
            
            # Check if the difference exceeds the threshold
            if abs(current_human - recent_avg) > threshold:
                shifted_dimensions.append(dim)
                
        return shifted_dimensions
    
    def generate_reflection_prompt(self, shifted_dimensions: List[str]) -> str:
        """
        Generate a reflection prompt about detected value shifts.
        
        Args:
            shifted_dimensions: List of dimensions where shifts were detected
            
        Returns:
            Reflection prompt text
        """
        if not shifted_dimensions:
            return ""
            
        # Generate a prompt for a single randomly selected shifted dimension
        dim = self.random_state.choice(shifted_dimensions)
        
        # Template prompts
        templates = [
            f"I notice your preference for {dim} seems to have changed. Is this important to you?",
            f"You seem to value {dim} differently now. Could you tell me more about this?",
            f"I'm curious about your changing perspective on {dim}. Would you like to discuss this?",
            f"My understanding is that your views on {dim} have shifted. Is this accurate?"
        ]
        
        return self.random_state.choice(templates)
    
    def record_feedback(self, dimension: str, feedback: float) -> None:
        """
        Record feedback from a human about a specific value dimension.
        
        Args:
            dimension: Value dimension
            feedback: Feedback value
        """
        self.feedback_history.append({"dimension": dimension, "feedback": feedback, "time": len(self.value_history)})
        
        # Update confidence based on feedback
        self.value_confidence[dimension] = min(1.0, self.value_confidence[dimension] + 0.2)
        
    def generate_response(self, query: str, human_values: ValueSystem) -> Dict:
        """
        Generate a response to a user query based on the current value system.
        
        Args:
            query: User query
            human_values: Human value system to consider
            
        Returns:
            Dictionary with response text and metadata
        """
        # Calculate alignment score
        alignment_score = self.calculate_alignment_score(human_values)
        
        # Detect value shifts for bidirectional feedback
        reflection_prompt = ""
        if self.bidirectional and self.feedback_mechanism == "reflection_prompting":
            shifted_dimensions = self.detect_value_shifts(human_values)
            
            # Record detected shifts
            if shifted_dimensions:
                self.detected_shifts.append({
                    "time": len(self.value_history),
                    "dimensions": shifted_dimensions
                })
                
                # Generate reflection prompt with some probability
                if shifted_dimensions and self.random_state.random() < 0.3:  # 30% chance to generate prompt
                    reflection_prompt = self.generate_reflection_prompt(shifted_dimensions)
        
        # Simplified response generation:
        # Response quality depends on alignment score
        # In a real implementation, this would involve more sophisticated language generation
        response_quality = alignment_score
        
        # Add penalty for too much reflection (don't want to overdo it)
        if reflection_prompt:
            response_quality *= 0.9  # Small penalty for adding reflection
        
        # Calculate average adaptation rate
        avg_adaptation = np.mean([self._get_adaptation_rate(dim) * self.value_confidence[dim] 
                                 for dim in self.value_dimensions])
        
        # Return response with metadata
        response = {
            "response": f"CEVA model response (alignment: {alignment_score:.2f})",
            "alignment_score": alignment_score,
            "satisfaction_score": response_quality,
            "adaptation": avg_adaptation
        }
        
        # Add reflection prompt if available
        if reflection_prompt:
            response["reflection_prompt"] = reflection_prompt
            
        return response
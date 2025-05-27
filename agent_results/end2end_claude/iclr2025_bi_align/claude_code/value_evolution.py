"""
Value evolution modeling for the CEVA framework.

This module implements the theoretical value evolution models described in the CEVA framework,
allowing simulation of how human values evolve through interactions with AI systems and society.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

class ValueSystem:
    """
    Represents a multidimensional value system where each dimension corresponds 
    to a specific value category (e.g., autonomy, fairness, efficiency).
    """
    def __init__(
        self, 
        dimensions: List[str], 
        initial_values: Optional[np.ndarray] = None,
        random_state: Optional[np.random.RandomState] = None
    ):
        """
        Initialize a value system with the given dimensions.
        
        Args:
            dimensions: List of value dimension names
            initial_values: Optional initial values for each dimension (if None, random values are generated)
            random_state: Optional random state for reproducibility
        """
        self.dimensions = dimensions
        self.n_dimensions = len(dimensions)
        
        if random_state is None:
            self.random_state = np.random.RandomState()
        else:
            self.random_state = random_state
            
        if initial_values is None:
            # Generate random values between 0 and 1
            self.values = self.random_state.rand(self.n_dimensions)
            # Normalize to sum to 1
            self.values = self.values / np.sum(self.values)
        else:
            if len(initial_values) != self.n_dimensions:
                raise ValueError(f"Expected {self.n_dimensions} values, got {len(initial_values)}")
            self.values = initial_values.copy()
            
        # Create a mapping from dimension names to indices
        self.dim_to_idx = {dim: i for i, dim in enumerate(dimensions)}
        
    def get_value(self, dimension: str) -> float:
        """Get the value of a specific dimension."""
        return self.values[self.dim_to_idx[dimension]]
    
    def set_value(self, dimension: str, value: float):
        """Set the value of a specific dimension."""
        self.values[self.dim_to_idx[dimension]] = value
        
    def as_dict(self) -> Dict[str, float]:
        """Return the value system as a dictionary mapping dimensions to values."""
        return {dim: self.values[i] for i, dim in enumerate(self.dimensions)}
    
    def distance(self, other: 'ValueSystem') -> float:
        """Calculate Euclidean distance between this value system and another."""
        if set(self.dimensions) != set(other.dimensions):
            raise ValueError("Value systems must have the same dimensions")
        
        # Re-order other's values to match this value system's dimension order
        other_values = np.array([other.get_value(dim) for dim in self.dimensions])
        
        # Calculate Euclidean distance
        return np.linalg.norm(self.values - other_values)
    
    def __str__(self) -> str:
        """String representation of the value system."""
        return " | ".join([f"{dim}: {val:.3f}" for dim, val in self.as_dict().items()])


class ValueEvolutionModel:
    """
    Models the evolution of human values over time based on external influences and 
    interaction experiences with AI systems.
    """
    def __init__(
        self,
        inertia_coefficient: float = 0.8,
        external_sensitivity: float = 0.1,
        interaction_sensitivity: float = 0.1,
        noise_level: float = 0.05,
        drift_rate: float = 0.02,
        random_seed: Optional[int] = None
    ):
        """
        Initialize a value evolution model.
        
        Args:
            inertia_coefficient: Tendency to maintain existing values (alpha)
            external_sensitivity: Sensitivity to external societal factors (beta)
            interaction_sensitivity: Sensitivity to AI interactions (gamma)
            noise_level: Level of random noise in value evolution
            drift_rate: Base rate of gradual value drift 
            random_seed: Random seed for reproducibility
        """
        self.alpha = inertia_coefficient
        self.beta = external_sensitivity
        self.gamma = interaction_sensitivity
        self.noise_level = noise_level
        self.drift_rate = drift_rate
        
        if random_seed is not None:
            self.random_state = np.random.RandomState(random_seed)
        else:
            self.random_state = np.random.RandomState()
            
    def _apply_external_factors(
        self, 
        values: np.ndarray, 
        external_factors: Optional[Dict[str, float]] = None,
        dimensions: List[str] = None
    ) -> np.ndarray:
        """
        Apply external societal factors to the value system.
        
        Args:
            values: Current value state
            external_factors: Dictionary mapping dimensions to influence strengths
            dimensions: List of value dimension names (for mapping)
            
        Returns:
            Updated values after applying external factors
        """
        if external_factors is None or not external_factors:
            # No external factors - return original values
            return values
        
        # Create a vector of external influences
        external_vector = np.zeros_like(values)
        for dim, strength in external_factors.items():
            if dim in dimensions:
                idx = dimensions.index(dim)
                external_vector[idx] = strength
        
        # Apply external factors (push values toward the direction of influence)
        updated_values = values + self.beta * external_vector
        
        # Normalize to ensure values remain valid
        return updated_values / np.sum(updated_values)
    
    def _apply_interaction_experiences(
        self, 
        values: np.ndarray, 
        interaction_data: Optional[Dict[str, float]] = None,
        dimensions: List[str] = None
    ) -> np.ndarray:
        """
        Apply AI interaction experiences to the value system.
        
        Args:
            values: Current value state
            interaction_data: Dictionary mapping dimensions to influence strengths
            dimensions: List of value dimension names (for mapping)
            
        Returns:
            Updated values after applying interaction experiences
        """
        if interaction_data is None or not interaction_data:
            # No interaction data - return original values
            return values
        
        # Create a vector of interaction influences
        interaction_vector = np.zeros_like(values)
        for dim, strength in interaction_data.items():
            if dim in dimensions:
                idx = dimensions.index(dim)
                interaction_vector[idx] = strength
        
        # Apply interaction influences
        updated_values = values + self.gamma * interaction_vector
        
        # Normalize to ensure values remain valid
        return updated_values / np.sum(updated_values)
    
    def _apply_value_drift(self, values: np.ndarray) -> np.ndarray:
        """
        Apply gradual value drift to simulate natural evolution of values over time.
        
        Args:
            values: Current value state
            
        Returns:
            Updated values after applying drift
        """
        # Generate random drift vector
        drift_vector = self.random_state.normal(0, self.drift_rate, size=len(values))
        
        # Apply drift
        updated_values = values + drift_vector
        
        # Ensure no negative values and normalize
        updated_values = np.maximum(0, updated_values)
        return updated_values / np.sum(updated_values)
    
    def _apply_noise(self, values: np.ndarray) -> np.ndarray:
        """Apply random noise to the value system."""
        # Generate random noise
        noise = self.random_state.normal(0, self.noise_level, size=len(values))
        
        # Apply noise
        updated_values = values + noise
        
        # Ensure no negative values and normalize
        updated_values = np.maximum(0, updated_values)
        return updated_values / np.sum(updated_values)
    
    def update(
        self, 
        value_system: ValueSystem,
        external_factors: Optional[Dict[str, float]] = None,
        interaction_data: Optional[Dict[str, float]] = None
    ) -> ValueSystem:
        """
        Update a value system based on external factors and interaction experiences.
        
        Args:
            value_system: Current value system
            external_factors: Dictionary mapping dimensions to influence strengths
            interaction_data: Dictionary mapping dimensions to influence strengths
            
        Returns:
            Updated value system
        """
        values = value_system.values.copy()
        dimensions = value_system.dimensions
        
        # Apply inertia (tendency to maintain existing values)
        updated_values = self.alpha * values
        
        # Apply external factors
        external_values = self._apply_external_factors(values, external_factors, dimensions)
        updated_values += self.beta * external_values
        
        # Apply interaction experiences
        interaction_values = self._apply_interaction_experiences(values, interaction_data, dimensions)
        updated_values += self.gamma * interaction_values
        
        # Apply value drift
        updated_values = self._apply_value_drift(updated_values)
        
        # Apply random noise
        updated_values = self._apply_noise(updated_values)
        
        # Create a new value system with the updated values
        new_value_system = ValueSystem(dimensions, updated_values, value_system.random_state)
        
        return new_value_system


class HumanAgent:
    """
    Simulates a human agent with evolving values and interaction history.
    """
    def __init__(
        self,
        value_dimensions: List[str],
        evolution_model: ValueEvolutionModel,
        initial_values: Optional[np.ndarray] = None,
        agent_id: Optional[str] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize a human agent.
        
        Args:
            value_dimensions: List of value dimension names
            evolution_model: Model for evolving values
            initial_values: Optional initial values (if None, random values are generated)
            agent_id: Optional identifier for the agent
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            self.random_state = np.random.RandomState(random_seed)
        else:
            self.random_state = np.random.RandomState()
            
        self.value_system = ValueSystem(
            value_dimensions,
            initial_values,
            self.random_state
        )
        self.evolution_model = evolution_model
        self.agent_id = agent_id or f"agent_{id(self)}"
        
        # Store history of value states
        self.value_history = [self.value_system.values.copy()]
        self.interaction_history = []
        
    def update_values(
        self,
        external_factors: Optional[Dict[str, float]] = None,
        interaction_data: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update the agent's values based on external factors and interaction experiences.
        
        Args:
            external_factors: Dictionary mapping dimensions to influence strengths
            interaction_data: Dictionary mapping dimensions to influence strengths
        """
        # Record interaction in history if provided
        if interaction_data:
            self.interaction_history.append(interaction_data)
            
        # Update value system
        self.value_system = self.evolution_model.update(
            self.value_system, 
            external_factors,
            interaction_data
        )
        
        # Record new value state in history
        self.value_history.append(self.value_system.values.copy())
        
    def get_value_history_df(self) -> pd.DataFrame:
        """
        Get the agent's value history as a DataFrame.
        
        Returns:
            DataFrame with columns for each value dimension and rows for each time step
        """
        history_array = np.array(self.value_history)
        return pd.DataFrame(
            history_array,
            columns=self.value_system.dimensions
        )


class ExternalEvent:
    """
    Represents an external event that can influence human values.
    """
    def __init__(
        self, 
        time: int, 
        affected_dimensions: List[str], 
        magnitude: float,
        direction: Optional[int] = 1  # 1 for positive, -1 for negative
    ):
        """
        Initialize an external event.
        
        Args:
            time: Time step at which the event occurs
            affected_dimensions: Value dimensions affected by the event
            magnitude: Strength of the event's influence
            direction: Direction of influence (1 for positive, -1 for negative)
        """
        self.time = time
        self.affected_dimensions = affected_dimensions
        self.magnitude = magnitude
        self.direction = direction
        
    def get_factors(self) -> Dict[str, float]:
        """
        Get the external factors induced by this event.
        
        Returns:
            Dictionary mapping dimensions to influence strengths
        """
        return {dim: self.magnitude * self.direction for dim in self.affected_dimensions}


class ValueEvolutionSimulation:
    """
    Simulates the evolution of human values over time for multiple agents.
    """
    def __init__(
        self,
        value_dimensions: List[str],
        n_agents: int,
        evolution_params: Dict[str, float],
        random_seed: Optional[int] = None
    ):
        """
        Initialize a value evolution simulation.
        
        Args:
            value_dimensions: List of value dimension names
            n_agents: Number of agents to simulate
            evolution_params: Parameters for the value evolution model
            random_seed: Random seed for reproducibility
        """
        self.value_dimensions = value_dimensions
        self.n_agents = n_agents
        
        if random_seed is not None:
            self.random_state = np.random.RandomState(random_seed)
        else:
            self.random_state = np.random.RandomState()
            
        # Create value evolution model
        self.evolution_model = ValueEvolutionModel(
            inertia_coefficient=evolution_params.get('inertia_coefficient', 0.8),
            external_sensitivity=evolution_params.get('external_sensitivity', 0.1),
            interaction_sensitivity=evolution_params.get('interaction_sensitivity', 0.1),
            noise_level=evolution_params.get('noise_level', 0.05),
            drift_rate=evolution_params.get('drift_rate', 0.02),
            random_seed=random_seed
        )
        
        # Create agents
        self.agents = []
        for i in range(n_agents):
            # Each agent gets different random seed derived from the main seed
            agent_seed = None if random_seed is None else random_seed + i
            
            agent = HumanAgent(
                value_dimensions,
                self.evolution_model,
                agent_id=f"agent_{i}",
                random_seed=agent_seed
            )
            self.agents.append(agent)
            
        # Store external events
        self.external_events = []
        
        # Current time step
        self.current_time = 0
        
    def add_external_event(self, event: ExternalEvent) -> None:
        """Add an external event to the simulation."""
        self.external_events.append(event)
        
    def get_events_at_time(self, time: int) -> List[ExternalEvent]:
        """Get all events occurring at the given time step."""
        return [event for event in self.external_events if event.time == time]
    
    def step(
        self, 
        interaction_data: Optional[Dict[str, Dict[str, float]]] = None
    ) -> None:
        """
        Advance the simulation by one time step.
        
        Args:
            interaction_data: Dictionary mapping agent IDs to their interaction data
        """
        # Get external events at the current time
        events = self.get_events_at_time(self.current_time)
        
        # Combine all external factors from events
        external_factors = {}
        for event in events:
            event_factors = event.get_factors()
            for dim, strength in event_factors.items():
                external_factors[dim] = external_factors.get(dim, 0) + strength
                
        # Update each agent
        for agent in self.agents:
            # Get interaction data for this agent, if any
            agent_interaction = None
            if interaction_data and agent.agent_id in interaction_data:
                agent_interaction = interaction_data[agent.agent_id]
                
            # Update agent's values
            agent.update_values(external_factors, agent_interaction)
            
        # Increment time
        self.current_time += 1
        
    def run(
        self, 
        n_steps: int,
        interaction_generator: Optional[callable] = None
    ) -> None:
        """
        Run the simulation for a specified number of time steps.
        
        Args:
            n_steps: Number of time steps to simulate
            interaction_generator: Optional function that generates interaction data for each time step
        """
        for _ in range(n_steps):
            # Generate interaction data if a generator is provided
            interaction_data = None
            if interaction_generator:
                interaction_data = interaction_generator(self.agents, self.current_time)
                
            # Advance the simulation
            self.step(interaction_data)
            
    def get_agent_value_history(self, agent_idx: int) -> pd.DataFrame:
        """Get the value history of a specific agent."""
        return self.agents[agent_idx].get_value_history_df()
    
    def get_average_values(self) -> Dict[str, np.ndarray]:
        """
        Get the average values across all agents at each time step.
        
        Returns:
            Dictionary mapping dimensions to arrays of average values
        """
        # Get the number of time steps (based on the first agent)
        n_steps = len(self.agents[0].value_history)
        
        # Initialize arrays for each dimension
        avg_values = {dim: np.zeros(n_steps) for dim in self.value_dimensions}
        
        # Compute average values
        for dim in self.value_dimensions:
            dim_idx = self.agents[0].value_system.dim_to_idx[dim]
            for t in range(n_steps):
                avg_values[dim][t] = np.mean([agent.value_history[t][dim_idx] for agent in self.agents])
                
        return avg_values
    
    def get_population_value_df(self) -> pd.DataFrame:
        """
        Get a DataFrame with the average values across the population at each time step.
        
        Returns:
            DataFrame with columns for each value dimension and rows for each time step
        """
        avg_values = self.get_average_values()
        return pd.DataFrame(avg_values)
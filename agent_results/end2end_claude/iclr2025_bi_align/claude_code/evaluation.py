"""
Evaluation metrics and scenarios for the CEVA framework.

This module provides tools for evaluating alignment models across different scenarios,
measuring key metrics such as adaptation accuracy, response time, and stability.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import time
from value_evolution import (
    ValueSystem, ValueEvolutionModel, HumanAgent, ExternalEvent, ValueEvolutionSimulation
)
from alignment_models import BaseAlignmentModel, StaticAlignmentModel, AdaptiveAlignmentModel, CEVAModel


class AlignmentScenario:
    """
    A scenario for evaluating alignment models.
    
    Scenarios define how human values evolve over time and provide mechanisms
    for measuring alignment quality across various conditions.
    """
    def __init__(
        self,
        name: str,
        description: str,
        value_dimensions: List[str],
        n_agents: int = 1,
        duration: int = 100,
        drift_multiplier: float = 1.0,
        external_events: Optional[List[Dict]] = None,
        evolution_params: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize an alignment scenario.
        
        Args:
            name: Scenario name
            description: Scenario description
            value_dimensions: List of value dimension names
            n_agents: Number of agents to simulate
            duration: Duration of the scenario in time steps
            drift_multiplier: Multiplier for value drift rate
            external_events: List of external events to apply
            evolution_params: Parameters for the value evolution model
            random_seed: Random seed for reproducibility
        """
        self.name = name
        self.description = description
        self.value_dimensions = value_dimensions
        self.n_agents = n_agents
        self.duration = duration
        self.drift_multiplier = drift_multiplier
        
        # Initialize with default evolution parameters if not provided
        if evolution_params is None:
            evolution_params = {
                "inertia_coefficient": 0.8,
                "external_sensitivity": 0.1,
                "interaction_sensitivity": 0.1,
                "noise_level": 0.05,
                "drift_rate": 0.02 * drift_multiplier,  # Apply drift multiplier
            }
        else:
            # Apply drift multiplier to provided parameters
            evolution_params = evolution_params.copy()
            if "drift_rate" in evolution_params:
                evolution_params["drift_rate"] *= drift_multiplier
        
        # Create simulation environment
        self.simulation = ValueEvolutionSimulation(
            value_dimensions=value_dimensions,
            n_agents=n_agents,
            evolution_params=evolution_params,
            random_seed=random_seed
        )
        
        # Add external events
        if external_events:
            for event_data in external_events:
                event = ExternalEvent(
                    time=event_data["time"],
                    affected_dimensions=event_data["dimensions"],
                    magnitude=event_data["magnitude"],
                    direction=event_data.get("direction", 1)
                )
                self.simulation.add_external_event(event)
                
    def run(self, models: List[BaseAlignmentModel]) -> Dict[str, Any]:
        """
        Run the scenario with the provided alignment models.
        
        Args:
            models: List of alignment models to evaluate
            
        Returns:
            Dictionary with evaluation results for each model
        """
        # Initialize results dictionary
        results = {
            "scenario_name": self.name,
            "scenario_description": self.description,
            "duration": self.duration,
            "models": {},
            "raw_data": {
                "human_values": [],
                "model_values": {},
                "alignment_scores": {},
                "response_times": {},
                "responses": {}
            }
        }
        
        # Run simulation step by step
        for t in range(self.duration):
            # Step the human value simulation forward
            self.simulation.step()
            
            # Record human values for each agent
            human_values_at_t = []
            for agent in self.simulation.agents:
                human_values_at_t.append(agent.value_system.values.copy())
            results["raw_data"]["human_values"].append(human_values_at_t)
            
            # Update each model with current human values
            for model in models:
                # We'll use the first agent's values for simplicity
                # In a more complex evaluation, we might consider aggregating across agents
                human_values = self.simulation.agents[0].value_system
                
                # Generate a query for response time measurement
                # (This is a placeholder - in a real system, we'd have actual queries)
                query = f"Query at time {t}"
                
                # Measure response time
                start_time = time.time()
                response = model.generate_response(query, human_values)
                end_time = time.time()
                response_time = end_time - start_time
                
                # Update model based on human values
                model.update(human_values)
                
                # Record model values and metrics
                if model.name not in results["raw_data"]["model_values"]:
                    results["raw_data"]["model_values"][model.name] = []
                    results["raw_data"]["alignment_scores"][model.name] = []
                    results["raw_data"]["response_times"][model.name] = []
                    results["raw_data"]["responses"][model.name] = []
                
                results["raw_data"]["model_values"][model.name].append(model.value_system.values.copy())
                results["raw_data"]["alignment_scores"][model.name].append(model.alignment_scores[-1])
                results["raw_data"]["response_times"][model.name].append(response_time)
                results["raw_data"]["responses"][model.name].append(response)
                
        # Calculate evaluation metrics for each model
        for model in models:
            model_metrics = self._calculate_metrics(model, results["raw_data"])
            results["models"][model.name] = {
                "model_description": model.description,
                "metrics": model_metrics
            }
            
        return results
    
    def _calculate_metrics(self, model: BaseAlignmentModel, raw_data: Dict) -> Dict[str, float]:
        """
        Calculate evaluation metrics for a model based on raw simulation data.
        
        Args:
            model: Alignment model to evaluate
            raw_data: Raw simulation data
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Calculate average alignment score
        alignment_scores = raw_data["alignment_scores"][model.name]
        metrics["avg_alignment_score"] = np.mean(alignment_scores)
        
        # Calculate final alignment score
        metrics["final_alignment_score"] = alignment_scores[-1]
        
        # Calculate adaptation accuracy
        # This is averaged across all time steps and measures how well the model tracks human values
        human_values_array = np.array(raw_data["human_values"])[:, 0, :]  # First agent only
        model_values_array = np.array(raw_data["model_values"][model.name])
        
        # Calculate Euclidean distance at each time step
        distances = np.linalg.norm(human_values_array - model_values_array, axis=1)
        avg_distance = np.mean(distances)
        
        # Convert to adaptation accuracy (1 - normalized distance)
        # Maximum distance in n-dimensional space with values summing to 1 is sqrt(2)
        max_distance = np.sqrt(2)
        metrics["adaptation_accuracy"] = 1 - (avg_distance / max_distance)
        
        # Calculate adaptation response time
        # Time to reduce value misalignment below threshold after shift
        # We'll use a simple heuristic: average time to reach alignment score above 0.8 after a drop below 0.7
        threshold = 0.7
        recovery_threshold = 0.8
        response_times = []
        
        below_threshold = False
        drop_time = 0
        
        for t, score in enumerate(alignment_scores):
            if not below_threshold and score < threshold:
                # Detected a drop in alignment
                below_threshold = True
                drop_time = t
            elif below_threshold and score > recovery_threshold:
                # Recovered from the drop
                recovery_time = t - drop_time
                response_times.append(recovery_time)
                below_threshold = False
        
        if response_times:
            metrics["adaptation_response_time"] = np.mean(response_times)
        else:
            # If no drops occurred, set to a perfect score
            metrics["adaptation_response_time"] = 0
            
        # Calculate stability
        # Resistance to spurious adaptation - measured by the variance of model values
        # Lower variance means more stability
        model_values_variance = np.var(model_values_array, axis=0)
        metrics["stability"] = 1 - np.mean(model_values_variance) * 10  # Scale to 0-1 range
        
        # Calculate user satisfaction
        # Average satisfaction score from responses
        satisfaction_scores = [r.get("satisfaction_score", 0) for r in raw_data["responses"][model.name]]
        metrics["user_satisfaction"] = np.mean(satisfaction_scores)
        
        # Calculate agency preservation
        # This is a theoretical construct - in a real system, we'd measure user perceptions
        # Here we'll use a heuristic: inversely related to number of reflection prompts
        reflection_count = sum(1 for r in raw_data["responses"][model.name] if "reflection_prompt" in r)
        reflection_ratio = reflection_count / len(raw_data["responses"][model.name])
        
        # Some reflection is good, but too much can be annoying
        # Optimal ratio might be around 0.1-0.2 (10-20% of interactions)
        if reflection_ratio <= 0.2:
            agency_score = 1.0 - (reflection_ratio / 0.4)  # Linear penalty up to 0.2
        else:
            agency_score = 0.5 - (reflection_ratio - 0.2)  # Steeper penalty after 0.2
            
        metrics["agency_preservation"] = max(0, min(1, agency_score))  # Clamp to 0-1
        
        return metrics


class EvaluationManager:
    """
    Manages evaluation of alignment models across multiple scenarios.
    """
    def __init__(
        self,
        value_dimensions: List[str],
        scenarios: List[Dict],
        models: List[Dict],
        value_adaptation_rates: Dict[str, float],
        random_seed: Optional[int] = None
    ):
        """
        Initialize an evaluation manager.
        
        Args:
            value_dimensions: List of value dimension names
            scenarios: List of scenario configurations
            models: List of model configurations
            value_adaptation_rates: Dictionary mapping value levels to adaptation rates
            random_seed: Random seed for reproducibility
        """
        self.value_dimensions = value_dimensions
        self.scenario_configs = scenarios
        self.model_configs = models
        self.value_adaptation_rates = value_adaptation_rates
        self.random_seed = random_seed
        
        # Set up value levels (mapping dimensions to levels)
        self.value_levels = self._create_value_levels()
        
    def _create_value_levels(self) -> Dict[str, List[str]]:
        """
        Create value level mappings based on dimension semantics.
        In a real system, this would be based on domain knowledge.
        Here we'll use a simple heuristic.
        """
        # For this demonstration, we'll categorize based on intuitive meanings
        # of the value dimensions defined in config.py
        
        value_levels = {
            "core_safety": [],
            "cultural": [],
            "preference": []
        }
        
        # Categorize dimensions into levels
        for dim in self.value_dimensions:
            if dim in ["security", "benevolence"]:
                value_levels["core_safety"].append(dim)
            elif dim in ["conformity"]:
                value_levels["cultural"].append(dim)
            else:
                value_levels["preference"].append(dim)
                
        return value_levels
    
    def create_models(self) -> List[BaseAlignmentModel]:
        """
        Create alignment models based on configurations.
        
        Returns:
            List of alignment model instances
        """
        models = []
        
        for config in self.model_configs:
            if config["name"] == "static_alignment":
                model = StaticAlignmentModel(
                    value_dimensions=self.value_dimensions,
                    name=config["name"],
                    description=config["description"],
                    random_seed=self.random_seed
                )
            elif config["name"] == "adaptive_alignment":
                model = AdaptiveAlignmentModel(
                    value_dimensions=self.value_dimensions,
                    update_rate=config["update_rate"],
                    name=config["name"],
                    description=config["description"],
                    random_seed=self.random_seed
                )
            elif config["name"].startswith("ceva"):
                model = CEVAModel(
                    value_dimensions=self.value_dimensions,
                    value_levels=self.value_levels,
                    adaptation_rates=self.value_adaptation_rates,
                    bidirectional=config["bidirectional"],
                    feedback_mechanism=config.get("feedback_mechanism", "none"),
                    name=config["name"],
                    description=config["description"],
                    random_seed=self.random_seed
                )
            else:
                raise ValueError(f"Unknown model type: {config['name']}")
                
            models.append(model)
            
        return models
    
    def create_scenarios(self) -> List[AlignmentScenario]:
        """
        Create alignment scenarios based on configurations.
        
        Returns:
            List of alignment scenario instances
        """
        scenarios = []
        
        for config in self.scenario_configs:
            # Each scenario gets a different seed derived from the main seed
            scenario_seed = None if self.random_seed is None else self.random_seed + len(scenarios)
            
            scenario = AlignmentScenario(
                name=config["name"],
                description=config["description"],
                value_dimensions=self.value_dimensions,
                n_agents=1,  # For simplicity
                duration=config["duration"],
                drift_multiplier=config.get("drift_multiplier", 1.0),
                external_events=config.get("external_events"),
                random_seed=scenario_seed
            )
            
            scenarios.append(scenario)
            
        return scenarios
    
    def run_evaluation(self) -> Dict:
        """
        Run evaluation of all models across all scenarios.
        
        Returns:
            Dictionary with evaluation results
        """
        models = self.create_models()
        scenarios = self.create_scenarios()
        
        results = {
            "overall": {
                "value_dimensions": self.value_dimensions,
                "n_scenarios": len(scenarios),
                "n_models": len(models),
            },
            "scenarios": {}
        }
        
        # Run each scenario
        for scenario in scenarios:
            # Create fresh model instances for each scenario to ensure fair comparison
            fresh_models = self.create_models()
            
            # Run the scenario with all models
            scenario_results = scenario.run(fresh_models)
            
            # Store results
            results["scenarios"][scenario.name] = scenario_results
            
        # Calculate aggregate metrics across scenarios
        results["aggregate_metrics"] = self._calculate_aggregate_metrics(results["scenarios"])
            
        return results
    
    def _calculate_aggregate_metrics(self, scenario_results: Dict) -> Dict:
        """
        Calculate aggregate metrics across all scenarios for each model.
        
        Args:
            scenario_results: Dictionary with results for each scenario
            
        Returns:
            Dictionary with aggregate metrics for each model
        """
        # Initialize aggregate metrics
        aggregate = {}
        
        # Get all model names
        model_names = []
        for scenario_name, data in scenario_results.items():
            model_names.extend(list(data["models"].keys()))
        model_names = list(set(model_names))
        
        # Initialize metrics for each model
        for model_name in model_names:
            aggregate[model_name] = {}
            
        # Calculate averages across scenarios
        for scenario_name, data in scenario_results.items():
            for model_name, model_data in data["models"].items():
                metrics = model_data["metrics"]
                
                # Initialize metric if not already present
                for metric_name, value in metrics.items():
                    if metric_name not in aggregate[model_name]:
                        aggregate[model_name][metric_name] = []
                        
                    aggregate[model_name][metric_name].append(value)
                    
        # Calculate averages
        for model_name in aggregate:
            for metric_name in list(aggregate[model_name].keys()):
                values = aggregate[model_name][metric_name]
                aggregate[model_name][f"avg_{metric_name}"] = np.mean(values)
                aggregate[model_name][f"std_{metric_name}"] = np.std(values)
                
                # Remove the list of values
                del aggregate[model_name][metric_name]
                
        return aggregate
"""
Dynamic Task Configuration Engine (DTCE) module

This module implements the Dynamic Task Configuration Engine, which transforms
test sets and evaluation criteria based on user-specified deployment contexts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import train_test_split
import json
import os


class ContextVector:
    """
    Class to represent a context vector for a specific domain.
    """
    
    def __init__(
        self,
        domain: str,
        constraints: Optional[Dict[str, Any]] = None,
        subgroup_weights: Optional[Dict[str, Dict[str, float]]] = None,
        performance_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize a context vector.
        
        Args:
            domain: Domain name (e.g., "healthcare", "finance")
            constraints: Dictionary of domain-specific constraints
            subgroup_weights: Dictionary mapping sensitive attributes to subgroup weights
            performance_thresholds: Dictionary of minimum performance thresholds
        """
        self.domain = domain
        self.constraints = constraints or {}
        self.subgroup_weights = subgroup_weights or {}
        self.performance_thresholds = performance_thresholds or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the context vector to a dictionary.
        
        Returns:
            dict: Dictionary representation of the context vector
        """
        return {
            "domain": self.domain,
            "constraints": self.constraints,
            "subgroup_weights": self.subgroup_weights,
            "performance_thresholds": self.performance_thresholds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextVector':
        """
        Create a ContextVector from a dictionary.
        
        Args:
            data: Dictionary representation of a context vector
            
        Returns:
            ContextVector: The created context vector
        """
        return cls(
            domain=data["domain"],
            constraints=data.get("constraints"),
            subgroup_weights=data.get("subgroup_weights"),
            performance_thresholds=data.get("performance_thresholds")
        )
    
    def to_json(self) -> str:
        """
        Convert the context vector to a JSON string.
        
        Returns:
            str: JSON representation of the context vector
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ContextVector':
        """
        Create a ContextVector from a JSON string.
        
        Args:
            json_str: JSON string representation of a context vector
            
        Returns:
            ContextVector: The created context vector
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save(self, directory: str, filename: Optional[str] = None) -> str:
        """
        Save the context vector to a JSON file.
        
        Args:
            directory: Directory to save the file
            filename: Filename (optional)
            
        Returns:
            str: Path to the saved file
        """
        os.makedirs(directory, exist_ok=True)
        
        if filename is None:
            filename = f"{self.domain}_context.json"
        
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'ContextVector':
        """
        Load a context vector from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            ContextVector: The loaded context vector
        """
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())


class DynamicTaskConfigurator:
    """
    Class to dynamically configure tasks based on contexts.
    """
    
    def __init__(self, metadata: Dict[str, Any]):
        """
        Initialize the task configurator.
        
        Args:
            metadata: Dataset metadata
        """
        self.metadata = metadata
    
    def create_context_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context_vector: ContextVector,
        feature_data: Optional[Dict[str, np.ndarray]] = None,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a context-specific test split.
        
        Args:
            X: Feature matrix
            y: Target vector
            context_vector: Context vector
            feature_data: Dictionary mapping feature names to arrays (optional)
            random_state: Random state for reproducibility
            
        Returns:
            tuple: Tuple of (X_test_context, y_test_context)
        """
        # Calculate sample weights based on subgroup weights in the context vector
        sample_weights = np.ones(len(y))
        
        if feature_data is not None and context_vector.subgroup_weights:
            for feature_name, subgroup_weights in context_vector.subgroup_weights.items():
                if feature_name in feature_data:
                    feature_values = feature_data[feature_name]
                    
                    for subgroup, weight in subgroup_weights.items():
                        # Find samples belonging to this subgroup
                        subgroup_mask = (feature_values == subgroup)
                        
                        # Apply subgroup weight
                        sample_weights[subgroup_mask] *= weight
        
        # Normalize weights
        if np.sum(sample_weights) > 0:
            sample_weights = sample_weights / np.sum(sample_weights)
        
        # Sample indices according to weights
        indices = np.arange(len(y))
        sampled_indices = np.random.choice(
            indices, size=len(y), replace=True, p=sample_weights
        )
        
        # Create the context-specific test split
        X_test_context = X[sampled_indices]
        y_test_context = y[sampled_indices]
        
        return X_test_context, y_test_context
    
    def create_domain_specific_split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        domain: str,
        feature_data: Optional[Dict[str, np.ndarray]] = None,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a domain-specific test split.
        
        Args:
            X: Feature matrix
            y: Target vector
            domain: Domain name
            feature_data: Dictionary mapping feature names to arrays (optional)
            random_state: Random state for reproducibility
            
        Returns:
            tuple: Tuple of (X_test_domain, y_test_domain)
        """
        # Define domain-specific context vectors
        domain_contexts = {
            "healthcare": ContextVector(
                domain="healthcare",
                constraints={
                    "max_false_negative_rate": 0.05,
                    "min_recall": 0.95
                },
                subgroup_weights={
                    "age": {
                        "elderly": 2.0,  # Prioritize elderly patients
                        "adult": 1.0,
                        "young": 1.0
                    },
                    "disease_rarity": {
                        "rare": 5.0,     # Emphasize rare diseases
                        "common": 1.0
                    }
                },
                performance_thresholds={
                    "recall": 0.95,      # High recall is critical in healthcare
                    "precision": 0.80
                }
            ),
            
            "finance": ContextVector(
                domain="finance",
                constraints={
                    "max_false_positive_rate": 0.01,
                    "min_precision": 0.99
                },
                subgroup_weights={
                    "transaction_size": {
                        "large": 3.0,    # Prioritize large transactions
                        "medium": 1.5,
                        "small": 1.0
                    },
                    "user_type": {
                        "new": 2.0,      # Emphasize new users
                        "existing": 1.0
                    }
                },
                performance_thresholds={
                    "precision": 0.99,   # High precision is critical in finance
                    "recall": 0.80
                }
            ),
            
            "vision": ContextVector(
                domain="vision",
                constraints={
                    "min_accuracy": 0.90
                },
                subgroup_weights={
                    "lighting_condition": {
                        "low_light": 2.0,  # Prioritize challenging lighting
                        "normal": 1.0,
                        "bright": 1.0
                    },
                    "object_size": {
                        "small": 2.0,      # Emphasize small objects
                        "medium": 1.0,
                        "large": 1.0
                    }
                },
                performance_thresholds={
                    "accuracy": 0.90
                }
            ),
            
            "nlp": ContextVector(
                domain="nlp",
                constraints={
                    "min_accuracy": 0.85
                },
                subgroup_weights={
                    "text_length": {
                        "short": 1.5,      # Slightly prioritize short texts
                        "medium": 1.0,
                        "long": 1.0
                    },
                    "language": {
                        "non_english": 2.0,  # Emphasize non-English texts
                        "english": 1.0
                    }
                },
                performance_thresholds={
                    "accuracy": 0.85
                }
            )
        }
        
        # Use the domain-specific context vector if available, otherwise use a default one
        if domain in domain_contexts:
            context_vector = domain_contexts[domain]
        else:
            # Default context vector with equal weights
            context_vector = ContextVector(
                domain=domain,
                constraints={},
                subgroup_weights={},
                performance_thresholds={}
            )
        
        # Create context-specific test split
        return self.create_context_test_split(X, y, context_vector, feature_data, random_state)
    
    def get_context_evaluation_weights(
        self,
        context_vector: ContextVector
    ) -> Dict[str, float]:
        """
        Get evaluation metric weights based on the context.
        
        Args:
            context_vector: Context vector
            
        Returns:
            dict: Dictionary mapping metric names to weights
        """
        # Default weights
        default_weights = {
            "performance": 1.0,
            "fairness": 1.0,
            "robustness": 1.0,
            "environmental_impact": 1.0,
            "interpretability": 1.0
        }
        
        # Domain-specific weights
        domain_weights = {
            "healthcare": {
                "performance": 1.0,
                "fairness": 1.2,     # Higher weight for fairness
                "robustness": 1.5,    # Higher weight for robustness
                "environmental_impact": 0.5,
                "interpretability": 1.8     # Higher weight for interpretability
            },
            
            "finance": {
                "performance": 1.2,    # Higher weight for performance
                "fairness": 1.0,
                "robustness": 1.5,    # Higher weight for robustness
                "environmental_impact": 0.6,
                "interpretability": 1.7     # Higher weight for interpretability
            },
            
            "vision": {
                "performance": 1.5,    # Higher weight for performance
                "fairness": 1.0,
                "robustness": 1.3,    # Higher weight for robustness
                "environmental_impact": 0.7,
                "interpretability": 0.5
            },
            
            "nlp": {
                "performance": 1.3,    # Higher weight for performance
                "fairness": 1.2,     # Higher weight for fairness
                "robustness": 1.0,
                "environmental_impact": 0.8,
                "interpretability": 0.7
            }
        }
        
        # Use domain-specific weights if available, otherwise use default
        if context_vector.domain in domain_weights:
            weights = domain_weights[context_vector.domain]
        else:
            weights = default_weights
        
        # Override with constraints if specified
        if "metric_weights" in context_vector.constraints:
            for metric, weight in context_vector.constraints["metric_weights"].items():
                if metric in weights:
                    weights[metric] = weight
        
        return weights
    
    def calculate_context_score(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        context_vector: ContextVector
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Calculate a context-specific score based on evaluation results.
        
        Args:
            evaluation_results: Results from the MultiMetricEvaluationSuite
            context_vector: Context vector
            
        Returns:
            dict: Dictionary containing the context score and component scores
        """
        # Get metric weights based on context
        weights = self.get_context_evaluation_weights(context_vector)
        
        # Extract key metrics from evaluation results
        metrics = {}
        
        # Performance metrics
        if "performance" in evaluation_results:
            perf = evaluation_results["performance"]
            if isinstance(perf, dict):
                # For classification
                if "accuracy" in perf:
                    metrics["performance_accuracy"] = perf["accuracy"]
                if "f1" in perf:
                    metrics["performance_f1"] = perf["f1"]
                if "precision" in perf:
                    metrics["performance_precision"] = perf["precision"]
                if "recall" in perf:
                    metrics["performance_recall"] = perf["recall"]
                # For regression
                if "mse" in perf:
                    metrics["performance_mse"] = 1.0 / (1.0 + perf["mse"])  # Invert for consistent direction
                if "r2" in perf:
                    metrics["performance_r2"] = max(0, perf["r2"])  # Clip negative R2
        
        # Fairness metrics
        if "fairness" in evaluation_results:
            fair = evaluation_results["fairness"]
            for feature, feature_metrics in fair.items():
                if isinstance(feature_metrics, dict):
                    if "demographic_parity_diff" in feature_metrics:
                        metrics[f"fairness_{feature}_dp_diff"] = 1.0 - feature_metrics["demographic_parity_diff"]
                    if "equal_opportunity_diff" in feature_metrics:
                        metrics[f"fairness_{feature}_eo_diff"] = 1.0 - feature_metrics["equal_opportunity_diff"]
                    if "disparate_impact" in feature_metrics:
                        # Transform to [0, 1] where 1 is perfect fairness
                        di = feature_metrics["disparate_impact"]
                        if di > 0:
                            metrics[f"fairness_{feature}_di"] = min(di, 1.0) if di <= 1.0 else 1.0 / di
        
        # Robustness metrics
        if "robustness" in evaluation_results:
            rob = evaluation_results["robustness"]
            if "noise" in rob and isinstance(rob["noise"], dict):
                if "retained_accuracy_pct" in rob["noise"]:
                    metrics["robustness_noise"] = rob["noise"]["retained_accuracy_pct"] / 100.0
            if "shift" in rob and isinstance(rob["shift"], dict):
                if "retained_accuracy_pct" in rob["shift"]:
                    metrics["robustness_shift"] = rob["shift"]["retained_accuracy_pct"] / 100.0
            if "adversarial" in rob and isinstance(rob["adversarial"], dict):
                if "adversarial_accuracy" in rob["adversarial"]:
                    metrics["robustness_adversarial"] = rob["adversarial"]["adversarial_accuracy"]
        
        # Environmental impact metrics
        if "environmental_impact" in evaluation_results:
            env = evaluation_results["environmental_impact"]
            if isinstance(env, dict):
                if "total_energy_kwh_per_sample" in env:
                    # Invert so that lower energy usage gets a higher score
                    energy = env["total_energy_kwh_per_sample"]
                    metrics["environmental_energy"] = 1.0 / (1.0 + 1000 * energy)  # Scale to make differences visible
                elif "total_energy_kwh" in env and "elapsed_time_seconds" in env:
                    # Calculate energy efficiency
                    energy = env["total_energy_kwh"]
                    time_sec = env["elapsed_time_seconds"]
                    samples = context_vector.constraints.get("num_samples", 1000)  # Default if not specified
                    energy_per_sample = energy / samples
                    metrics["environmental_energy"] = 1.0 / (1.0 + 1000 * energy_per_sample)
        
        # Interpretability metrics
        if "interpretability" in evaluation_results:
            interp = evaluation_results["interpretability"]
            if "stability" in interp and isinstance(interp["stability"], dict):
                if "mean_stability" in interp["stability"]:
                    metrics["interpretability_stability"] = interp["stability"]["mean_stability"]
            if "concentration" in interp and isinstance(interp["concentration"], dict):
                if "feature_concentration_ratio" in interp["concentration"]:
                    metrics["interpretability_concentration"] = 1.0 - interp["concentration"]["feature_concentration_ratio"]
            if "clarity" in interp and isinstance(interp["clarity"], dict):
                if "mean_gradient_magnitude" in interp["clarity"]:
                    # Normalize to [0, 1] range (assuming gradient magnitudes are in a reasonable range)
                    magnitude = interp["clarity"]["mean_gradient_magnitude"]
                    metrics["interpretability_clarity"] = min(1.0, magnitude / 10.0)  # Cap at 1.0
        
        # Calculate category scores
        category_scores = {}
        
        # Performance score
        perf_metrics = [v for k, v in metrics.items() if k.startswith("performance_")]
        category_scores["performance"] = sum(perf_metrics) / len(perf_metrics) if perf_metrics else 0.5
        
        # Fairness score
        fair_metrics = [v for k, v in metrics.items() if k.startswith("fairness_")]
        category_scores["fairness"] = sum(fair_metrics) / len(fair_metrics) if fair_metrics else 0.5
        
        # Robustness score
        rob_metrics = [v for k, v in metrics.items() if k.startswith("robustness_")]
        category_scores["robustness"] = sum(rob_metrics) / len(rob_metrics) if rob_metrics else 0.5
        
        # Environmental impact score
        env_metrics = [v for k, v in metrics.items() if k.startswith("environmental_")]
        category_scores["environmental_impact"] = sum(env_metrics) / len(env_metrics) if env_metrics else 0.5
        
        # Interpretability score
        interp_metrics = [v for k, v in metrics.items() if k.startswith("interpretability_")]
        category_scores["interpretability"] = sum(interp_metrics) / len(interp_metrics) if interp_metrics else 0.5
        
        # Calculate weighted sum for overall score
        overall_score = 0.0
        total_weight = 0.0
        
        for category, score in category_scores.items():
            if category in weights:
                weight = weights[category]
                overall_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_score /= total_weight
        
        return {
            "overall_score": float(overall_score),
            "category_scores": category_scores,
            "metrics": metrics
        }


def create_example_contexts():
    """
    Create example context vectors for various domains.
    
    Returns:
        dict: Dictionary mapping domain names to ContextVector objects
    """
    contexts = {}
    
    # Healthcare context
    healthcare_context = ContextVector(
        domain="healthcare",
        constraints={
            "max_false_negative_rate": 0.05,
            "min_recall": 0.95
        },
        subgroup_weights={
            "age": {
                "elderly": 2.0,  # Prioritize elderly patients
                "adult": 1.0,
                "young": 1.0
            },
            "disease_rarity": {
                "rare": 5.0,     # Emphasize rare diseases
                "common": 1.0
            }
        },
        performance_thresholds={
            "recall": 0.95,      # High recall is critical in healthcare
            "precision": 0.80
        }
    )
    contexts["healthcare"] = healthcare_context
    
    # Finance context
    finance_context = ContextVector(
        domain="finance",
        constraints={
            "max_false_positive_rate": 0.01,
            "min_precision": 0.99
        },
        subgroup_weights={
            "transaction_size": {
                "large": 3.0,    # Prioritize large transactions
                "medium": 1.5,
                "small": 1.0
            },
            "user_type": {
                "new": 2.0,      # Emphasize new users
                "existing": 1.0
            }
        },
        performance_thresholds={
            "precision": 0.99,   # High precision is critical in finance
            "recall": 0.80
        }
    )
    contexts["finance"] = finance_context
    
    # Vision context
    vision_context = ContextVector(
        domain="vision",
        constraints={
            "min_accuracy": 0.90
        },
        subgroup_weights={
            "lighting_condition": {
                "low_light": 2.0,  # Prioritize challenging lighting
                "normal": 1.0,
                "bright": 1.0
            },
            "object_size": {
                "small": 2.0,      # Emphasize small objects
                "medium": 1.0,
                "large": 1.0
            }
        },
        performance_thresholds={
            "accuracy": 0.90
        }
    )
    contexts["vision"] = vision_context
    
    # NLP context
    nlp_context = ContextVector(
        domain="nlp",
        constraints={
            "min_accuracy": 0.85
        },
        subgroup_weights={
            "text_length": {
                "short": 1.5,      # Slightly prioritize short texts
                "medium": 1.0,
                "long": 1.0
            },
            "language": {
                "non_english": 2.0,  # Emphasize non-English texts
                "english": 1.0
            }
        },
        performance_thresholds={
            "accuracy": 0.85
        }
    )
    contexts["nlp"] = nlp_context
    
    return contexts


def save_example_contexts(directory: str):
    """
    Create and save example context vectors.
    
    Args:
        directory: Directory to save contexts
    """
    contexts = create_example_contexts()
    
    os.makedirs(directory, exist_ok=True)
    
    for domain, context in contexts.items():
        context.save(directory)


if __name__ == "__main__":
    # Create and save example contexts
    save_example_contexts("../data/contexts")
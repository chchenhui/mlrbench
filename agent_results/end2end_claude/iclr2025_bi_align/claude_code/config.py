"""
Configuration for CEVA framework experiments.
"""
import os
from pathlib import Path

# Directory paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = ROOT_DIR / "data"

# Ensure directories exist
for directory in [RESULTS_DIR, FIGURES_DIR, DATA_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Simulation parameters
SIMULATION_CONFIG = {
    "n_dimensions": 5,  # Number of value dimensions
    "n_iterations": 100,  # Number of iterations for simulation
    "n_agents": 50,      # Number of simulated human agents
    "random_seed": 42,   # Random seed for reproducibility
}

# Value dimensions representation
# These are the dimensions we'll use for our value space
VALUE_DIMENSIONS = [
    "autonomy",      # Preference for independence and self-direction
    "benevolence",   # Concern for others' welfare
    "security",      # Safety, stability and order
    "achievement",   # Personal success and competence
    "conformity"     # Adherence to norms and expectations
]

# Value evolution parameters
VALUE_EVOLUTION_PARAMS = {
    "inertia_coefficient": 0.8,        # Alpha: tendency to maintain existing values
    "external_sensitivity": 0.1,       # Beta: sensitivity to external social factors
    "interaction_sensitivity": 0.1,    # Gamma: sensitivity to AI interactions
    "noise_level": 0.05,               # Random variation in value evolution
    "drift_rate": 0.02,                # Base rate of gradual value drift
}

# Multi-level value representation weights
# (Used by the CEVA model to determine how quickly different types of values adapt)
VALUE_ADAPTATION_RATES = {
    "core_safety": 0.1,     # Core safety values adapt very slowly
    "cultural": 0.3,        # Cultural values adapt at moderate pace
    "preference": 0.7,      # Personal preferences adapt more quickly
}

# Experimental scenarios
SCENARIOS = [
    {
        "name": "gradual_drift",
        "description": "Gradual preference drift scenario - values change slowly over time",
        "drift_multiplier": 1.0,
        "external_events": None,
        "duration": 100
    },
    {
        "name": "rapid_shift",
        "description": "Rapid value shift scenario - sudden change in response to critical event",
        "drift_multiplier": 1.0,
        "external_events": [
            {"time": 50, "dimensions": ["security", "autonomy"], "magnitude": 0.3}
        ],
        "duration": 100
    },
    {
        "name": "value_conflict",
        "description": "Value conflict scenario - tension between different value levels",
        "drift_multiplier": 1.0,
        "external_events": [
            {"time": 30, "dimensions": ["security"], "magnitude": 0.25},
            {"time": 60, "dimensions": ["autonomy"], "magnitude": 0.25},
        ],
        "duration": 100
    }
]

# Models to compare
MODELS = [
    {
        "name": "static_alignment",
        "description": "Traditional static alignment model - no adaptation",
        "update_rate": 0.0,
        "multi_level": False,
        "bidirectional": False
    },
    {
        "name": "adaptive_alignment",
        "description": "Simple adaptive alignment model - uniform adaptation",
        "update_rate": 0.5,
        "multi_level": False,
        "bidirectional": False
    },
    {
        "name": "ceva_basic",
        "description": "Basic CEVA model with multi-level value adaptation",
        "update_rate": 0.5,
        "multi_level": True,
        "bidirectional": False,
        "feedback_mechanism": "none"
    },
    {
        "name": "ceva_full",
        "description": "Full CEVA model with bidirectional feedback",
        "update_rate": 0.5,
        "multi_level": True,
        "bidirectional": True,
        "feedback_mechanism": "reflection_prompting"
    }
]

# Evaluation metrics
METRICS = [
    "adaptation_accuracy",          # How well the AI model's values match human values
    "adaptation_response_time",     # Time to reduce value misalignment below threshold
    "stability",                    # Resistance to spurious adaptation
    "user_satisfaction",            # Simulated user satisfaction with responses
    "agency_preservation"           # How well human agency is preserved in the process
]

# Visualization settings
VISUALIZATION_CONFIG = {
    "figsize": (12, 8),
    "dpi": 100,
    "palette": "viridis",
    "style": "whitegrid",
    "context": "paper",
    "save_format": "png"
}
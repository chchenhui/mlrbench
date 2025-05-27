"""
Configuration settings for AEB experiments.
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    model_type: str
    pretrained: bool = False
    checkpoint_path: Optional[str] = None
    
    # Training parameters
    lr: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 30
    early_stopping: Optional[int] = 5
    scheduler: Optional[Dict[str, Any]] = None

@dataclass
class DataConfig:
    """Configuration for datasets."""
    dataset: str
    data_dir: str = './data'
    val_split: float = 0.1
    batch_size: int = 128
    num_workers: int = 4
    seed: int = 42

@dataclass
class EvolverConfig:
    """Configuration for the Benchmark Evolver."""
    pop_size: int = 50
    max_generations: int = 30
    tournament_size: int = 3
    crossover_prob: float = 0.7
    mutation_prob: float = 0.3
    elitism_count: int = 2
    min_transformations: int = 1
    max_transformations: int = 5
    fitness_weights: Tuple[float, float, float] = (0.6, 0.2, 0.2)  # challenge, diversity, novelty
    save_dir: str = './results/evolutionary_runs'
    seed: int = 42

@dataclass
class ExperimentConfig:
    """Main configuration for an experiment."""
    name: str
    output_dir: str
    data: DataConfig
    models: Dict[str, ModelConfig]
    evolver: EvolverConfig
    device: str = 'cuda'
    
    # Adversarial hardening parameters
    hardening_epochs: int = 15
    hardening_alpha: float = 0.5  # Weight for original data loss


def load_config(config_path):
    """Load configuration from a YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create data config
    data_config = DataConfig(**config_dict.get('data', {}))
    
    # Create model configs
    model_configs = {}
    for model_name, model_dict in config_dict.get('models', {}).items():
        model_configs[model_name] = ModelConfig(name=model_name, **model_dict)
    
    # Create evolver config
    evolver_config = EvolverConfig(**config_dict.get('evolver', {}))
    
    # Create main experiment config
    experiment_config = ExperimentConfig(
        name=config_dict.get('name', 'aeb_experiment'),
        output_dir=config_dict.get('output_dir', './results'),
        data=data_config,
        models=model_configs,
        evolver=evolver_config,
        device=config_dict.get('device', 'cuda'),
        hardening_epochs=config_dict.get('hardening_epochs', 15),
        hardening_alpha=config_dict.get('hardening_alpha', 0.5)
    )
    
    logger.info(f"Configuration loaded: experiment '{experiment_config.name}'")
    return experiment_config

def save_config(config, config_path):
    """Save configuration to a YAML file."""
    # Convert config to dictionary
    config_dict = {
        'name': config.name,
        'output_dir': config.output_dir,
        'device': config.device,
        'hardening_epochs': config.hardening_epochs,
        'hardening_alpha': config.hardening_alpha,
        'data': vars(config.data),
        'evolver': vars(config.evolver),
        'models': {name: vars(model_config) for name, model_config in config.models.items()}
    }
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    logger.info(f"Configuration saved to {config_path}")

# Default configuration for CIFAR-10 experiment
DEFAULT_CIFAR10_CONFIG = ExperimentConfig(
    name='aeb_cifar10',
    output_dir='./results',
    data=DataConfig(
        dataset='cifar10',
        data_dir='./data',
        val_split=0.1,
        batch_size=128,
        num_workers=4,
        seed=42
    ),
    models={
        'standard_cnn': ModelConfig(
            name='standard_cnn',
            model_type='simplecnn',
            pretrained=False,
            lr=0.001,
            weight_decay=1e-5,
            batch_size=128,
            epochs=30,
            early_stopping=5,
            scheduler={'type': 'reduce_on_plateau', 'patience': 3, 'factor': 0.5}
        ),
        'standard_resnet': ModelConfig(
            name='standard_resnet',
            model_type='resnet18',
            pretrained=True,
            lr=0.0005,
            weight_decay=1e-5,
            batch_size=128,
            epochs=30,
            early_stopping=5,
            scheduler={'type': 'reduce_on_plateau', 'patience': 3, 'factor': 0.5}
        )
    },
    evolver=EvolverConfig(
        pop_size=30,
        max_generations=20,
        tournament_size=3,
        crossover_prob=0.7,
        mutation_prob=0.3,
        elitism_count=2,
        min_transformations=1,
        max_transformations=3,
        fitness_weights=(0.6, 0.2, 0.2),
        save_dir='./results/evolutionary_runs',
        seed=42
    ),
    device='cuda',
    hardening_epochs=15,
    hardening_alpha=0.5
)
import os
import yaml
import logging
import re
from typing import Dict, List, Any, Optional, Union
import torch
import random
import numpy as np

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Utility class for loading and managing experiment configurations."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize ConfigLoader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir
    
    def _resolve_extends(self, config: Dict[str, Any], config_path: str) -> Dict[str, Any]:
        """
        Resolve the 'extends' directive by loading and merging the parent config.
        
        Args:
            config: Configuration dictionary that may contain an 'extends' key
            config_path: Path to the current configuration file
            
        Returns:
            Merged configuration dictionary
        """
        if 'extends' not in config:
            return config
        
        parent_file = config.pop('extends')
        parent_path = os.path.join(os.path.dirname(config_path), parent_file)
        
        if not os.path.exists(parent_path):
            parent_path = os.path.join(self.config_dir, parent_file)
        
        if not os.path.exists(parent_path):
            logger.warning(f"Parent config file {parent_file} not found, ignoring 'extends' directive")
            return config
        
        parent_config = self.load_config(parent_path)
        merged_config = self._merge_configs(parent_config, config)
        
        return merged_config
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base with
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            # If both base and override have dict values, merge them recursively
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                # Otherwise, override or add the key-value pair
                merged[key] = value
        
        return merged
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Resolve 'extends' directive
            config = self._resolve_extends(config, config_path)
            
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise
    
    def save_config(self, config: Dict[str, Any], save_path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration dictionary
            save_path: Path to save the configuration
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
            logger.info(f"Saved config to {save_path}")
        except Exception as e:
            logger.error(f"Error saving config to {save_path}: {e}")
            raise
    
    def update_paths(self, config: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
        """
        Update relative paths in config to absolute paths based on base_dir.
        
        Args:
            config: Configuration dictionary
            base_dir: Base directory for resolving relative paths
            
        Returns:
            Configuration with updated paths
        """
        updated_config = config.copy()
        
        # Update experiment paths
        if 'experiment' in updated_config:
            for key in ['log_dir', 'save_dir', 'figures_dir', 'tensorboard_dir']:
                if key in updated_config['experiment']:
                    path = updated_config['experiment'][key]
                    if not os.path.isabs(path):
                        updated_config['experiment'][key] = os.path.join(base_dir, path)
        
        # Update data paths
        if 'data' in updated_config:
            for key in ['data_dir', 'processed_dir']:
                if key in updated_config['data']:
                    path = updated_config['data'][key]
                    if not os.path.isabs(path):
                        updated_config['data'][key] = os.path.join(base_dir, path)
        
        return updated_config
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        # Check for required sections
        required_sections = ['experiment', 'data', 'model', 'training']
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False
        
        # Check for required experiment parameters
        required_exp_params = ['name', 'device']
        for param in required_exp_params:
            if param not in config['experiment']:
                logger.error(f"Missing required experiment parameter: {param}")
                return False
        
        # Check for valid device
        if config['experiment']['device'] not in ['cuda', 'cpu', 'auto']:
            logger.error(f"Invalid device: {config['experiment']['device']}")
            return False
        
        # Check data configuration
        required_data_params = ['model_properties', 'train_ratio', 'val_ratio', 'test_ratio', 'batch_size']
        for param in required_data_params:
            if param not in config['data']:
                logger.error(f"Missing required data parameter: {param}")
                return False
        
        # Check that ratios sum to 1
        ratios_sum = config['data']['train_ratio'] + config['data']['val_ratio'] + config['data']['test_ratio']
        if not np.isclose(ratios_sum, 1.0, atol=1e-6):
            logger.error(f"Data split ratios must sum to 1.0, got {ratios_sum}")
            return False
        
        # Check model configuration
        if config['model']['type'] not in ['weightnet', 'mlp', 'stats']:
            logger.error(f"Invalid model type: {config['model']['type']}")
            return False
        
        # Check training configuration
        required_training_params = ['num_epochs', 'optimizer']
        for param in required_training_params:
            if param not in config['training']:
                logger.error(f"Missing required training parameter: {param}")
                return False
        
        return True
    
    def get_device(self, config: Dict[str, Any]) -> torch.device:
        """
        Get the device to use for the experiment.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            PyTorch device
        """
        device_name = config['experiment']['device']
        
        if device_name == 'auto':
            device_name = 'cuda' if torch.cuda.is_available() and config['experiment'].get('use_gpu', True) else 'cpu'
        
        if device_name == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device_name = 'cpu'
        
        device = torch.device(device_name)
        logger.info(f"Using device: {device}")
        
        return device
    
    def set_seed(self, config: Dict[str, Any]) -> None:
        """
        Set random seeds for reproducibility.
        
        Args:
            config: Configuration dictionary
        """
        seed = config['experiment'].get('seed', 42)
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.info(f"Random seed set to {seed}")
    
    def create_experiment_dirs(self, config: Dict[str, Any]) -> None:
        """
        Create directories for the experiment.
        
        Args:
            config: Configuration dictionary
        """
        experiment_name = config['experiment']['name']
        
        # Create experiment directories
        for key in ['log_dir', 'save_dir', 'figures_dir', 'tensorboard_dir']:
            if key in config['experiment']:
                dir_path = config['experiment'][key]
                
                # Add experiment name subdirectory
                dir_path = os.path.join(dir_path, experiment_name)
                config['experiment'][key] = dir_path
                
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")

def load_experiment_config(config_path: str, base_dir: str = ".") -> Dict[str, Any]:
    """
    Load and prepare an experiment configuration.
    
    Args:
        config_path: Path to the configuration file
        base_dir: Base directory for resolving relative paths
        
    Returns:
        Prepared configuration dictionary
    """
    config_loader = ConfigLoader()
    
    # Load configuration
    config = config_loader.load_config(config_path)
    
    # Update paths to absolute
    config = config_loader.update_paths(config, base_dir)
    
    # Validate configuration
    if not config_loader.validate_config(config):
        raise ValueError(f"Invalid configuration: {config_path}")
    
    # Create experiment directories
    config_loader.create_experiment_dirs(config)
    
    # Set random seeds
    config_loader.set_seed(config)
    
    # Save the final merged config for reference
    experiment_dir = config['experiment']['save_dir']
    config_save_path = os.path.join(experiment_dir, "config.yaml")
    config_loader.save_config(config, config_save_path)
    
    return config

def get_model_config(config: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get model configuration.
    
    Args:
        config: Full configuration dictionary
        model_name: Optional model name to get config for (for baseline models)
        
    Returns:
        Model configuration dictionary
    """
    if model_name is None:
        # Return the main model config
        return config['model']
    else:
        # Look for the model in baselines
        if 'baselines' in config and 'models' in config['baselines']:
            for model_config in config['baselines']['models']:
                if model_config['name'] == model_name:
                    return model_config
        
        # Look for the model in ablation studies
        if 'ablation' in config and 'experiments' in config['ablation']:
            for ablation_config in config['ablation']['experiments']:
                if ablation_config['name'] == model_name:
                    # Merge with base model config
                    base_model_config = config['model'].copy()
                    if 'model' in ablation_config:
                        base_model_config.update(ablation_config['model'])
                    
                    # Add the ablation name to the model config
                    base_model_config['name'] = model_name
                    return base_model_config
        
        raise ValueError(f"Model config not found for {model_name}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test config loading
    config_path = "configs/base_config.yaml"
    config = load_experiment_config(config_path)
    
    print(f"Loaded config for experiment: {config['experiment']['name']}")
    print(f"Model type: {config['model']['type']}")
    print(f"Data properties: {config['data']['model_properties']}")
    
    # Test loading model configs
    main_model_config = get_model_config(config)
    print(f"Main model config: {main_model_config['type']}")
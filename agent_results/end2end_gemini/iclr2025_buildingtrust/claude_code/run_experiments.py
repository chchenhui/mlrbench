#!/usr/bin/env python
"""
Main script to run Concept-Graph experiments.

This script sets up and runs experiments for the Concept-Graph explanations approach,
processing datasets, generating visualizations, and producing analysis reports.
"""

import os
import sys
import json
import time
import logging
import argparse
import torch
from datetime import datetime

from utils.logging_utils import setup_logger
from experiments.experiment_runner import ExperimentRunner

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Concept-Graph experiments")
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on ('cpu' or 'cuda')"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="gsm8k",
        help="Comma-separated list of datasets to use"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples per dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiment_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--use_openai",
        type=bool,
        default=True,
        help="Whether to use OpenAI for concept labeling"
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for concept labeling"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()

def load_config(config_path=None, args=None):
    """
    Load configuration from file and/or command line arguments.
    
    Args:
        config_path: Path to configuration file
        args: Command line arguments
        
    Returns:
        Dictionary with configuration
    """
    # Default configuration
    config = {
        "models_config": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "layers_to_extract": None,
            "cache_dir": "cache"
        },
        "dataset_config": {
            "datasets": ["gsm8k"],
            "max_samples": 100,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "cache_dir": "cache/datasets",
            "test_splits": ["test"]
        },
        "experiment_config": {
            "num_samples_per_dataset": 10,
            "use_openai": True,
            "openai_model": "gpt-4o-mini",
            "seed": 42,
            "generation": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9
            },
            "concept_mapping": {
                "num_concepts": 10,
                "pca_components": 50,
                "umap_components": 2,
                "clustering_method": "kmeans",
                "min_edge_weight": 0.1,
                "graph_layout": "temporal"
            }
        }
    }
    
    # Load from config file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                
                # Update nested dictionaries
                for section, section_config in file_config.items():
                    if section in config and isinstance(config[section], dict):
                        config[section].update(section_config)
                    else:
                        config[section] = section_config
                
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {str(e)}")
    
    # Update from command line arguments if provided
    if args:
        # Update model config
        if hasattr(args, 'model_name'):
            config["models_config"]["model_name"] = args.model_name
        
        if hasattr(args, 'device'):
            config["models_config"]["device"] = args.device
        
        # Update dataset config
        if hasattr(args, 'datasets'):
            config["dataset_config"]["datasets"] = args.datasets.split(",")
        
        if hasattr(args, 'num_samples'):
            config["experiment_config"]["num_samples_per_dataset"] = args.num_samples
        
        # Update experiment config
        if hasattr(args, 'use_openai'):
            config["experiment_config"]["use_openai"] = args.use_openai
        
        if hasattr(args, 'openai_model'):
            config["experiment_config"]["openai_model"] = args.openai_model
        
        if hasattr(args, 'seed'):
            config["experiment_config"]["seed"] = args.seed
    
    return config

def setup_experiment_dir(base_dir="experiment_results"):
    """
    Set up experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to the experiment directory
    """
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir

def main():
    """Main function to run experiments."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, args)
    
    # Setup experiment directory
    experiment_dir = setup_experiment_dir(args.output_dir)
    
    # Setup logging
    log_file = os.path.join(experiment_dir, "log.txt")
    logger = setup_logger(log_file, getattr(logging, args.log_level))
    
    # Log start of experiment
    logger.info("="*80)
    logger.info("Starting Concept-Graph experiments")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Model: {config['models_config']['model_name']}")
    logger.info(f"Device: {config['models_config']['device']}")
    logger.info(f"Datasets: {config['dataset_config']['datasets']}")
    logger.info(f"Samples per dataset: {config['experiment_config']['num_samples_per_dataset']}")
    logger.info("="*80)
    
    # Save configuration
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved configuration to {config_path}")
    
    # Create experiment runner
    try:
        runner = ExperimentRunner(
            experiment_dir=experiment_dir,
            models_config=config["models_config"],
            dataset_config=config["dataset_config"],
            experiment_config=config["experiment_config"]
        )
        
        logger.info("Created experiment runner")
        
        # Run full pipeline
        start_time = time.time()
        results = runner.run_full_pipeline()
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"Completed experiments in {execution_time:.2f} seconds")
        
        if "error" in results:
            logger.error(f"Error in experiment pipeline: {results['error']}")
            return 1
        
        # Log summary and results location
        logger.info("="*80)
        logger.info("Experiment Summary")
        logger.info(f"Results directory: {experiment_dir}")
        
        if "report_path" in results and os.path.exists(results["report_path"]):
            logger.info(f"Report: {results['report_path']}")
        
        logger.info("="*80)
        
        # Copy results to the results folder
        results_dir = os.path.join(os.path.dirname(os.path.dirname(experiment_dir)), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create a symlink to the experiment results in the results folder
        symlink_path = os.path.join(results_dir, f"concept_graph_{os.path.basename(experiment_dir)}")
        if os.path.exists(symlink_path):
            os.remove(symlink_path)
        
        os.symlink(experiment_dir, symlink_path, target_is_directory=True)
        logger.info(f"Created symlink to results in {symlink_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running experiments: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
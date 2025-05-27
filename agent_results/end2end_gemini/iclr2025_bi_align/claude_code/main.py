#!/usr/bin/env python3
"""
Main script for running the AI Cognitive Tutor experiment.

This script orchestrates the entire experimental workflow, including:
1. Setting up the experiment
2. Running the simulation for control and treatment groups
3. Analyzing and visualizing the results
4. Generating reports
"""

import os
import sys
import logging
import argparse
import yaml
import json
import numpy as np
import pandas as pd
import torch
import random
from datetime import datetime
from pathlib import Path

# Local imports
from models.ai_diagnostic import AIDiagnosticSystem
from models.cognitive_tutor import AICognitiveTutor
from models.baselines import StandardExplanation, NoExplanation, StaticTutorial
from simulation.participant import SimulatedParticipant
from simulation.experiment import Experiment
from evaluation.metrics import evaluate_experiment
from visualization.visualizer import ExperimentVisualizer

# Set up logging
def setup_logging(log_file):
    """Set up logging configuration"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def set_random_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    """Load experiment configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_results(results, save_path):
    """Save experiment results to JSON and CSV files"""
    os.makedirs(save_path, exist_ok=True)
    
    # Save raw results as JSON
    with open(os.path.join(save_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary metrics as CSV
    metrics_df = pd.DataFrame(results['summary_metrics'])
    metrics_df.to_csv(os.path.join(save_path, 'metrics.csv'), index=False)
    
    # Save participant-level data
    participant_df = pd.DataFrame(results['participant_data'])
    participant_df.to_csv(os.path.join(save_path, 'participant_data.csv'), index=False)
    
    # Save trial-level data
    trial_df = pd.DataFrame(results['trial_data'])
    trial_df.to_csv(os.path.join(save_path, 'trial_data.csv'), index=False)

def main(args):
    """Main function to run the experiment"""
    # Load configuration
    config = load_config(args.config)
    
    # Setup experiment directory and logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(config['experiment']['save_path'], f"{config['experiment']['name']}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    log_file = os.path.join(experiment_dir, "experiment.log")
    logger = setup_logging(log_file)
    
    # Set random seeds for reproducibility
    set_random_seeds(config['experiment']['random_seed'])
    
    logger.info("Starting AI Cognitive Tutor experiment")
    logger.info(f"Configuration: {config}")
    
    # Set up the AI diagnostic system and different explanation methods
    logger.info("Initializing AI Diagnostic System")
    ai_diagnostic = AIDiagnosticSystem(
        model_type=config['ai_diagnostic']['model_type'],
        accuracy=config['ai_diagnostic']['accuracy'],
        uncertainty_levels=config['ai_diagnostic']['uncertainty_levels'],
        explanation_types=config['ai_diagnostic']['explanation_types']
    )
    
    # Initialize baseline methods
    logger.info("Initializing baseline methods")
    baselines = {}
    if config['baselines']['standard_explanation']:
        baselines['standard_explanation'] = StandardExplanation(ai_diagnostic)
    if config['baselines']['no_explanation']:
        baselines['no_explanation'] = NoExplanation(ai_diagnostic)
    if config['baselines']['static_tutorial']:
        baselines['static_tutorial'] = StaticTutorial(ai_diagnostic)
    
    # Initialize the AI Cognitive Tutor for the treatment group
    logger.info("Initializing AI Cognitive Tutor")
    cognitive_tutor = AICognitiveTutor(
        ai_diagnostic=ai_diagnostic,
        activation_threshold=config['tutor']['activation_threshold'],
        strategies=config['tutor']['strategies'],
        triggers=config['triggers']
    )
    
    # Set up and run the experiment
    logger.info("Setting up the experiment")
    experiment = Experiment(
        ai_diagnostic=ai_diagnostic,
        cognitive_tutor=cognitive_tutor,
        baselines=baselines,
        config=config,
        logger=logger
    )
    
    logger.info("Running the experiment")
    results = experiment.run()
    
    # Save the results
    logger.info("Saving experiment results")
    save_results(results, experiment_dir)
    
    # Evaluate the results
    logger.info("Evaluating experiment results")
    evaluation_results = evaluate_experiment(results, config)
    
    # Generate visualizations
    logger.info("Generating visualizations")
    visualizer = ExperimentVisualizer(results, config, save_dir=experiment_dir)
    visualization_paths = visualizer.generate_all_visualizations()
    
    # Generate results.md
    logger.info("Generating results markdown")
    from reports.report_generator import generate_results_markdown
    results_md_path = os.path.join(args.results_dir, "results.md")
    generate_results_markdown(
        results=results,
        evaluation_results=evaluation_results,
        visualization_paths=visualization_paths,
        config=config,
        output_path=results_md_path,
        relative_path_prefix="../results/"
    )
    
    # Move log file to results directory
    import shutil
    shutil.copy(log_file, os.path.join(args.results_dir, "log.txt"))
    
    # Move generated figures to results directory
    for fig_path in visualization_paths:
        filename = os.path.basename(fig_path)
        shutil.copy(fig_path, os.path.join(args.results_dir, filename))
    
    logger.info("Experiment completed successfully")
    logger.info(f"Results saved to {experiment_dir}")
    logger.info(f"Results summary saved to {results_md_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Cognitive Tutor experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--results_dir", type=str, default="../results", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    main(args)
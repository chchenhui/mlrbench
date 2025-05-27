"""
Main Experiment Runner for Contextual Dataset Deprecation Framework

This script runs the complete experiment to evaluate the effectiveness of the
Contextual Dataset Deprecation Framework compared to baseline approaches.
"""

import os
import json
import time
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from experimental_design import (
    WarningLevel, DeprecationRecord, SyntheticDataset, 
    DatasetVersion, DeprecationStrategy, run_experiment,
    create_synthetic_datasets, create_deprecation_records
)
from dataset_generator import (
    generate_dataset_collection, save_datasets, 
    generate_deprecation_records, save_deprecation_records
)
from framework import (
    run_framework_simulation, User, ContextualDeprecationFramework,
    create_simulated_users, simulate_initial_dataset_usage
)
from baselines import (
    run_traditional_simulation, run_basic_simulation,
    TraditionalDeprecation, BasicDeprecation
)
from evaluation import (
    EvaluationMetrics, load_experiment_results
)

# Set up logging
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log.txt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("experiment_runner")

def setup_experiment_environment() -> Tuple[Dict[str, SyntheticDataset], Dict[str, DeprecationRecord]]:
    """
    Set up the experiment environment by creating synthetic datasets and deprecation records.
    
    Returns:
        Tuple of (datasets, deprecation_records)
    """
    logger.info("Setting up experiment environment")
    
    # Generate synthetic datasets
    datasets = generate_dataset_collection(save_to_disk=True)
    logger.info(f"Generated {len(datasets)} synthetic datasets")
    
    # Generate deprecation records
    deprecation_records = generate_deprecation_records(datasets)
    save_deprecation_records(deprecation_records)
    logger.info(f"Generated {len(deprecation_records)} deprecation records")
    
    return datasets, deprecation_records

def run_all_experiments(
    datasets: Dict[str, SyntheticDataset],
    deprecation_records: Dict[str, DeprecationRecord],
    n_simulations: int = 50,
    output_dir: str = None
) -> Dict[str, str]:
    """
    Run experiments for all deprecation strategies.
    
    Args:
        datasets: Dictionary of synthetic datasets
        deprecation_records: Dictionary of deprecation records
        n_simulations: Number of simulated research groups/users
        output_dir: Directory to save results
        
    Returns:
        Dictionary mapping strategy names to result directories
    """
    logger.info(f"Running experiments with {n_simulations} simulated users")
    
    if output_dir is None:
        timestamp = int(time.time())
        output_dir = os.path.join(os.path.dirname(__file__), f'experiment_results_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store result directories
    result_dirs = {}
    
    # 1. Run simulation for Traditional approach (CONTROL)
    logger.info("Running Traditional (CONTROL) simulation")
    traditional = run_traditional_simulation(datasets, deprecation_records, n_accesses=n_simulations * 5)
    traditional_dir = os.path.join(output_dir, f"control_{int(time.time())}")
    os.makedirs(traditional_dir, exist_ok=True)
    traditional.save_evaluation_data(traditional_dir)
    result_dirs["CONTROL"] = traditional_dir
    
    # 2. Run simulation for Basic approach (BASIC)
    logger.info("Running Basic Framework (BASIC) simulation")
    basic = run_basic_simulation(datasets, deprecation_records, n_accesses=n_simulations * 5)
    basic_dir = os.path.join(output_dir, f"basic_{int(time.time())}")
    os.makedirs(basic_dir, exist_ok=True)
    basic.save_evaluation_data(basic_dir)
    result_dirs["BASIC"] = basic_dir
    
    # 3. Run simulation for Full Framework approach (FULL)
    logger.info("Running Full Framework (FULL) simulation")
    # Create a new dataset and record copy for the full simulation
    datasets_copy = {k: v for k, v in datasets.items()}
    records_copy = {k: v for k, v in deprecation_records.items()}
    
    framework = run_framework_simulation(
        DeprecationStrategy.FULL,
        datasets_copy,
        records_copy,
        n_users=n_simulations
    )
    
    full_dir = os.path.join(output_dir, f"full_{int(time.time())}")
    os.makedirs(full_dir, exist_ok=True)
    framework.save_evaluation_data(full_dir)
    result_dirs["FULL"] = full_dir
    
    # 4. Run experimental simulation from experimental_design.py
    logger.info("Running experimental design simulation")
    
    # Create fresh synthetic datasets and deprecation records
    exp_datasets = create_synthetic_datasets()
    exp_records = create_deprecation_records()
    
    # Run the experiment for all strategies
    experiment_results = run_experiment(
        exp_datasets, 
        exp_records,
        n_simulations=n_simulations,
        strategies=[DeprecationStrategy.CONTROL, DeprecationStrategy.BASIC, DeprecationStrategy.FULL]
    )
    
    # Save experiment results
    experiment_dir = os.path.join(output_dir, f"experiment_{int(time.time())}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    experiment_file = os.path.join(experiment_dir, "experiment_results.json")
    with open(experiment_file, 'w') as f:
        # Convert numpy values to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj
        
        json.dump(convert_numpy(experiment_results), f, indent=2)
    
    result_dirs["EXPERIMENT"] = experiment_dir
    
    logger.info(f"All experiments completed and results saved to {output_dir}")
    return result_dirs

def evaluate_results(result_dirs: Dict[str, str], output_dir: str = None) -> Tuple[str, Dict[str, str]]:
    """
    Evaluate the results of all experiments and generate visualizations and reports.
    
    Args:
        result_dirs: Dictionary mapping strategy names to result directories
        output_dir: Directory to save evaluation results
        
    Returns:
        Tuple of (report_path, figure_paths)
    """
    logger.info("Evaluating experiment results")
    
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create evaluation metrics object
    metrics = EvaluationMetrics(output_dir)
    
    # Load results for each strategy
    for strategy_name, result_dir in result_dirs.items():
        if strategy_name == "EXPERIMENT":
            # Load experiment results separately
            continue
        
        strategy = DeprecationStrategy[strategy_name]
        metrics.load_results(strategy, result_dir)
    
    # If experiment results exist, load them too
    if "EXPERIMENT" in result_dirs:
        experiment_file = os.path.join(result_dirs["EXPERIMENT"], "experiment_results.json")
        
        if os.path.exists(experiment_file):
            logger.info(f"Found experimental results at {experiment_file}")
            
            # Load experiment results directly
            with open(experiment_file, 'r') as f:
                experiment_results = json.load(f)
            
            # Save a copy of the experiment results in the output directory
            output_file = os.path.join(output_dir, "experiment_results.json")
            with open(output_file, 'w') as f:
                json.dump(experiment_results, f, indent=2)
    
    # Generate all figures
    figure_paths = metrics.generate_all_figures()
    logger.info(f"Generated {len(figure_paths)} figures")
    
    # Generate summary report
    report_path = metrics.generate_report(os.path.join(output_dir, "results.md"))
    logger.info(f"Generated evaluation report at {report_path}")
    
    return report_path, figure_paths

def main(args):
    """Main function to run the complete experiment workflow."""
    start_time = time.time()
    logger.info("Starting Contextual Dataset Deprecation Framework experiment")
    
    # Step 1: Setup the experiment environment
    datasets, deprecation_records = setup_experiment_environment()
    
    # Step 2: Run all experiments
    result_dirs = run_all_experiments(
        datasets, 
        deprecation_records,
        n_simulations=args.simulations,
        output_dir=args.output_dir
    )
    
    # Step 3: Evaluate results and generate reports
    report_path, figure_paths = evaluate_results(result_dirs, RESULTS_DIR)
    
    # Step 4: Log completion
    elapsed_time = time.time() - start_time
    logger.info(f"Experiment completed in {elapsed_time:.2f} seconds")
    logger.info(f"Results saved to {RESULTS_DIR}")
    logger.info(f"Evaluation report: {report_path}")
    
    return report_path, figure_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Contextual Dataset Deprecation Framework experiment")
    parser.add_argument("--simulations", type=int, default=50, help="Number of simulated users/groups")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save experiment results")
    args = parser.parse_args()
    
    main(args)
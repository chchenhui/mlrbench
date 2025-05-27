#!/usr/bin/env python
"""
Main script to run the adaptive code assistant experiment.
"""

import os
import sys
import time
import logging
import argparse
import shutil
from pathlib import Path
from typing import Tuple

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import experiment modules
from utils import ensure_dir
from simulation import ExperimentRunner

def setup_logging(log_file: str) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file
    """
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def prepare_output_directories(base_dir: str) -> Tuple[str, str]:
    """
    Prepare output directories for experiment results.
    
    Args:
        base_dir: Base directory for outputs
        
    Returns:
        Tuple of (results_dir, log_file)
    """
    # Create results directory inside claude_code
    results_dir = os.path.join(base_dir, "results")
    ensure_dir(results_dir)
    
    # Create log file
    log_file = os.path.join(base_dir, "log.txt")
    
    return results_dir, log_file

def move_results(results_dir: str, base_dir: str) -> None:
    """
    Move results to the final location.
    
    Args:
        results_dir: Source directory with results
        base_dir: Base directory for outputs
    """
    # Create final results directory at the parent level
    final_results_dir = os.path.join(os.path.dirname(base_dir), "results")
    ensure_dir(final_results_dir)
    
    # Copy results.md
    shutil.copy(
        os.path.join(results_dir, "results.md"),
        os.path.join(final_results_dir, "results.md")
    )
    
    # Copy log.txt
    shutil.copy(
        os.path.join(base_dir, "log.txt"),
        os.path.join(final_results_dir, "log.txt")
    )
    
    # Copy all PNG files (figures)
    for png_file in Path(results_dir).glob("*.png"):
        shutil.copy(
            png_file,
            os.path.join(final_results_dir, png_file.name)
        )
    
    print(f"Results moved to {final_results_dir}")

def main():
    """Main function to run the experiment."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run adaptive code assistant experiment")
    parser.add_argument("--developers", type=int, default=3, help="Number of developer profiles")
    parser.add_argument("--tasks", type=int, default=5, help="Number of tasks per model")
    parser.add_argument("--iterations", type=int, default=3, help="Maximum iterations per task")
    parser.add_argument("--small-models", action="store_true", default=True, help="Use small models for faster experimentation")
    
    args = parser.parse_args()
    
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir, log_file = prepare_output_directories(base_dir)
    
    # Set up logging
    setup_logging(log_file)
    
    # Log start time and configuration
    logger = logging.getLogger("experiment_runner")
    logger.info("="*80)
    logger.info("Starting adaptive code assistant experiment")
    logger.info(f"Configuration: {args}")
    logger.info(f"Results directory: {results_dir}")
    start_time = time.time()
    
    try:
        # Run experiment
        experiment_runner = ExperimentRunner(
            num_developers=args.developers,
            num_tasks=args.tasks,
            max_iterations=args.iterations,
            output_dir=results_dir,
            use_small_models=args.small_models
        )
        
        # Run experiment
        experiment_data = experiment_runner.run_experiment()
        
        # Evaluate results
        evaluation_results = experiment_runner.evaluate_results()
        
        # Move results to final location
        move_results(results_dir, base_dir)
        
        # Log completion
        end_time = time.time()
        logger.info(f"Experiment completed successfully in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logger.exception(f"Error during experiment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    from typing import Tuple
    main()
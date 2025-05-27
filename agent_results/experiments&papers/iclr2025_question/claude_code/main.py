"""
Main script to run the UAD experiments.
"""

import os
import sys
import logging
import argparse
import json
import shutil
from pathlib import Path
import time
from experiment import Experiment
from config import MODEL_CONFIGS, DATASET_CONFIGS, EXPERIMENT_CONFIGS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run UAD experiments")
    
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=MODEL_CONFIGS.keys(),
        help="Model configuration to use"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="squad",
        choices=DATASET_CONFIGS.keys(),
        help="Dataset to use"
    )
    
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=list(EXPERIMENT_CONFIGS.keys()),
        help="Experiments to run"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--organize_results",
        action="store_true",
        help="Organize results into a separate directory"
    )
    
    return parser.parse_args()

def main():
    """Run the experiments."""
    # Parse arguments
    args = parse_args()
    
    logger.info(f"Starting UAD experiments with model={args.model}, dataset={args.dataset}")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Filter experiment configurations
        filtered_configs = {name: config for name, config in EXPERIMENT_CONFIGS.items() if name in args.experiments}
        
        if not filtered_configs:
            logger.error(f"No valid experiment configurations found for {args.experiments}")
            return
        
        # Create experiment
        experiment = Experiment(
            config_name=args.model,
            dataset_name=args.dataset,
            experiment_configs=filtered_configs,
            results_dir=args.results_dir
        )
        
        # Run experiments
        logger.info("Running experiments")
        results = experiment.run_all_experiments()
        
        # Visualize results
        logger.info("Visualizing results")
        visualizations = experiment.visualize_results(results)
        
        # Save results
        logger.info("Saving results")
        experiment.save_results(results)
        
        # Generate report
        logger.info("Generating report")
        report_path = experiment.generate_markdown_report(results, visualizations)
        
        # Organize results if requested
        if args.organize_results:
            logger.info("Organizing results")
            organize_results("claude_exp2/iclr2025_question/results")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(f"Experiments completed in {execution_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during experiment: {e}", exc_info=True)
        return
    
def organize_results(target_dir):
    """
    Organize results into a separate directory.
    
    Args:
        target_dir: The target directory for organizing results.
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy results.md to the target directory
    shutil.copy("results/results.md", f"{target_dir}/results.md")
    
    # Copy log.txt to the target directory
    shutil.copy("log.txt", f"{target_dir}/log.txt")
    
    # Copy all figures to the target directory
    for file in os.listdir("results"):
        if file.endswith(".png"):
            shutil.copy(f"results/{file}", f"{target_dir}/{file}")
    
    logger.info(f"Results organized in {target_dir}")

if __name__ == "__main__":
    main()
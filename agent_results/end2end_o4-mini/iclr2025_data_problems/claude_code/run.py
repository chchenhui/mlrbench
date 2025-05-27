#!/usr/bin/env python
"""
Main runner script for Gradient-Informed Fingerprinting (GIF) experiments.

This script orchestrates the execution of different experiment types:
1. The simplified experiment (for quick testing)
2. The full-scale experiment (for comprehensive evaluation)

It also handles logging, timing, and resource monitoring.
"""

import os
import sys
import argparse
import logging
import time
import subprocess
import json
from datetime import datetime
import shutil

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/run.log")
    ]
)
logger = logging.getLogger(__name__)


def check_gpu_availability() -> bool:
    """Check if GPU is available for acceleration."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        logger.warning("PyTorch not installed, cannot check GPU availability")
        return False


def monitor_process(process) -> None:
    """Monitor a subprocess, capturing and logging its output."""
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            logger.info(output.strip())
    
    # Check return code
    return_code = process.poll()
    if return_code != 0:
        logger.error(f"Process exited with return code {return_code}")
        # Capture any error output
        error = process.stderr.read()
        if error:
            logger.error(f"Error output: {error}")


def ensure_results_directory(output_dir: str = "results") -> None:
    """Ensure results directory exists, creating it if needed."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")


def run_simplified_experiment(config_path: str, output_dir: str = "results") -> None:
    """Run the simplified GIF experiment."""
    logger.info("Running simplified experiment...")
    
    # Ensure output directory exists
    ensure_results_directory(output_dir)
    
    start_time = time.time()
    
    # Run the experiment
    cmd = [
        sys.executable, 
        "run_simplified_experiment.py", 
        "--config", config_path
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        monitor_process(process)
        
        # Check if results were generated
        if os.path.exists("results/results.md"):
            logger.info("Simplified experiment completed successfully")
        else:
            logger.error("Simplified experiment failed to generate results")
            return
        
        # Copy results to output directory if different
        if output_dir != "results":
            for file in os.listdir("results"):
                shutil.copy2(os.path.join("results", file), output_dir)
            logger.info(f"Copied results to {output_dir}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Simplified experiment completed in {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error running simplified experiment: {str(e)}")


def run_full_experiment(config_path: str, output_dir: str = "results") -> None:
    """Run the full-scale GIF experiment."""
    logger.info("Running full-scale experiment...")
    
    # Ensure output directory exists
    ensure_results_directory(output_dir)
    
    start_time = time.time()
    
    # Run the experiment
    cmd = [
        sys.executable, 
        "run_experiments.py", 
        "--config", config_path
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        monitor_process(process)
        
        # Check if results were generated
        if os.path.exists("results/results.md"):
            logger.info("Full-scale experiment completed successfully")
        else:
            logger.error("Full-scale experiment failed to generate results")
            return
        
        # Copy results to output directory if different
        if output_dir != "results":
            for file in os.listdir("results"):
                shutil.copy2(os.path.join("results", file), output_dir)
            logger.info(f"Copied results to {output_dir}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Full-scale experiment completed in {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error running full-scale experiment: {str(e)}")


def create_experiment_summary(output_dir: str = "results") -> None:
    """Create a summary of experiment results."""
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    
    # Check if results exist
    if not os.path.exists(os.path.join(output_dir, "results.md")):
        logger.error("Cannot create summary: No results found")
        return
    
    # Load metrics if they exist
    metrics_path = os.path.join(output_dir, "attribution_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}
    
    # Create summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_available": check_gpu_availability(),
        "metrics": metrics
    }
    
    # Write summary
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Created experiment summary: {summary_path}")


def move_results_to_destination(source_dir: str = "results", dest_dir: str = "../results") -> None:
    """Move results from source to destination directory."""
    # Ensure destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Check if source directory exists and has results
    if not os.path.exists(source_dir):
        logger.error(f"Source directory {source_dir} does not exist")
        return
    
    if not os.path.exists(os.path.join(source_dir, "results.md")):
        logger.error(f"No results found in {source_dir}")
        return
    
    # Create a timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(dest_dir, f"experiment_{timestamp}")
    os.makedirs(exp_dir)
    
    # Copy all files from source to destination
    for file in os.listdir(source_dir):
        source_path = os.path.join(source_dir, file)
        dest_path = os.path.join(exp_dir, file)
        
        if os.path.isfile(source_path):
            shutil.copy2(source_path, dest_path)
    
    logger.info(f"Moved results to {exp_dir}")
    
    # Create a symbolic link to the latest results
    latest_link = os.path.join(dest_dir, "latest")
    if os.path.exists(latest_link):
        os.remove(latest_link)
    
    try:
        os.symlink(exp_dir, latest_link)
        logger.info(f"Created symbolic link to latest results: {latest_link}")
    except OSError:
        logger.warning(f"Could not create symbolic link to latest results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GIF attribution experiments")
    parser.add_argument("--mode", type=str, default="simplified", choices=["simplified", "full"],
                        help="Experiment mode: simplified (quicker, synthetic data) or full (comprehensive)")
    parser.add_argument("--config", type=str, help="Path to configuration file (optional)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--move-results", action="store_true", help="Move results to a timestamped directory")
    parser.add_argument("--move-dir", type=str, default="../results", 
                        help="Destination directory for results (only used with --move-results)")
    
    args = parser.parse_args()
    
    # Display experiment info
    logger.info(f"Starting experiment in {args.mode} mode")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"GPU available: {check_gpu_availability()}")
    
    # Select default config path based on mode
    if args.config is None:
        if args.mode == "simplified":
            config_path = "config_simplified.json"
        else:
            config_path = "config.json"
    else:
        config_path = args.config
    
    logger.info(f"Using configuration file: {config_path}")
    
    # Run the appropriate experiment
    start_time = time.time()
    
    if args.mode == "simplified":
        run_simplified_experiment(config_path, args.output_dir)
    else:
        run_full_experiment(config_path, args.output_dir)
    
    # Create experiment summary
    create_experiment_summary(args.output_dir)
    
    # Move results if requested
    if args.move_results:
        move_results_to_destination(args.output_dir, args.move_dir)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Experiment completed in {elapsed_time:.2f} seconds")
#!/usr/bin/env python3
"""
Main script for running all ContractGPT experiments.

This script runs the entire experimental pipeline:
1. Generate benchmarks
2. Run all methods on all benchmarks
3. Generate visualizations and tables
4. Create results.md
"""

import os
import sys
import time
import json
import argparse
import logging
import shutil
from pathlib import Path
import subprocess

# Set up base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")


def setup_logging(log_file: str) -> logging.Logger:
    """
    Set up logging.
    
    Args:
        log_file: Path to log file.
        
    Returns:
        Configured logger.
    """
    # Create logger
    logger = logging.getLogger("ContractGPT")
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def run_command(command: list, logger: logging.Logger) -> int:
    """
    Run a command and log the output.
    
    Args:
        command: Command to run, as a list of strings.
        logger: Logger to use.
        
    Returns:
        Return code of the command.
    """
    logger.info(f"Running command: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Log output in real-time
    for line in process.stdout:
        logger.info(line.strip())
    
    return process.wait()


def main():
    """Main function for running the entire experimental pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run ContractGPT experiments')
    parser.add_argument('--target-language', type=str, default='python', help='Target programming language')
    parser.add_argument('--model-name', type=str, default='gpt-4o-mini', help='Name of the LLM to use')
    parser.add_argument('--max-iterations', type=int, default=5, help='Maximum number of iterations for synthesis')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature for LLM generation')
    parser.add_argument('--skip-benchmarks', action='store_true', help='Skip benchmark generation')
    parser.add_argument('--methods', type=str, nargs='+', default=['ContractGPT', 'LLMOnly', 'VeCoGenLike', 'LLM4CodeLike'], help='Methods to run')
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(BASE_DIR, "log.txt")
    logger = setup_logging(log_file)
    
    logger.info("Starting ContractGPT experimental pipeline")
    logger.info(f"Using model: {args.model_name}")
    logger.info(f"Target language: {args.target_language}")
    
    # Time the entire pipeline
    start_time = time.time()
    
    # Run experiments
    methods_str = " ".join(args.methods)
    experiment_cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "scripts", "run_experiments.py"),
        "--target-language", args.target_language,
        "--model-name", args.model_name,
        "--max-iterations", str(args.max_iterations),
        "--temperature", str(args.temperature),
        "--methods", *args.methods
    ]
    
    if not args.skip_benchmarks:
        experiment_cmd.append("--generate-benchmarks")
    
    return_code = run_command(experiment_cmd, logger)
    
    if return_code != 0:
        logger.error(f"Experiment failed with return code {return_code}")
        return return_code
    
    # Move results to the results directory
    source_results = os.path.join(BASE_DIR, "results")
    if os.path.exists(source_results):
        logger.info(f"Moving results from {source_results} to {RESULTS_DIR}")
        
        # Copy results.md
        results_md = os.path.join(source_results, "results.md")
        if os.path.exists(results_md):
            shutil.copy2(results_md, os.path.join(RESULTS_DIR, "results.md"))
        
        # Copy figures
        for file in os.listdir(source_results):
            if file.endswith(".png"):
                shutil.copy2(
                    os.path.join(source_results, file),
                    os.path.join(RESULTS_DIR, file)
                )
    
    # Copy log file to results directory
    if os.path.exists(log_file):
        shutil.copy2(log_file, os.path.join(RESULTS_DIR, "log.txt"))
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total experiment time: {total_time:.2f} seconds")
    
    logger.info(f"All experiments complete. Results saved to {RESULTS_DIR}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
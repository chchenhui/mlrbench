#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script to run a single experiment with the IETA framework.
This script demonstrates how to use the framework with a small dataset.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import run_experiments and run_all_experiments
from run_experiments import run_experiment, parse_arguments as parse_exp_arguments
from run_all_experiments import run_all_experiments, parse_arguments as parse_all_arguments

def main():
    """Main entry point for the demo."""
    print("Interactive Execution-Trace Alignment (IETA) Framework Demo")
    print("=" * 80)
    print("\nThis demo will run a simplified experiment to demonstrate the framework.")
    print("It will use synthetic data for demonstration purposes.")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run IETA demo")
    parser.add_argument("--method", type=str, default="all", 
                        choices=["all", "baseline", "dpo", "rlaif"], 
                        help="Method to run (default: all)")
    parser.add_argument("--iterations", type=int, default=3, 
                        help="Number of iterations (default: 3)")
    parser.add_argument("--samples", type=int, default=5, 
                        help="Number of dataset samples (default: 5)")
    parser.add_argument("--steps", type=int, default=100, 
                        help="Number of training steps (default: 100)")
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = Path("../results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nRunning with the following configuration:")
    print(f"  Method: {args.method}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Dataset samples: {args.samples}")
    print(f"  Training steps: {args.steps}")
    print(f"  Output directory: {output_dir}")
    print("\nThis will take a few minutes to complete...")
    
    if args.method == "all":
        # Run all experiments
        all_args = parse_all_arguments([
            "--dataset", "humaneval",
            "--num_samples", str(args.samples),
            "--model_type", "api",
            "--model_name", "claude-3-7-sonnet",
            "--num_iterations", str(args.iterations),
            "--batch_size", "4",
            "--training_steps", str(args.steps),
            "--output_dir", str(output_dir),
            "--use_synthetic"
        ])
        
        run_all_experiments(all_args)
    else:
        # Run a single experiment
        exp_args = parse_exp_arguments([
            "--method", args.method,
            "--dataset", "humaneval",
            "--num_samples", str(args.samples),
            "--model_type", "api",
            "--model_name", "claude-3-7-sonnet",
            "--num_iterations", str(args.iterations),
            "--batch_size", "4",
            "--training_steps", str(args.steps),
            "--output_dir", str(output_dir),
            "--use_synthetic"
        ])
        
        run_experiment(exp_args)
    
    print("\nDemo completed successfully!")
    print(f"Results are available in the '{output_dir}' directory.")
    print("\nTo view the results, check the following files:")
    print(f"  - {output_dir}/results.md: Comprehensive summary of results")
    print(f"  - {output_dir}/method_comparison.png: Comparison of different methods")
    print(f"  - {output_dir}/comparison_dashboard.png: Detailed comparison dashboard")

if __name__ == "__main__":
    main()
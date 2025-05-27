"""
Wrapper script to run the full experiment pipeline.
"""

import os
import argparse
import logging
import subprocess
import time
import torch
from utils import setup_logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run LASS experiment pipeline')
    
    # General settings
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_file', type=str, default='./log.txt', help='Log file path')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    # Experiment settings
    parser.add_argument('--dataset', type=str, default='waterbirds', choices=['waterbirds', 'celeba', 'civilcomments'], help='Dataset name')
    parser.add_argument('--llm_provider', type=str, default='anthropic', choices=['openai', 'anthropic'], help='LLM provider')
    parser.add_argument('--skip_baselines', action='store_true', help='Skip running baseline models')
    parser.add_argument('--fast_run', action='store_true', help='Run with reduced epochs for faster execution')
    
    return parser.parse_args()

def run_experiments(args):
    """Run the full experiment pipeline."""
    logger = logging.getLogger("LASS_Pipeline")
    
    # Start timing
    start_time = time.time()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up the experiment command
    cmd_args = [
        "python", "run_experiments.py",
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--dataset", args.dataset,
        "--seed", str(args.seed),
        "--log_file", args.log_file,
        "--llm_provider", args.llm_provider
    ]
    
    # Add GPU flag if requested
    if args.gpu and torch.cuda.is_available():
        cmd_args.extend(["--cuda"])
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")
    
    # Determine which experiment parts to run
    if args.skip_baselines:
        cmd_args.extend(["--run_lass"])
        logger.info("Skipping baseline models, running only LASS")
    else:
        cmd_args.extend(["--run_baselines", "--run_lass"])
        logger.info("Running both baseline models and LASS")
    
    # Adjust parameters for fast run
    if args.fast_run:
        cmd_args.extend([
            "--num_epochs", "5",
            "--patience", "2",
            "--batch_size", "16"
        ])
        logger.info("Running in fast mode with reduced epochs")
    else:
        cmd_args.extend([
            "--num_epochs", "30",
            "--patience", "5",
            "--batch_size", "32"
        ])
    
    # Log the command
    logger.info(f"Running command: {' '.join(cmd_args)}")
    
    # Run the experiment
    try:
        process = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Stream the output
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for completion
        exit_code = process.wait()
        
        if exit_code != 0:
            logger.error(f"Experiment failed with exit code: {exit_code}")
            return False
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        return False
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Experiment completed in: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    
    return True

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logger
    logger = setup_logger(args.log_file)
    logger.info(f"Starting LASS experiment pipeline with args: {args}")
    
    # Run the experiments
    success = run_experiments(args)
    
    if success:
        logger.info("All experiments completed successfully!")
    else:
        logger.error("Experiments failed. Check logs for details.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Main script to run the full experiment suite for the Adaptive Code Assistant.
This script sets up all necessary directories, downloads datasets, trains models,
runs evaluations, and generates visualizations and analysis.
"""

import os
import argparse
import logging
import time
import json
import subprocess
import sys
from pathlib import Path

# Make sure the current directory is in the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def setup_directories():
    """Setup necessary directories for the experiment."""
    dirs = ['data', 'logs', 'models', 'results']
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Create subdirectories
    os.makedirs('models/baseline', exist_ok=True)
    os.makedirs('models/adaptive', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)

def check_gpu():
    """Check if GPU is available and print info."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"GPU available: {device_count} devices")
            print(f"Primary device: {device_name}")
            return True
        else:
            print("No GPU available, using CPU")
            return False
    except Exception as e:
        print(f"Error checking GPU: {e}")
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Adaptive Code Assistant experiments')
    
    # Experiment configuration
    parser.add_argument('--experiment_mode', type=str, choices=['full', 'simplified', 'minimal'], default='full',
                        help='Experiment mode: full, simplified, or minimal')
    
    # Directories
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing the datasets')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate')
    parser.add_argument('--ppo_epochs', type=int, default=4,
                        help='Number of PPO epochs')
    
    # Device selection
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    
    # Experiment parameters
    parser.add_argument('--num_developers', type=int, default=30,
                        help='Number of simulated developers')
    parser.add_argument('--num_tasks', type=int, default=12,
                        help='Number of coding tasks per developer')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Execution control
    parser.add_argument('--download_only', action='store_true',
                        help='Only download datasets and models')
    parser.add_argument('--eval_only', action='store_true',
                        help='Run only evaluation, no training')
    parser.add_argument('--visualize_only', action='store_true',
                        help='Only generate visualizations from existing results')
    
    return parser.parse_args()

def run_experiment(args):
    """Run the appropriate experiment based on arguments."""
    # Start timing
    start_time = time.time()
    
    # Set up logging
    log_file = os.path.join(args.log_dir, 'experiment.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('adaptive_code_assistant')
    
    # Log experiment start
    logger.info("=" * 80)
    logger.info("STARTING ADAPTIVE CODE ASSISTANT EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Experiment mode: {args.experiment_mode}")
    logger.info(f"Arguments: {args}")
    
    # Check for GPU
    has_gpu = check_gpu()
    if args.gpu and not has_gpu:
        logger.warning("GPU requested but not available. Using CPU instead.")
    
    # Create experiment command based on mode
    if args.experiment_mode == 'minimal':
        logger.info("Running minimal experiment (faster, fewer samples)")
        cmd_args = [
            'python', 'run_simplified.py',
            '--synthetic_samples', '10',
            '--num_developers', '3',
            '--num_tasks', '2',
            '--epochs', '1',
            '--batch_size', '2',
            f'--output_dir={args.output_dir}',
            f'--log_dir={args.log_dir}',
            f'--data_dir={args.data_dir}',
            f'--seed={args.seed}'
        ]
    elif args.experiment_mode == 'simplified':
        logger.info("Running simplified experiment (moderate size)")
        cmd_args = [
            'python', 'run_simplified.py',
            '--synthetic_samples', '20',
            '--num_developers', '5',
            '--num_tasks', '3',
            '--epochs', '2',
            '--batch_size', '4',
            f'--output_dir={args.output_dir}',
            f'--log_dir={args.log_dir}',
            f'--data_dir={args.data_dir}',
            f'--seed={args.seed}'
        ]
    else:  # full
        logger.info("Running full experiment")
        cmd_args = [
            'python', 'run_experiments.py',
            f'--epochs={args.epochs}',
            f'--batch_size={args.batch_size}',
            f'--lr={args.lr}',
            f'--ppo_epochs={args.ppo_epochs}',
            f'--num_developers={args.num_developers}',
            f'--num_tasks={args.num_tasks}',
            f'--output_dir={args.output_dir}',
            f'--log_dir={args.log_dir}',
            f'--data_dir={args.data_dir}',
            f'--seed={args.seed}'
        ]
    
    # Add conditional arguments
    if args.gpu and has_gpu:
        cmd_args.append('--gpu')
    
    if args.download_only:
        logger.info("Download-only mode enabled, skipping experiment execution")
        return
    
    if args.eval_only:
        cmd_args.append('--eval_only')
        logger.info("Evaluation-only mode enabled, skipping training")
    
    if args.visualize_only:
        cmd_args.append('--visualize_only')
        logger.info("Visualization-only mode enabled, skipping training and evaluation")
    
    # Run the experiment
    logger.info(f"Executing: {' '.join(cmd_args)}")
    try:
        process = subprocess.Popen(
            cmd_args, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output to both console and log
        for line in iter(process.stdout.readline, ''):
            if line:
                logger.info(line.strip())
        
        for line in iter(process.stderr.readline, ''):
            if line:
                logger.error(line.strip())
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Experiment failed with code {process.returncode}")
            return False
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        return False
    
    # Calculate and log execution time
    end_time = time.time()
    duration = end_time - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info("=" * 80)
    logger.info(f"EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info(f"Total execution time: {int(hours):02d}h:{int(minutes):02d}m:{int(seconds):02d}s")
    logger.info("=" * 80)
    
    return True

def main():
    """Main entry point for the experiment."""
    # Parse arguments
    args = parse_args()
    
    # Setup directories
    setup_directories()
    
    # Run experiment
    success = run_experiment(args)
    
    # Copy results to parent result directory
    if success and not args.download_only:
        # Create parent results directory
        parent_results_dir = '../results'
        os.makedirs(parent_results_dir, exist_ok=True)
        
        try:
            # Copy log file
            subprocess.run(['cp', 'logs/experiment.log', f'{parent_results_dir}/log.txt'])
            
            # Copy visualization files
            subprocess.run(['cp', '-r', 'results/figures', parent_results_dir])
            
            # Copy results.md file or create one with experiment summary
            if os.path.exists('results/tables/summary_results.md'):
                with open(f'{parent_results_dir}/results.md', 'w') as out_file:
                    # Add header
                    out_file.write("# Adaptive Code Assistant Experiment Results\n\n")
                    
                    # Include summary from summary_results.md
                    with open('results/tables/summary_results.md', 'r') as in_file:
                        out_file.write(in_file.read())
                    
                    # Add figures
                    out_file.write("\n\n## Visualizations\n\n")
                    
                    # Add figures from the figures directory
                    figures = [f for f in os.listdir('results/figures') if f.endswith('.png')]
                    for fig in figures:
                        fig_path = f"figures/{fig}"
                        fig_name = fig.replace('_', ' ').replace('.png', '')
                        out_file.write(f"### {fig_name.title()}\n\n")
                        out_file.write(f"![{fig_name}]({fig_path})\n\n")
                    
                    # Add limitations and future work
                    out_file.write("## Limitations and Future Work\n\n")
                    out_file.write("### Limitations\n\n")
                    out_file.write("- The experiments were conducted with simulated developer feedback rather than real developer interactions.\n")
                    out_file.write("- The implementation uses a smaller variant of CodeT5+ to keep training time reasonable, which may limit performance.\n")
                    out_file.write("- Current reward function is a weighted combination of different signals, which may not optimally capture developer preferences.\n\n")
                    
                    out_file.write("### Future Work\n\n")
                    out_file.write("- Extend the experiment to real developer interactions in a controlled study.\n")
                    out_file.write("- Explore different reward functions and weighting strategies for implicit feedback signals.\n")
                    out_file.write("- Investigate the impact of different model scales on adaptation effectiveness.\n")
                    out_file.write("- Develop methods to more accurately capture and model individualized developer preferences.\n")
            
            print(f"Results successfully copied to {parent_results_dir}")
            
        except Exception as e:
            print(f"Error copying results: {e}")

if __name__ == "__main__":
    main()
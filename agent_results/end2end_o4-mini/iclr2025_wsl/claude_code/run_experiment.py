"""
Main runner script for weight graph embedding experiments.
"""
import os
import sys
import argparse
import subprocess
import logging


def setup_logging():
    """Set up logging to file and console."""
    os.makedirs('logs', exist_ok=True)
    log_file = os.path.join('logs', 'runner.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run weight graph embedding experiments')
    
    # General settings
    parser.add_argument('--experiment_name', type=str, default='default_experiment',
                        help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--small', action='store_true',
                        help='Run with smaller dataset and fewer epochs for testing')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training if available')
    
    return parser.parse_args()


def main(args):
    """Main function to run experiments."""
    logger = setup_logging()
    logger.info(f"Starting runner for experiment: {args.experiment_name}")
    
    # Build command for the training script
    cmd = [sys.executable, '-m', 'scripts.train']
    
    # Add experiment name
    cmd.extend(['--experiment_name', args.experiment_name])
    
    # Add seed
    cmd.extend(['--seed', str(args.seed)])
    
    # Set device if GPU requested
    if args.gpu:
        cmd.extend(['--device', 'cuda'])
    else:
        cmd.extend(['--device', 'cpu'])
    
    # Adjust parameters for small run if requested
    if args.small:
        logger.info("Running with reduced dataset and epochs (small mode)")
        cmd.extend([
            '--num_models', '200',
            '--num_epochs', '20',
            '--regressor_epochs', '10',
            '--decoder_epochs', '10'
        ])
    
    # Log the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Execute command
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            logger.info("Experiment completed successfully!")
        else:
            logger.error(f"Experiment failed with return code {process.returncode}")
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
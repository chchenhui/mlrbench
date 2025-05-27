"""
Simplified script to run MeLPA experiments.
"""

import os
import logging
import argparse
import subprocess
import datetime
import shutil

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run MeLPA experiments")
    
    # General settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="../results", help="Output directory")
    parser.add_argument("--log_file", type=str, default="experiment.log", help="Log file name")
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Base model name")
    parser.add_argument("--adapter_type", type=str, default="pfeiffer", choices=["pfeiffer", "lora"], help="Type of adapter to use")
    
    # Dataset settings
    parser.add_argument("--dataset_names", nargs="+", default=["glue/sst2", "imdb", "ag_news", "tweet_eval"], help="Datasets to use")
    
    # Experiment selection
    parser.add_argument("--run_meta_learning", action="store_true", help="Run meta-learning experiments")
    parser.add_argument("--run_baselines", action="store_true", help="Run baseline experiments")
    parser.add_argument("--run_melpa", action="store_true", help="Run MeLPA experiments")
    parser.add_argument("--run_analysis", action="store_true", help="Run analysis experiments")
    
    # Quick mode (faster experiments with reduced scale)
    parser.add_argument("--quick", action="store_true", help="Run in quick mode with reduced scale")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Default to running all experiments if none specified
    if not any([args.run_meta_learning, args.run_baselines, args.run_melpa, args.run_analysis]):
        args.run_meta_learning = True
        args.run_baselines = True
        args.run_melpa = True
        args.run_analysis = True
    
    return args


def setup_logging(log_dir, log_file="run.log"):
    """Set up logging configuration."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_file)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def run_command(cmd, logger):
    """Run a command and log its output."""
    logger.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        logger.info(line.strip())
    
    process.wait()
    return process.returncode


def main():
    """Main function to run MeLPA experiments."""
    # Parse arguments
    args = parse_args()
    
    # Get current timestamp for experiment ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"melpa_exp_{timestamp}"
    
    # Create output directories
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(".", "run_log.txt")
    
    logger.info("=" * 80)
    logger.info(f"Starting MeLPA experiment: {experiment_id}")
    logger.info("=" * 80)
    
    # Build command for run_experiments.py
    cmd = ["python", "run_experiments.py"]
    
    # Add arguments
    cmd.extend(["--seed", str(args.seed)])
    cmd.extend(["--output_dir", output_dir])
    cmd.extend(["--log_file", args.log_file])
    cmd.extend(["--model_name", args.model_name])
    cmd.extend(["--adapter_type", args.adapter_type])
    cmd.extend(["--dataset_names"] + args.dataset_names)
    
    # Add experiment selection flags
    if args.run_meta_learning:
        cmd.append("--run_meta_learning")
    if args.run_baselines:
        cmd.append("--run_baselines")
    if args.run_melpa:
        cmd.append("--run_melpa")
    if args.run_analysis:
        cmd.append("--run_analysis")
    
    # Add quick mode settings if requested
    if args.quick:
        cmd.extend([
            "--n_meta_epochs", "5",
            "--n_meta_train_tasks", "20",
            "--n_meta_val_tasks", "5",
            "--n_tasks", "3",
            "--n_examples_per_task", "50",
            "--n_epochs_per_task", "3"
        ])
    
    # Run the experiment
    start_time = datetime.datetime.now()
    logger.info(f"Experiment started at: {start_time}")
    
    try:
        returncode = run_command(cmd, logger)
        if returncode != 0:
            logger.error(f"Experiment failed with return code: {returncode}")
        else:
            logger.info("Experiment completed successfully")
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
    
    end_time = datetime.datetime.now()
    logger.info(f"Experiment ended at: {end_time}")
    logger.info(f"Total duration: {end_time - start_time}")
    
    # Copy log file to results directory
    try:
        shutil.copy("run_log.txt", os.path.join(output_dir, "run_log.txt"))
    except Exception as e:
        logger.error(f"Error copying log file: {e}")
    
    logger.info("=" * 80)
    logger.info(f"MeLPA experiment {experiment_id} completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
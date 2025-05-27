#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import subprocess
import time
import json
import shutil
from pathlib import Path
import torch

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
    parser = argparse.ArgumentParser(description="Run KV cache compression experiments.")
    
    # General settings
    parser.add_argument("--model_name_or_path", type=str, default="gpt2",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="Dataset name for evaluation")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-v1",
                        help="Dataset configuration")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--sample_size", type=int, default=10,
                        help="Number of examples to use for evaluation")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    
    # Experiment selection
    parser.add_argument("--run_training", action="store_true",
                        help="Run training experiment")
    parser.add_argument("--run_evaluation", action="store_true",
                        help="Run evaluation experiment")
    parser.add_argument("--run_ablations", action="store_true",
                        help="Run ablation studies")
    parser.add_argument("--run_summarization", action="store_true",
                        help="Run summarization evaluation")
    
    # Compression settings
    parser.add_argument("--max_cache_size", type=int, default=1024,
                        help="Maximum number of KV pairs to retain after pruning.")
    parser.add_argument("--num_clusters", type=int, default=256,
                        help="Number of cluster centroids for low-rank summarization.")
    parser.add_argument("--pruning_interval", type=int, default=512,
                        help="Interval (in tokens) between pruning operations.")
    parser.add_argument("--lookback_window", type=int, default=256,
                        help="Number of recent positions to consider for importance.")
    
    args = parser.parse_args()
    return args

def setup_environment():
    """Set up directories and environment for experiments."""
    logger.info("Setting up environment...")
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Check for GPU
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.warning("No GPU available, running on CPU")
    
    # Log system information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Python version: {subprocess.check_output('python --version', shell=True).decode().strip()}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    return os.getcwd()

def run_training_experiment(args):
    """
    Run training experiment with KV cache compression.
    
    Args:
        args: Command line arguments
    """
    logger.info("Running training experiment...")
    
    cmd = [
        "python", "train.py",
        "--model_name_or_path", args.model_name_or_path,
        "--dataset_name", f"{args.dataset_name}/{args.dataset_config}",
        "--output_dir", "results/training",
        "--max_length", str(args.max_length),
        "--train_batch_size", "4",
        "--eval_batch_size", "4",
        "--learning_rate", "5e-5",
        "--num_train_epochs", "3",
        "--warmup_steps", "500",
        "--evaluate_during_training",
        "--evaluation_strategy", "epoch",
        "--max_cache_size", str(args.max_cache_size),
        "--num_clusters", str(args.num_clusters),
        "--pruning_interval", str(args.pruning_interval),
        "--lookback_window", str(args.lookback_window),
        "--use_compression",
        "--seed", "42",
        "--sample_size", str(args.sample_size),
    ]
    
    if args.fp16:
        cmd.append("--fp16")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        universal_newlines=True
    )
    
    # Stream output
    with open("logs/training.log", "w") as log_file:
        for line in process.stdout:
            log_file.write(line)
            print(line, end="")
    
    process.wait()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time:.2f} seconds")
    
    if process.returncode != 0:
        logger.error("Training failed")
        return False
    
    logger.info("Training completed successfully")
    return True

def run_evaluation_experiment(args):
    """
    Run evaluation experiment on various compression methods.
    
    Args:
        args: Command line arguments
    """
    logger.info("Running evaluation experiment...")
    
    cmd = [
        "python", "evaluate.py",
        "--model_name_or_path", args.model_name_or_path,
        "--dataset_name", f"{args.dataset_name}/{args.dataset_config}",
        "--output_dir", "results/evaluation",
        "--max_length", str(args.max_length),
        "--batch_size", "1",
        "--seed", "42",
        "--max_cache_size", str(args.max_cache_size),
        "--num_clusters", str(args.num_clusters),
        "--pruning_interval", str(args.pruning_interval),
        "--lookback_window", str(args.lookback_window),
        "--sample_size", str(args.sample_size),
    ]
    
    # Add methods to evaluate
    cmd.extend(["--methods", "full", "ours", "zack", "dynamic_kv", "razor", "uncomp"])
    
    # Add sequence lengths for scaling analysis
    cmd.extend(["--sequence_lengths", "512", "1024", "2048", "4096", "8192"])
    
    if args.run_ablations:
        cmd.append("--run_ablations")
    
    if args.run_summarization:
        cmd.append("--summarization")
        cmd.extend(["--summarization_dataset", "cnn_dailymail"])
    
    if args.fp16:
        cmd.append("--fp16")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        universal_newlines=True
    )
    
    # Stream output
    with open("logs/evaluation.log", "w") as log_file:
        for line in process.stdout:
            log_file.write(line)
            print(line, end="")
    
    process.wait()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
    
    if process.returncode != 0:
        logger.error("Evaluation failed")
        return False
    
    logger.info("Evaluation completed successfully")
    return True

def organize_results(args):
    """
    Organize results into the final output directory structure.
    
    Args:
        args: Command line arguments
    """
    logger.info("Organizing results...")
    
    # Create results directory
    results_dir = os.path.join(args.output_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Move relevant files
    files_to_move = [
        "results/evaluation/results.md",
        "log.txt",
        "results/evaluation/visualizations/perplexity_comparison.png",
        "results/evaluation/visualizations/throughput_comparison.png",
        "results/evaluation/visualizations/memory_usage_comparison.png",
        "results/evaluation/visualizations/compression_ratio_comparison.png",
        "results/evaluation/visualizations/latency_vs_sequence_length.png",
        "results/evaluation/visualizations/tradeoff_analysis.png",
    ]
    
    for file_path in files_to_move:
        if os.path.exists(file_path):
            dest_path = os.path.join(results_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)
            logger.info(f"Copied {file_path} to {dest_path}")
    
    # Copy evaluation results.json
    if os.path.exists("results/evaluation/evaluation_results.json"):
        # Read the results
        with open("results/evaluation/evaluation_results.json", "r") as f:
            results = json.load(f)
        
        # Save a cleaned version with just the essential metrics
        essential_metrics = {}
        for method, metrics in results.items():
            if method != "ablations":
                essential_metrics[method] = {
                    "perplexity": metrics.get("perplexity", 0),
                    "tokens_per_second": metrics.get("tokens_per_second", 0),
                    "peak_memory_gb": metrics.get("peak_memory_gb", 0),
                    "average_compression_ratio": metrics.get("average_compression_ratio", 1.0),
                }
        
        # Save essential metrics
        with open(os.path.join(results_dir, "metrics_summary.json"), "w") as f:
            json.dump(essential_metrics, f, indent=2)
    
    logger.info(f"Results organized in {results_dir}")
    return results_dir

def main():
    """Main function to run all experiments."""
    args = parse_args()
    
    # Setup environment
    working_dir = setup_environment()
    
    # Log experiment configuration
    logger.info("Experiment configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save experiment configuration
    with open(os.path.join(args.output_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Run experiments
    success = True
    
    if args.run_training:
        success = success and run_training_experiment(args)
    
    if args.run_evaluation:
        success = success and run_evaluation_experiment(args)
    
    if success:
        logger.info("All experiments completed successfully!")
        
        # Organize results
        results_dir = organize_results(args)
        
        logger.info(f"Results saved to {results_dir}")
    else:
        logger.error("Some experiments failed. Check logs for details.")
    
    return success

if __name__ == "__main__":
    success = main()
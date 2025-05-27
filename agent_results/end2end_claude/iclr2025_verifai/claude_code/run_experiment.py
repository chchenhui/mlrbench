"""
Main experiment runner for the VERIL framework.
"""

import os
import sys
import argparse
import random
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import shutil

import torch
import numpy as np
from tqdm import tqdm

from config import (
    MODELS, 
    TRAINING, 
    VERIFICATION_TOOLS, 
    DATASET_NAME, 
    DATASET_SIZE,
    SAMPLE_GENERATIONS,
    EXPERIMENT,
    ROOT_DIR,
    RESULTS_DIR,
    CODE_DIR,
    LOG_FILE,
)
from data import load_dataset_by_name
from model import create_model, RecursiveImprovementLearning
from verification import VerificationIntegrationLayer
from evaluation import Evaluator, EvaluationResult
from utils import logger, time_function, save_json, setup_gpu


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run VERIL experiment")
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=DATASET_NAME,
        help=f"Dataset to use (default: {DATASET_NAME})"
    )
    
    parser.add_argument(
        "--dataset_size", 
        type=int, 
        default=DATASET_SIZE,
        help=f"Number of examples to use (default: {DATASET_SIZE})"
    )
    
    parser.add_argument(
        "--run_baseline", 
        action="store_true",
        default=EXPERIMENT["run_baseline"],
        help="Run baseline model"
    )
    
    parser.add_argument(
        "--run_veril_static", 
        action="store_true",
        default=EXPERIMENT["run_veril_static"],
        help="Run VERIL model with static verification"
    )
    
    parser.add_argument(
        "--run_veril_dynamic", 
        action="store_true",
        default=EXPERIMENT["run_veril_dynamic"],
        help="Run VERIL model with dynamic verification"
    )
    
    parser.add_argument(
        "--run_veril_full", 
        action="store_true",
        default=EXPERIMENT["run_veril_full"],
        help="Run VERIL model with full verification"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=EXPERIMENT["seed"],
        help=f"Random seed (default: {EXPERIMENT['seed']})"
    )
    
    parser.add_argument(
        "--num_trials", 
        type=int, 
        default=EXPERIMENT["num_trials"],
        help=f"Number of trials (default: {EXPERIMENT['num_trials']})"
    )
    
    parser.add_argument(
        "--gpu", 
        action="store_true",
        default=True,
        help="Use GPU if available"
    )
    
    return parser.parse_args()


@time_function
def run_experiment(args):
    """
    Run the VERIL experiment.
    
    Args:
        args: Command line arguments
    """
    # Set random seed
    set_seed(args.seed)
    
    # Setup GPU if available
    device = setup_gpu(use_gpu=args.gpu, gpu_ids=[0])
    
    # Log experiment configuration
    logger.info(f"Running experiment with configuration:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Dataset size: {args.dataset_size}")
    logger.info(f"  Run baseline: {args.run_baseline}")
    logger.info(f"  Run VERIL static: {args.run_veril_static}")
    logger.info(f"  Run VERIL dynamic: {args.run_veril_dynamic}")
    logger.info(f"  Run VERIL full: {args.run_veril_full}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Number of trials: {args.num_trials}")
    logger.info(f"  Device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset {args.dataset}...")
    problems = load_dataset_by_name(args.dataset, max_problems=args.dataset_size)
    logger.info(f"Loaded {len(problems)} problems")
    
    # Split dataset
    train_size = int(len(problems) * 0.8)
    train_problems = problems[:train_size]
    test_problems = problems[train_size:]
    logger.info(f"Split dataset into {len(train_problems)} training problems and {len(test_problems)} test problems")
    
    # Initialize evaluator
    evaluator = Evaluator(verification_types=["static", "dynamic"])
    
    # Initialize results dict
    evaluation_results = {}
    
    # Run baseline model
    if args.run_baseline:
        logger.info("Running baseline model...")
        
        baseline_model = create_model(MODELS["baseline"], device=device)
        baseline_result = evaluator.evaluate_model(baseline_model, test_problems, n_samples=1)
        evaluation_results["baseline"] = baseline_result
        
        # Save baseline results
        save_json(baseline_result.to_dict(), RESULTS_DIR / "baseline_results.json")
    
    # Run VERIL with static verification
    if args.run_veril_static:
        logger.info("Running VERIL model with static verification...")
        
        veril_static_model = create_model(MODELS["veril_static"], device=device)
        ril = RecursiveImprovementLearning(
            veril_static_model,
            verification_types=["static"],
            num_iterations=TRAINING["num_iterations"],
        )
        
        # Train the model
        learning_metrics = ril.train(train_problems)
        
        # Evaluate the model
        veril_static_result = evaluator.evaluate_model(veril_static_model, test_problems, n_samples=1)
        veril_static_result.learning_metrics = learning_metrics
        
        evaluation_results["veril_static"] = veril_static_result
        
        # Save VERIL static results
        save_json(veril_static_result.to_dict(), RESULTS_DIR / "veril_static_results.json")
    
    # Run VERIL with dynamic verification
    if args.run_veril_dynamic:
        logger.info("Running VERIL model with dynamic verification...")
        
        veril_dynamic_model = create_model(MODELS["veril_dynamic"], device=device)
        ril = RecursiveImprovementLearning(
            veril_dynamic_model,
            verification_types=["dynamic"],
            num_iterations=TRAINING["num_iterations"],
        )
        
        # Train the model
        learning_metrics = ril.train(train_problems)
        
        # Evaluate the model
        veril_dynamic_result = evaluator.evaluate_model(veril_dynamic_model, test_problems, n_samples=1)
        veril_dynamic_result.learning_metrics = learning_metrics
        
        evaluation_results["veril_dynamic"] = veril_dynamic_result
        
        # Save VERIL dynamic results
        save_json(veril_dynamic_result.to_dict(), RESULTS_DIR / "veril_dynamic_results.json")
    
    # Run VERIL with full verification
    if args.run_veril_full:
        logger.info("Running VERIL model with full verification...")
        
        veril_full_model = create_model(MODELS["veril_full"], device=device)
        ril = RecursiveImprovementLearning(
            veril_full_model,
            verification_types=["static", "dynamic"],
            num_iterations=TRAINING["num_iterations"],
        )
        
        # Train the model
        learning_metrics = ril.train(train_problems)
        
        # Evaluate the model
        veril_full_result = evaluator.evaluate_model(veril_full_model, test_problems, n_samples=1)
        veril_full_result.learning_metrics = learning_metrics
        
        evaluation_results["veril_full"] = veril_full_result
        
        # Save VERIL full results
        save_json(veril_full_result.to_dict(), RESULTS_DIR / "veril_full_results.json")
    
    # Generate evaluation report
    logger.info("Generating evaluation report...")
    report = evaluator.generate_report(
        evaluation_results,
        output_path=RESULTS_DIR / "results.md",
    )
    
    # Save all results
    save_json(
        {name: result.to_dict() for name, result in evaluation_results.items()},
        RESULTS_DIR / "all_results.json",
    )
    
    logger.info(f"Experiment completed. Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Run the experiment
        run_experiment(args)
    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
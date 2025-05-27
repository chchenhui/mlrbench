#!/usr/bin/env python3
# Main script to run the experiments

import os
import argparse
import time
import logging
import torch
import json
from pathlib import Path

from utils.logger import setup_logger
from utils.data_utils import load_dataset, prepare_datasets
from models.baseline import StaticCodeT5Plus
from models.adaptive import AdaptiveCodeAssistant
from utils.evaluation import evaluate_models
from utils.visualization import plot_results, create_tables

# Setup argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Run adaptive code assistant experiments')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing the datasets')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate')
    parser.add_argument('--ppo_epochs', type=int, default=4,
                        help='Number of PPO epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--num_developers', type=int, default=30,
                        help='Number of simulated developers')
    parser.add_argument('--num_tasks', type=int, default=12,
                        help='Number of coding tasks per developer')
    parser.add_argument('--eval_only', action='store_true',
                        help='Run only evaluation, no training')
    parser.add_argument('--visualize_only', action='store_true',
                        help='Only generate visualizations from existing results')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.log_dir, 'experiments.log')
    logger = setup_logger('adaptive_code_assistant', log_file)
    logger.info(f"Starting experiments with args: {args}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.gpu:
        torch.cuda.manual_seed_all(args.seed)
        logger.info("Using GPU for training")
        device = torch.device('cuda')
    else:
        logger.info("Using CPU for training")
        device = torch.device('cpu')
    
    # If only visualization is requested, skip training and evaluation
    if args.visualize_only:
        results_file = os.path.join(args.output_dir, 'experiment_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            plot_results(results, args.output_dir)
            create_tables(results, args.output_dir)
            logger.info("Visualizations generated successfully")
            return
        else:
            logger.error(f"Results file {results_file} not found. Cannot generate visualizations.")
            return
    
    # Load and prepare datasets
    logger.info("Loading datasets...")
    raw_data = load_dataset(args.data_dir)
    train_data, valid_data, test_data = prepare_datasets(raw_data, args.seed)
    logger.info(f"Datasets loaded. Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    
    # Initialize models
    logger.info("Initializing models...")
    baseline_model = StaticCodeT5Plus(device=device)
    adaptive_model = AdaptiveCodeAssistant(
        device=device, 
        learning_rate=args.lr,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size
    )
    
    # Skip training if eval_only is specified
    if not args.eval_only:
        # Train the baseline model
        logger.info("Training baseline model...")
        baseline_model.train(train_data, valid_data, args.epochs, args.batch_size)
        
        # Train the adaptive model
        logger.info("Training adaptive model...")
        adaptive_model.train(train_data, valid_data, args.epochs, args.batch_size)
    
    # Evaluate models
    logger.info("Evaluating models...")
    results = evaluate_models(
        baseline_model=baseline_model,
        adaptive_model=adaptive_model,
        test_data=test_data,
        num_developers=args.num_developers,
        num_tasks=args.num_tasks,
        device=device
    )
    
    # Save results
    results_file = os.path.join(args.output_dir, 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations and tables
    logger.info("Generating visualizations and tables...")
    plot_results(results, args.output_dir)
    create_tables(results, args.output_dir)
    
    logger.info("Experiments completed successfully")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

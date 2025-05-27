#!/usr/bin/env python3
"""
Simplified experiment runner with reduced parameters for faster testing.
"""

import os
import argparse
import time
import logging
import torch
import json
from pathlib import Path

from utils.logger import setup_logger
from utils.data_utils import load_dataset, prepare_datasets, _generate_synthetic_dataset
from models.baseline import StaticCodeT5Plus
from models.adaptive import AdaptiveCodeAssistant
from utils.evaluation import evaluate_models
from utils.visualization import plot_results, create_tables

def parse_args():
    parser = argparse.ArgumentParser(description='Run simplified adaptive code assistant experiments')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing the datasets')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--num_developers', type=int, default=5,
                        help='Number of simulated developers')
    parser.add_argument('--num_tasks', type=int, default=3,
                        help='Number of coding tasks per developer')
    parser.add_argument('--synthetic_samples', type=int, default=20,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.log_dir, 'simplified_experiment.log')
    logger = setup_logger('adaptive_code_assistant', log_file)
    logger.info(f"Starting simplified experiment with args: {args}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.gpu:
        torch.cuda.manual_seed_all(args.seed)
        logger.info("Using GPU for training")
        device = torch.device('cuda')
    else:
        logger.info("Using CPU for training")
        device = torch.device('cpu')
    
    # Generate synthetic dataset for testing
    logger.info(f"Generating synthetic dataset with {args.synthetic_samples} samples")
    synthetic_data = _generate_synthetic_dataset(args.synthetic_samples)

    # Add needed fields for HuggingFace Dataset format
    from datasets import Dataset
    dataset_dict = {
        'task_id': [f"task_{i}" for i in range(len(synthetic_data))],
        'context': [item['context'] for item in synthetic_data],
        'solution': [item['solution'] for item in synthetic_data],
        'description': [item['description'] for item in synthetic_data],
        'tags': [item['tags'] for item in synthetic_data]
    }
    synthetic_dataset = Dataset.from_dict(dataset_dict)

    # Split dataset
    train_data, valid_data, test_data = prepare_datasets(synthetic_dataset, args.seed)
    logger.info(f"Datasets created. Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    
    # Initialize models with simpler configurations
    logger.info("Initializing simplified models...")
    
    # Use smaller models for faster testing
    baseline_model = StaticCodeT5Plus(
        model_name="Salesforce/codet5p-220m-py",  # Smaller CodeT5+ model
        device=device,
        max_length=128,  # Reduced length for faster processing
        temperature=0.7
    )
    
    adaptive_model = AdaptiveCodeAssistant(
        model_name="Salesforce/codet5p-220m-py",  # Smaller CodeT5+ model
        device=device,
        max_length=128,  # Reduced length for faster processing
        embedding_dim=16,  # Smaller embedding dimension
        learning_rate=5e-5,
        ppo_epochs=2,  # Fewer PPO epochs
        batch_size=args.batch_size
    )
    
    # Train models with reduced epochs
    logger.info("Training baseline model...")
    baseline_model.train(
        train_data=train_data,
        valid_data=valid_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=f"{args.output_dir}/baseline"
    )
    
    logger.info("Training adaptive model...")
    adaptive_model.train(
        train_data=train_data,
        valid_data=valid_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=f"{args.output_dir}/adaptive"
    )
    
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
    
    logger.info("Simplified experiment completed successfully")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
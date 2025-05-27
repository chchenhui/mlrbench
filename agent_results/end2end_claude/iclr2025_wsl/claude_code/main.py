#!/usr/bin/env python3
"""
Neural Weight Archeology: Decoding Model Behaviors from Weight Patterns
Main experiment runner
"""

import os
import sys
import json
import time
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
from data.data_generator import ModelZooDataset
from models.baseline_models import WeightStatisticsModel, PCAPredictionModel
from models.nwpa import NWPA
from utils.evaluation import evaluate_model, compute_metrics
from utils.visualization import (
    plot_training_curves, 
    plot_comparison_bar_chart, 
    plot_property_correlation, 
    visualize_weight_patterns
)

# Set up logging
def setup_logging(log_path):
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Neural Weight Archeology Experiments')
    
    # General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--log_dir', type=str, default='../results', help='Directory to save logs')
    parser.add_argument('--output_dir', type=str, default='../results', help='Directory to save outputs')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for dataset')
    
    # Dataset settings
    parser.add_argument('--num_models', type=int, default=100, help='Number of models to generate')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.15, help='Test split ratio')
    
    # Training settings
    parser.add_argument('--model_type', type=str, default='nwpa', choices=['nwpa', 'statistics', 'pca'], 
                        help='Model type to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use')
    
    # Model-specific settings
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use_attention', action='store_true', help='Use attention in GNN')
    
    # Evaluation settings
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation on existing model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model for evaluation')
    
    return parser.parse_args()

def create_model(args, num_features, num_classes=None, num_regression_targets=None):
    """Create model based on arguments"""
    if args.model_type == 'nwpa':
        model = NWPA(
            input_dim=num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=num_classes,
            num_regression_targets=num_regression_targets,
            dropout=args.dropout,
            use_attention=args.use_attention
        )
    elif args.model_type == 'statistics':
        model = WeightStatisticsModel(
            input_dim=num_features,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_regression_targets=num_regression_targets,
            dropout=args.dropout
        )
    elif args.model_type == 'pca':
        model = PCAPredictionModel(
            input_dim=num_features,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_regression_targets=num_regression_targets,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    return model.to(args.device)

def train(args, model, train_loader, val_loader, optimizer, criterion, logger):
    """Train the model"""
    logger.info("Starting training...")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            # Move batch to device
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            
            # Compute loss
            loss = criterion(outputs, batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Compute average loss over epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = model(batch)
                
                # Compute loss
                loss = criterion(outputs, batch)
                val_loss += loss.item()
        
        # Compute average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}")
        
        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if args.save_model:
                model_save_path = os.path.join(args.output_dir, f"best_{args.model_type}_model.pt")
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"Model saved to {model_save_path}")
    
    return train_losses, val_losses

def run_experiment(args, logger):
    """Run the experiment based on arguments"""
    logger.info("Initializing experiment...")
    logger.info(f"Using device: {args.device}")
    
    # Use existing dataset if provided
    if hasattr(args, 'dataset_info') and args.dataset_info is not None:
        logger.info("Using provided dataset...")
        dataset = args.dataset_info['dataset']
        num_features = args.dataset_info['num_features']
        num_classes = args.dataset_info['num_classes']
        num_regression_targets = args.dataset_info['num_regression_targets']
        train_size = args.dataset_info['train_size']
        val_size = args.dataset_info['val_size']
        test_size = args.dataset_info['test_size']
    else:
        # Create dataset
        logger.info(f"Creating dataset with {args.num_models} models...")
        dataset = ModelZooDataset(
            num_models=args.num_models, 
            data_dir=args.data_dir,
            create_if_not_exists=True
        )
        
        # Get dataset properties
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        num_regression_targets = dataset.num_regression_targets
        
        logger.info(f"Dataset created. Features: {num_features}, "
                  f"Classes: {num_classes}, Regression targets: {num_regression_targets}")
        
        # Split dataset
        train_size = int(len(dataset) * (1 - args.val_split - args.test_split))
        val_size = int(len(dataset) * args.val_split)
        test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    logger.info(f"Data split: Train: {len(train_dataset)}, "
               f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    model = create_model(
        args, 
        num_features=num_features, 
        num_classes=num_classes,
        num_regression_targets=num_regression_targets
    )
    
    # Create optimizer and loss criterion
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Use model's loss criterion
    criterion = model.loss_function
    
    # Train or load model
    if args.eval_only and args.model_path:
        logger.info(f"Loading model from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path))
        train_losses, val_losses = [], []
    else:
        logger.info("Starting training...")
        train_losses, val_losses = train(
            args, model, train_loader, val_loader, optimizer, criterion, logger
        )
        
        # Plot training curves
        plot_training_curves(
            train_losses, 
            val_losses, 
            title=f"{args.model_type.upper()} Training Curves",
            save_path=os.path.join(args.output_dir, f"{args.model_type}_training_curves.png")
        )
    
    # Evaluate model
    logger.info("Evaluating model on test set...")
    test_results = evaluate_model(model, test_loader, args.device)
    
    # Save results
    # Convert args to a serializable dict and remove non-serializable objects
    args_dict = vars(args).copy()
    if 'dataset_info' in args_dict:
        del args_dict['dataset_info']  # Remove non-serializable dataset
    
    results = {
        "model_type": args.model_type,
        "test_metrics": test_results,
        "args": args_dict
    }
    
    with open(os.path.join(args.output_dir, f"{args.model_type}_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {os.path.join(args.output_dir, f'{args.model_type}_results.json')}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    visualize_results(args, model, test_loader, test_results)
    
    return results

def visualize_results(args, model, test_loader, test_results):
    """Generate and save visualizations of the results"""
    
    # Generate property correlation plot if we have regression targets
    plot_property_correlation(
        test_results,
        title=f"{args.model_type.upper()} Property Correlations",
        save_path=os.path.join(args.output_dir, f"{args.model_type}_property_correlations.png")
    )
    
    # Generate weight pattern visualizations for NWPA model
    if args.model_type == 'nwpa':
        visualize_weight_patterns(
            model, 
            test_loader, 
            args.device,
            save_path=os.path.join(args.output_dir, f"{args.model_type}_weight_patterns.png")
        )

def run_all_models(args, logger):
    """Run experiments for all model types"""
    all_results = {}
    
    # Save original model type
    original_model_type = args.model_type
    
    # Create dataset once to be reused
    logger.info("Creating dataset to be shared across all models...")
    data_dir = args.data_dir
    dataset_common = ModelZooDataset(
        num_models=args.num_models, 
        data_dir=data_dir,
        create_if_not_exists=True
    )
    
    # Get dataset properties
    num_features = dataset_common.num_features
    num_classes = dataset_common.num_classes
    num_regression_targets = dataset_common.num_regression_targets
    
    logger.info(f"Dataset created. Features: {num_features}, "
               f"Classes: {num_classes}, Regression targets: {num_regression_targets}")
    
    # Split dataset
    train_size = int(len(dataset_common) * (1 - args.val_split - args.test_split))
    val_size = int(len(dataset_common) * args.val_split)
    test_size = len(dataset_common) - train_size - val_size
    
    # Run each model type
    for model_type in ['statistics', 'pca', 'nwpa']:
        logger.info(f"\n{'='*50}\nRunning experiment with {model_type} model\n{'='*50}")
        args.model_type = model_type
        
        # We'll pass the dataset information to avoid recreating it
        args.dataset_info = {
            'dataset': dataset_common,
            'num_features': num_features,
            'num_classes': num_classes,
            'num_regression_targets': num_regression_targets,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size
        }
        
        results = run_experiment(args, logger)
        all_results[model_type] = results
        
        # Remove the dataset info to avoid memory issues
        args.dataset_info = None
    
    # Restore original model type
    args.model_type = original_model_type
    
    # Generate comparison plots
    logger.info("Generating comparison plots...")
    compare_models(all_results, args.output_dir)
    
    return all_results

def compare_models(results, output_dir):
    """Compare the performance of different models"""
    model_names = list(results.keys())
    accuracies = []
    f1_scores = []
    r2_scores = []
    
    for model_name in model_names:
        model_results = results[model_name]["test_metrics"]
        
        # Extract classification metrics
        if "classification" in model_results and "average" in model_results["classification"]:
            accuracies.append(model_results["classification"]["average"]["accuracy"])
            f1_scores.append(model_results["classification"]["average"]["f1_score"])
        else:
            accuracies.append(0)
            f1_scores.append(0)
        
        # Extract regression metrics
        if "regression" in model_results:
            r2_scores.append(model_results["regression"]["r2_score"])
        else:
            r2_scores.append(0)
    
    # Plot bar chart of classification metrics
    plot_comparison_bar_chart(
        model_names,
        [accuracies, f1_scores],
        labels=["Accuracy", "F1 Score"],
        title="Classification Performance Comparison",
        save_path=os.path.join(output_dir, "model_classification_comparison.png")
    )
    
    # Plot bar chart of regression metrics
    plot_comparison_bar_chart(
        model_names,
        [r2_scores],
        labels=["R² Score"],
        title="Regression Performance Comparison",
        save_path=os.path.join(output_dir, "model_regression_comparison.png")
    )

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Set up logging
    log_path = os.path.join(args.log_dir, 'log.txt')
    logger = setup_logging(log_path)
    
    # Set random seed
    set_seed(args.seed)
    
    try:
        logger.info(f"Starting Neural Weight Archeology experiments")
        
        # Run experiments
        if args.model_type == 'all':
            results = run_all_models(args, logger)
        else:
            results = run_experiment(args, logger)
        
        logger.info("Experiments completed successfully")
        
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
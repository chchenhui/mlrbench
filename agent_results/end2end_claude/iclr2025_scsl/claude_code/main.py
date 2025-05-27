#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for running the CIMRL experiments.
"""

import os
import argparse
import logging
import torch
import numpy as np
import random
import json
from datetime import datetime
from pathlib import Path

from data.dataloader import get_dataloaders
from models.cimrl import CIMRL
from models.baselines import StandardMultiModal, GroupDRO, JTT, CCRMultiModal
from utils.training import train_model, evaluate_model
from utils.visualization import plot_training_curves, plot_performance_comparison, plot_feature_visualizations
from utils.metrics import compute_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'log.txt')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_model(model_name, config, device):
    """Get the appropriate model based on name."""
    if model_name == 'cimrl':
        return CIMRL(config).to(device)
    elif model_name == 'standard':
        return StandardMultiModal(config).to(device)
    elif model_name == 'groupdro':
        return GroupDRO(config).to(device)
    elif model_name == 'jtt':
        return JTT(config).to(device)
    elif model_name == 'ccr':
        return CCRMultiModal(config).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def main(args):
    # Create output directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs(os.path.join('results', 'figures'), exist_ok=True)
    os.makedirs(os.path.join('results', 'checkpoints'), exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load config from JSON file
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Log GPU info if using CUDA
    if device.type == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    # Get dataloaders
    train_loader, val_loader, test_loader, ood_test_loader = get_dataloaders(
        config['data'], 
        batch_size=config['training']['batch_size'],
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = get_model(args.model, config, device)
    logger.info(f"Initialized {args.model.upper()} model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Setup optimizer and scheduler
    optimizer_class = getattr(torch.optim, config['training']['optimizer'])
    optimizer = optimizer_class(model.parameters(), **config['training']['optimizer_params'])
    
    scheduler = None
    if config['training'].get('scheduler'):
        scheduler_class = getattr(torch.optim.lr_scheduler, config['training']['scheduler'])
        scheduler = scheduler_class(optimizer, **config['training']['scheduler_params'])
    
    # Start time for logging
    start_time = datetime.now()
    logger.info(f"Starting training at {start_time}")
    
    # Train the model
    train_losses, val_losses, train_metrics, val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config['training'],
        logger=logger,
        model_name=args.model
    )
    
    # End time and duration
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Training completed at {end_time}")
    logger.info(f"Total training time: {duration}")
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'training_time': str(duration)
    }
    with open(os.path.join('results', f'{args.model}_training_history.json'), 'w') as f:
        json.dump(training_history, f)
    
    # Evaluate on test set
    logger.info("Evaluating on in-distribution test set...")
    test_metrics = evaluate_model(model, test_loader, device, config['training'])
    
    # Evaluate on OOD test set
    logger.info("Evaluating on out-of-distribution test set...")
    ood_test_metrics = evaluate_model(model, ood_test_loader, device, config['training'])
    
    # Log and save results
    results = {
        'model': args.model,
        'dataset': config['data']['dataset'],
        'seed': args.seed,
        'in_distribution_metrics': test_metrics,
        'out_of_distribution_metrics': ood_test_metrics,
        'config': config,
        'training_time': str(duration)
    }
    
    with open(os.path.join('results', f'{args.model}_results.json'), 'w') as f:
        json.dump(results, f)
    
    # Plot results
    plot_training_curves(
        train_losses, 
        val_losses, 
        train_metrics, 
        val_metrics, 
        save_path=os.path.join('results', 'figures', f'{args.model}_training_curves.png')
    )
    
    # Generate feature visualizations
    plot_feature_visualizations(
        model, 
        test_loader, 
        device, 
        save_path=os.path.join('results', 'figures', f'{args.model}_feature_viz.png')
    )
    
    logger.info(f"Results saved to results/{args.model}_results.json")
    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIMRL Experiments')
    parser.add_argument('--config', type=str, default='configs/default.json', help='Path to config file')
    parser.add_argument('--model', type=str, default='cimrl', choices=['cimrl', 'standard', 'groupdro', 'jtt', 'ccr'], help='Model to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    main(args)
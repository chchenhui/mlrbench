#!/usr/bin/env python
"""
Simplified experiment runner for Cluster-Driven Certified Unlearning.
This script uses smaller models, reduced datasets, and simplified parameters
to allow for faster experimentation.
"""

import os
import sys
import json
import time
import torch
import numpy as np
import argparse
import logging
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config
)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code.models import ClusterDrivenCertifiedUnlearning
from claude_code.baselines import (
    RelearnUnlearningMethod,
    UnlearnWhatYouWantMethod
)
from claude_code.data import (
    load_webtext_data,
    create_deletion_sets
)
from claude_code.evaluation import (
    compute_perplexity,
    compute_knowledge_forgetting_rate,
    compute_knowledge_retention_rate,
    evaluate_membership_inference
)
from claude_code.visualization import create_summary_dashboard
from claude_code.utils import (
    set_seed,
    setup_logging,
    save_json,
    create_results_summary,
    get_available_device
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run simplified unlearning experiments')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='gpt2',
                        choices=['gpt2'],
                        help='Model to use')
    
    # Data arguments
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--stride', type=int, default=64,
                        help='Stride for tokenization window')
    parser.add_argument('--data_subset', type=float, default=0.01,
                        help='Fraction of data to use (0-1)')
    
    # Unlearning arguments
    parser.add_argument('--method', type=str, default='all',
                        choices=['cluster_driven', 'relearn', 'unlearn_what_you_want', 'all'],
                        help='Unlearning method to use')
    parser.add_argument('--num_clusters', type=int, default=5,
                        help='Number of clusters for Cluster-Driven method')
    parser.add_argument('--deletion_set_size', type=int, default=20,
                        help='Size of each deletion set')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs')
    
    # Experiment arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    
    return parser.parse_args()


def load_model_and_tokenizer(args, device):
    """Load model and tokenizer."""
    logger.info(f"Loading model {args.model_name}...")
    
    # Load from Hugging Face
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to device
    model = model.to(device)
    
    return model, tokenizer


def load_data(args, tokenizer):
    """Load and prepare dataset."""
    logger.info(f"Loading WebText dataset (subset: {args.data_subset * 100}%)...")
    
    # Load WebText data
    train_data, val_data, test_data = load_webtext_data(
        tokenizer,
        max_length=args.max_length,
        stride=args.stride
    )
    
    # Use subset of data for faster experimentation
    subset_size_train = int(len(train_data) * args.data_subset)
    subset_size_val = int(len(val_data) * args.data_subset)
    subset_size_test = int(len(test_data) * args.data_subset)
    
    train_indices = torch.randperm(len(train_data))[:subset_size_train]
    val_indices = torch.randperm(len(val_data))[:subset_size_val]
    test_indices = torch.randperm(len(test_data))[:subset_size_test]
    
    train_data_subset = torch.utils.data.Subset(train_data, train_indices)
    val_data_subset = torch.utils.data.Subset(val_data, val_indices)
    test_data_subset = torch.utils.data.Subset(test_data, test_indices)
    
    logger.info(f"Dataset sizes - Train: {len(train_data_subset)}, Val: {len(val_data_subset)}, Test: {len(test_data_subset)}")
    
    # Create deletion sets
    logger.info(f"Creating deletion set of size {args.deletion_set_size}...")
    deletion_sets = create_deletion_sets(
        train_data_subset,
        num_sets=1,
        set_sizes=[args.deletion_set_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_data_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data_subset, batch_size=args.batch_size, shuffle=False)
    
    return train_data_subset, val_data_subset, test_data_subset, deletion_sets, train_loader, val_loader, test_loader


def run_experiment(args, device):
    """Run the simplified experiment."""
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args, device)
    
    # Load data
    train_data, val_data, test_data, deletion_sets, train_loader, val_loader, test_loader = load_data(args, tokenizer)
    deletion_set = deletion_sets[0]  # Just use one deletion set for simplicity
    
    # Initialize results dictionary
    results = {
        'method_comparison': {},
        'cluster_assignments': None
    }
    
    # Compute original model perplexity as reference
    logger.info("Computing original model perplexity...")
    original_perplexity = compute_perplexity(model, test_loader, device)
    results['method_comparison']['original_model'] = {
        'perplexity': original_perplexity
    }
    
    # Define methods to run
    methods = []
    if args.method == 'all':
        methods = ['cluster_driven', 'relearn', 'unlearn_what_you_want']
    else:
        methods = [args.method]
    
    # Run each method
    for method_name in methods:
        logger.info(f"Running {method_name} unlearning method...")
        
        if method_name == 'cluster_driven':
            # Our proposed method
            start_time = time.time()
            
            unlearning_method = ClusterDrivenCertifiedUnlearning(
                model=model,
                n_clusters=args.num_clusters,
                embedding_dim=32,  # Reduced for simplicity
                influence_threshold=0.1,
                learning_rates=None,
                epsilon=0.1,
                adapter_rank=2,  # Reduced for simplicity
                device=device
            )
            
            # Extract activations and cluster representations
            logger.info("Extracting activations and clustering representations...")
            sample_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            
            # Use a smaller subset for clustering to speed up
            sample_indices = torch.randperm(len(train_data))[:min(500, len(train_data))]
            sample_data = torch.utils.data.Subset(train_data, sample_indices)
            sample_loader = DataLoader(sample_data, batch_size=args.batch_size, shuffle=True)
            
            activations = unlearning_method.extract_activations(sample_loader)
            cluster_assignments = unlearning_method.cluster_representations(activations)
            
            # Store for visualization
            results['cluster_activations'] = activations.cpu().numpy()
            results['cluster_assignments'] = cluster_assignments
            
            # Perform unlearning
            logger.info("Unlearning with Cluster-Driven method...")
            unlearned_model, certificate, metrics = unlearning_method.unlearn(
                deletion_set, val_data, test_loader
            )
            
            # Compute evaluation metrics
            logger.info("Computing evaluation metrics...")
            metrics.update({
                'perplexity': compute_perplexity(unlearned_model, test_loader, device),
                'KFR': compute_knowledge_forgetting_rate(model, unlearned_model, deletion_set, device),
                'KRR': compute_knowledge_retention_rate(model, unlearned_model, test_loader, device),
                'compute_time': time.time() - start_time
            })
            
            # Add membership inference metrics
            logger.info("Evaluating membership inference attack resistance...")
            membership_metrics = evaluate_membership_inference(
                model, unlearned_model, deletion_set, test_data, device
            )
            metrics.update(membership_metrics)
            
            # Add certificate details
            if certificate:
                metrics['certified'] = certificate['is_certified']
                metrics['kl_divergence'] = certificate['kl_divergence']
            
            results['method_comparison'][method_name] = metrics
            
        elif method_name == 'relearn':
            # ReLearn baseline
            start_time = time.time()
            
            unlearning_method = RelearnUnlearningMethod(
                model=model,
                optimizer_class=torch.optim.AdamW,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                device=device
            )
            
            # Perform unlearning
            logger.info("Unlearning with ReLearn method...")
            unlearned_model, metrics = unlearning_method.unlearn(
                val_data, deletion_set, tokenizer
            )
            
            # Compute evaluation metrics
            logger.info("Computing evaluation metrics...")
            metrics.update({
                'perplexity': compute_perplexity(unlearned_model, test_loader, device),
                'KFR': compute_knowledge_forgetting_rate(model, unlearned_model, deletion_set, device),
                'KRR': compute_knowledge_retention_rate(model, unlearned_model, test_loader, device),
                'compute_time': time.time() - start_time
            })
            
            # Add membership inference metrics
            logger.info("Evaluating membership inference attack resistance...")
            membership_metrics = evaluate_membership_inference(
                model, unlearned_model, deletion_set, test_data, device
            )
            metrics.update(membership_metrics)
            
            results['method_comparison'][method_name] = metrics
            
        elif method_name == 'unlearn_what_you_want':
            # Unlearn What You Want baseline
            start_time = time.time()
            
            unlearning_method = UnlearnWhatYouWantMethod(
                model=model,
                optimizer_class=torch.optim.AdamW,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                distillation_temp=2.0,
                distillation_alpha=0.5,
                device=device
            )
            
            # Perform unlearning
            logger.info("Unlearning with Unlearn What You Want method...")
            unlearned_model, metrics = unlearning_method.unlearn(
                val_data, deletion_set
            )
            
            # Compute evaluation metrics
            logger.info("Computing evaluation metrics...")
            metrics.update({
                'perplexity': compute_perplexity(unlearned_model, test_loader, device),
                'KFR': compute_knowledge_forgetting_rate(model, unlearned_model, deletion_set, device),
                'KRR': compute_knowledge_retention_rate(model, unlearned_model, test_loader, device),
                'compute_time': time.time() - start_time
            })
            
            # Add membership inference metrics
            logger.info("Evaluating membership inference attack resistance...")
            membership_metrics = evaluate_membership_inference(
                model, unlearned_model, deletion_set, test_data, device
            )
            metrics.update(membership_metrics)
            
            results['method_comparison'][method_name] = metrics
    
    return results


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_available_device()
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiment
    results = run_experiment(args, device)
    
    # Save results
    logger.info("Saving results...")
    save_json(results, os.path.join(args.output_dir, 'results.json'))
    
    # Create results summary
    create_results_summary(results, os.path.join(args.output_dir, 'results.md'))
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_summary_dashboard(results, os.path.join(args.output_dir, 'visualizations'))
    
    logger.info("Experiment completed successfully!")


if __name__ == "__main__":
    # Setup logging
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    logger = setup_logging(log_dir, 'simplified_experiment')
    
    main()
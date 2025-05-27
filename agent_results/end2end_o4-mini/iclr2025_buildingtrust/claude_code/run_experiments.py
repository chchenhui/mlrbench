#!/usr/bin/env python
"""
Main experiment runner for Cluster-Driven Certified Unlearning.
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
    UnlearnWhatYouWantMethod,
    CodeUnlearnMethod,
    UNDIALMethod,
    O3Framework
)
from claude_code.data import (
    load_webtext_data,
    load_domain_specific_data,
    create_deletion_sets,
    create_sequential_deletion_requests
)
from claude_code.evaluation import (
    compute_perplexity,
    compute_knowledge_forgetting_rate,
    compute_knowledge_retention_rate,
    evaluate_downstream_task,
    compute_computational_cost,
    evaluate_membership_inference
)
from claude_code.visualization import create_summary_dashboard


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'logs', 'experiment.log'
        )),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run unlearning experiments')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium'],
                        help='Model to use')
    parser.add_argument('--load_model_path', type=str, default=None,
                        help='Path to load a pre-trained model')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='webtext',
                        choices=['webtext', 'medical', 'legal', 'code'],
                        help='Dataset to use')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--stride', type=int, default=256,
                        help='Stride for tokenization window')
    
    # Unlearning arguments
    parser.add_argument('--method', type=str, default='cluster_driven',
                        choices=['cluster_driven', 'relearn', 'unlearn_what_you_want',
                                'code_unlearn', 'undial', 'o3_framework', 'all'],
                        help='Unlearning method to use')
    parser.add_argument('--num_clusters', type=int, default=10,
                        help='Number of clusters for Cluster-Driven method')
    parser.add_argument('--n_deletion_sets', type=int, default=1,
                        help='Number of deletion sets to create')
    parser.add_argument('--deletion_set_size', type=int, default=100,
                        help='Size of each deletion set')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    
    # Experiment arguments
    parser.add_argument('--run_sequential', action='store_true',
                        help='Run sequential unlearning experiment')
    parser.add_argument('--run_size_impact', action='store_true',
                        help='Run deletion set size impact experiment')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Make cudnn deterministic (slightly reduces performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_tokenizer(args):
    """Load model and tokenizer."""
    logger.info(f"Loading model {args.model_name}...")
    
    if args.load_model_path and os.path.exists(args.load_model_path):
        # Load from specified path
        model = GPT2LMHeadModel.from_pretrained(args.load_model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(args.load_model_path)
    else:
        # Load from Hugging Face
        model = GPT2LMHeadModel.from_pretrained(args.model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to device
    model = model.to(args.device)
    
    return model, tokenizer


def load_data(args, tokenizer):
    """Load and prepare dataset."""
    logger.info(f"Loading {args.dataset} dataset...")
    
    # Load data based on dataset argument
    if args.dataset == 'webtext':
        train_data, val_data, test_data = load_webtext_data(
            tokenizer,
            max_length=args.max_length,
            stride=args.stride
        )
    else:
        train_data, val_data, test_data = load_domain_specific_data(
            args.dataset,
            tokenizer,
            max_length=args.max_length,
            stride=args.stride
        )
    
    # Create deletion sets
    logger.info(f"Creating {args.n_deletion_sets} deletion sets of size {args.deletion_set_size}...")
    deletion_sets = create_deletion_sets(
        train_data,
        num_sets=args.n_deletion_sets,
        set_sizes=[args.deletion_set_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    return train_data, val_data, test_data, deletion_sets, train_loader, val_loader, test_loader


def run_unlearning_experiment(args, model, tokenizer, train_data, val_data, test_data, deletion_sets, val_loader, test_loader):
    """Run unlearning experiment with the specified method."""
    results = {}
    
    # Define methods to run
    methods = []
    if args.method == 'all':
        methods = ['cluster_driven', 'relearn', 'unlearn_what_you_want', 'code_unlearn', 'undial', 'o3_framework']
    else:
        methods = [args.method]
    
    # Iterate through methods
    for method_name in methods:
        logger.info(f"Running {method_name} unlearning experiment...")
        
        # Initialize method
        if method_name == 'cluster_driven':
            # Our proposed method
            unlearning_method = ClusterDrivenCertifiedUnlearning(
                model=model,
                n_clusters=args.num_clusters,
                embedding_dim=64,
                influence_threshold=0.1,
                learning_rates=None,
                epsilon=0.1,
                adapter_rank=4,
                device=args.device
            )
            
            # Extract activations and cluster representations
            logger.info("Extracting activations and clustering representations...")
            sample_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            activations = unlearning_method.extract_activations(sample_loader)
            cluster_assignments = unlearning_method.cluster_representations(activations)
            
            # Perform unlearning for each deletion set
            deletion_results = []
            for i, deletion_set in enumerate(deletion_sets):
                logger.info(f"Unlearning deletion set {i+1}/{len(deletion_sets)}...")
                start_time = time.time()
                
                unlearned_model, certificate, metrics = unlearning_method.unlearn(
                    deletion_set, val_data, test_loader
                )
                
                # Compute evaluation metrics
                metrics.update({
                    'perplexity': compute_perplexity(unlearned_model, test_loader, args.device),
                    'KFR': compute_knowledge_forgetting_rate(model, unlearned_model, deletion_set, args.device),
                    'KRR': compute_knowledge_retention_rate(model, unlearned_model, test_loader, args.device),
                    'compute_time': time.time() - start_time
                })
                
                # Add membership inference metrics
                membership_metrics = evaluate_membership_inference(
                    model, unlearned_model, deletion_set, test_data, args.device
                )
                metrics.update(membership_metrics)
                
                # Add certificate details
                if certificate:
                    metrics['certified'] = certificate['is_certified']
                    metrics['kl_divergence'] = certificate['kl_divergence']
                
                deletion_results.append(metrics)
            
            # Average metrics across deletion sets
            avg_results = {}
            for key in deletion_results[0].keys():
                if isinstance(deletion_results[0][key], (int, float)):
                    avg_results[key] = sum(result[key] for result in deletion_results) / len(deletion_results)
            
            results[method_name] = avg_results
            
            # Store clustering information for visualization
            results['cluster_activations'] = activations.cpu().numpy()
            results['cluster_assignments'] = cluster_assignments
            
        elif method_name == 'relearn':
            # ReLearn baseline
            unlearning_method = RelearnUnlearningMethod(
                model=model,
                optimizer_class=torch.optim.AdamW,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                device=args.device
            )
            
            # Perform unlearning for each deletion set
            deletion_results = []
            for i, deletion_set in enumerate(deletion_sets):
                logger.info(f"Unlearning deletion set {i+1}/{len(deletion_sets)}...")
                start_time = time.time()
                
                unlearned_model, metrics = unlearning_method.unlearn(
                    val_data, deletion_set, tokenizer
                )
                
                # Compute evaluation metrics
                metrics.update({
                    'perplexity': compute_perplexity(unlearned_model, test_loader, args.device),
                    'KFR': compute_knowledge_forgetting_rate(model, unlearned_model, deletion_set, args.device),
                    'KRR': compute_knowledge_retention_rate(model, unlearned_model, test_loader, args.device),
                    'compute_time': time.time() - start_time
                })
                
                # Add membership inference metrics
                membership_metrics = evaluate_membership_inference(
                    model, unlearned_model, deletion_set, test_data, args.device
                )
                metrics.update(membership_metrics)
                
                deletion_results.append(metrics)
            
            # Average metrics across deletion sets
            avg_results = {}
            for key in deletion_results[0].keys():
                if isinstance(deletion_results[0][key], (int, float)):
                    avg_results[key] = sum(result[key] for result in deletion_results) / len(deletion_results)
            
            results[method_name] = avg_results
            
        elif method_name == 'unlearn_what_you_want':
            # Unlearn What You Want baseline
            unlearning_method = UnlearnWhatYouWantMethod(
                model=model,
                optimizer_class=torch.optim.AdamW,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                distillation_temp=2.0,
                distillation_alpha=0.5,
                device=args.device
            )
            
            # Perform unlearning for each deletion set
            deletion_results = []
            for i, deletion_set in enumerate(deletion_sets):
                logger.info(f"Unlearning deletion set {i+1}/{len(deletion_sets)}...")
                start_time = time.time()
                
                unlearned_model, metrics = unlearning_method.unlearn(
                    val_data, deletion_set
                )
                
                # Compute evaluation metrics
                metrics.update({
                    'perplexity': compute_perplexity(unlearned_model, test_loader, args.device),
                    'KFR': compute_knowledge_forgetting_rate(model, unlearned_model, deletion_set, args.device),
                    'KRR': compute_knowledge_retention_rate(model, unlearned_model, test_loader, args.device),
                    'compute_time': time.time() - start_time
                })
                
                # Add membership inference metrics
                membership_metrics = evaluate_membership_inference(
                    model, unlearned_model, deletion_set, test_data, args.device
                )
                metrics.update(membership_metrics)
                
                deletion_results.append(metrics)
            
            # Average metrics across deletion sets
            avg_results = {}
            for key in deletion_results[0].keys():
                if isinstance(deletion_results[0][key], (int, float)):
                    avg_results[key] = sum(result[key] for result in deletion_results) / len(deletion_results)
            
            results[method_name] = avg_results
            
        elif method_name == 'code_unlearn':
            # CodeUnlearn baseline
            unlearning_method = CodeUnlearnMethod(
                model=model,
                optimizer_class=torch.optim.AdamW,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                codebook_size=1024,
                num_bottleneck_layers=2,
                device=args.device
            )
            
            # Perform unlearning for each deletion set
            deletion_results = []
            for i, deletion_set in enumerate(deletion_sets):
                logger.info(f"Unlearning deletion set {i+1}/{len(deletion_sets)}...")
                start_time = time.time()
                
                unlearned_model, metrics = unlearning_method.unlearn(
                    val_data, deletion_set
                )
                
                # Compute evaluation metrics
                metrics.update({
                    'perplexity': compute_perplexity(unlearned_model, test_loader, args.device),
                    'KFR': compute_knowledge_forgetting_rate(model, unlearned_model, deletion_set, args.device),
                    'KRR': compute_knowledge_retention_rate(model, unlearned_model, test_loader, args.device),
                    'compute_time': time.time() - start_time
                })
                
                # Add membership inference metrics
                membership_metrics = evaluate_membership_inference(
                    model, unlearned_model, deletion_set, test_data, args.device
                )
                metrics.update(membership_metrics)
                
                deletion_results.append(metrics)
            
            # Average metrics across deletion sets
            avg_results = {}
            for key in deletion_results[0].keys():
                if isinstance(deletion_results[0][key], (int, float)):
                    avg_results[key] = sum(result[key] for result in deletion_results) / len(deletion_results)
            
            results[method_name] = avg_results
            
        elif method_name == 'undial':
            # UNDIAL baseline
            unlearning_method = UNDIALMethod(
                model=model,
                optimizer_class=torch.optim.AdamW,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                distillation_temp=2.0,
                reduction_factor=0.5,
                device=args.device
            )
            
            # Perform unlearning for each deletion set
            deletion_results = []
            for i, deletion_set in enumerate(deletion_sets):
                logger.info(f"Unlearning deletion set {i+1}/{len(deletion_sets)}...")
                start_time = time.time()
                
                unlearned_model, metrics = unlearning_method.unlearn(
                    val_data, deletion_set
                )
                
                # Compute evaluation metrics
                metrics.update({
                    'perplexity': compute_perplexity(unlearned_model, test_loader, args.device),
                    'KFR': compute_knowledge_forgetting_rate(model, unlearned_model, deletion_set, args.device),
                    'KRR': compute_knowledge_retention_rate(model, unlearned_model, test_loader, args.device),
                    'compute_time': time.time() - start_time
                })
                
                # Add membership inference metrics
                membership_metrics = evaluate_membership_inference(
                    model, unlearned_model, deletion_set, test_data, args.device
                )
                metrics.update(membership_metrics)
                
                deletion_results.append(metrics)
            
            # Average metrics across deletion sets
            avg_results = {}
            for key in deletion_results[0].keys():
                if isinstance(deletion_results[0][key], (int, float)):
                    avg_results[key] = sum(result[key] for result in deletion_results) / len(deletion_results)
            
            results[method_name] = avg_results
            
        elif method_name == 'o3_framework':
            # O3 Framework baseline
            unlearning_method = O3Framework(
                model=model,
                optimizer_class=torch.optim.AdamW,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                num_detector_layers=2,
                adapter_dim=4,
                ood_threshold=0.5,
                orthogonal_weight=0.1,
                device=args.device
            )
            
            # Perform unlearning for each deletion set
            deletion_results = []
            for i, deletion_set in enumerate(deletion_sets):
                logger.info(f"Unlearning deletion set {i+1}/{len(deletion_sets)}...")
                start_time = time.time()
                
                unlearned_model, metrics = unlearning_method.unlearn(
                    val_data, deletion_set
                )
                
                # Compute evaluation metrics
                metrics.update({
                    'perplexity': compute_perplexity(unlearned_model, test_loader, args.device),
                    'KFR': compute_knowledge_forgetting_rate(model, unlearned_model, deletion_set, args.device),
                    'KRR': compute_knowledge_retention_rate(model, unlearned_model, test_loader, args.device),
                    'compute_time': time.time() - start_time
                })
                
                # Add membership inference metrics
                membership_metrics = evaluate_membership_inference(
                    model, unlearned_model, deletion_set, test_data, args.device
                )
                metrics.update(membership_metrics)
                
                deletion_results.append(metrics)
            
            # Average metrics across deletion sets
            avg_results = {}
            for key in deletion_results[0].keys():
                if isinstance(deletion_results[0][key], (int, float)):
                    avg_results[key] = sum(result[key] for result in deletion_results) / len(deletion_results)
            
            results[method_name] = avg_results
    
    # Add original model perplexity as reference
    results['original_model'] = {
        'perplexity': compute_perplexity(model, test_loader, args.device)
    }
    
    return results


def run_sequential_unlearning_experiment(args, model, tokenizer, train_data, val_data, test_data, val_loader, test_loader):
    """Run sequential unlearning experiment."""
    logger.info("Running sequential unlearning experiment...")
    
    # Create sequential deletion requests
    deletion_requests = create_sequential_deletion_requests(
        train_data,
        num_requests=5,
        request_sizes=[50, 50, 50, 50, 50]
    )
    
    methods_to_run = []
    if args.method == 'all':
        methods_to_run = ['cluster_driven', 'o3_framework']  # Only methods that support sequential unlearning
    elif args.method in ['cluster_driven', 'o3_framework']:
        methods_to_run = [args.method]
    else:
        logger.warning(f"Method {args.method} does not support sequential unlearning. Using cluster_driven instead.")
        methods_to_run = ['cluster_driven']
    
    results = {'sequential_results': {}}
    
    # Run experiment for each method
    for method_name in methods_to_run:
        logger.info(f"Running sequential unlearning with {method_name}...")
        
        # Initialize method
        if method_name == 'cluster_driven':
            unlearning_method = ClusterDrivenCertifiedUnlearning(
                model=model,
                n_clusters=args.num_clusters,
                embedding_dim=64,
                influence_threshold=0.1,
                learning_rates=None,
                epsilon=0.1,
                adapter_rank=4,
                device=args.device
            )
            
            # Extract activations and cluster representations
            logger.info("Extracting activations and clustering representations...")
            sample_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            activations = unlearning_method.extract_activations(sample_loader)
            cluster_assignments = unlearning_method.cluster_representations(activations)
            
            # Sequential unlearning
            sequential_metrics = []
            current_model = model
            
            for i, deletion_set in enumerate(deletion_requests):
                logger.info(f"Processing deletion request {i+1}/{len(deletion_requests)}...")
                start_time = time.time()
                
                # Perform unlearning
                if i == 0:
                    # First request
                    unlearned_model, certificate, metrics = unlearning_method.unlearn(
                        deletion_set, val_data, test_loader
                    )
                else:
                    # Subsequent requests
                    unlearned_model, certificate, metrics = unlearning_method.sequential_unlearn(
                        deletion_set, val_data, test_loader
                    )
                
                # Compute evaluation metrics
                metrics.update({
                    'request_idx': i+1,
                    'perplexity': compute_perplexity(unlearned_model, test_loader, args.device),
                    'KFR': compute_knowledge_forgetting_rate(model, unlearned_model, deletion_set, args.device),
                    'KRR': compute_knowledge_retention_rate(model, unlearned_model, test_loader, args.device),
                    'compute_time': time.time() - start_time
                })
                
                # Add membership inference metrics
                membership_metrics = evaluate_membership_inference(
                    model, unlearned_model, deletion_set, test_data, args.device
                )
                metrics.update(membership_metrics)
                
                # Add certificate details
                if certificate:
                    metrics['certified'] = certificate['is_certified']
                    metrics['kl_divergence'] = certificate['kl_divergence']
                
                sequential_metrics.append(metrics)
                current_model = unlearned_model
            
            results['sequential_results'][method_name] = sequential_metrics
            
        elif method_name == 'o3_framework':
            unlearning_method = O3Framework(
                model=model,
                optimizer_class=torch.optim.AdamW,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                num_detector_layers=2,
                adapter_dim=4,
                ood_threshold=0.5,
                orthogonal_weight=0.1,
                device=args.device
            )
            
            # Sequential unlearning
            sequential_metrics = []
            current_model = model
            
            for i, deletion_set in enumerate(deletion_requests):
                logger.info(f"Processing deletion request {i+1}/{len(deletion_requests)}...")
                start_time = time.time()
                
                # Perform unlearning
                unlearned_model, metrics = unlearning_method.sequential_unlearn(
                    val_data, deletion_set
                )
                
                # Compute evaluation metrics
                metrics.update({
                    'request_idx': i+1,
                    'perplexity': compute_perplexity(unlearned_model, test_loader, args.device),
                    'KFR': compute_knowledge_forgetting_rate(model, unlearned_model, deletion_set, args.device),
                    'KRR': compute_knowledge_retention_rate(model, unlearned_model, test_loader, args.device),
                    'compute_time': time.time() - start_time
                })
                
                # Add membership inference metrics
                membership_metrics = evaluate_membership_inference(
                    model, unlearned_model, deletion_set, test_data, args.device
                )
                metrics.update(membership_metrics)
                
                sequential_metrics.append(metrics)
                current_model = unlearned_model
            
            results['sequential_results'][method_name] = sequential_metrics
    
    return results


def run_deletion_size_impact_experiment(args, model, tokenizer, train_data, val_data, test_data, val_loader, test_loader):
    """Run experiment to measure the impact of deletion set size."""
    logger.info("Running deletion set size impact experiment...")
    
    # Define deletion set sizes
    deletion_sizes = [10, 50, 100, 500, 1000]
    
    # Create deletion sets of various sizes
    all_deletion_sets = []
    for size in deletion_sizes:
        deletion_sets = create_deletion_sets(
            train_data,
            num_sets=1,  # Just one set per size to save time
            set_sizes=[size]
        )
        all_deletion_sets.extend(deletion_sets)
    
    methods_to_run = []
    if args.method == 'all':
        methods_to_run = ['cluster_driven', 'relearn', 'unlearn_what_you_want']  # Use a subset to save time
    else:
        methods_to_run = [args.method]
    
    results = {'deletion_size_impact': {}}
    
    # Initialize size results for each size
    for size in deletion_sizes:
        results['deletion_size_impact'][size] = {}
    
    # Run experiment for each method and size
    for method_name in methods_to_run:
        logger.info(f"Measuring {method_name} performance across deletion set sizes...")
        
        for i, size in enumerate(deletion_sizes):
            logger.info(f"Testing with deletion set size {size}...")
            deletion_set = all_deletion_sets[i]
            
            # Initialize method
            if method_name == 'cluster_driven':
                unlearning_method = ClusterDrivenCertifiedUnlearning(
                    model=model,
                    n_clusters=args.num_clusters,
                    embedding_dim=64,
                    influence_threshold=0.1,
                    learning_rates=None,
                    epsilon=0.1,
                    adapter_rank=4,
                    device=args.device
                )
                
                # Extract activations and cluster representations (once is enough)
                if i == 0:
                    logger.info("Extracting activations and clustering representations...")
                    sample_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
                    activations = unlearning_method.extract_activations(sample_loader)
                    cluster_assignments = unlearning_method.cluster_representations(activations)
                
                # Perform unlearning
                start_time = time.time()
                unlearned_model, certificate, metrics = unlearning_method.unlearn(
                    deletion_set, val_data, test_loader
                )
                
                # Compute evaluation metrics
                metrics.update({
                    'deletion_set_size': size,
                    'perplexity': compute_perplexity(unlearned_model, test_loader, args.device),
                    'KFR': compute_knowledge_forgetting_rate(model, unlearned_model, deletion_set, args.device),
                    'KRR': compute_knowledge_retention_rate(model, unlearned_model, test_loader, args.device),
                    'compute_time': time.time() - start_time
                })
                
                # Add membership inference metrics
                membership_metrics = evaluate_membership_inference(
                    model, unlearned_model, deletion_set, test_data, args.device
                )
                metrics.update(membership_metrics)
                
                # Add certificate details
                if certificate:
                    metrics['certified'] = certificate['is_certified']
                    metrics['kl_divergence'] = certificate['kl_divergence']
                
                results['deletion_size_impact'][size][method_name] = metrics
                
            elif method_name == 'relearn':
                unlearning_method = RelearnUnlearningMethod(
                    model=model,
                    optimizer_class=torch.optim.AdamW,
                    learning_rate=args.learning_rate,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    device=args.device
                )
                
                # Perform unlearning
                start_time = time.time()
                unlearned_model, metrics = unlearning_method.unlearn(
                    val_data, deletion_set, tokenizer
                )
                
                # Compute evaluation metrics
                metrics.update({
                    'deletion_set_size': size,
                    'perplexity': compute_perplexity(unlearned_model, test_loader, args.device),
                    'KFR': compute_knowledge_forgetting_rate(model, unlearned_model, deletion_set, args.device),
                    'KRR': compute_knowledge_retention_rate(model, unlearned_model, test_loader, args.device),
                    'compute_time': time.time() - start_time
                })
                
                # Add membership inference metrics
                membership_metrics = evaluate_membership_inference(
                    model, unlearned_model, deletion_set, test_data, args.device
                )
                metrics.update(membership_metrics)
                
                results['deletion_size_impact'][size][method_name] = metrics
                
            elif method_name == 'unlearn_what_you_want':
                unlearning_method = UnlearnWhatYouWantMethod(
                    model=model,
                    optimizer_class=torch.optim.AdamW,
                    learning_rate=args.learning_rate,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    distillation_temp=2.0,
                    distillation_alpha=0.5,
                    device=args.device
                )
                
                # Perform unlearning
                start_time = time.time()
                unlearned_model, metrics = unlearning_method.unlearn(
                    val_data, deletion_set
                )
                
                # Compute evaluation metrics
                metrics.update({
                    'deletion_set_size': size,
                    'perplexity': compute_perplexity(unlearned_model, test_loader, args.device),
                    'KFR': compute_knowledge_forgetting_rate(model, unlearned_model, deletion_set, args.device),
                    'KRR': compute_knowledge_retention_rate(model, unlearned_model, test_loader, args.device),
                    'compute_time': time.time() - start_time
                })
                
                # Add membership inference metrics
                membership_metrics = evaluate_membership_inference(
                    model, unlearned_model, deletion_set, test_data, args.device
                )
                metrics.update(membership_metrics)
                
                results['deletion_size_impact'][size][method_name] = metrics
    
    return results


def normalize_metrics(results):
    """Normalize metrics for radar plot."""
    normalized_metrics = {}
    
    # Extract metrics to normalize
    metrics_to_normalize = ['KFR', 'KRR', 'compute_time', 'perplexity']
    
    # Create normalized_metrics structure
    for method in results.keys():
        if method == 'original_model':
            continue
        normalized_metrics[method] = {}
    
    # Collect min/max values for each metric
    min_values = {metric: float('inf') for metric in metrics_to_normalize}
    max_values = {metric: float('-inf') for metric in metrics_to_normalize}
    
    for method, method_results in results.items():
        if method == 'original_model':
            continue
        
        for metric in metrics_to_normalize:
            if metric in method_results:
                min_values[metric] = min(min_values[metric], method_results[metric])
                max_values[metric] = max(max_values[metric], method_results[metric])
    
    # Normalize metrics
    for method, method_results in results.items():
        if method == 'original_model':
            continue
        
        for metric in metrics_to_normalize:
            if metric in method_results:
                # For metrics where higher is better (KFR, KRR)
                if metric in ['KFR', 'KRR']:
                    normalized_value = (method_results[metric] - min_values[metric]) / (max_values[metric] - min_values[metric]) if max_values[metric] > min_values[metric] else 0.5
                # For metrics where lower is better (compute_time, perplexity)
                else:
                    normalized_value = 1 - (method_results[metric] - min_values[metric]) / (max_values[metric] - min_values[metric]) if max_values[metric] > min_values[metric] else 0.5
                
                # Add to normalized_metrics with modified name
                normalized_metrics[method][f"{metric}_Norm"] = normalized_value
    
    return normalized_metrics


def save_results(args, results):
    """Save experiment results."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save raw results as JSON
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary dashboard
    if 'deletion_size_impact' in results:
        results['methods'] = list(next(iter(results['deletion_size_impact'].values())).keys())
    
    # Add normalized metrics for radar plot
    if 'method_comparison' in results:
        results['normalized_metrics'] = normalize_metrics(results['method_comparison'])
    
    # Create visualizations
    create_summary_dashboard(results, os.path.join(args.output_dir, 'visualizations'))
    
    logger.info(f"Results saved to {args.output_dir}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Load data
    train_data, val_data, test_data, deletion_sets, train_loader, val_loader, test_loader = load_data(args, tokenizer)
    
    # Initialize results dictionary
    all_results = {}
    
    # Run unlearning experiment
    unlearning_results = run_unlearning_experiment(
        args, model, tokenizer, train_data, val_data, test_data, deletion_sets, val_loader, test_loader
    )
    all_results['method_comparison'] = unlearning_results
    
    # Run sequential unlearning experiment if requested
    if args.run_sequential:
        sequential_results = run_sequential_unlearning_experiment(
            args, model, tokenizer, train_data, val_data, test_data, val_loader, test_loader
        )
        all_results.update(sequential_results)
    
    # Run deletion set size impact experiment if requested
    if args.run_size_impact:
        size_impact_results = run_deletion_size_impact_experiment(
            args, model, tokenizer, train_data, val_data, test_data, val_loader, test_loader
        )
        all_results.update(size_impact_results)
    
    # Save results
    save_results(args, all_results)
    
    logger.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()
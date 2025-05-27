#!/usr/bin/env python
"""
Synthetic experiment runner for Cluster-Driven Certified Unlearning.
This script uses synthetic data to test the unlearning methods.
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
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_synthetic.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing unlearning methods.
    """
    
    def __init__(self, size=1000, seq_length=128, vocab_size=50257):
        """
        Initialize the synthetic dataset.
        
        Args:
            size (int): Number of examples
            seq_length (int): Sequence length
            vocab_size (int): Vocabulary size
        """
        # Create random input_ids and attention_mask
        self.input_ids = torch.randint(0, vocab_size, (size, seq_length))
        self.attention_mask = torch.ones_like(self.input_ids)
        
        # For targets, just shift the input_ids right by one position (like language modeling)
        self.targets = torch.roll(self.input_ids, -1, dims=1)
        # Fill the last position with a random token
        self.targets[:, -1] = torch.randint(0, vocab_size, (size,))
        
    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'targets': self.targets[idx]
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run synthetic unlearning experiments')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='gpt2',
                        choices=['gpt2'],
                        help='Model to use')
    
    # Data arguments
    parser.add_argument('--dataset_size', type=int, default=1000,
                        help='Number of examples in the synthetic dataset')
    parser.add_argument('--seq_length', type=int, default=128,
                        help='Sequence length')
    
    # Unlearning arguments
    parser.add_argument('--method', type=str, default='all',
                        choices=['cluster_driven', 'relearn', 'unlearn_what_you_want', 'all'],
                        help='Unlearning method to use')
    parser.add_argument('--num_clusters', type=int, default=3,
                        help='Number of clusters for Cluster-Driven method')
    parser.add_argument('--deletion_set_size', type=int, default=10,
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


def load_data(args):
    """Load and prepare dataset."""
    logger.info(f"Creating synthetic dataset (size: {args.dataset_size})...")
    
    # Create synthetic dataset
    full_dataset = SyntheticDataset(
        size=args.dataset_size,
        seq_length=args.seq_length
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_data, val_data, test_data = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create deletion set
    logger.info(f"Creating deletion set (size: {args.deletion_set_size})...")
    
    # Sample random indices
    indices = torch.randperm(len(train_data))[:args.deletion_set_size]
    deletion_set = [train_data[i] for i in indices]
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    return train_data, val_data, test_data, deletion_set, train_loader, val_loader, test_loader


def custom_compute_perplexity(model, data_loader, device):
    """
    Simplified perplexity computation that works with our synthetic data.
    
    Args:
        model: Language model
        data_loader: DataLoader providing examples
        device: Device for computation
        
    Returns:
        perplexity (float): Perplexity score (or a proxy for it)
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    # Use cross-entropy loss
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for batch in data_loader:
            # Move inputs to device
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Make targets the same shape as logits for loss computation
            targets = batch['targets'].to(device)
            
            # Reshape logits to [batch_size * seq_length, vocab_size]
            batch_size, seq_length = inputs['input_ids'].shape
            vocab_size = logits.shape[-1]
            reshaped_logits = logits.reshape(-1, vocab_size)
            
            # Reshape targets to [batch_size * seq_length]
            reshaped_targets = targets.reshape(-1)
            
            # Compute loss
            loss = loss_fn(reshaped_logits, reshaped_targets)
            
            # Accumulate loss
            total_loss += loss.item()
            total_tokens += torch.sum(batch['attention_mask']).item()
    
    # Compute average loss
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    # Return raw loss instead of perplexity (which would be torch.exp(avg_loss))
    # This avoids overflow for very large losses
    return avg_loss


def custom_knowledge_forgetting_rate(original_model, unlearned_model, deletion_set, device):
    """
    Simplified KFR computation that works with our synthetic data.
    
    Args:
        original_model: Original model
        unlearned_model: Unlearned model
        deletion_set: Set of examples to delete
        device: Device for computation
        
    Returns:
        kfr (float): Knowledge Forgetting Rate proxy
    """
    # Compute loss on deletion set for both models
    original_loss = 0
    unlearned_loss = 0
    total_tokens = 0
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    original_model.eval()
    unlearned_model.eval()
    
    with torch.no_grad():
        for example in deletion_set:
            # Move inputs to device
            inputs = {
                'input_ids': example['input_ids'].unsqueeze(0).to(device),
                'attention_mask': example['attention_mask'].unsqueeze(0).to(device)
            }
            targets = example['targets'].unsqueeze(0).to(device)
            
            # Forward pass for original model
            original_outputs = original_model(**inputs)
            original_logits = original_outputs.logits
            
            # Forward pass for unlearned model
            unlearned_outputs = unlearned_model(**inputs)
            unlearned_logits = unlearned_outputs.logits
            
            # Reshape logits and targets
            batch_size, seq_length = inputs['input_ids'].shape
            vocab_size = original_logits.shape[-1]
            
            reshaped_original_logits = original_logits.reshape(-1, vocab_size)
            reshaped_unlearned_logits = unlearned_logits.reshape(-1, vocab_size)
            reshaped_targets = targets.reshape(-1)
            
            # Compute losses
            original_example_loss = loss_fn(reshaped_original_logits, reshaped_targets)
            unlearned_example_loss = loss_fn(reshaped_unlearned_logits, reshaped_targets)
            
            # Accumulate losses
            original_loss += original_example_loss.item()
            unlearned_loss += unlearned_example_loss.item()
            total_tokens += torch.sum(inputs['attention_mask']).item()
    
    # Compute average losses
    avg_original_loss = original_loss / total_tokens if total_tokens > 0 else 0
    avg_unlearned_loss = unlearned_loss / total_tokens if total_tokens > 0 else 0
    
    # KFR is higher when unlearned model has higher loss on deletion set
    # Normalize to [0, 1] range
    relative_increase = (avg_unlearned_loss - avg_original_loss) / (avg_original_loss + 1e-10)
    kfr = min(max(relative_increase, 0), 1)  # Clip to [0, 1]
    
    return kfr


def custom_knowledge_retention_rate(original_model, unlearned_model, data_loader, device):
    """
    Simplified KRR computation that works with our synthetic data.
    
    Args:
        original_model: Original model
        unlearned_model: Unlearned model
        data_loader: DataLoader for test data
        device: Device for computation
        
    Returns:
        krr (float): Knowledge Retention Rate proxy
    """
    # Compute loss on test set for both models
    original_loss = 0
    unlearned_loss = 0
    total_tokens = 0
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    original_model.eval()
    unlearned_model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
            # Move inputs to device
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            targets = batch['targets'].to(device)
            
            # Forward pass for original model
            original_outputs = original_model(**inputs)
            original_logits = original_outputs.logits
            
            # Forward pass for unlearned model
            unlearned_outputs = unlearned_model(**inputs)
            unlearned_logits = unlearned_outputs.logits
            
            # Reshape logits and targets
            batch_size, seq_length = inputs['input_ids'].shape
            vocab_size = original_logits.shape[-1]
            
            reshaped_original_logits = original_logits.reshape(-1, vocab_size)
            reshaped_unlearned_logits = unlearned_logits.reshape(-1, vocab_size)
            reshaped_targets = targets.reshape(-1)
            
            # Compute losses
            original_batch_loss = loss_fn(reshaped_original_logits, reshaped_targets)
            unlearned_batch_loss = loss_fn(reshaped_unlearned_logits, reshaped_targets)
            
            # Accumulate losses
            original_loss += original_batch_loss.item()
            unlearned_loss += unlearned_batch_loss.item()
            total_tokens += torch.sum(batch['attention_mask']).item()
    
    # Compute average losses
    avg_original_loss = original_loss / total_tokens if total_tokens > 0 else float('inf')
    avg_unlearned_loss = unlearned_loss / total_tokens if total_tokens > 0 else float('inf')
    
    # KRR is higher when unlearned model has similar loss to original model
    # Normalized similarity: 1 - |diff|/original
    relative_diff = abs(avg_unlearned_loss - avg_original_loss) / (avg_original_loss + 1e-10)
    krr = max(1 - relative_diff, 0)  # Ensure it's non-negative
    
    return krr


def run_experiment(args, device):
    """Run the synthetic experiment."""
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args, device)
    
    # Load data
    train_data, val_data, test_data, deletion_set, train_loader, val_loader, test_loader = load_data(args)
    
    # Initialize results dictionary
    results = {
        'method_comparison': {},
        'cluster_assignments': None
    }
    
    # Compute original model perplexity as reference
    logger.info("Computing original model perplexity...")
    original_perplexity = custom_compute_perplexity(model, test_loader, device)
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
            sample_loader = DataLoader(
                torch.utils.data.Subset(train_data, torch.randperm(len(train_data))[:100]),
                batch_size=args.batch_size,
                shuffle=True
            )
            
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
                'perplexity': custom_compute_perplexity(unlearned_model, test_loader, device),
                'KFR': custom_knowledge_forgetting_rate(model, unlearned_model, deletion_set, device),
                'KRR': custom_knowledge_retention_rate(model, unlearned_model, test_loader, device),
                'compute_time': time.time() - start_time
            })
            
            # Skip membership inference metrics for synthetic data
            
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
                'perplexity': custom_compute_perplexity(unlearned_model, test_loader, device),
                'KFR': custom_knowledge_forgetting_rate(model, unlearned_model, deletion_set, device),
                'KRR': custom_knowledge_retention_rate(model, unlearned_model, test_loader, device),
                'compute_time': time.time() - start_time
            })
            
            # Skip membership inference metrics for synthetic data
            
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
                'perplexity': custom_compute_perplexity(unlearned_model, test_loader, device),
                'KFR': custom_knowledge_forgetting_rate(model, unlearned_model, deletion_set, device),
                'KRR': custom_knowledge_retention_rate(model, unlearned_model, test_loader, device),
                'compute_time': time.time() - start_time
            })
            
            # Skip membership inference metrics for synthetic data
            
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
    main()
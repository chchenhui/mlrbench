"""
Main experiment runner script for SpurGen benchmark.
"""

import argparse
import json
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import time
from typing import Dict, List, Tuple, Optional, Union

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generator import SpurGen
from data.dataset import get_data_loaders, get_shuffled_dataloader
from models.models import SimpleImageClassifier, SimpleTextClassifier, MultimodalClassifier, Adversary
from models.robustification import (IRMModel, GroupDROModel, AdversarialModel, ContrastiveModel,
                                  create_adversarial_model)
from utils.training import Trainer, IRMTrainer, GroupDROTrainer, AdversarialDebiasing, ContrastiveTrainer
from utils.metrics import compute_spurious_sensitivity_score, compute_invariance_gap, compute_worst_group_accuracy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log.txt')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments on SpurGen benchmark')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory to save/load the dataset')
    parser.add_argument('--generate_data', action='store_true',
                        help='Whether to generate new data or use existing data')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes in the dataset')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to generate')
    
    # Model parameters
    parser.add_argument('--modality', type=str, default='image',
                        choices=['image', 'text', 'multimodal'],
                        help='Data modality to use')
    parser.add_argument('--feature_dim', type=int, default=512,
                        help='Dimension of feature vectors')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    
    # Robustification parameters
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['erm', 'irm', 'group_dro', 'adversarial', 'contrastive'],
                        help='Robustification methods to evaluate')
    parser.add_argument('--irm_lambda', type=float, default=1.0,
                        help='IRM penalty weight')
    parser.add_argument('--adv_lambda', type=float, default=1.0,
                        help='Adversarial penalty weight')
    parser.add_argument('--cont_lambda', type=float, default=0.1,
                        help='Contrastive loss weight')
    
    # Experiment parameters
    parser.add_argument('--exp_name', type=str, default='spurgen_experiment',
                        help='Experiment name')
    parser.add_argument('--save_dir', type=str, default='../results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    return parser.parse_args()

def generate_dataset(args):
    """Generate synthetic dataset with spurious correlations."""
    logger.info("Generating synthetic dataset...")
    
    # Define class names
    class_names = ["dog", "cat", "car", "tree", "flower", 
                   "building", "airplane", "bird", "boat", "bicycle"][:args.num_classes]
    
    # Initialize SpurGen
    spurgen = SpurGen(
        num_classes=args.num_classes,
        num_samples=args.num_samples,
        class_names=class_names,
        save_dir=args.data_dir
    )
    
    # Generate dataset
    metadata = spurgen.generate_dataset()
    logger.info(f"Generated dataset with {metadata['num_samples']} samples")
    
    # Visualize samples
    spurgen.visualize_samples(num_samples=5, save_path=os.path.join(args.save_dir, "sample_visualization.png"))
    
    # Create shuffled versions for evaluating sensitivity to each channel
    for channel in spurgen.spurious_channels.keys():
        logger.info(f"Creating shuffled dataset for channel: {channel}")
        shuffled_data = spurgen.perform_attribute_shuffling(channel=channel)
        logger.info(f"Created shuffled dataset with {len(shuffled_data)} samples")
    
    return metadata

def create_model(args, num_classes, metadata=None):
    """Create model based on the specified modality."""
    if args.modality == "image":
        model = SimpleImageClassifier(num_classes=num_classes, feature_dim=args.feature_dim)
    elif args.modality == "text":
        model = SimpleTextClassifier(num_classes=num_classes, hidden_dim=args.feature_dim)
    else:  # multimodal
        model = MultimodalClassifier(num_classes=num_classes, fusion_dim=args.feature_dim)
    
    return model

def run_experiment(args, metadata):
    """Run experiments with different robustification methods."""
    logger.info(f"Running experiments with modality: {args.modality}")
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get data loaders
    data_loaders = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        modality=args.modality
    )
    
    # Get shuffled data loaders for evaluation
    shuffled_loaders = {}
    for channel in metadata["spurious_channels"].keys():
        shuffled_loaders[channel] = get_shuffled_dataloader(
            data_dir=args.data_dir,
            channel=channel,
            batch_size=args.batch_size,
            modality=args.modality
        )
    
    # Create directory for saving results
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Dictionary to store results for all methods
    all_results = {}
    
    # Run each robustification method
    for method in args.methods:
        logger.info(f"Starting experiment with method: {method}")
        
        # Create model and optimizer
        base_model = create_model(args, metadata["num_classes"], metadata)
        
        if method == "erm":
            model = base_model
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            
            trainer = Trainer(
                model=model,
                train_loader=data_loaders["train"],
                val_loader=data_loaders["val"],
                test_loader=data_loaders["test"],
                criterion=nn.CrossEntropyLoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=args.num_epochs,
                modality=args.modality,
                save_dir=args.save_dir,
                exp_name=f"{args.exp_name}_{method}",
                metadata=metadata
            )
            
        elif method == "irm":
            model = IRMModel(base_model, modality=args.modality)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            
            trainer = IRMTrainer(
                model=model,
                train_loader=data_loaders["train"],
                val_loader=data_loaders["val"],
                test_loader=data_loaders["test"],
                criterion=nn.CrossEntropyLoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=args.num_epochs,
                modality=args.modality,
                save_dir=args.save_dir,
                exp_name=f"{args.exp_name}_{method}",
                metadata=metadata,
                irm_lambda=args.irm_lambda
            )
            
        elif method == "group_dro":
            # Need data loaders with attributes
            group_data_loaders = get_data_loaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                modality=args.modality,
                return_attributes=True
            )
            
            model = GroupDROModel(base_model, modality=args.modality)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            
            trainer = GroupDROTrainer(
                model=model,
                train_loader=group_data_loaders["train"],
                val_loader=data_loaders["val"],
                test_loader=data_loaders["test"],
                criterion=nn.CrossEntropyLoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=args.num_epochs,
                modality=args.modality,
                save_dir=args.save_dir,
                exp_name=f"{args.exp_name}_{method}",
                metadata=metadata
            )
            
        elif method == "adversarial":
            # Need data loaders with attributes
            adv_data_loaders = get_data_loaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                modality=args.modality,
                return_attributes=True
            )
            
            # Create adversary for the background channel
            adv_channel = "background"
            num_attributes = len(metadata["spurious_channels"][adv_channel]["attributes"])
            
            model = AdversarialModel(base_model, modality=args.modality)
            adversary = Adversary(
                input_dim=args.feature_dim,
                hidden_dim=args.feature_dim // 2,
                num_attributes=num_attributes
            )
            
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            adv_optimizer = optim.Adam(adversary.parameters(), lr=args.lr * 10, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            
            trainer = AdversarialDebiasing(
                model=model,
                train_loader=adv_data_loaders["train"],
                val_loader=data_loaders["val"],
                test_loader=data_loaders["test"],
                criterion=nn.CrossEntropyLoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=args.num_epochs,
                modality=args.modality,
                save_dir=args.save_dir,
                exp_name=f"{args.exp_name}_{method}",
                metadata=metadata,
                adversary=adversary,
                adv_optimizer=adv_optimizer,
                lambda_adv=args.adv_lambda,
                adv_channel=adv_channel
            )
            
        elif method == "contrastive":
            # Need data loaders with shuffled pairs
            channel_to_shuffle = "background"  # Choose a channel to shuffle
            contrastive_loader = get_shuffled_dataloader(
                data_dir=args.data_dir,
                channel=channel_to_shuffle,
                batch_size=args.batch_size,
                split="train",
                modality=args.modality
            )
            
            model = ContrastiveModel(base_model, modality=args.modality)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            
            trainer = ContrastiveTrainer(
                model=model,
                train_loader=contrastive_loader,
                val_loader=data_loaders["val"],
                test_loader=data_loaders["test"],
                criterion=nn.CrossEntropyLoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=args.num_epochs,
                modality=args.modality,
                save_dir=args.save_dir,
                exp_name=f"{args.exp_name}_{method}",
                metadata=metadata,
                lambda_contrastive=args.cont_lambda
            )
        
        # Train the model
        model.to(device)
        history = trainer.train()
        
        # Evaluate sensitivity for each channel
        sss_scores = {}
        for channel, loader in shuffled_loaders.items():
            sss = compute_spurious_sensitivity_score(
                model=model,
                shuffled_loader=loader,
                device=device,
                modality=args.modality
            )
            sss_scores[channel] = sss
            logger.info(f"Method: {method}, Channel: {channel}, Spurious Sensitivity Score: {sss:.4f}")
        
        # Create a loader with controlled spurious alignments
        # For simplicity, we'll use the standard test loader as controlled
        # and a shuffled loader as uncontrolled
        uncontrolled_loader = shuffled_loaders["background"]
        
        # Compute Invariance Gap
        ig = compute_invariance_gap(
            model=model,
            ctrl_loader=data_loaders["test"],
            uncontrolled_loader=uncontrolled_loader,
            device=device,
            criterion=nn.CrossEntropyLoss(),
            modality=args.modality
        )
        logger.info(f"Method: {method}, Invariance Gap: {ig:.4f}")
        
        # Store results
        all_results[method] = {
            "test_accuracy": history["test_acc"],
            "worst_group_accuracy": history.get("worst_group_acc", None),
            "spurious_sensitivity_scores": sss_scores,
            "invariance_gap": ig,
            "training_time": history["training_time"],
            "best_epoch": history["best_epoch"]
        }
        
        # Plot sensitivity scores
        plt.figure(figsize=(10, 6))
        channels = list(sss_scores.keys())
        scores = [sss_scores[c] for c in channels]
        
        plt.bar(channels, scores)
        plt.xlabel("Spurious Channel")
        plt.ylabel("Spurious Sensitivity Score (SSS)")
        plt.title(f"{method.upper()} - Spurious Sensitivity Scores")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(args.save_dir, f"{args.exp_name}_{method}_sss.png"))
        plt.close()
    
    # Compare methods
    logger.info("Comparison of methods:")
    logger.info(f"{'Method':<15} {'Test Acc':<10} {'Worst Group':<15} {'Avg SSS':<10} {'IG':<10}")
    logger.info("-" * 60)
    
    for method, results in all_results.items():
        avg_sss = np.mean(list(results["spurious_sensitivity_scores"].values()))
        logger.info(f"{method:<15} {results['test_accuracy']:<10.4f} "
                    f"{results.get('worst_group_accuracy', 0.0):<15.4f} "
                    f"{avg_sss:<10.4f} {results['invariance_gap']:<10.4f}")
    
    # Plot comparison of methods
    plt.figure(figsize=(12, 8))
    
    # Bar width
    width = 0.2
    
    # x locations for groups
    ind = np.arange(len(args.methods))
    
    # Extract metrics for each method
    test_accs = [all_results[m]["test_accuracy"] for m in args.methods]
    worst_group_accs = [all_results[m].get("worst_group_accuracy", 0.0) for m in args.methods]
    avg_sss = [np.mean(list(all_results[m]["spurious_sensitivity_scores"].values())) for m in args.methods]
    igs = [all_results[m]["invariance_gap"] for m in args.methods]
    
    # Plot bars
    plt.bar(ind - width*1.5, test_accs, width, label="Test Accuracy")
    plt.bar(ind - width*0.5, worst_group_accs, width, label="Worst Group Accuracy")
    plt.bar(ind + width*0.5, avg_sss, width, label="Avg. Spurious Sensitivity Score")
    plt.bar(ind + width*1.5, igs, width, label="Invariance Gap")
    
    plt.xlabel("Method")
    plt.ylabel("Value")
    plt.title("Comparison of Robustification Methods")
    plt.xticks(ind, [m.upper() for m in args.methods])
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(args.save_dir, f"{args.exp_name}_comparison.png"))
    plt.close()
    
    # Save results to JSON
    with open(os.path.join(args.save_dir, f"{args.exp_name}_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def generate_results_markdown(args, all_results):
    """Generate a markdown file with results summary."""
    results_md = f"# SpurGen Experiment Results: {args.exp_name}\n\n"
    
    # Add experiment details
    results_md += "## Experiment Details\n\n"
    results_md += f"- **Modality**: {args.modality}\n"
    results_md += f"- **Number of classes**: {args.num_classes}\n"
    results_md += f"- **Number of samples**: {args.num_samples}\n"
    results_md += f"- **Batch size**: {args.batch_size}\n"
    results_md += f"- **Learning rate**: {args.lr}\n"
    results_md += f"- **Number of epochs**: {args.num_epochs}\n"
    results_md += f"- **Feature dimension**: {args.feature_dim}\n"
    results_md += f"- **Methods evaluated**: {', '.join(args.methods)}\n\n"
    
    # Add summary table
    results_md += "## Results Summary\n\n"
    results_md += "| Method | Test Accuracy | Worst Group Accuracy | Avg. Spurious Sensitivity | Invariance Gap |\n"
    results_md += "|--------|--------------|----------------------|---------------------------|---------------|\n"
    
    for method, results in all_results.items():
        avg_sss = np.mean(list(results["spurious_sensitivity_scores"].values()))
        results_md += f"| {method.upper()} | {results['test_accuracy']:.4f} | "
        results_md += f"{results.get('worst_group_accuracy', 0.0):.4f} | {avg_sss:.4f} | {results['invariance_gap']:.4f} |\n"
    
    results_md += "\n"
    
    # Add figures
    results_md += "## Performance Comparison\n\n"
    results_md += f"![Comparison of Methods]({args.exp_name}_comparison.png)\n\n"
    
    results_md += "## Training Curves\n\n"
    for method in args.methods:
        results_md += f"### {method.upper()}\n\n"
        results_md += f"![Training Curves for {method.upper()}]({args.exp_name}_{method}_training_curves.png)\n\n"
    
    results_md += "## Spurious Sensitivity Scores\n\n"
    for method in args.methods:
        results_md += f"### {method.upper()}\n\n"
        results_md += f"![Spurious Sensitivity Scores for {method.upper()}]({args.exp_name}_{method}_sss.png)\n\n"
    
    # Add sample visualization
    results_md += "## Sample Visualization\n\n"
    results_md += "![Sample Visualization](sample_visualization.png)\n\n"
    
    # Add conclusion
    results_md += "## Conclusion\n\n"
    
    # Find best method based on worst group accuracy
    best_method = max(all_results, key=lambda m: all_results[m].get("worst_group_accuracy", 0.0))
    best_worst_group_acc = all_results[best_method].get("worst_group_accuracy", 0.0)
    
    # Find method with lowest sensitivity
    best_sss_method = min(all_results, key=lambda m: np.mean(list(all_results[m]["spurious_sensitivity_scores"].values())))
    best_sss = np.mean(list(all_results[best_sss_method]["spurious_sensitivity_scores"].values()))
    
    results_md += f"Based on our experiments, the **{best_method.upper()}** method achieved the best worst-group accuracy "
    results_md += f"of {best_worst_group_acc:.4f}, indicating the highest robustness to spurious correlations. "
    results_md += f"The **{best_sss_method.upper()}** method demonstrated the lowest average spurious sensitivity score "
    results_md += f"of {best_sss:.4f}, suggesting the least reliance on spurious features.\n\n"
    
    results_md += "These results demonstrate the effectiveness of robustification methods in mitigating "
    results_md += "the effects of spurious correlations in machine learning models. The SpurGen benchmark "
    results_md += "provides a controlled environment for evaluating these methods across different modalities "
    results_md += "and spurious channels.\n\n"
    
    # Add limitations and future work
    results_md += "## Limitations and Future Work\n\n"
    results_md += "- The current experiments are limited to synthetic data and may not fully capture the complexity of real-world spurious correlations.\n"
    results_md += "- Future work could explore the effectiveness of these methods on larger, more diverse datasets.\n"
    results_md += "- Investigating the interplay between different spurious channels and their impact on model performance would be valuable.\n"
    results_md += "- Developing and evaluating more sophisticated robustification methods, particularly for multimodal data, remains an important direction for future research.\n"
    
    return results_md

def main():
    """Main function to run the experiments."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Generate or load dataset
    if args.generate_data:
        metadata = generate_dataset(args)
    else:
        # Load metadata from existing dataset
        with open(os.path.join(args.data_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
    
    # Run experiments
    all_results = run_experiment(args, metadata)
    
    # Generate results markdown
    results_md = generate_results_markdown(args, all_results)
    
    # Write results.md
    with open(os.path.join(args.save_dir, "results.md"), "w") as f:
        f.write(results_md)
    
    logger.info(f"Results saved to {os.path.join(args.save_dir, 'results.md')}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info(f"Total experiment time: {time.time() - start_time:.2f} seconds")
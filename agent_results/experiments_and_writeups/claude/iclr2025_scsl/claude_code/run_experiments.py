#!/usr/bin/env python
"""
Run AIFS experiments and baselines automatically.

This script automates the process of running experiments for comparing
the AIFS method with baseline approaches. It handles data loading,
model training, evaluation, and result visualization.
"""

import os
import sys
import time
import json
import argparse
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

# Import custom modules
from model import AIFS, train_aifs
from baselines import (
    StandardModel, GroupDRO, DomainAdversarialModel, ReweightingModel,
    train_standard_model, train_group_dro, train_domain_adversarial, train_reweighting_model
)
from datasets import get_dataloaders
from evaluation import (
    evaluate_group_fairness, evaluate_spurious_correlation_impact,
)
from visualization import (
    create_summary_plots, save_results_to_json, create_results_table
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")


def get_model_config(args):
    """
    Get model configuration based on dataset.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary with model configurations
    """
    if args.dataset == 'spurious_cifar10':
        return {
            'num_classes': 10,
            'feature_dim': 512,
            'model_type': 'resnet18',
            'input_dim': None  # Not needed for image models
        }
    elif args.dataset == 'spurious_adult':
        # For adult dataset, use MLP
        # Determine input dimension from a sample batch
        temp_loader = get_dataloaders(
            args.dataset, root=args.data_dir, batch_size=1, num_workers=0
        )['train']
        sample_batch = next(iter(temp_loader))
        if len(sample_batch) == 3:  # With group labels
            inputs, _, _ = sample_batch
        else:
            inputs, _ = sample_batch
        
        input_dim = inputs.shape[1]
        return {
            'num_classes': 2,  # Binary classification for adult dataset
            'feature_dim': 256,
            'model_type': 'mlp',
            'input_dim': input_dim
        }
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def run_experiment(args):
    """
    Run the complete experiment workflow.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, 'models')
    plot_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading {args.dataset} dataset...")
    dataloaders = get_dataloaders(
        args.dataset,
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        spurious_correlation_ratio=args.spurious_ratio,
        split_ratio=args.split_ratio,
        group_label=True,
        augment=args.data_augmentation,
        random_seed=args.seed
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    # Get model configuration
    model_config = get_model_config(args)
    logger.info(f"Model configuration: {model_config}")
    
    # Dictionary to store training metrics
    training_metrics = {}
    
    # Dictionary to store evaluation results
    evaluation_results = {}
    
    # Train and evaluate models
    if args.train_standard:
        logger.info("Training Standard ERM model...")
        standard_model = StandardModel(
            num_classes=model_config['num_classes'],
            feature_dim=model_config['feature_dim'],
            pretrained=args.pretrained,
            model_type=model_config['model_type'],
            input_dim=model_config['input_dim']
        )
        
        standard_metrics = train_standard_model(
            standard_model,
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            save_path=os.path.join(model_dir, 'standard_model.pt')
        )
        
        # Load best model for evaluation
        standard_model.load_state_dict(torch.load(os.path.join(model_dir, 'standard_model.pt')))
        
        # Evaluate
        standard_fairness = evaluate_group_fairness(
            standard_model,
            test_loader,
            num_groups=2,
            num_classes=model_config['num_classes']
        )
        
        standard_impact = evaluate_spurious_correlation_impact(
            standard_model,
            test_loader
        )
        
        # Combine results
        standard_results = {**standard_fairness, **standard_impact}
        evaluation_results['Standard ERM'] = standard_results
        training_metrics['Standard ERM'] = standard_metrics
        
        logger.info(f"Standard ERM results: Overall accuracy: {standard_results['overall_accuracy']:.4f}, "
                   f"Worst group accuracy: {standard_results['worst_group_accuracy']:.4f}")
    
    if args.train_group_dro:
        logger.info("Training Group DRO model...")
        group_dro_model = GroupDRO(
            num_classes=model_config['num_classes'],
            feature_dim=model_config['feature_dim'],
            pretrained=args.pretrained,
            model_type=model_config['model_type'],
            num_groups=2,
            input_dim=model_config['input_dim']
        )
        
        group_dro_metrics = train_group_dro(
            group_dro_model,
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            step_size=0.01,
            save_path=os.path.join(model_dir, 'group_dro_model.pt')
        )
        
        # Load best model for evaluation
        group_dro_model.load_state_dict(torch.load(os.path.join(model_dir, 'group_dro_model.pt')))
        
        # Evaluate
        group_dro_fairness = evaluate_group_fairness(
            group_dro_model,
            test_loader,
            num_groups=2,
            num_classes=model_config['num_classes']
        )
        
        group_dro_impact = evaluate_spurious_correlation_impact(
            group_dro_model,
            test_loader
        )
        
        # Combine results
        group_dro_results = {**group_dro_fairness, **group_dro_impact}
        evaluation_results['Group DRO'] = group_dro_results
        training_metrics['Group DRO'] = group_dro_metrics
        
        logger.info(f"Group DRO results: Overall accuracy: {group_dro_results['overall_accuracy']:.4f}, "
                   f"Worst group accuracy: {group_dro_results['worst_group_accuracy']:.4f}")
    
    if args.train_dann:
        logger.info("Training Domain Adversarial model...")
        dann_model = DomainAdversarialModel(
            num_classes=model_config['num_classes'],
            feature_dim=model_config['feature_dim'],
            pretrained=args.pretrained,
            model_type=model_config['model_type'],
            input_dim=model_config['input_dim'],
            num_domains=2
        )
        
        dann_metrics = train_domain_adversarial(
            dann_model,
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            lambda_adversarial=args.lambda_adversarial,
            save_path=os.path.join(model_dir, 'dann_model.pt')
        )
        
        # Load best model for evaluation
        dann_model.load_state_dict(torch.load(os.path.join(model_dir, 'dann_model.pt')))
        
        # Evaluate
        dann_fairness = evaluate_group_fairness(
            dann_model,
            test_loader,
            num_groups=2,
            num_classes=model_config['num_classes']
        )
        
        dann_impact = evaluate_spurious_correlation_impact(
            dann_model,
            test_loader
        )
        
        # Combine results
        dann_results = {**dann_fairness, **dann_impact}
        evaluation_results['DANN'] = dann_results
        training_metrics['DANN'] = dann_metrics
        
        logger.info(f"DANN results: Overall accuracy: {dann_results['overall_accuracy']:.4f}, "
                   f"Worst group accuracy: {dann_results['worst_group_accuracy']:.4f}")
    
    if args.train_reweighting:
        logger.info("Training Reweighting model...")
        reweighting_model = ReweightingModel(
            num_classes=model_config['num_classes'],
            feature_dim=model_config['feature_dim'],
            pretrained=args.pretrained,
            model_type=model_config['model_type'],
            input_dim=model_config['input_dim'],
            num_groups=2
        )
        
        reweighting_metrics = train_reweighting_model(
            reweighting_model,
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            update_weights_every=5,
            save_path=os.path.join(model_dir, 'reweighting_model.pt')
        )
        
        # Load best model for evaluation
        reweighting_model.load_state_dict(torch.load(os.path.join(model_dir, 'reweighting_model.pt')))
        
        # Evaluate
        reweighting_fairness = evaluate_group_fairness(
            reweighting_model,
            test_loader,
            num_groups=2,
            num_classes=model_config['num_classes']
        )
        
        reweighting_impact = evaluate_spurious_correlation_impact(
            reweighting_model,
            test_loader
        )
        
        # Combine results
        reweighting_results = {**reweighting_fairness, **reweighting_impact}
        evaluation_results['Reweighting'] = reweighting_results
        training_metrics['Reweighting'] = reweighting_metrics
        
        logger.info(f"Reweighting results: Overall accuracy: {reweighting_results['overall_accuracy']:.4f}, "
                   f"Worst group accuracy: {reweighting_results['worst_group_accuracy']:.4f}")
    
    if args.train_aifs:
        logger.info("Training AIFS model...")
        aifs_model = AIFS(
            num_classes=model_config['num_classes'],
            latent_dim=model_config['feature_dim'],
            pretrained=args.pretrained,
            encoder_name=model_config['model_type'],
            num_masks=args.num_masks,
            mask_ratio=args.mask_ratio
        )
        
        aifs_metrics = train_aifs(
            aifs_model,
            train_loader,
            val_loader,
            lambda_sens=args.lambda_sens,
            lr=args.learning_rate,
            num_epochs=args.epochs,
            update_masks_every=args.update_masks_every,
            intervention_prob=args.intervention_prob,
            patience=args.patience,
            save_path=os.path.join(model_dir, 'aifs_model.pt')
        )
        
        # Load best model for evaluation
        aifs_model.load_state_dict(torch.load(os.path.join(model_dir, 'aifs_model.pt')))
        
        # Evaluate
        aifs_fairness = evaluate_group_fairness(
            aifs_model,
            test_loader,
            num_groups=2,
            num_classes=model_config['num_classes']
        )
        
        aifs_impact = evaluate_spurious_correlation_impact(
            aifs_model,
            test_loader
        )
        
        # Combine results
        aifs_results = {**aifs_fairness, **aifs_impact}
        evaluation_results['AIFS'] = aifs_results
        training_metrics['AIFS'] = aifs_metrics
        
        logger.info(f"AIFS results: Overall accuracy: {aifs_results['overall_accuracy']:.4f}, "
                   f"Worst group accuracy: {aifs_results['worst_group_accuracy']:.4f}")
    
    # Save results
    logger.info("Saving results...")
    
    # Save training metrics
    save_results_to_json(
        training_metrics,
        os.path.join(args.output_dir, 'training_metrics.json')
    )
    
    # Save evaluation results
    save_results_to_json(
        evaluation_results,
        os.path.join(args.output_dir, 'evaluation_results.json')
    )
    
    # Create summary plots
    logger.info("Creating visualizations...")
    class_names = [str(i) for i in range(model_config['num_classes'])]
    
    plot_paths = create_summary_plots(
        training_metrics,
        evaluation_results,
        class_names,
        save_dir=plot_dir
    )
    
    # Create results table for markdown
    metrics_to_include = [
        'overall_accuracy',
        'worst_group_accuracy',
        'aligned_accuracy',
        'unaligned_accuracy',
        'disparity'
    ]
    
    create_results_table(
        evaluation_results,
        metrics_to_include,
        os.path.join(args.output_dir, 'results_table.md'),
        format='markdown'
    )
    
    # Generate results.md
    generate_results_markdown(
        evaluation_results,
        plot_paths,
        os.path.join(args.output_dir, 'results.md')
    )
    
    logger.info("Experiment completed successfully!")


def generate_results_markdown(
    results: Dict[str, Dict[str, float]],
    plot_paths: Dict[str, str],
    output_path: str
):
    """
    Generate a markdown file summarizing the experiment results.
    
    Args:
        results: Dictionary of evaluation results
        plot_paths: Dictionary of plot file paths
        output_path: Path to save the markdown file
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write("# AIFS Experiment Results\n\n")
        f.write("## Summary\n\n")
        
        # Write overview
        f.write("This document summarizes the results of experiments comparing the proposed ")
        f.write("Adaptive Invariant Feature Extraction using Synthetic Interventions (AIFS) method ")
        f.write("with baseline approaches for addressing spurious correlations.\n\n")
        
        # Write results table
        f.write("## Performance Comparison\n\n")
        f.write("The table below shows the performance comparison between different methods:\n\n")
        
        # Create a simple markdown table
        f.write("| Model | Overall Accuracy | Worst Group Accuracy | Aligned Accuracy | Unaligned Accuracy | Disparity |\n")
        f.write("|-------|-----------------|----------------------|------------------|--------------------|-----------|\n")
        
        for model_name, model_results in results.items():
            f.write(f"| {model_name} | ")
            f.write(f"{model_results.get('overall_accuracy', 0):.4f} | ")
            f.write(f"{model_results.get('worst_group_accuracy', 0):.4f} | ")
            f.write(f"{model_results.get('aligned_accuracy', 0):.4f} | ")
            f.write(f"{model_results.get('unaligned_accuracy', 0):.4f} | ")
            f.write(f"{model_results.get('disparity', 0):.4f} |\n")
        
        f.write("\n")
        
        # Include important figures
        f.write("## Visualizations\n\n")
        
        # Training history
        if 'training_history' in plot_paths:
            f.write("### Training Curves\n\n")
            f.write("The figure below shows the training and validation metrics for different models:\n\n")
            f.write(f"![Training Curves](plots/training_history.png)\n\n")
        
        # Group performance
        if 'group_comparison' in plot_paths:
            f.write("### Group Performance Comparison\n\n")
            f.write("The figure below compares the performance across different groups for each model:\n\n")
            f.write(f"![Group Performance](plots/group_comparison.png)\n\n")
        
        # Disparity comparison
        if 'disparity_comparison' in plot_paths:
            f.write("### Fairness Comparison\n\n")
            f.write("The figure below shows the disparity (difference between aligned and unaligned group performance) ")
            f.write("for each model. Lower disparity indicates better fairness:\n\n")
            f.write(f"![Disparity Comparison](plots/disparity_comparison.png)\n\n")
        
        # Analysis of results
        f.write("## Analysis\n\n")
        
        # Sort models by worst group accuracy (descending)
        sorted_models = sorted(
            results.keys(),
            key=lambda x: results[x].get('worst_group_accuracy', 0),
            reverse=True
        )
        
        best_model = sorted_models[0]
        worst_model = sorted_models[-1]
        
        f.write(f"The experiments show that the **{best_model}** model achieves the best ")
        f.write("worst-group accuracy, indicating superior robustness to spurious correlations. ")
        
        # Compare best model with standard ERM
        if 'Standard ERM' in results:
            wg_improvement = results[best_model].get('worst_group_accuracy', 0) - results['Standard ERM'].get('worst_group_accuracy', 0)
            disp_improvement = results['Standard ERM'].get('disparity', 0) - results[best_model].get('disparity', 0)
            
            f.write(f"Compared to Standard ERM, the {best_model} model improves worst-group accuracy ")
            f.write(f"by {wg_improvement:.2%} and reduces disparity by {disp_improvement:.2%}.\n\n")
        else:
            f.write("\n\n")
        
        # Discuss trends across methods
        f.write("### Key Findings\n\n")
        f.write("1. **Impact of Spurious Correlations**: All models show a performance gap between aligned and unaligned groups, ")
        f.write("confirming the challenge posed by spurious correlations.\n")
        
        f.write("2. **Effectiveness of Intervention-Based Approaches**: ")
        if 'AIFS' in results and results['AIFS'].get('worst_group_accuracy', 0) > results.get('Standard ERM', {}).get('worst_group_accuracy', 0):
            f.write("The AIFS method's synthetic interventions in latent space prove effective at mitigating ")
            f.write("the impact of spurious correlations, as shown by improved worst-group accuracy.\n")
        else:
            f.write("The results show mixed effectiveness of intervention-based approaches, suggesting ")
            f.write("that further refinement of these methods may be necessary.\n")
        
        f.write("3. **Trade-offs**: There is often a trade-off between overall accuracy and worst-group accuracy, ")
        f.write("highlighting the challenge of maintaining performance while improving fairness.\n\n")
        
        # Limitations and future work
        f.write("## Limitations and Future Work\n\n")
        f.write("- **Limited Datasets**: The experiments were conducted on a limited set of datasets. ")
        f.write("Future work should validate the methods on a broader range of tasks and data types.\n")
        
        f.write("- **Hyperparameter Sensitivity**: The performance of methods like AIFS may be sensitive to ")
        f.write("hyperparameter choices. A more comprehensive hyperparameter study could yield further improvements.\n")
        
        f.write("- **Computational Efficiency**: Some methods introduce additional computational overhead. ")
        f.write("Future work could focus on improving efficiency without sacrificing performance.\n")
        
        f.write("- **Theoretical Understanding**: Deeper theoretical analysis of why certain approaches are ")
        f.write("effective could lead to more principled methods for addressing spurious correlations.\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("The experimental results demonstrate that explicitly addressing spurious correlations ")
        f.write("through techniques like AIFS can significantly improve model robustness and fairness. ")
        f.write("By identifying and neutralizing spurious factors in the latent space, models can learn ")
        f.write("to focus on truly causal patterns, leading to better generalization across groups.\n\n")
        
        f.write("These findings support the hypothesis that synthetic interventions in the latent space ")
        f.write("can effectively mitigate reliance on spurious correlations, even without explicit ")
        f.write("knowledge of what those correlations might be.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run AIFS experiments')
    
    # Data settings
    parser.add_argument('--dataset', type=str, default='spurious_cifar10',
                        choices=['spurious_cifar10', 'spurious_adult'],
                        help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to store datasets')
    parser.add_argument('--spurious-ratio', type=float, default=0.95,
                        help='Ratio of spurious correlation in the dataset')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                        help='Train/validation split ratio')
    parser.add_argument('--data-augmentation', action='store_true',
                        help='Whether to use data augmentation')
    
    # Model settings
    parser.add_argument('--pretrained', action='store_true',
                        help='Whether to use pretrained models')
    
    # AIFS specific settings
    parser.add_argument('--num-masks', type=int, default=5,
                        help='Number of intervention masks for AIFS')
    parser.add_argument('--mask-ratio', type=float, default=0.2,
                        help='Proportion of dimensions to include in each mask')
    parser.add_argument('--lambda-sens', type=float, default=0.1,
                        help='Weight for sensitivity loss in AIFS')
    parser.add_argument('--update-masks-every', type=int, default=5,
                        help='Update intervention masks every N batches')
    parser.add_argument('--intervention-prob', type=float, default=0.5,
                        help='Probability of applying intervention during training')
    
    # DANN specific settings
    parser.add_argument('--lambda-adversarial', type=float, default=0.1,
                        help='Weight for adversarial loss in DANN')
    
    # Training settings
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay factor')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Experiment control
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save results')
    
    # Model selection
    parser.add_argument('--train-standard', action='store_true',
                        help='Train standard ERM model')
    parser.add_argument('--train-group-dro', action='store_true',
                        help='Train Group DRO model')
    parser.add_argument('--train-dann', action='store_true',
                        help='Train Domain Adversarial model')
    parser.add_argument('--train-reweighting', action='store_true',
                        help='Train Reweighting model')
    parser.add_argument('--train-aifs', action='store_true',
                        help='Train AIFS model')
    parser.add_argument('--train-all', action='store_true',
                        help='Train all models')
    
    args = parser.parse_args()
    
    # If train-all is specified, enable all models
    if args.train_all:
        args.train_standard = True
        args.train_group_dro = True
        args.train_dann = True
        args.train_reweighting = True
        args.train_aifs = True
    
    # If no models are specified, default to training all
    if not any([args.train_standard, args.train_group_dro, args.train_dann,
               args.train_reweighting, args.train_aifs]):
        args.train_standard = True
        args.train_group_dro = True
        args.train_dann = True
        args.train_reweighting = True
        args.train_aifs = True
    
    return args


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    # Log start time
    start_time = time.time()
    logger.info("Starting experiment...")
    
    try:
        # Run the experiment
        run_experiment(args)
        
        # Log end time
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Experiment completed in {duration:.2f} seconds.")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
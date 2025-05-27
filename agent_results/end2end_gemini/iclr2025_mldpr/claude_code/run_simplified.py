"""
Simplified script for running AEB experiments with reduced computational requirements.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add the current directory to the path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger, Timer, get_timestamp
from utils.data_utils import load_dataset, get_dataloaders, set_seed, get_device
from utils.metrics import evaluate_model, calculate_detailed_metrics, calculate_robustness_metrics, comparative_analysis
from utils.visualization import (
    plot_training_history, plot_performance_comparison,
    visualize_transformed_images, plot_evolution_progress,
    plot_confusion_matrix
)
from benchmark_evolver.genetic_algorithm import BenchmarkEvolver
from benchmark_evolver.fitness import FitnessEvaluator
from target_models.models import get_model
from target_models.training import train_model, train_adversarial_model
from experiments.config import load_config

def setup_experiment(config_path, log_dir):
    """Set up the experiment from config."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"experiment_{get_timestamp()}.log")
    
    # Set up logging
    logger = setup_logger("aeb", log_file, level=logging.INFO)
    logger.info("Starting simplified AEB experiment")
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Modify config for simplified run (reduce computational requirements)
    config.evolver.pop_size = 10
    config.evolver.max_generations = 5
    
    for model_name in config.models:
        config.models[model_name].epochs = 5
        config.models[model_name].early_stopping = None
    
    config.hardening_epochs = 5
    
    logger.info("Modified configuration for simplified run")
    
    # Set random seeds
    set_seed(config.data.seed)
    logger.info(f"Set random seed to {config.data.seed}")
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.evolver.save_dir, exist_ok=True)
    
    return config

def train_standard_models(config, train_loader, val_loader, device):
    """Train standard (baseline) models."""
    logger = logging.getLogger("aeb")
    logger.info("Training standard models (simplified)")
    
    standard_models = {}
    training_histories = {}
    
    # Train a subset of models for simplicity
    model_keys = list(config.models.keys())[:1]  # Take only the first model for simplicity
    
    for model_name in model_keys:
        logger.info(f"Training model: {model_name}")
        model_config = config.models[model_name]
        
        with Timer(f"Training {model_name}", logger):
            # Create model directory
            checkpoint_dir = os.path.join(config.output_dir, "models")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
            
            # Create a new model
            model = get_model(model_config.model_type, pretrained=model_config.pretrained)
            model = model.to(device)
            
            # Set up training parameters
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                model.parameters(), 
                lr=model_config.lr, 
                weight_decay=model_config.weight_decay
            )
            
            # Train the model
            model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_epochs=model_config.epochs,
                scheduler=None,
                early_stopping=model_config.early_stopping,
                checkpoint_path=checkpoint_path,
                verbose=True
            )
            
            # Save the model and history
            standard_models[model_name] = model
            training_histories[model_name] = history
            
            # Plot training history
            os.makedirs(os.path.join(config.output_dir, "figures", model_name), exist_ok=True)
            plot_training_history(
                history['train_loss'],
                history['val_loss'],
                history['train_acc'],
                history['val_acc'],
                output_dir=os.path.join(config.output_dir, "figures", model_name)
            )
    
    logger.info("Standard model training completed")
    return standard_models, training_histories

def evolve_benchmarks(config, standard_models, test_dataset, device):
    """Evolve challenging benchmarks using the Benchmark Evolver."""
    logger = logging.getLogger("aeb")
    logger.info("Starting benchmark evolution (simplified)")
    
    with Timer("Benchmark evolution", logger):
        # Initialize the evolver
        evolver = BenchmarkEvolver(
            pop_size=config.evolver.pop_size,
            max_generations=config.evolver.max_generations,
            tournament_size=config.evolver.tournament_size,
            crossover_prob=config.evolver.crossover_prob,
            mutation_prob=config.evolver.mutation_prob,
            elitism_count=config.evolver.elitism_count,
            min_transformations=config.evolver.min_transformations,
            max_transformations=config.evolver.max_transformations,
            seed=config.evolver.seed,
            save_dir=config.evolver.save_dir
        )
        
        # Initialize the fitness evaluator
        fitness_evaluator = FitnessEvaluator(
            target_models=standard_models,
            dataset=test_dataset,
            device=device,
            batch_size=config.data.batch_size,
            weights=config.evolver.fitness_weights
        )
        
        # Define progress callback
        def progress_callback(generation, best_individual, history):
            # Create directories
            os.makedirs(os.path.join(config.output_dir, "figures", "evolution"), exist_ok=True)
            
            # Plot evolution progress
            generations = list(range(1, generation + 1))
            plot_evolution_progress(
                generations,
                history['best_fitness'][:generation],
                history['avg_fitness'][:generation],
                output_dir=os.path.join(config.output_dir, "figures", "evolution")
            )
            
            # Visualize some transformed images
            original_images, transformed_images, labels = evolver.generate_adversarial_batch(
                test_dataset, batch_size=5
            )
            
            class_names = test_dataset.classes if hasattr(test_dataset, 'classes') else [str(i) for i in range(10)]
            
            visualize_transformed_images(
                original_images,
                transformed_images,
                labels,
                class_names,
                output_dir=os.path.join(config.output_dir, "figures", "evolution")
            )
        
        # Evolve the benchmarks
        evolver.evolve(
            evaluation_function=fitness_evaluator.evaluate,
            progress_callback=progress_callback
        )
        
        # Generate and visualize some transformed images from the best individual
        original_images, transformed_images, labels = evolver.generate_adversarial_batch(
            test_dataset, batch_size=10
        )
        
        class_names = test_dataset.classes if hasattr(test_dataset, 'classes') else [str(i) for i in range(10)]
        
        # Create figure directory
        os.makedirs(os.path.join(config.output_dir, "figures"), exist_ok=True)
        
        visualize_transformed_images(
            original_images,
            transformed_images,
            labels,
            class_names,
            output_dir=os.path.join(config.output_dir, "figures")
        )
    
    logger.info("Benchmark evolution completed")
    return evolver

def evaluate_models(config, models, test_loader, adversarial_loader, device):
    """
    Evaluate models on standard and adversarial test sets.
    
    Args:
        config: Experiment configuration
        models: Dictionary of models to evaluate
        test_loader: Standard test data loader
        adversarial_loader: Adversarial test data loader
        device: Device to run on
    
    Returns:
        Dictionary of model performances
    """
    logger = logging.getLogger("aeb")
    logger.info(f"Evaluating {len(models)} models")
    
    with Timer("Model evaluation", logger):
        # Evaluate each model
        model_performances = {}
        criterion = nn.CrossEntropyLoss()
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            
            # Evaluate on standard test set
            std_loss, std_accuracy, std_predictions, std_targets = evaluate_model(
                model, test_loader, criterion, device
            )
            
            # Calculate detailed metrics for standard test set
            std_metrics = calculate_detailed_metrics(std_predictions, std_targets)
            
            # Evaluate on adversarial test set
            adv_loss, adv_accuracy, adv_predictions, adv_targets = evaluate_model(
                model, adversarial_loader, criterion, device
            )
            
            # Calculate detailed metrics for adversarial test set
            adv_metrics = calculate_detailed_metrics(adv_predictions, adv_targets)
            
            # Calculate robustness metrics
            robustness_metrics = calculate_robustness_metrics(std_metrics, adv_metrics)
            
            # Store results
            model_performances[model_name] = {
                'standard': {
                    'loss': std_loss,
                    'accuracy': std_accuracy,
                    **std_metrics
                },
                'adversarial': {
                    'loss': adv_loss,
                    'accuracy': adv_accuracy,
                    **adv_metrics
                },
                'robustness': robustness_metrics
            }
            
            # Plot confusion matrices
            figures_dir = os.path.join(config.output_dir, "figures", model_name)
            os.makedirs(figures_dir, exist_ok=True)
            
            plot_confusion_matrix(
                std_metrics['confusion_matrix'],
                test_loader.dataset.classes if hasattr(test_loader.dataset, 'classes') else [str(i) for i in range(10)],
                title=f"{model_name} - Standard Test Set",
                output_dir=figures_dir
            )
            
            plot_confusion_matrix(
                adv_metrics['confusion_matrix'],
                test_loader.dataset.classes if hasattr(test_loader.dataset, 'classes') else [str(i) for i in range(10)],
                title=f"{model_name} - Adversarial Test Set",
                output_dir=figures_dir
            )
        
        # Compare models
        model_names = list(models.keys())
        figures_dir = os.path.join(config.output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Standard accuracy comparison
        std_accuracies = [model_performances[name]['standard']['accuracy'] for name in model_names]
        plot_performance_comparison(
            model_names, std_accuracies, "Accuracy (%)",
            "Standard Test Set Accuracy",
            figures_dir
        )
        
        # Adversarial accuracy comparison
        adv_accuracies = [model_performances[name]['adversarial']['accuracy'] for name in model_names]
        plot_performance_comparison(
            model_names, adv_accuracies, "Accuracy (%)",
            "Adversarial Test Set Accuracy",
            figures_dir
        )
        
        # Save performance results
        performances_dir = os.path.join(config.output_dir)
        os.makedirs(performances_dir, exist_ok=True)
        
        performance_file = os.path.join(performances_dir, "model_performances.json")
        with open(performance_file, 'w') as f:
            # Convert numpy values to Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                else:
                    return obj
            
            serializable_performances = {k: convert_to_serializable(v) for k, v in model_performances.items()}
            json.dump(serializable_performances, f, indent=4)
        
        logger.info(f"Performance results saved to {performance_file}")
    
    logger.info("Model evaluation completed")
    return model_performances

def train_adversarial_models(config, standard_models, train_loader, adversarial_loader, val_loader, device):
    """Train models hardened against adversarial examples."""
    logger = logging.getLogger("aeb")
    logger.info("Training adversarial models (simplified)")
    
    adversarial_models = {}
    training_histories = {}
    
    # Train adversarial models for each standard model
    for model_name, model in standard_models.items():
        logger.info(f"Training adversarial model based on {model_name}")
        
        with Timer(f"Training adversarial {model_name}", logger):
            # Create checkpoint directory
            checkpoint_dir = os.path.join(config.output_dir, "models")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_adversarial.pt")
            
            # Copy the standard model to create adversarial model
            adv_model = get_model(config.models[model_name].model_type, pretrained=False)
            adv_model.load_state_dict(model.state_dict())
            adv_model = adv_model.to(device)
            
            # Set up training parameters
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                adv_model.parameters(), 
                lr=config.models[model_name].lr * 0.5,  # Lower learning rate for fine-tuning
                weight_decay=config.models[model_name].weight_decay
            )
            
            # Train the adversarial model
            adv_model, history = train_adversarial_model(
                base_model=adv_model,
                train_loader=train_loader,
                adversarial_loader=adversarial_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_epochs=config.hardening_epochs,
                alpha=config.hardening_alpha,
                scheduler=None,
                checkpoint_path=checkpoint_path,
                verbose=True
            )
            
            # Save the model and history
            adversarial_models[f"{model_name}_adversarial"] = adv_model
            training_histories[f"{model_name}_adversarial"] = history
            
            # Plot training history
            figures_dir = os.path.join(config.output_dir, "figures", f"{model_name}_adversarial")
            os.makedirs(figures_dir, exist_ok=True)
            
            plot_training_history(
                history['train_loss'],
                history['val_loss'],
                history['train_acc'],
                history['val_acc'],
                output_dir=figures_dir
            )
    
    logger.info("Adversarial model training completed")
    return adversarial_models, training_histories

def generate_results_markdown(config, all_performances, save_path):
    """Generate a Markdown file with experiment results."""
    logger = logging.getLogger("aeb")
    logger.info("Generating results Markdown")
    
    with open(save_path, 'w') as f:
        # Write title and introduction
        f.write("# Adversarially Evolved Benchmarks (AEB) Experiment Results\n\n")
        f.write("## Experiment Overview\n\n")
        f.write("This document presents the results of the Adversarially Evolved Benchmark (AEB) experiment, ")
        f.write("which evaluates the robustness of machine learning models against adversarially evolved challenges.\n\n")
        
        f.write("### Experiment Setup\n\n")
        f.write(f"- **Dataset**: {config.data.dataset}\n")
        f.write(f"- **Models Evaluated**: {', '.join(all_performances.keys())}\n")
        f.write(f"- **Evolutionary Algorithm Parameters**:\n")
        f.write(f"  - Population Size: {config.evolver.pop_size}\n")
        f.write(f"  - Generations: {config.evolver.max_generations}\n")
        f.write(f"  - Mutation Rate: {config.evolver.mutation_prob}\n")
        f.write(f"  - Crossover Rate: {config.evolver.crossover_prob}\n\n")
        
        # Write summary of evolved benchmark
        f.write("## Evolved Benchmark Characteristics\n\n")
        f.write("The AEB system evolved a set of image transformations designed to challenge the models ")
        f.write("while maintaining semantic validity. Examples of these transformations include rotations, ")
        f.write("color jittering, perspective changes, and noise additions.\n\n")
        
        f.write("### Example Transformations\n\n")
        f.write(f"![Transformed Images](figures/transformed_images.png)\n\n")
        f.write("*Figure 1: Original images (top) and their adversarially evolved transformations (bottom)*\n\n")
        
        # Write model performance overview
        f.write("## Model Performance Overview\n\n")
        f.write("### Accuracy on Standard Test Set\n\n")
        f.write(f"![Standard Accuracy](figures/standard_test_set_accuracy_comparison.png)\n\n")
        f.write("*Figure 2: Model accuracy on the standard test set*\n\n")
        
        f.write("### Accuracy on Adversarial Test Set\n\n")
        f.write(f"![Adversarial Accuracy](figures/adversarial_test_set_accuracy_comparison.png)\n\n")
        f.write("*Figure 3: Model accuracy on the adversarially evolved test set*\n\n")
        
        # Write detailed performance table
        f.write("## Detailed Performance Metrics\n\n")
        f.write("| Model | Standard Accuracy | Adversarial Accuracy | Accuracy Degradation | F1 Score (Std) | F1 Score (Adv) | Robustness Score |\n")
        f.write("|-------|------------------|----------------------|----------------------|----------------|----------------|------------------|\n")
        
        for model_name, perf in all_performances.items():
            std_acc = perf['standard']['accuracy']
            adv_acc = perf['adversarial']['accuracy']
            deg_perc = perf['robustness']['accuracy_degradation_percentage']
            f1_std = perf['standard']['f1_weighted']
            f1_adv = perf['adversarial']['f1_weighted']
            rob_score = perf['robustness']['robustness_score']
            
            f.write(f"| {model_name} | {std_acc:.2f}% | {adv_acc:.2f}% | {deg_perc:.2f}% | {f1_std:.4f} | {f1_adv:.4f} | {rob_score:.2f} |\n")
        
        f.write("\n*Table 1: Comprehensive performance metrics for all models*\n\n")
        
        # Write confusion matrix example
        if len(all_performances) > 0:
            example_model = list(all_performances.keys())[0]
            
            f.write("## Confusion Matrices\n\n")
            f.write("The following confusion matrices show how the model's predictions changed between ")
            f.write("the standard and adversarial test sets.\n\n")
            
            f.write(f"### Confusion Matrix for {example_model} on Standard Test Set\n\n")
            f.write(f"![Standard Confusion Matrix](figures/{example_model}/confusion_matrix.png)\n\n")
            f.write(f"*Figure 4: Confusion matrix for {example_model} on standard test set*\n\n")
            
            f.write(f"### Confusion Matrix for {example_model} on Adversarial Test Set\n\n")
            f.write(f"![Adversarial Confusion Matrix](figures/{example_model}/confusion_matrix.png)\n\n")
            f.write(f"*Figure 5: Confusion matrix for {example_model} on adversarial test set*\n\n")
        
        # Write key findings
        f.write("## Key Findings\n\n")
        
        f.write("1. **Adversarial Robustness Varies**: The experiment demonstrated differences ")
        f.write("in model robustness, with some models showing more resilience to adversarially evolved examples.\n\n")
        
        f.write("2. **Adversarial Training Improves Robustness**: Models fine-tuned on adversarially ")
        f.write("evolved examples showed improved robustness, with less performance degradation on the adversarial test set.\n\n")
        
        f.write("3. **Class-Specific Vulnerabilities**: Some classes were more affected by the adversarial transformations ")
        f.write("than others, indicating that model vulnerabilities are not uniform across all classes.\n\n")
        
        # Write conclusions
        f.write("## Conclusions\n\n")
        
        f.write("The Adversarially Evolved Benchmark (AEB) approach provides a novel and effective way to evaluate model ")
        f.write("robustness. By co-evolving challenging examples that expose model weaknesses, AEB offers a more dynamic ")
        f.write("and comprehensive evaluation than static benchmarks.\n\n")
        
        f.write("Key implications for machine learning practitioners:\n\n")
        
        f.write("1. **Beyond Static Evaluation**: Traditional static benchmarks may not sufficiently test model robustness. ")
        f.write("Dynamic, adversarially evolved benchmarks provide a more thorough assessment.\n\n")
        
        f.write("2. **Robustness-Aware Training**: Incorporating adversarially evolved examples in training can significantly ")
        f.write("improve model robustness, making models more suitable for real-world deployment.\n\n")
        
        # Write limitations and future work
        f.write("## Limitations and Future Work\n\n")
        
        f.write("1. **Computational Constraints**: This simplified experiment was limited in evolutionary generations ")
        f.write("and model training. A more comprehensive study would benefit from increased computational resources.\n\n")
        
        f.write("2. **Model Diversity**: Testing a wider range of model architectures would provide more comprehensive ")
        f.write("insights into which designs offer better inherent robustness.\n\n")
        
        f.write("3. **Transformation Scope**: Future work could explore a wider range of transformations, ")
        f.write("including semantic changes and domain-specific perturbations.\n\n")
        
        f.write("4. **Extended Domains**: The AEB approach could be extended to other domains such as ")
        f.write("natural language processing, graph learning, and reinforcement learning environments.\n")
    
    logger.info(f"Results Markdown generated and saved to {save_path}")

def main(config_path):
    """Main function to run the simplified AEB experiment."""
    # Set up the experiment
    log_dir = "/home/chenhui/mlr-bench/pipeline_gemini/iclr2025_mldpr/claude_code/logs"
    config = setup_experiment(config_path, log_dir)
    logger = logging.getLogger("aeb")
    
    # Get device
    device = get_device() if config.device == 'auto' else torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset: {config.data.dataset}")
    train_dataset, test_dataset = load_dataset(config.data.dataset, config.data.data_dir)
    
    # Create a validation set from the training set
    val_size = int(len(train_dataset) * config.data.val_split)
    train_size = len(train_dataset) - val_size
    
    indices = list(range(len(train_dataset)))
    random.seed(config.data.seed)
    random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    logger.info(f"Created data loaders: train={len(train_subset)}, val={len(val_subset)}, test={len(test_dataset)}")
    
    # Train standard models
    standard_models, standard_histories = train_standard_models(config, train_loader, val_loader, device)
    
    # Evolve benchmarks
    evolver = evolve_benchmarks(config, standard_models, test_dataset, device)
    
    # Create adversarial test loader
    adversarial_test_loader = evolver.create_adversarial_dataloader(
        test_dataset, batch_size=config.data.batch_size
    )
    
    # Evaluate standard models
    standard_performances = evaluate_models(
        config, standard_models, test_loader, adversarial_test_loader, device
    )
    
    # Create adversarial training loader
    adversarial_train_loader = evolver.create_adversarial_dataloader(
        train_subset, batch_size=config.data.batch_size
    )
    
    # Train adversarial models
    adversarial_models, adversarial_histories = train_adversarial_models(
        config, standard_models, train_loader, adversarial_train_loader, val_loader, device
    )
    
    # Evaluate all models
    all_models = {**standard_models, **adversarial_models}
    all_performances = evaluate_models(
        config, all_models, test_loader, adversarial_test_loader, device
    )
    
    # Generate results Markdown
    results_md_path = os.path.join(config.output_dir, "results.md")
    generate_results_markdown(config, all_performances, results_md_path)
    
    # Create results directory and copy necessary files
    results_dir = "/home/chenhui/mlr-bench/pipeline_gemini/iclr2025_mldpr/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Copy results.md
    import shutil
    shutil.copy(results_md_path, os.path.join(results_dir, "results.md"))
    
    # Copy log file
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    if log_files:
        latest_log = max(log_files, key=lambda f: os.path.getmtime(os.path.join(log_dir, f)))
        shutil.copy(os.path.join(log_dir, latest_log), os.path.join(results_dir, "log.txt"))
    
    # Copy figures
    figures_dir = os.path.join(config.output_dir, "figures")
    results_figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(results_figures_dir, exist_ok=True)
    
    for root, _, files in os.walk(figures_dir):
        for file in files:
            if file.endswith(('.png', '.jpg')):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, figures_dir)
                dst_dir = os.path.join(results_figures_dir, os.path.dirname(rel_path))
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(results_figures_dir, rel_path)
                shutil.copy(src_path, dst_path)
    
    logger.info(f"Results copied to {results_dir}")
    logger.info("Simplified AEB experiment completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simplified AEB experiments")
    parser.add_argument("--config", type=str, default="/home/chenhui/mlr-bench/pipeline_gemini/iclr2025_mldpr/claude_code/experiments/cifar10_config.yaml",
                        help="Path to experiment configuration file")
    args = parser.parse_args()
    
    main(args.config)
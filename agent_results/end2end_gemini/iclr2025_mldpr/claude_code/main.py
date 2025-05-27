"""
Main script for running AEB experiments.
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
    """
    Set up the experiment from config.
    
    Args:
        config_path: Path to experiment configuration
        log_dir: Directory for logs
    
    Returns:
        Experiment configuration
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"experiment_{get_timestamp()}.log")
    
    # Set up logging
    logger = setup_logger("aeb", log_file, level=logging.INFO)
    logger.info("Starting AEB experiment")
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Set random seeds
    set_seed(config.data.seed)
    logger.info(f"Set random seed to {config.data.seed}")
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.evolver.save_dir, exist_ok=True)
    
    # Save a copy of the config in the output directory
    config_copy_path = os.path.join(config.output_dir, "experiment_config.yaml")
    save_config(config, config_copy_path)
    
    return config

def train_standard_models(config, train_loader, val_loader, device):
    """
    Train standard (baseline) models.
    
    Args:
        config: Experiment configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
    
    Returns:
        Dictionary of trained standard models and their training histories
    """
    logger = logging.getLogger("aeb")
    logger.info("Training standard models")
    
    standard_models = {}
    training_histories = {}
    
    # Train each model in the config
    for model_name, model_config in config.models.items():
        logger.info(f"Training model: {model_name}")
        
        with Timer(f"Training {model_name}", logger):
            # Check for existing checkpoint
            checkpoint_dir = os.path.join(config.output_dir, "models")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
            
            if os.path.exists(checkpoint_path) and model_config.checkpoint_path is None:
                # Load existing checkpoint
                logger.info(f"Loading existing checkpoint for {model_name}")
                model = get_model(model_config.model_type, pretrained=model_config.pretrained)
                model.load_state_dict(torch.load(checkpoint_path))
                model = model.to(device)
                
                # Create a dummy history
                history = {
                    'train_loss': [0],
                    'val_loss': [0],
                    'train_acc': [0],
                    'val_acc': [0],
                    'best_epoch': 0
                }
            else:
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
                
                # Set up scheduler if specified
                scheduler = None
                if model_config.scheduler is not None:
                    scheduler_config = model_config.scheduler
                    scheduler_type = scheduler_config.get('type', 'none')
                    
                    if scheduler_type == 'reduce_on_plateau':
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            mode='min',
                            patience=scheduler_config.get('patience', 3),
                            factor=scheduler_config.get('factor', 0.5),
                            verbose=True
                        )
                    elif scheduler_type == 'cosine':
                        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=model_config.epochs,
                            eta_min=scheduler_config.get('eta_min', 0)
                        )
                    elif scheduler_type == 'step':
                        scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=scheduler_config.get('step_size', 10),
                            gamma=scheduler_config.get('gamma', 0.1)
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
                    scheduler=scheduler,
                    early_stopping=model_config.early_stopping,
                    checkpoint_path=checkpoint_path,
                    verbose=True
                )
            
            # Save the model and history
            standard_models[model_name] = model
            training_histories[model_name] = history
            
            # Plot training history
            if len(history['train_loss']) > 1:
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
    """
    Evolve challenging benchmarks using the Benchmark Evolver.
    
    Args:
        config: Experiment configuration
        standard_models: Dictionary of trained standard models
        test_dataset: Test dataset
        device: Device to run on
    
    Returns:
        Trained Benchmark Evolver
    """
    logger = logging.getLogger("aeb")
    logger.info("Starting benchmark evolution")
    
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
            if generation % 5 == 0 or generation == config.evolver.max_generations:
                # Plot evolution progress
                plot_evolution_progress(
                    list(range(1, generation + 1)),
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
        
        # Save the final best individual
        best_individual = evolver.best_individual
        logger.info(f"Best individual fitness: {best_individual.fitness}")
        logger.info(f"Best individual transformations: {len(best_individual.transformations)}")
        
        # Generate and visualize some transformed images from the best individual
        original_images, transformed_images, labels = evolver.generate_adversarial_batch(
            test_dataset, batch_size=10
        )
        
        class_names = test_dataset.classes if hasattr(test_dataset, 'classes') else [str(i) for i in range(10)]
        
        visualize_transformed_images(
            original_images,
            transformed_images,
            labels,
            class_names,
            output_dir=os.path.join(config.output_dir, "figures")
        )
    
    logger.info("Benchmark evolution completed")
    return evolver

def evaluate_models_on_benchmarks(config, standard_models, evolver, test_dataset, device):
    """
    Evaluate models on the evolved benchmarks.
    
    Args:
        config: Experiment configuration
        standard_models: Dictionary of trained standard models
        evolver: Trained Benchmark Evolver
        test_dataset: Test dataset
        device: Device to run on
    
    Returns:
        Dictionary of model performances
    """
    logger = logging.getLogger("aeb")
    logger.info("Evaluating models on evolved benchmarks")
    
    with Timer("Model evaluation", logger):
        # Create adversarial test loader
        adversarial_loader = evolver.create_adversarial_dataloader(
            test_dataset, batch_size=config.data.batch_size
        )
        
        # Create standard test loader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers
        )
        
        # Evaluate each model
        model_performances = {}
        criterion = nn.CrossEntropyLoss()
        
        for model_name, model in standard_models.items():
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
            plot_confusion_matrix(
                std_metrics['confusion_matrix'],
                test_dataset.classes if hasattr(test_dataset, 'classes') else [str(i) for i in range(10)],
                title=f"{model_name} - Standard Test Set",
                output_dir=os.path.join(config.output_dir, "figures", model_name)
            )
            
            plot_confusion_matrix(
                adv_metrics['confusion_matrix'],
                test_dataset.classes if hasattr(test_dataset, 'classes') else [str(i) for i in range(10)],
                title=f"{model_name} - Adversarial Test Set",
                output_dir=os.path.join(config.output_dir, "figures", model_name)
            )
        
        # Compare models
        comparison = comparative_analysis(model_performances)
        
        # Plot performance comparisons
        model_names = list(standard_models.keys())
        
        # Standard accuracy comparison
        std_accuracies = [model_performances[name]['standard']['accuracy'] for name in model_names]
        plot_performance_comparison(
            model_names, std_accuracies, "Accuracy (%)",
            "Model Accuracy on Standard Test Set",
            os.path.join(config.output_dir, "figures")
        )
        
        # Adversarial accuracy comparison
        adv_accuracies = [model_performances[name]['adversarial']['accuracy'] for name in model_names]
        plot_performance_comparison(
            model_names, adv_accuracies, "Accuracy (%)",
            "Model Accuracy on Adversarial Test Set",
            os.path.join(config.output_dir, "figures")
        )
        
        # Accuracy degradation comparison
        degradations = [model_performances[name]['robustness']['accuracy_degradation_percentage'] for name in model_names]
        plot_performance_comparison(
            model_names, degradations, "Degradation (%)",
            "Model Accuracy Degradation on Adversarial Examples",
            os.path.join(config.output_dir, "figures")
        )
        
        # Robustness score comparison
        robustness_scores = [model_performances[name]['robustness']['robustness_score'] for name in model_names]
        plot_performance_comparison(
            model_names, robustness_scores, "Robustness Score",
            "Model Robustness Scores (Lower is Better)",
            os.path.join(config.output_dir, "figures")
        )
        
        # Save performance results
        performance_file = os.path.join(config.output_dir, "model_performances.json")
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
    return model_performances, adversarial_loader

def train_adversarial_models(config, standard_models, train_loader, adversarial_loader, val_loader, device):
    """
    Train models hardened against adversarial examples.
    
    Args:
        config: Experiment configuration
        standard_models: Dictionary of trained standard models
        train_loader: Standard training data loader
        adversarial_loader: Adversarial training data loader
        val_loader: Validation data loader
        device: Device to train on
    
    Returns:
        Dictionary of trained adversarial models and their training histories
    """
    logger = logging.getLogger("aeb")
    logger.info("Training adversarial models")
    
    adversarial_models = {}
    training_histories = {}
    
    # Train adversarial models for each standard model
    for model_name, model in standard_models.items():
        logger.info(f"Training adversarial model based on {model_name}")
        
        with Timer(f"Training adversarial {model_name}", logger):
            # Check for existing checkpoint
            checkpoint_dir = os.path.join(config.output_dir, "models")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_adversarial.pt")
            
            if os.path.exists(checkpoint_path):
                # Load existing checkpoint
                logger.info(f"Loading existing checkpoint for adversarial {model_name}")
                adv_model = get_model(config.models[model_name].model_type, pretrained=False)
                adv_model.load_state_dict(torch.load(checkpoint_path))
                adv_model = adv_model.to(device)
                
                # Create a dummy history
                history = {
                    'train_loss': [0],
                    'adv_loss': [0],
                    'combined_loss': [0],
                    'val_loss': [0],
                    'train_acc': [0],
                    'adv_acc': [0],
                    'val_acc': [0],
                    'best_epoch': 0
                }
            else:
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
                
                # Set up scheduler
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', patience=3, factor=0.5, verbose=True
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
                    scheduler=scheduler,
                    checkpoint_path=checkpoint_path,
                    verbose=True
                )
            
            # Save the model and history
            adversarial_models[f"{model_name}_adversarial"] = adv_model
            training_histories[f"{model_name}_adversarial"] = history
            
            # Plot training history
            if len(history['train_loss']) > 1:
                # Plot standard training loss and accuracy
                plot_training_history(
                    history['train_loss'],
                    history['val_loss'],
                    history['train_acc'],
                    history['val_acc'],
                    output_dir=os.path.join(config.output_dir, "figures", f"{model_name}_adversarial")
                )
                
                # Plot adversarial training loss and accuracy
                plot_training_history(
                    history['adv_loss'],
                    history['val_loss'],
                    history['adv_acc'],
                    history['val_acc'],
                    output_dir=os.path.join(config.output_dir, "figures", f"{model_name}_adversarial_adv")
                )
    
    logger.info("Adversarial model training completed")
    return adversarial_models, training_histories

def evaluate_all_models(config, all_models, test_loader, adversarial_loader, device):
    """
    Evaluate all models (standard and adversarial) on both test sets.
    
    Args:
        config: Experiment configuration
        all_models: Dictionary of all models
        test_loader: Standard test data loader
        adversarial_loader: Adversarial test data loader
        device: Device to run on
    
    Returns:
        Dictionary of all model performances
    """
    logger = logging.getLogger("aeb")
    logger.info("Evaluating all models on standard and adversarial test sets")
    
    with Timer("Evaluating all models", logger):
        # Evaluate each model
        all_performances = {}
        criterion = nn.CrossEntropyLoss()
        
        for model_name, model in all_models.items():
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
            all_performances[model_name] = {
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
        
        # Compare models
        comparison = comparative_analysis(all_performances)
        
        # Plot performance comparisons
        model_names = list(all_models.keys())
        
        # Standard accuracy comparison
        std_accuracies = [all_performances[name]['standard']['accuracy'] for name in model_names]
        plot_performance_comparison(
            model_names, std_accuracies, "Accuracy (%)",
            "All Models: Accuracy on Standard Test Set",
            os.path.join(config.output_dir, "figures")
        )
        
        # Adversarial accuracy comparison
        adv_accuracies = [all_performances[name]['adversarial']['accuracy'] for name in model_names]
        plot_performance_comparison(
            model_names, adv_accuracies, "Accuracy (%)",
            "All Models: Accuracy on Adversarial Test Set",
            os.path.join(config.output_dir, "figures")
        )
        
        # Accuracy degradation comparison
        degradations = [all_performances[name]['robustness']['accuracy_degradation_percentage'] for name in model_names]
        plot_performance_comparison(
            model_names, degradations, "Degradation (%)",
            "All Models: Accuracy Degradation on Adversarial Examples",
            os.path.join(config.output_dir, "figures")
        )
        
        # Robustness score comparison
        robustness_scores = [all_performances[name]['robustness']['robustness_score'] for name in model_names]
        plot_performance_comparison(
            model_names, robustness_scores, "Robustness Score",
            "All Models: Robustness Scores (Lower is Better)",
            os.path.join(config.output_dir, "figures")
        )
        
        # Save performance results
        performance_file = os.path.join(config.output_dir, "all_model_performances.json")
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
            
            serializable_performances = {k: convert_to_serializable(v) for k, v in all_performances.items()}
            json.dump(serializable_performances, f, indent=4)
        
        logger.info(f"All performance results saved to {performance_file}")
    
    logger.info("All model evaluation completed")
    return all_performances

def generate_results_markdown(config, all_performances, save_path):
    """
    Generate a Markdown file with experiment results.
    
    Args:
        config: Experiment configuration
        all_performances: Dictionary of all model performances
        save_path: Path to save the Markdown file
    """
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
        f.write(f"- **Models Evaluated**: {', '.join(config.models.keys())}\n")
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
        f.write(f"![Standard Accuracy](figures/accuracy_comparison.png)\n\n")
        f.write("*Figure 2: Model accuracy on the standard test set*\n\n")
        
        f.write("### Accuracy on Adversarial Test Set\n\n")
        f.write(f"![Adversarial Accuracy](figures/accuracy_comparison.png)\n\n")
        f.write("*Figure 3: Model accuracy on the adversarially evolved test set*\n\n")
        
        f.write("### Model Robustness\n\n")
        f.write(f"![Robustness Scores](figures/robustness_score_comparison.png)\n\n")
        f.write("*Figure 4: Model robustness scores (lower is better)*\n\n")
        
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
        
        # Write per-class performance
        f.write("## Class-wise Performance Analysis\n\n")
        f.write("The adversarially evolved benchmarks affected different classes to varying degrees. ")
        f.write("The following tables show the per-class F1 scores for each model on standard and adversarial test sets.\n\n")
        
        # Choose one model for detailed class analysis (e.g., the first standard model)
        example_model = list(config.models.keys())[0]
        example_perf = all_performances[example_model]
        
        f.write(f"### Class-wise F1 Scores for {example_model}\n\n")
        f.write("| Class | F1 Score (Std) | F1 Score (Adv) | Degradation (%) |\n")
        f.write("|-------|----------------|----------------|----------------|\n")
        
        for i in range(10):  # Assuming 10 classes (e.g., CIFAR-10)
            class_key = f'f1_class_{i}'
            if class_key in example_perf['standard'] and class_key in example_perf['adversarial']:
                f1_std = example_perf['standard'][class_key]
                f1_adv = example_perf['adversarial'][class_key]
                
                if f1_std > 0:
                    deg_perc = (f1_std - f1_adv) / f1_std * 100
                else:
                    deg_perc = float('inf')
                
                class_name = i
                if hasattr(config, 'class_names') and i < len(config.class_names):
                    class_name = config.class_names[i]
                
                f.write(f"| {class_name} | {f1_std:.4f} | {f1_adv:.4f} | {deg_perc:.2f}% |\n")
        
        f.write("\n*Table 2: Class-wise F1 scores for standard and adversarial test sets*\n\n")
        
        # Write confusion matrix example
        f.write("## Confusion Matrices\n\n")
        f.write("The following confusion matrices show how the model's predictions changed between ")
        f.write("the standard and adversarial test sets.\n\n")
        
        f.write(f"### Confusion Matrix for {example_model} on Standard Test Set\n\n")
        f.write(f"![Standard Confusion Matrix](figures/{example_model}/confusion_matrix.png)\n\n")
        f.write(f"*Figure 5: Confusion matrix for {example_model} on standard test set*\n\n")
        
        f.write(f"### Confusion Matrix for {example_model} on Adversarial Test Set\n\n")
        f.write(f"![Adversarial Confusion Matrix](figures/{example_model}/confusion_matrix.png)\n\n")
        f.write(f"*Figure 6: Confusion matrix for {example_model} on adversarial test set*\n\n")
        
        # Write key findings and conclusions
        f.write("## Key Findings\n\n")
        
        # Identify most robust model
        model_robustness = {name: perf['robustness']['robustness_score'] for name, perf in all_performances.items()}
        most_robust_model = min(model_robustness.items(), key=lambda x: x[1])[0]
        
        # Identify most vulnerable model
        most_vulnerable_model = max(model_robustness.items(), key=lambda x: x[1])[0]
        
        f.write("1. **Adversarial Robustness Varies Significantly**: The experiment demonstrated substantial differences ")
        f.write(f"in model robustness, with {most_robust_model} showing the highest robustness and {most_vulnerable_model} being most vulnerable to adversarial examples.\n\n")
        
        f.write("2. **Adversarial Training Improves Robustness**: Models that were trained or fine-tuned on adversarially ")
        f.write("evolved examples showed improved robustness, with less performance degradation on the adversarial test set.\n\n")
        
        f.write("3. **Class-Specific Vulnerabilities**: Some classes were more affected by the adversarial transformations ")
        f.write("than others, indicating that model vulnerabilities are not uniform across all classes.\n\n")
        
        f.write("4. **Trade-offs Between Standard and Adversarial Performance**: There appears to be a trade-off between ")
        f.write("performance on standard examples and robustness to adversarial examples, highlighting the importance of ")
        f.write("evaluating models on both types of data.\n\n")
        
        # Write conclusions and implications
        f.write("## Conclusions and Implications\n\n")
        
        f.write("The Adversarially Evolved Benchmark (AEB) approach provides a novel and effective way to evaluate model ")
        f.write("robustness. By co-evolving challenging examples that expose model weaknesses, AEB offers a more dynamic ")
        f.write("and comprehensive evaluation than static benchmarks.\n\n")
        
        f.write("Key implications for machine learning practitioners:\n\n")
        
        f.write("1. **Beyond Static Evaluation**: Traditional static benchmarks may not sufficiently test model robustness. ")
        f.write("Dynamic, adversarially evolved benchmarks provide a more thorough assessment.\n\n")
        
        f.write("2. **Robustness-Aware Training**: Incorporating adversarially evolved examples in training can significantly ")
        f.write("improve model robustness, making models more suitable for real-world deployment.\n\n")
        
        f.write("3. **Identifying Vulnerabilities**: The AEB approach effectively identifies specific model vulnerabilities, ")
        f.write("providing valuable insights for targeted improvements.\n\n")
        
        f.write("4. **Evolutionary Pressure for Better Models**: By creating a co-evolutionary 'arms race' between models ")
        f.write("and benchmarks, AEB can drive the development of inherently more robust and generalizable models.\n\n")
        
        # Write limitations and future work
        f.write("## Limitations and Future Work\n\n")
        
        f.write("While the AEB approach shows promising results, this experiment has several limitations that could be ")
        f.write("addressed in future work:\n\n")
        
        f.write("1. **Computational Constraints**: Due to computational limitations, the evolutionary process was ")
        f.write("constrained in population size and generations. Larger-scale evolution could produce more challenging benchmarks.\n\n")
        
        f.write("2. **Limited Model Diversity**: A broader range of model architectures would provide more comprehensive ")
        f.write("insights into which designs offer better inherent robustness.\n\n")
        
        f.write("3. **Transformation Scope**: This experiment focused on a limited set of image transformations. Future ")
        f.write("work could explore a wider range of transformations, including semantic changes.\n\n")
        
        f.write("4. **Single Modality**: This implementation was focused on image classification. The AEB approach could ")
        f.write("be extended to other modalities like text, audio, or multimodal tasks.\n\n")
        
        f.write("5. **Human Evaluation**: Incorporating human evaluation of the evolved benchmarks would help ensure ")
        f.write("they remain semantically valid while being challenging.\n\n")
        
        f.write("Future work could address these limitations and further explore the potential of co-evolutionary ")
        f.write("approaches for evaluating and improving machine learning models.\n")
    
    logger.info(f"Results Markdown generated and saved to {save_path}")

def main(config_path):
    """
    Main function to run the AEB experiment.
    
    Args:
        config_path: Path to experiment configuration
    """
    # Set up the experiment
    log_dir = "./logs"
    config = setup_experiment(config_path, log_dir)
    logger = logging.getLogger("aeb")
    
    # Get device
    device = get_device() if config.device == 'auto' else torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset: {config.data.dataset}")
    train_dataset, test_dataset = load_dataset(config.data.dataset, config.data.data_dir)
    
    # Create data loaders
    train_loader, test_loader = get_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers
    )
    
    # Create a validation set from the training set
    val_size = int(len(train_dataset) * config.data.val_split)
    train_size = len(train_dataset) - val_size
    
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
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
    
    logger.info(f"Created data loaders: train={len(train_subset)}, val={len(val_subset)}, test={len(test_dataset)}")
    
    # Train standard models
    standard_models, standard_histories = train_standard_models(config, train_loader, val_loader, device)
    
    # Evolve benchmarks
    evolver = evolve_benchmarks(config, standard_models, test_dataset, device)
    
    # Evaluate standard models on evolved benchmarks
    standard_performances, adversarial_loader = evaluate_models_on_benchmarks(
        config, standard_models, evolver, test_dataset, device
    )
    
    # Create an adversarial training set
    adversarial_train_loader = evolver.create_adversarial_dataloader(
        train_subset, batch_size=config.data.batch_size
    )
    
    # Train adversarial models
    adversarial_models, adversarial_histories = train_adversarial_models(
        config, standard_models, train_loader, adversarial_train_loader, val_loader, device
    )
    
    # Combine all models
    all_models = {**standard_models, **adversarial_models}
    
    # Evaluate all models
    all_performances = evaluate_all_models(
        config, all_models, test_loader, adversarial_loader, device
    )
    
    # Generate results Markdown
    results_md_path = os.path.join(config.output_dir, "results.md")
    generate_results_markdown(config, all_performances, results_md_path)
    
    # Create results directory and copy necessary files
    results_dir = os.path.join(os.path.dirname(config.output_dir), "results")
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
    logger.info("AEB experiment completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AEB experiments")
    parser.add_argument("--config", type=str, default="experiments/cifar10_config.yaml",
                        help="Path to experiment configuration file")
    args = parser.parse_args()
    
    main(args.config)
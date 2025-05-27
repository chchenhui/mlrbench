import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import time
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from models.weight_net import WeightNetTransformer, MLPBaseline, StatsBaseline
from data.dataset import (
    create_model_zoo, 
    load_model_zoo, 
    prepare_datasets, 
    create_data_loaders
)
from utils.trainer import Trainer
from utils.visualization import (
    plot_training_history,
    plot_predictions_vs_targets,
    plot_error_distributions,
    plot_model_comparison,
    create_comparative_predictions_plot,
    plot_property_correlations,
    create_radar_chart
)
from utils.config import load_experiment_config, get_model_config

# Set up logging
def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """
    Set up logging for the experiment.
    
    Args:
        log_dir: Directory to save log files
        experiment_name: Name of the experiment
        
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_model(
    model_config: Dict[str, Any], 
    num_properties: int, 
    token_dim: int,
    device: torch.device
) -> nn.Module:
    """
    Create a model based on the configuration.
    
    Args:
        model_config: Model configuration dictionary
        num_properties: Number of properties to predict
        token_dim: Dimension of input tokens
        device: Device to put the model on
        
    Returns:
        PyTorch model
    """
    model_type = model_config['type']
    
    if model_type == 'weightnet':
        # Create WeightNet model
        model = WeightNetTransformer(
            d_model=model_config.get('d_model', 256),
            num_intra_layer_heads=model_config.get('num_intra_layer_heads', 4),
            num_cross_layer_heads=model_config.get('num_cross_layer_heads', 8),
            num_intra_layer_blocks=model_config.get('num_intra_layer_blocks', 2),
            num_cross_layer_blocks=model_config.get('num_cross_layer_blocks', 2),
            d_ff=model_config.get('d_ff', 1024),
            dropout=model_config.get('dropout', 0.1),
            max_seq_length=model_config.get('max_seq_length', 4096),
            num_segments=model_config.get('num_segments', 100),
            num_properties=num_properties,
            token_dim=token_dim - 4,  # Subtract metadata dimensions
        )
    elif model_type == 'mlp':
        # Create MLP baseline
        model = MLPBaseline(
            input_dim=token_dim,
            hidden_dims=model_config.get('hidden_dims', [1024, 512, 256]),
            output_dim=num_properties,
            dropout=model_config.get('dropout', 0.2),
        )
    elif model_type == 'stats':
        # Create statistics baseline
        model = StatsBaseline(
            token_dim=token_dim,
            num_features=model_config.get('num_features', 20),
            hidden_dims=model_config.get('hidden_dims', [256, 128, 64]),
            output_dim=num_properties,
            dropout=model_config.get('dropout', 0.2),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move model to device
    model = model.to(device)
    
    return model

def create_optimizer(
    model: nn.Module, 
    optimizer_config: Dict[str, Any]
) -> optim.Optimizer:
    """
    Create an optimizer based on the configuration.
    
    Args:
        model: PyTorch model
        optimizer_config: Optimizer configuration dictionary
        
    Returns:
        PyTorch optimizer
    """
    optimizer_type = optimizer_config.get('type', 'adam').lower()
    lr = optimizer_config.get('lr', 0.001)
    weight_decay = optimizer_config.get('weight_decay', 0.0001)
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer

def create_scheduler(
    optimizer: optim.Optimizer, 
    scheduler_config: Dict[str, Any], 
    num_epochs: int
) -> Optional[Any]:
    """
    Create a learning rate scheduler based on the configuration.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_config: Scheduler configuration dictionary
        num_epochs: Number of training epochs
        
    Returns:
        PyTorch scheduler or None
    """
    if scheduler_config is None or 'type' not in scheduler_config:
        return None
    
    scheduler_type = scheduler_config.get('type', '').lower()
    
    if scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 10)
        gamma = scheduler_config.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'multistep':
        milestones = scheduler_config.get('milestones', [30, 60, 90])
        gamma = scheduler_config.get('gamma', 0.1)
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler_type == 'cosine':
        T_max = scheduler_config.get('T_max', num_epochs)
        eta_min = scheduler_config.get('min_lr', 0)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_type == 'plateau':
        mode = scheduler_config.get('mode', 'min')
        factor = scheduler_config.get('factor', 0.1)
        patience = scheduler_config.get('patience', 10)
        min_lr = scheduler_config.get('min_lr', 0.000001)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr
        )
    
    elif scheduler_type == 'warmup':
        from transformers import get_cosine_schedule_with_warmup
        
        # Calculate number of training steps
        num_warmup_steps = int(scheduler_config.get('warmup_ratio', 0.1) * num_epochs)
        num_training_steps = num_epochs
        
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    elif scheduler_type == 'none' or not scheduler_type:
        return None
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def create_criterion(criterion_name: str) -> nn.Module:
    """
    Create a loss function based on the name.
    
    Args:
        criterion_name: Name of the loss function
        
    Returns:
        PyTorch loss function
    """
    criterion_name = criterion_name.lower()
    
    if criterion_name == 'mse':
        return nn.MSELoss()
    elif criterion_name == 'mae':
        return nn.L1Loss()
    elif criterion_name == 'huber':
        return nn.SmoothL1Loss()
    elif criterion_name == 'bce':
        return nn.BCELoss()
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")

def run_experiment(
    config: Dict[str, Any],
    model_name: Optional[str] = None,
    skip_training: bool = False,
    resume_from: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run an experiment based on the configuration.
    
    Args:
        config: Configuration dictionary
        model_name: Optional name of a specific model to run (for baselines)
        skip_training: Whether to skip training (for evaluation only)
        resume_from: Optional path to a checkpoint to resume from
        
    Returns:
        Dictionary of results
    """
    # Get experiment configuration
    experiment_config = config['experiment']
    data_config = config['data']
    training_config = config['training']
    
    # Get model configuration
    if model_name is None:
        model_config = config['model']
        model_name = model_config.get('name', 'weightnet')
    else:
        model_config = get_model_config(config, model_name)
    
    # Set up logging
    logger = setup_logging(experiment_config['log_dir'], f"{experiment_config['name']}_{model_name}")
    
    # Log experiment start
    logger.info(f"Starting experiment: {experiment_config['name']} - {model_name}")
    logger.info(f"Config: {json.dumps(model_config, indent=2)}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and experiment_config.get('use_gpu', True) else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data directories
    os.makedirs(data_config['data_dir'], exist_ok=True)
    os.makedirs(data_config['processed_dir'], exist_ok=True)
    
    # Create model zoo if needed
    if data_config.get('synthetic', {}).get('create', True):
        logger.info("Creating synthetic model zoo")
        create_model_zoo(
            output_dir=data_config['data_dir'],
            num_models_per_architecture=data_config['synthetic'].get('num_models_per_architecture', 10),
            architectures=data_config['synthetic'].get('architectures'),
            generate_variations=data_config['synthetic'].get('generate_variations', True),
            random_seed=experiment_config.get('seed', 42)
        )
    
    # Prepare datasets
    logger.info("Preparing datasets")
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        data_dir=data_config['data_dir'],
        model_properties=data_config['model_properties'],
        canonicalization_method=data_config.get('canonicalization_method'),
        tokenization_strategy=data_config.get('tokenization_strategy', 'neuron_centric'),
        max_token_length=data_config.get('max_token_length', 4096),
        train_ratio=data_config.get('train_ratio', 0.7),
        val_ratio=data_config.get('val_ratio', 0.15),
        test_ratio=data_config.get('test_ratio', 0.15),
        split_by_architecture=data_config.get('split_by_architecture', False),
        seed=experiment_config.get('seed', 42)
    )
    
    # Create data loaders
    logger.info("Creating data loaders")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=data_config.get('batch_size', 32),
        num_workers=data_config.get('num_workers', 4)
    )
    
    # Sample a batch to determine token dimension
    for tokens, _ in train_loader:
        token_dim = tokens.shape[2]
        break
    
    # Create model
    logger.info(f"Creating model: {model_name}")
    model = create_model(
        model_config=model_config,
        num_properties=len(data_config['model_properties']),
        token_dim=token_dim,
        device=device
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, training_config['optimizer'])
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer, 
        training_config.get('scheduler'), 
        training_config['num_epochs']
    )
    
    # Create criterion
    criterion = create_criterion(training_config.get('criterion', 'mse'))
    
    # Create directories for saving results
    save_dir = os.path.join(experiment_config['save_dir'], model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    tensorboard_dir = os.path.join(experiment_config['tensorboard_dir'], model_name) if 'tensorboard_dir' in experiment_config else None
    if tensorboard_dir:
        os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        model_name=model_name,
        save_dir=save_dir,
        property_names=data_config['model_properties'],
        tensorboard_dir=tensorboard_dir,
        early_stopping_patience=training_config.get('early_stopping_patience', 10)
    )
    
    # Resume from checkpoint if specified
    if resume_from is not None:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)
    
    # Train model if not skipping training
    if not skip_training:
        logger.info("Training model")
        trainer.train(num_epochs=training_config['num_epochs'])
    
    # Load best model for evaluation
    best_checkpoint_path = os.path.join(save_dir, f"{model_name}_best.pth")
    if os.path.exists(best_checkpoint_path):
        logger.info(f"Loading best model from {best_checkpoint_path}")
        trainer.load_checkpoint(best_checkpoint_path)
    
    # Evaluate model
    logger.info("Evaluating model")
    test_results = trainer.detailed_evaluation(test_loader, "Test Evaluation")
    
    # Create results table
    results_table = trainer.create_results_table(test_results)
    results_table_path = os.path.join(save_dir, f"{model_name}_results.csv")
    results_table.to_csv(results_table_path)
    logger.info(f"Results table saved to {results_table_path}")
    
    # Visualize results
    if config.get('visualization', {}).get('create_history_plots', True):
        history_path = os.path.join(save_dir, f"{model_name}_history.json")
        if os.path.exists(history_path):
            figures_dir = os.path.join(experiment_config['figures_dir'], model_name)
            os.makedirs(figures_dir, exist_ok=True)
            
            plot_paths = plot_training_history(
                history_path=history_path,
                save_dir=figures_dir,
                model_name=model_name
            )
            logger.info(f"Created {len(plot_paths)} training history plots")
    
    if config.get('visualization', {}).get('create_prediction_plots', True):
        figures_dir = os.path.join(experiment_config['figures_dir'], model_name)
        os.makedirs(figures_dir, exist_ok=True)
        
        plot_paths = plot_predictions_vs_targets(
            predictions=test_results['predictions'],
            targets=test_results['targets'],
            property_names=data_config['model_properties'],
            save_dir=figures_dir,
            model_name=model_name
        )
        logger.info(f"Created {len(plot_paths)} prediction vs target plots")
    
    if config.get('visualization', {}).get('create_error_plots', True):
        figures_dir = os.path.join(experiment_config['figures_dir'], model_name)
        os.makedirs(figures_dir, exist_ok=True)
        
        plot_paths = plot_error_distributions(
            predictions=test_results['predictions'],
            targets=test_results['targets'],
            property_names=data_config['model_properties'],
            save_dir=figures_dir,
            model_name=model_name
        )
        logger.info(f"Created {len(plot_paths)} error distribution plots")
    
    # Save predictions
    if config.get('evaluation', {}).get('save_predictions', True):
        predictions_dir = os.path.join(save_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        
        np.save(os.path.join(predictions_dir, f"{model_name}_predictions.npy"), test_results['predictions'])
        np.save(os.path.join(predictions_dir, f"{model_name}_targets.npy"), test_results['targets'])
        logger.info("Saved predictions and targets")
    
    # Return results
    return {
        'model_name': model_name,
        'test_results': test_results,
        'model_config': model_config,
        'save_dir': save_dir
    }

def run_all_experiments(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run all experiments specified in the configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of results dictionaries
    """
    all_results = []
    
    # Run main model experiment
    main_results = run_experiment(config)
    all_results.append(main_results)
    
    # Run baseline experiments if specified
    if config.get('baselines', {}).get('run', True):
        for baseline_config in config.get('baselines', {}).get('models', []):
            baseline_name = baseline_config['name']
            baseline_results = run_experiment(config, model_name=baseline_name)
            all_results.append(baseline_results)
    
    # Run ablation experiments if specified
    if config.get('ablation', {}).get('run', False):
        for ablation_config in config.get('ablation', {}).get('experiments', []):
            ablation_name = ablation_config['name']
            # Create a new config with the ablation-specific settings
            ablation_full_config = copy.deepcopy(config)
            
            # Update config with ablation-specific settings
            if 'data' in ablation_config:
                ablation_full_config['data'].update(ablation_config['data'])
            
            ablation_results = run_experiment(ablation_full_config, model_name=ablation_name)
            all_results.append(ablation_results)
    
    return all_results

def compare_models(
    results: List[Dict[str, Any]], 
    config: Dict[str, Any]
) -> None:
    """
    Create comparative visualizations for multiple models.
    
    Args:
        results: List of results dictionaries from experiments
        config: Configuration dictionary
    """
    if len(results) <= 1:
        return
    
    # Extract model names and property names
    model_names = [result['model_name'] for result in results]
    property_names = config['data']['model_properties']
    
    # Set up directories
    experiment_name = config['experiment']['name']
    figures_dir = os.path.join(config['experiment']['figures_dir'], "comparisons")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create a dictionary of model metrics for comparison
    model_metrics = {}
    for result in results:
        model_name = result['model_name']
        model_metrics[model_name] = {}
        
        for prop in property_names:
            prop_metrics = result['test_results']['property_metrics'][prop]
            model_metrics[model_name][prop] = {
                'mae': prop_metrics['mae'],
                'rmse': prop_metrics['rmse'],
                'r2': prop_metrics['r2']
            }
    
    # Create comparison plots
    if config.get('visualization', {}).get('create_model_comparison', True):
        for metric in ['mae', 'rmse', 'r2']:
            plot_path = plot_model_comparison(
                model_results=model_metrics,
                property_names=property_names,
                metric_name=metric,
                save_dir=figures_dir
            )
            logging.info(f"Created model comparison plot for {metric}: {plot_path}")
    
    # Create radar charts
    if config.get('visualization', {}).get('create_radar_charts', True):
        for metric in ['r2', 'mae']:
            radar_path = create_radar_chart(
                model_results=model_metrics,
                property_names=property_names,
                metric_name=metric,
                save_dir=figures_dir
            )
            logging.info(f"Created radar chart for {metric}: {radar_path}")
    
    # Create comparative prediction plots
    if config.get('visualization', {}).get('create_prediction_plots', True):
        predictions_dict = {}
        targets = None
        
        for result in results:
            model_name = result['model_name']
            predictions_dict[model_name] = result['test_results']['predictions']
            if targets is None:
                targets = result['test_results']['targets']
        
        plot_paths = create_comparative_predictions_plot(
            predictions_dict=predictions_dict,
            targets=targets,
            property_names=property_names,
            save_dir=figures_dir
        )
        logging.info(f"Created {len(plot_paths)} comparative prediction plots")
    
    # Create a summary table
    summary_data = []
    for prop in property_names:
        for metric in ['mae', 'rmse', 'r2']:
            row = {'Property': prop, 'Metric': metric.upper()}
            
            for model_name in model_names:
                row[model_name] = model_metrics[model_name][prop][metric]
            
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(figures_dir, "model_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"Created summary table: {summary_path}")

def create_results_markdown(
    results: List[Dict[str, Any]], 
    config: Dict[str, Any]
) -> str:
    """
    Create a markdown file summarizing experiment results.
    
    Args:
        results: List of results dictionaries from experiments
        config: Configuration dictionary
        
    Returns:
        Path to the created markdown file
    """
    experiment_name = config['experiment']['name']
    property_names = config['data']['model_properties']
    
    # Create results directory
    results_dir = os.path.join("/home/chenhui/mlr-bench/pipeline_gemini/iclr2025_wsl/results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create markdown content
    md_content = [
        f"# {experiment_name} Experiment Results\n",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Experiment Summary\n",
        f"This experiment evaluated the performance of the WeightNet permutation-invariant transformer model for predicting model properties from neural network weights, as described in the proposal. The experiment compared the main WeightNet model with baseline approaches and conducted ablation studies.\n",
        f"Properties predicted: {', '.join(property_names)}\n",
        "## Models Evaluated\n"
    ]
    
    # Add model descriptions
    for result in results:
        model_name = result['model_name']
        model_config = result['model_config']
        model_type = model_config.get('type', 'unknown')
        
        md_content.append(f"### {model_name}\n")
        
        if model_type == 'weightnet':
            md_content.append("**Model Type**: Permutation-Invariant Transformer (WeightNet)\n")
            md_content.append(f"- d_model: {model_config.get('d_model', 256)}\n")
            md_content.append(f"- Intra-layer attention heads: {model_config.get('num_intra_layer_heads', 4)}\n")
            md_content.append(f"- Cross-layer attention heads: {model_config.get('num_cross_layer_heads', 8)}\n")
            md_content.append(f"- Intra-layer blocks: {model_config.get('num_intra_layer_blocks', 2)}\n")
            md_content.append(f"- Cross-layer blocks: {model_config.get('num_cross_layer_blocks', 2)}\n")
        elif model_type == 'mlp':
            md_content.append("**Model Type**: MLP Baseline\n")
            md_content.append(f"- Hidden dimensions: {model_config.get('hidden_dims', [1024, 512, 256])}\n")
            md_content.append(f"- Dropout: {model_config.get('dropout', 0.2)}\n")
        elif model_type == 'stats':
            md_content.append("**Model Type**: Statistics Baseline\n")
            md_content.append(f"- Number of features: {model_config.get('num_features', 20)}\n")
            md_content.append(f"- Hidden dimensions: {model_config.get('hidden_dims', [256, 128, 64])}\n")
            md_content.append(f"- Dropout: {model_config.get('dropout', 0.2)}\n")
        
        if model_name in config.get('ablation', {}).get('experiments', []):
            for ablation in config['ablation']['experiments']:
                if ablation['name'] == model_name:
                    md_content.append(f"**Ablation**: {ablation.get('description', 'No description provided')}\n")
        
        md_content.append("\n")
    
    # Add results tables
    md_content.append("## Results Summary\n")
    
    # Create a summary table of all models and properties
    md_content.append("### Model Performance Comparison\n")
    md_content.append("#### Mean Absolute Error (MAE)\n")
    md_content.append("| Property | " + " | ".join([result['model_name'] for result in results]) + " |\n")
    md_content.append("| --- | " + " | ".join(["---" for _ in results]) + " |\n")
    
    for prop in property_names:
        row = f"| {prop} | "
        for result in results:
            mae = result['test_results']['property_metrics'][prop]['mae']
            row += f"{mae:.4f} | "
        md_content.append(row + "\n")
    
    md_content.append("\n#### R² Score\n")
    md_content.append("| Property | " + " | ".join([result['model_name'] for result in results]) + " |\n")
    md_content.append("| --- | " + " | ".join(["---" for _ in results]) + " |\n")
    
    for prop in property_names:
        row = f"| {prop} | "
        for result in results:
            r2 = result['test_results']['property_metrics'][prop]['r2']
            row += f"{r2:.4f} | "
        md_content.append(row + "\n")
    
    # Add detailed results for each model
    md_content.append("\n## Detailed Model Results\n")
    
    for result in results:
        model_name = result['model_name']
        md_content.append(f"### {model_name}\n")
        
        # Property-specific metrics
        md_content.append("#### Property Metrics\n")
        md_content.append("| Property | MAE | RMSE | R² |\n")
        md_content.append("| --- | --- | --- | --- |\n")
        
        for prop in property_names:
            metrics = result['test_results']['property_metrics'][prop]
            md_content.append(f"| {prop} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | {metrics['r2']:.4f} |\n")
        
        # Overall metrics
        md_content.append("\n#### Overall Metrics\n")
        md_content.append(f"- MAE: {result['test_results']['mae']:.4f}\n")
        md_content.append(f"- RMSE: {result['test_results']['rmse']:.4f}\n")
        
        md_content.append("\n")
    
    # Add visualizations
    md_content.append("## Visualizations\n")
    
    # Add predictions vs targets plots
    md_content.append("### Predictions vs Targets\n")
    
    for prop in property_names:
        best_model = results[0]['model_name']  # Assume first model (WeightNet) is best
        figures_dir = os.path.join(config['experiment']['figures_dir'], best_model)
        rel_path = os.path.join("figures", best_model, f"{best_model}_{prop}_preds_vs_targets.png")
        
        md_content.append(f"#### {prop} Predictions\n")
        md_content.append(f"![{prop} Predictions]({rel_path})\n\n")
    
    # Add error distribution plots
    md_content.append("### Error Distributions\n")
    
    for prop in property_names:
        best_model = results[0]['model_name']
        figures_dir = os.path.join(config['experiment']['figures_dir'], best_model)
        rel_path = os.path.join("figures", best_model, f"{best_model}_{prop}_error_kde.png")
        
        md_content.append(f"#### {prop} Error Distribution\n")
        md_content.append(f"![{prop} Error Distribution]({rel_path})\n\n")
    
    # Add model comparison plots
    if len(results) > 1:
        md_content.append("### Model Comparisons\n")
        
        # Add comparison bar chart
        comp_dir = os.path.join(config['experiment']['figures_dir'], "comparisons")
        rel_path = os.path.join("figures", "comparisons", "model_comparison_mae.png")
        md_content.append("#### MAE Comparison\n")
        md_content.append(f"![MAE Comparison]({rel_path})\n\n")
        
        # Add radar chart
        rel_path = os.path.join("figures", "comparisons", "radar_chart_r2.png")
        md_content.append("#### R² Score Comparison (Radar Chart)\n")
        md_content.append(f"![R² Radar Chart]({rel_path})\n\n")
    
    # Add training history plots
    md_content.append("### Training History\n")
    
    for result in results:
        model_name = result['model_name']
        figures_dir = os.path.join(config['experiment']['figures_dir'], model_name)
        rel_path = os.path.join("figures", model_name, f"{model_name}_combined_metrics.png")
        
        md_content.append(f"#### {model_name} Training History\n")
        md_content.append(f"![{model_name} Training History]({rel_path})\n\n")
    
    # Add analysis and findings
    md_content.append("## Analysis and Findings\n")
    
    # Determine best model for each property
    best_models = {}
    for prop in property_names:
        best_r2 = -float('inf')
        best_model = None
        
        for result in results:
            r2 = result['test_results']['property_metrics'][prop]['r2']
            if r2 > best_r2:
                best_r2 = r2
                best_model = result['model_name']
        
        best_models[prop] = (best_model, best_r2)
    
    md_content.append("### Key Findings\n")
    
    # Add key findings based on results
    weightnet_results = None
    mlp_results = None
    stats_results = None
    
    for result in results:
        if result['model_name'] == 'weightnet_main' or result['model_config']['type'] == 'weightnet':
            weightnet_results = result
        elif result['model_name'] == 'mlp_baseline' or result['model_config']['type'] == 'mlp':
            mlp_results = result
        elif result['model_name'] == 'stats_baseline' or result['model_config']['type'] == 'stats':
            stats_results = result
    
    if weightnet_results and mlp_results:
        # Compare WeightNet vs MLP baseline
        weightnet_avg_r2 = np.mean([weightnet_results['test_results']['property_metrics'][prop]['r2'] for prop in property_names])
        mlp_avg_r2 = np.mean([mlp_results['test_results']['property_metrics'][prop]['r2'] for prop in property_names])
        
        if weightnet_avg_r2 > mlp_avg_r2:
            improvement = ((weightnet_avg_r2 - mlp_avg_r2) / mlp_avg_r2) * 100
            md_content.append(f"1. **WeightNet outperforms MLP baseline**: The permutation-invariant WeightNet model achieves an average R² score of {weightnet_avg_r2:.4f} across all properties, compared to {mlp_avg_r2:.4f} for the MLP baseline, representing a {improvement:.1f}% improvement.\n")
        else:
            md_content.append(f"1. **MLP baseline performs competitively**: The MLP baseline achieves an average R² score of {mlp_avg_r2:.4f} across all properties, compared to {weightnet_avg_r2:.4f} for the WeightNet model.\n")
    
    # Add property-specific findings
    for i, prop in enumerate(property_names):
        best_model, best_r2 = best_models[prop]
        md_content.append(f"{i+2}. **{prop} prediction**: The best model for predicting {prop} is {best_model} with an R² score of {best_r2:.4f}.\n")
    
    # Add ablation study findings if available
    ablation_results = [r for r in results if r['model_name'] in [a['name'] for a in config.get('ablation', {}).get('experiments', [])]]
    
    if ablation_results:
        md_content.append("\n### Ablation Study Findings\n")
        
        for ablation in ablation_results:
            model_name = ablation['model_name']
            avg_r2 = np.mean([ablation['test_results']['property_metrics'][prop]['r2'] for prop in property_names])
            
            for abl_config in config.get('ablation', {}).get('experiments', []):
                if abl_config['name'] == model_name:
                    description = abl_config.get('description', model_name)
                    md_content.append(f"- **{description}**: Average R² score = {avg_r2:.4f}\n")
    
    # Add conclusions
    md_content.append("\n## Conclusions\n")
    
    if weightnet_results:
        weightnet_avg_r2 = np.mean([weightnet_results['test_results']['property_metrics'][prop]['r2'] for prop in property_names])
        
        if weightnet_avg_r2 > 0.7:
            md_content.append("The WeightNet model demonstrates strong predictive performance for model properties from weights, confirming the feasibility of the proposed approach. The permutation-invariant design enables the model to effectively handle the symmetry inherent in neural network weights.\n")
        elif weightnet_avg_r2 > 0.5:
            md_content.append("The WeightNet model shows moderate predictive performance for model properties from weights. While the permutation-invariant design helps with handling weight symmetry, there is room for improvement in the model architecture and training process.\n")
        else:
            md_content.append("The WeightNet model shows limited predictive performance for model properties from weights. Further research is needed to improve the architecture and training methodology to better capture the relationship between weights and model properties.\n")
    
    # Add limitations and future work
    md_content.append("\n### Limitations and Future Work\n")
    
    md_content.append("1. **Synthetic Data**: This experiment used synthetic data, which may not fully capture the complexity and diversity of real-world models. Future work should involve evaluation on real model weights from diverse sources.\n")
    md_content.append("2. **Model Size**: The current implementation is limited in handling very large models due to memory constraints. Developing more memory-efficient versions would be valuable for practical applications.\n")
    md_content.append("3. **Property Range**: The experiment focused on a limited set of properties. Expanding to more properties like fairness, adversarial robustness, and specific task performance would enhance the utility of this approach.\n")
    md_content.append("4. **Architecture Diversity**: Including a wider range of model architectures, especially non-convolutional ones like transformers, would provide a more comprehensive evaluation.\n")
    
    # Join all content
    full_content = "\n".join(md_content)
    
    # Write to file
    results_path = os.path.join(results_dir, "results.md")
    with open(results_path, 'w') as f:
        f.write(full_content)
    
    logging.info(f"Created results markdown file: {results_path}")
    
    return results_path

def copy_visualizations_to_results(config: Dict[str, Any]) -> None:
    """
    Copy visualization files to the results directory.
    
    Args:
        config: Configuration dictionary
    """
    source_dir = config['experiment']['figures_dir']
    dest_dir = os.path.join("/home/chenhui/mlr-bench/pipeline_gemini/iclr2025_wsl/results")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    # Create figures directory in results
    figures_dir = os.path.join(dest_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Copy all subdirectories
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        if os.path.isdir(subdir_path):
            dest_subdir = os.path.join(figures_dir, subdir)
            os.makedirs(dest_subdir, exist_ok=True)
            
            # Copy PNG files
            for file in os.listdir(subdir_path):
                if file.endswith('.png'):
                    source_file = os.path.join(subdir_path, file)
                    dest_file = os.path.join(dest_subdir, file)
                    import shutil
                    shutil.copy2(source_file, dest_file)
    
    logging.info(f"Copied visualizations to {figures_dir}")

def copy_log_to_results(config: Dict[str, Any]) -> None:
    """
    Copy log file to the results directory.
    
    Args:
        config: Configuration dictionary
    """
    source_dir = config['experiment']['log_dir']
    dest_dir = os.path.join("/home/chenhui/mlr-bench/pipeline_gemini/iclr2025_wsl/results")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    # Find main log file
    experiment_name = config['experiment']['name']
    log_file = os.path.join(source_dir, f"{experiment_name}.log")
    
    if os.path.exists(log_file):
        import shutil
        dest_file = os.path.join(dest_dir, "log.txt")
        shutil.copy2(log_file, dest_file)
        logging.info(f"Copied log file to {dest_file}")

def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description="Run WeightNet experiments")
    parser.add_argument("--config", type=str, default="configs/weightnet_experiment.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--model", type=str, default=None,
                        help="Run experiment for a specific model")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and only run evaluation")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from a checkpoint")
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        if os.path.exists(os.path.join("configs", args.config)):
            config_path = os.path.join("configs", args.config)
        else:
            config_path = args.config
    else:
        config_path = args.config

    print(f"Using config file: {config_path}")
    config = load_experiment_config(config_path, os.getcwd())
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.model:
        # Run experiment for a specific model
        results = [run_experiment(config, model_name=args.model, skip_training=args.skip_training, resume_from=args.resume)]
    else:
        # Run all experiments
        results = run_all_experiments(config)
    
    # Compare models if multiple were run
    if len(results) > 1:
        compare_models(results, config)
    
    # Create results markdown
    results_md_path = create_results_markdown(results, config)
    
    # Copy visualizations to results directory
    copy_visualizations_to_results(config)
    
    # Copy log file to results directory
    copy_log_to_results(config)
    
    logging.info(f"All experiments completed. Results saved to {results_md_path}")

if __name__ == "__main__":
    main()
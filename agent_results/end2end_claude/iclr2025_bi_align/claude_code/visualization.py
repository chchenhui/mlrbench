"""
Visualization utilities for the CEVA framework experiments.

This module provides functions for visualizing experimental results,
including value evolution, alignment metrics, and comparative performance.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import os
from pathlib import Path


def setup_visualization(config: Dict) -> None:
    """
    Set up visualization environment based on configuration.
    
    Args:
        config: Visualization configuration dictionary
    """
    plt.rcParams['figure.figsize'] = config['figsize']
    plt.rcParams['figure.dpi'] = config['dpi']
    sns.set_style(config['style'])
    sns.set_context(config['context'])
    
    
def save_figure(fig: plt.Figure, filename: str, figures_dir: Path, config: Dict) -> str:
    """
    Save a figure to disk.
    
    Args:
        fig: Figure to save
        filename: Base filename (without extension)
        figures_dir: Directory to save figure in
        config: Visualization configuration
        
    Returns:
        Path to the saved figure
    """
    # Ensure directory exists
    os.makedirs(figures_dir, exist_ok=True)
    
    # Add file extension
    filepath = figures_dir / f"{filename}.{config['save_format']}"
    
    # Save figure
    fig.savefig(filepath, bbox_inches='tight', dpi=config['dpi'])
    
    # Close figure to free memory
    plt.close(fig)
    
    return str(filepath)


def plot_value_evolution(
    value_data: pd.DataFrame, 
    filename: str, 
    figures_dir: Path,
    config: Dict,
    title: str = "Value Evolution Over Time"
) -> str:
    """
    Plot the evolution of values over time.
    
    Args:
        value_data: DataFrame with value evolution data
        filename: Base filename for saving the plot
        figures_dir: Directory to save figures in
        config: Visualization configuration
        title: Plot title
        
    Returns:
        Path to the saved figure
    """
    fig, ax = plt.subplots()
    
    # Plot each value dimension
    for column in value_data.columns:
        ax.plot(value_data.index, value_data[column], label=column)
        
    # Add labels and legend
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value Importance")
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    
    # Save and return figure path
    return save_figure(fig, filename, figures_dir, config)


def plot_comparative_alignment(
    scenario_results: Dict, 
    filename: str, 
    figures_dir: Path,
    config: Dict,
    title: str = "Alignment Score Comparison"
) -> str:
    """
    Plot alignment scores for multiple models over time.
    
    Args:
        scenario_results: Results dictionary for a scenario
        filename: Base filename for saving the plot
        figures_dir: Directory to save figures in
        config: Visualization configuration
        title: Plot title
        
    Returns:
        Path to the saved figure
    """
    fig, ax = plt.subplots()
    
    # Extract alignment scores for each model
    models = list(scenario_results["models"].keys())
    n_steps = len(scenario_results["raw_data"]["alignment_scores"][models[0]])
    
    # Plot alignment scores for each model
    for model_name in models:
        scores = scenario_results["raw_data"]["alignment_scores"][model_name]
        ax.plot(range(n_steps), scores, label=model_name)
        
    # Add labels and legend
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Alignment Score")
    ax.set_title(f"{title}: {scenario_results['scenario_name']}")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    
    # Save and return figure path
    return save_figure(fig, filename, figures_dir, config)


def plot_metric_radar(
    metrics: Dict, 
    filename: str, 
    figures_dir: Path,
    config: Dict,
    title: str = "Model Performance Metrics"
) -> str:
    """
    Plot a radar chart of model performance metrics.
    
    Args:
        metrics: Dictionary mapping models to their metrics
        filename: Base filename for saving the plot
        figures_dir: Directory to save figures in
        config: Visualization configuration
        title: Plot title
        
    Returns:
        Path to the saved figure
    """
    # Get list of all metrics and models
    all_metrics = []
    for model_name, model_metrics in metrics.items():
        all_metrics.extend(list(model_metrics.keys()))
    all_metrics = list(set([m for m in all_metrics if not m.startswith('std_')]))  # Exclude std metrics
    
    models = list(metrics.keys())
    
    # Convert metrics to angle in the chart
    angles = np.linspace(0, 2*np.pi, len(all_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot each model
    for model_name in models:
        values = []
        for metric in all_metrics:
            # Get avg_ metrics
            avg_metric = f"avg_{metric}"
            if avg_metric in metrics[model_name]:
                values.append(metrics[model_name][avg_metric])
            else:
                values.append(0)  # Use 0 if metric not available
                
        # Close the loop
        values += values[:1]
        
        # Plot the model
        ax.plot(angles, values, linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.1)
        
    # Set labels for each metric
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_metrics)
    
    # Add legend and title
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=15, y=1.1)
    
    # Save and return figure path
    return save_figure(fig, filename, figures_dir, config)


def plot_scenario_performance(
    scenario_results: Dict, 
    metric_name: str,
    filename: str, 
    figures_dir: Path,
    config: Dict,
    title: Optional[str] = None
) -> str:
    """
    Plot performance of models on a specific metric across scenarios.
    
    Args:
        scenario_results: Dictionary with results for each scenario
        metric_name: Name of the metric to plot
        filename: Base filename for saving the plot
        figures_dir: Directory to save figures in
        config: Visualization configuration
        title: Optional plot title
        
    Returns:
        Path to the saved figure
    """
    fig, ax = plt.subplots()
    
    # Extract metric values for each model and scenario
    scenarios = list(scenario_results.keys())
    model_metrics = {}
    
    for scenario_name in scenarios:
        scenario_data = scenario_results[scenario_name]
        for model_name, model_data in scenario_data["models"].items():
            if model_name not in model_metrics:
                model_metrics[model_name] = []
                
            model_metrics[model_name].append(model_data["metrics"].get(metric_name, 0))
            
    # Plot metric values for each model
    bar_width = 0.8 / len(model_metrics)
    x = np.arange(len(scenarios))
    
    for i, (model_name, values) in enumerate(model_metrics.items()):
        ax.bar(x + i * bar_width, values, bar_width, label=model_name)
        
    # Add labels and legend
    ax.set_xlabel("Scenario")
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{metric_name.replace('_', ' ').title()} Comparison")
    ax.set_xticks(x + bar_width * (len(model_metrics) - 1) / 2)
    ax.set_xticklabels(scenarios)
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and return figure path
    return save_figure(fig, filename, figures_dir, config)


def plot_aggregate_metrics(
    aggregate_metrics: Dict, 
    filename: str, 
    figures_dir: Path,
    config: Dict,
    title: str = "Aggregate Model Performance"
) -> str:
    """
    Plot aggregate metrics for all models.
    
    Args:
        aggregate_metrics: Dictionary with aggregate metrics for each model
        filename: Base filename for saving the plot
        figures_dir: Directory to save figures in
        config: Visualization configuration
        title: Plot title
        
    Returns:
        Path to the saved figure
    """
    # Prepare DataFrame for plotting
    metrics_to_plot = ['avg_adaptation_accuracy', 'avg_stability', 
                      'avg_user_satisfaction', 'avg_agency_preservation']
    
    data = []
    for model_name, metrics in aggregate_metrics.items():
        for metric in metrics_to_plot:
            if metric in metrics:
                # Strip 'avg_' prefix for cleaner labels
                metric_label = metric[4:].replace('_', ' ').title()
                data.append({
                    'Model': model_name,
                    'Metric': metric_label,
                    'Value': metrics[metric]
                })
    
    df = pd.DataFrame(data)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Metric', y='Value', hue='Model', data=df, ax=ax)
    
    # Add labels and legend
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(title="Model")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and return figure path
    return save_figure(fig, filename, figures_dir, config)


def visualize_experiment_results(
    results: Dict, 
    figures_dir: Path,
    config: Dict
) -> Dict[str, str]:
    """
    Generate all visualization figures for experiment results.
    
    Args:
        results: Dictionary with experiment results
        figures_dir: Directory to save figures in
        config: Visualization configuration
        
    Returns:
        Dictionary mapping figure types to file paths
    """
    # Set up visualization
    setup_visualization(config)
    
    # Initialize figure paths
    figure_paths = {}
    
    # Plot aggregate metrics comparison
    figure_paths['aggregate_metrics'] = plot_aggregate_metrics(
        results['aggregate_metrics'],
        'aggregate_metrics_comparison',
        figures_dir,
        config
    )
    
    # Plot radar chart comparison
    figure_paths['metric_radar'] = plot_metric_radar(
        results['aggregate_metrics'],
        'model_metrics_radar',
        figures_dir,
        config
    )
    
    # Plot individual scenario results
    for scenario_name, scenario_results in results['scenarios'].items():
        # Plot alignment comparison for this scenario
        figure_paths[f'alignment_{scenario_name}'] = plot_comparative_alignment(
            scenario_results,
            f'alignment_comparison_{scenario_name}',
            figures_dir,
            config,
            title="Alignment Score Comparison"
        )
        
        # Plot human value evolution
        human_values = np.array(scenario_results['raw_data']['human_values'])[:, 0, :]  # First agent only
        human_df = pd.DataFrame(
            human_values,
            columns=results['overall']['value_dimensions']
        )
        
        figure_paths[f'human_values_{scenario_name}'] = plot_value_evolution(
            human_df,
            f'human_value_evolution_{scenario_name}',
            figures_dir,
            config,
            title="Human Value Evolution"
        )
        
        # Plot model value evolution for each model
        for model_name in scenario_results['models']:
            model_values = np.array(scenario_results['raw_data']['model_values'][model_name])
            model_df = pd.DataFrame(
                model_values,
                columns=results['overall']['value_dimensions']
            )
            
            figure_paths[f'model_values_{model_name}_{scenario_name}'] = plot_value_evolution(
                model_df,
                f'model_value_evolution_{model_name}_{scenario_name}',
                figures_dir,
                config,
                title=f"{model_name} Value Evolution"
            )
            
    # Plot key metric comparisons across scenarios
    for metric in ['adaptation_accuracy', 'stability', 'user_satisfaction', 'agency_preservation']:
        figure_paths[f'metric_{metric}'] = plot_scenario_performance(
            results['scenarios'],
            metric,
            f'scenario_comparison_{metric}',
            figures_dir,
            config
        )
        
    return figure_paths


def generate_results_tables(results: Dict) -> Dict[str, pd.DataFrame]:
    """
    Generate tables summarizing experiment results.
    
    Args:
        results: Dictionary with experiment results
        
    Returns:
        Dictionary mapping table names to DataFrames
    """
    tables = {}
    
    # Table 1: Overall performance comparison
    # Create aggregated metrics DataFrame
    metrics_to_include = [
        'avg_adaptation_accuracy', 'avg_adaptation_response_time', 
        'avg_stability', 'avg_user_satisfaction', 'avg_agency_preservation'
    ]
    
    rows = []
    for model_name, metrics in results['aggregate_metrics'].items():
        row = {'Model': model_name}
        for metric in metrics_to_include:
            if metric in metrics:
                # Remove 'avg_' prefix
                clean_metric = metric[4:].replace('_', ' ').title()
                row[clean_metric] = f"{metrics[metric]:.3f} ± {metrics[f'std_{metric[4:]}']:.3f}"
        rows.append(row)
        
    tables['overall_performance'] = pd.DataFrame(rows)
    
    # Table 2: Scenario-specific performance
    # For each scenario, show the best model for each metric
    for scenario_name, scenario_data in results['scenarios'].items():
        rows = []
        metrics_to_compare = ['adaptation_accuracy', 'adaptation_response_time', 
                             'stability', 'user_satisfaction', 'agency_preservation']
        
        for metric in metrics_to_compare:
            best_model = None
            best_value = -float('inf')
            
            for model_name, model_data in scenario_data['models'].items():
                value = model_data['metrics'].get(metric, 0)
                
                # For response time, lower is better
                if metric == 'adaptation_response_time':
                    value = -value
                    
                if value > best_value:
                    best_value = value
                    best_model = model_name
                    
            # Revert the negation for response time
            if metric == 'adaptation_response_time':
                best_value = -best_value
                
            rows.append({
                'Metric': metric.replace('_', ' ').title(),
                'Best Model': best_model,
                'Value': f"{best_value:.3f}"
            })
            
        tables[f'scenario_{scenario_name}'] = pd.DataFrame(rows)
        
    # Table 3: Experimental setup summary
    table_rows = [
        {'Parameter': 'Value Dimensions', 'Value': ', '.join(results['overall']['value_dimensions'])},
        {'Parameter': 'Number of Scenarios', 'Value': results['overall']['n_scenarios']},
        {'Parameter': 'Number of Models', 'Value': results['overall']['n_models']},
    ]
    
    # Add scenario details
    for scenario_name, scenario_data in results['scenarios'].items():
        table_rows.append({
            'Parameter': f'Scenario: {scenario_name}', 
            'Value': scenario_data['scenario_description']
        })
        
    # Add model details
    for scenario_name, scenario_data in results['scenarios'].items():
        for model_name, model_data in scenario_data['models'].items():
            table_rows.append({
                'Parameter': f'Model: {model_name}', 
                'Value': model_data['model_description']
            })
        break  # Only need to do this once
        
    tables['experimental_setup'] = pd.DataFrame(table_rows)
    
    return tables
"""
Visualization module

This module implements functions for visualizing experimental results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import logging
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set Seaborn style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def plot_learning_curves(
    histories: Dict[str, Dict[str, List[float]]],
    title: str = 'Learning Curves',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot learning curves for multiple models.
    
    Args:
        histories: Dictionary mapping model names to training histories
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, history in histories.items():
        if 'train_scores' in history and history['train_scores']:
            ax.plot(
                history['train_scores'],
                label=f"{model_name} (Train)",
                linestyle='-'
            )
        
        if 'val_scores' in history and history['val_scores']:
            ax.plot(
                history['val_scores'],
                label=f"{model_name} (Validation)",
                linestyle='--'
            )
    
    ax.set_xlabel('Epochs / Iterations')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)
    
    # Add horizontal line at y=1.0 to indicate perfect score
    ax.axhline(y=1.0, color='r', linestyle=':', alpha=0.3)
    
    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_performance_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy',
    contexts: Optional[List[str]] = None,
    title: str = 'Performance Comparison',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot performance comparison across models and contexts.
    
    Args:
        results: Dictionary mapping model names to dictionaries of metrics
        metric: Metric to plot
        contexts: List of contexts to include (optional)
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Extract model names and metric values
    model_names = list(results.keys())
    
    if contexts is None:
        # If contexts are not provided, assume there's only one context
        # and extract values directly
        metric_values = [
            results[model].get('performance', {}).get(metric, np.nan)
            for model in model_names
        ]
        
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'Model': model_names,
            metric: metric_values
        })
        
        # Plot bar chart
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.barplot(data=df, x='Model', y=metric, ax=ax)
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
    else:
        # If contexts are provided, create a grouped bar chart
        data = []
        
        for model in model_names:
            for context in contexts:
                if context in results[model]:
                    value = results[model][context].get('performance', {}).get(metric, np.nan)
                    data.append({
                        'Model': model,
                        'Context': context,
                        metric: value
                    })
        
        # Create a DataFrame for plotting
        df = pd.DataFrame(data)
        
        # Plot grouped bar chart
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.barplot(data=df, x='Model', y=metric, hue='Context', ax=ax)
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title='Context')
    
    # Add horizontal line at y=1.0 to indicate perfect score (for metrics like accuracy)
    if metric.lower() in ['accuracy', 'f1', 'precision', 'recall', 'auc', 'r2']:
        ax.axhline(y=1.0, color='r', linestyle=':', alpha=0.3)
    
    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_radar_chart(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metrics: List[str] = ['performance', 'fairness', 'robustness', 'environmental_impact', 'interpretability'],
    title: str = 'Multi-Metric Comparison',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot radar chart for multi-metric comparison.
    
    Args:
        results: Dictionary mapping model names to dictionaries of category scores
        metrics: List of metrics to include
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Extract model names
    model_names = list(results.keys())
    
    # Number of metrics
    num_metrics = len(metrics)
    
    # Create angle for each metric
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    
    # Close the polygon
    angles += angles[:1]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Create color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    
    # Plot each model
    for i, model_name in enumerate(model_names):
        # Extract category scores for each metric
        values = []
        
        for metric in metrics:
            # Try to get the category score, default to 0.5 if not found
            score = results[model_name].get('category_scores', {}).get(metric, 0.5)
            values.append(score)
        
        # Close the polygon
        values += values[:1]
        
        # Plot the model
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=model_name)
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Set metrics as labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric.capitalize() for metric in metrics])
    
    # Set y limits
    ax.set_ylim(0, 1)
    
    # Add labels for y ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # Add title and legend
    ax.set_title(title, size=16, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_fairness_comparison(
    results: Dict[str, Dict[str, float]],
    sensitive_attributes: List[str],
    title: str = 'Fairness Comparison',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot fairness comparison across models.
    
    Args:
        results: Dictionary mapping model names to dictionaries of fairness metrics
        sensitive_attributes: List of sensitive attributes
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Extract model names
    model_names = list(results.keys())
    
    # Create a figure with one row per sensitive attribute
    fig, axes = plt.subplots(
        len(sensitive_attributes), 1,
        figsize=figsize,
        sharex=True
    )
    
    # If there's only one sensitive attribute, make axes a list
    if len(sensitive_attributes) == 1:
        axes = [axes]
    
    # For each sensitive attribute, plot fairness metrics
    for i, attribute in enumerate(sensitive_attributes):
        # Extract fairness metrics for this attribute
        fairness_data = []
        
        for model in model_names:
            if attribute in results[model].get('fairness', {}):
                metrics = results[model]['fairness'][attribute]
                
                # Get key fairness metrics
                demographic_parity = metrics.get('demographic_parity_diff', np.nan)
                equal_opportunity = metrics.get('equal_opportunity_diff', np.nan)
                
                fairness_data.append({
                    'Model': model,
                    'Demographic Parity Difference': demographic_parity,
                    'Equal Opportunity Difference': equal_opportunity
                })
        
        # Create a DataFrame for plotting
        df = pd.DataFrame(fairness_data)
        
        # Convert to long format for plotting
        df_long = pd.melt(
            df,
            id_vars=['Model'],
            value_vars=['Demographic Parity Difference', 'Equal Opportunity Difference'],
            var_name='Metric',
            value_name='Value'
        )
        
        # Plot grouped bar chart
        ax = axes[i]
        sns.barplot(data=df_long, x='Model', y='Value', hue='Metric', ax=ax)
        
        ax.set_title(f'Fairness Metrics for {attribute.capitalize()}')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Difference (lower is better)')
        
        if i == len(sensitive_attributes) - 1:
            ax.set_xlabel('Model')
        else:
            ax.set_xlabel('')
        
        ax.legend(title='Metric')
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_robustness_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]],
    title: str = 'Robustness Comparison',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot robustness comparison across models.
    
    Args:
        results: Dictionary mapping model names to dictionaries of robustness metrics
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Extract model names
    model_names = list(results.keys())
    
    # Create data for plotting
    robustness_data = []
    
    for model in model_names:
        rob = results[model].get('robustness', {})
        
        # Get noise robustness
        if 'noise' in rob and isinstance(rob['noise'], dict):
            noise_robustness = rob['noise'].get('retained_accuracy_pct', np.nan)
            if not np.isnan(noise_robustness):
                noise_robustness /= 100.0  # Convert percentage to fraction
        else:
            noise_robustness = np.nan
        
        # Get shift robustness
        if 'shift' in rob and isinstance(rob['shift'], dict):
            shift_robustness = rob['shift'].get('retained_accuracy_pct', np.nan)
            if not np.isnan(shift_robustness):
                shift_robustness /= 100.0  # Convert percentage to fraction
        else:
            shift_robustness = np.nan
        
        # Get adversarial robustness
        if 'adversarial' in rob and isinstance(rob['adversarial'], dict):
            adv_robustness = rob['adversarial'].get('adversarial_accuracy', np.nan)
        else:
            adv_robustness = np.nan
        
        robustness_data.append({
            'Model': model,
            'Noise Robustness': noise_robustness,
            'Shift Robustness': shift_robustness,
            'Adversarial Robustness': adv_robustness
        })
    
    # Create a DataFrame for plotting
    df = pd.DataFrame(robustness_data)
    
    # Convert to long format for plotting
    df_long = pd.melt(
        df,
        id_vars=['Model'],
        value_vars=['Noise Robustness', 'Shift Robustness', 'Adversarial Robustness'],
        var_name='Metric',
        value_name='Value'
    )
    
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.barplot(data=df_long, x='Model', y='Value', hue='Metric', ax=ax)
    
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Model')
    ax.set_ylabel('Robustness (higher is better)')
    ax.legend(title='Robustness Type')
    
    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_environmental_impact(
    results: Dict[str, Dict[str, float]],
    title: str = 'Environmental Impact Comparison',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot environmental impact comparison across models.
    
    Args:
        results: Dictionary mapping model names to dictionaries of environmental metrics
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Extract model names
    model_names = list(results.keys())
    
    # Create data for plotting
    env_data = []
    
    for model in model_names:
        env = results[model].get('environmental_impact', {})
        
        # Get key environmental metrics
        energy_kwh = env.get('total_energy_kwh', np.nan)
        energy_per_sample = env.get('total_energy_kwh_per_sample', np.nan)
        carbon_emissions = env.get('carbon_emissions_kg', np.nan)
        training_time = env.get('elapsed_time_seconds', np.nan)
        
        env_data.append({
            'Model': model,
            'Total Energy (kWh)': energy_kwh,
            'Energy per Sample (kWh)': energy_per_sample,
            'Carbon Emissions (kg CO2e)': carbon_emissions,
            'Training Time (s)': training_time
        })
    
    # Create a DataFrame for plotting
    df = pd.DataFrame(env_data)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot total energy
    sns.barplot(data=df, x='Model', y='Total Energy (kWh)', ax=axes[0, 0])
    axes[0, 0].set_title('Total Energy Consumption')
    axes[0, 0].set_ylabel('Energy (kWh)')
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot energy per sample
    sns.barplot(data=df, x='Model', y='Energy per Sample (kWh)', ax=axes[0, 1])
    axes[0, 1].set_title('Energy per Sample')
    axes[0, 1].set_ylabel('Energy per Sample (kWh)')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
    
    # Plot carbon emissions
    sns.barplot(data=df, x='Model', y='Carbon Emissions (kg CO2e)', ax=axes[1, 0])
    axes[1, 0].set_title('Carbon Emissions')
    axes[1, 0].set_ylabel('Carbon Emissions (kg CO2e)')
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot training time
    sns.barplot(data=df, x='Model', y='Training Time (s)', ax=axes[1, 1])
    axes[1, 1].set_title('Training Time')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_interpretability_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]],
    title: str = 'Interpretability Comparison',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot interpretability comparison across models.
    
    Args:
        results: Dictionary mapping model names to dictionaries of interpretability metrics
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Extract model names
    model_names = list(results.keys())
    
    # Create data for plotting
    interp_data = []
    
    for model in model_names:
        interp = results[model].get('interpretability', {})
        
        # Get stability metrics
        if 'stability' in interp and isinstance(interp['stability'], dict):
            stability = interp['stability'].get('mean_stability', np.nan)
        else:
            stability = np.nan
        
        # Get concentration metrics
        if 'concentration' in interp and isinstance(interp['concentration'], dict):
            feature_concentration = interp['concentration'].get('feature_concentration_ratio', np.nan)
            gini_coefficient = interp['concentration'].get('gini_coefficient', np.nan)
        else:
            feature_concentration = np.nan
            gini_coefficient = np.nan
        
        interp_data.append({
            'Model': model,
            'Attribution Stability': stability,
            'Feature Concentration': feature_concentration,
            'Gini Coefficient': gini_coefficient
        })
    
    # Create a DataFrame for plotting
    df = pd.DataFrame(interp_data)
    
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to long format
    df_long = pd.melt(
        df,
        id_vars=['Model'],
        value_vars=['Attribution Stability', 'Feature Concentration', 'Gini Coefficient'],
        var_name='Metric',
        value_name='Value'
    )
    
    # Plot
    sns.barplot(data=df_long, x='Model', y='Value', hue='Metric', ax=ax)
    
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Model')
    ax.set_ylabel('Value')
    ax.legend(title='Interpretability Metric')
    
    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_context_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]],
    contexts: List[str],
    title: str = 'Context-Specific Performance',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot context-specific performance comparison across models.
    
    Args:
        results: Dictionary mapping model names to dictionaries of context scores
        contexts: List of contexts to include
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Extract model names
    model_names = list(results.keys())
    
    # Create data for plotting
    context_data = []
    
    for model in model_names:
        for context in contexts:
            if context in results[model]:
                overall_score = results[model][context].get('overall_score', np.nan)
                
                context_data.append({
                    'Model': model,
                    'Context': context,
                    'Overall Score': overall_score
                })
    
    # Create a DataFrame for plotting
    df = pd.DataFrame(context_data)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.barplot(data=df, x='Model', y='Overall Score', hue='Context', ax=ax)
    
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Model')
    ax.set_ylabel('Overall Score')
    ax.legend(title='Context')
    
    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_trade_offs(
    results: Dict[str, Dict[str, float]],
    x_metric: str = 'performance',
    y_metric: str = 'fairness',
    title: str = 'Performance-Fairness Trade-off',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot trade-offs between two metrics.
    
    Args:
        results: Dictionary mapping model names to dictionaries of metrics
        x_metric: Metric for x-axis
        y_metric: Metric for y-axis
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Extract model names
    model_names = list(results.keys())
    
    # Create data for plotting
    trade_off_data = []
    
    for model in model_names:
        x_value = results[model].get('category_scores', {}).get(x_metric, np.nan)
        y_value = results[model].get('category_scores', {}).get(y_metric, np.nan)
        
        trade_off_data.append({
            'Model': model,
            x_metric.capitalize(): x_value,
            y_metric.capitalize(): y_value
        })
    
    # Create a DataFrame for plotting
    df = pd.DataFrame(trade_off_data)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    sns.scatterplot(
        data=df,
        x=x_metric.capitalize(),
        y=y_metric.capitalize(),
        s=100,
        ax=ax
    )
    
    # Add model names as labels
    for i, row in df.iterrows():
        ax.annotate(
            row['Model'],
            (row[x_metric.capitalize()], row[y_metric.capitalize()]),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Add horizontal and vertical lines to indicate "perfect" performance
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
    
    # Add diagonal line to indicate equal performance
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
    
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def create_context_profile_visualization(
    results: Dict[str, Any],
    title: str = 'Context Profile',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a comprehensive context profile visualization.
    
    Args:
        results: Dictionary of evaluation results for a model
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Create a figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5])
    
    # 1. Radar chart for category scores
    category_scores = results.get('category_scores', {})
    metrics = [
        'performance',
        'fairness',
        'robustness',
        'environmental_impact',
        'interpretability'
    ]
    
    # Extract values for each metric
    values = []
    for metric in metrics:
        values.append(category_scores.get(metric, 0.5))
    
    # Create angles for each metric
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    
    # Close the polygon
    values += values[:1]
    angles += angles[:1]
    
    # Create radar chart
    ax_radar = fig.add_subplot(gs[0, 0], polar=True)
    
    ax_radar.plot(angles, values, 'o-', linewidth=2)
    ax_radar.fill(angles, values, alpha=0.25)
    
    # Set metrics as labels
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([metric.capitalize() for metric in metrics])
    
    # Set y limits
    ax_radar.set_ylim(0, 1)
    
    # Add title
    ax_radar.set_title('Category Scores', size=12)
    
    # 2. Bar chart for metrics details
    metrics_details = results.get('metrics', {})
    
    # Extract a subset of metrics for visualization
    selected_metrics = [
        'performance_accuracy',
        'fairness_race_dp_diff',
        'robustness_noise',
        'environmental_energy',
        'interpretability_stability'
    ]
    
    # Create lists for plotting
    metric_names = []
    metric_values = []
    
    for metric in selected_metrics:
        if metric in metrics_details:
            metric_names.append(metric.replace('_', ' ').title())
            metric_values.append(metrics_details[metric])
    
    # Create bar chart
    ax_bar = fig.add_subplot(gs[0, 1])
    
    y_pos = np.arange(len(metric_names))
    ax_bar.barh(y_pos, metric_values)
    
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(metric_names)
    ax_bar.invert_yaxis()  # Labels read top-to-bottom
    ax_bar.set_xlabel('Value')
    ax_bar.set_title('Selected Metrics', size=12)
    
    # 3. Table with key information
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    
    # Create table data
    table_data = []
    
    # Add overall score
    table_data.append(['Overall Score', f"{results.get('overall_score', 0.0):.3f}"])
    
    # Add performance metrics
    perf = results.get('performance', {})
    for metric, value in perf.items():
        if isinstance(value, (int, float)):
            table_data.append([f"Performance: {metric.capitalize()}", f"{value:.3f}"])
    
    # Add fairness metrics
    fair = results.get('fairness', {})
    for feature, metrics in fair.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    table_data.append([f"Fairness: {feature} {metric}", f"{value:.3f}"])
    
    # Add robustness metrics
    rob = results.get('robustness', {})
    for rob_type, metrics in rob.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    table_data.append([f"Robustness: {rob_type} {metric}", f"{value:.3f}"])
    
    # Create table
    table = ax_table.table(
        cellText=table_data,
        colWidths=[0.7, 0.3],
        loc='center',
        cellLoc='left'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Add title to the whole figure
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def generate_all_visualizations(
    results: Dict[str, Dict[str, Any]],
    context_results: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: str,
    contexts: List[str]
) -> Dict[str, str]:
    """
    Generate all visualizations for experimental results.
    
    Args:
        results: Dictionary mapping model names to dictionaries of metrics
        context_results: Dictionary mapping model names to dictionaries of context scores
        output_dir: Directory to save visualizations
        contexts: List of contexts
        
    Returns:
        dict: Dictionary mapping visualization names to file paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dictionary to store visualization paths
    visualization_paths = {}
    
    # 1. Performance comparison across models
    try:
        fig_perf = plot_performance_comparison(
            {model: res.get('performance', {}) for model, res in results.items()},
            metric='accuracy',
            title='Performance Comparison (Accuracy)',
            save_path=os.path.join(output_dir, 'performance_comparison.png')
        )
        plt.close(fig_perf)
        visualization_paths['performance_comparison'] = os.path.join(output_dir, 'performance_comparison.png')
    except Exception as e:
        logger.error(f"Error generating performance comparison: {str(e)}")
    
    # 2. Radar chart for multi-metric comparison
    try:
        category_scores = {
            model: {'category_scores': res.get('category_scores', {})}
            for model, res in results.items()
        }
        
        fig_radar = plot_radar_chart(
            category_scores,
            title='Multi-Metric Comparison',
            save_path=os.path.join(output_dir, 'radar_chart.png')
        )
        plt.close(fig_radar)
        visualization_paths['radar_chart'] = os.path.join(output_dir, 'radar_chart.png')
    except Exception as e:
        logger.error(f"Error generating radar chart: {str(e)}")
    
    # 3. Fairness comparison
    try:
        fig_fair = plot_fairness_comparison(
            {model: res for model, res in results.items()},
            sensitive_attributes=['race', 'sex', 'age'],
            title='Fairness Comparison',
            save_path=os.path.join(output_dir, 'fairness_comparison.png')
        )
        plt.close(fig_fair)
        visualization_paths['fairness_comparison'] = os.path.join(output_dir, 'fairness_comparison.png')
    except Exception as e:
        logger.error(f"Error generating fairness comparison: {str(e)}")
    
    # 4. Robustness comparison
    try:
        fig_rob = plot_robustness_comparison(
            {model: res for model, res in results.items()},
            title='Robustness Comparison',
            save_path=os.path.join(output_dir, 'robustness_comparison.png')
        )
        plt.close(fig_rob)
        visualization_paths['robustness_comparison'] = os.path.join(output_dir, 'robustness_comparison.png')
    except Exception as e:
        logger.error(f"Error generating robustness comparison: {str(e)}")
    
    # 5. Environmental impact comparison
    try:
        fig_env = plot_environmental_impact(
            {model: res for model, res in results.items()},
            title='Environmental Impact Comparison',
            save_path=os.path.join(output_dir, 'environmental_impact.png')
        )
        plt.close(fig_env)
        visualization_paths['environmental_impact'] = os.path.join(output_dir, 'environmental_impact.png')
    except Exception as e:
        logger.error(f"Error generating environmental impact comparison: {str(e)}")
    
    # 6. Interpretability comparison
    try:
        fig_interp = plot_interpretability_comparison(
            {model: res for model, res in results.items()},
            title='Interpretability Comparison',
            save_path=os.path.join(output_dir, 'interpretability_comparison.png')
        )
        plt.close(fig_interp)
        visualization_paths['interpretability_comparison'] = os.path.join(output_dir, 'interpretability_comparison.png')
    except Exception as e:
        logger.error(f"Error generating interpretability comparison: {str(e)}")
    
    # 7. Context comparison
    try:
        fig_context = plot_context_comparison(
            context_results,
            contexts=contexts,
            title='Context-Specific Performance',
            save_path=os.path.join(output_dir, 'context_comparison.png')
        )
        plt.close(fig_context)
        visualization_paths['context_comparison'] = os.path.join(output_dir, 'context_comparison.png')
    except Exception as e:
        logger.error(f"Error generating context comparison: {str(e)}")
    
    # 8. Trade-offs between performance and fairness
    try:
        fig_tradeoff1 = plot_trade_offs(
            {model: res for model, res in results.items()},
            x_metric='performance',
            y_metric='fairness',
            title='Performance-Fairness Trade-off',
            save_path=os.path.join(output_dir, 'performance_fairness_tradeoff.png')
        )
        plt.close(fig_tradeoff1)
        visualization_paths['performance_fairness_tradeoff'] = os.path.join(output_dir, 'performance_fairness_tradeoff.png')
    except Exception as e:
        logger.error(f"Error generating performance-fairness trade-off: {str(e)}")
    
    # 9. Trade-offs between performance and robustness
    try:
        fig_tradeoff2 = plot_trade_offs(
            {model: res for model, res in results.items()},
            x_metric='performance',
            y_metric='robustness',
            title='Performance-Robustness Trade-off',
            save_path=os.path.join(output_dir, 'performance_robustness_tradeoff.png')
        )
        plt.close(fig_tradeoff2)
        visualization_paths['performance_robustness_tradeoff'] = os.path.join(output_dir, 'performance_robustness_tradeoff.png')
    except Exception as e:
        logger.error(f"Error generating performance-robustness trade-off: {str(e)}")
    
    # 10. Context profiles for each model
    for model, contexts_results in context_results.items():
        for context, results in contexts_results.items():
            try:
                fig_profile = create_context_profile_visualization(
                    results,
                    title=f'{model} - {context} Context Profile',
                    save_path=os.path.join(output_dir, f'{model}_{context}_profile.png')
                )
                plt.close(fig_profile)
                visualization_paths[f'{model}_{context}_profile'] = os.path.join(output_dir, f'{model}_{context}_profile.png')
            except Exception as e:
                logger.error(f"Error generating context profile for {model} in {context}: {str(e)}")
    
    return visualization_paths


if __name__ == "__main__":
    # Test visualization functions with dummy data
    
    # Create dummy results
    models = ['LogisticRegression', 'RandomForest', 'SVM', 'MLP', 'XGBoost']
    contexts = ['healthcare', 'finance', 'vision']
    
    results = {}
    context_results = {}
    
    for model in models:
        # Create random results for each model
        model_results = {
            'performance': {
                'accuracy': np.random.uniform(0.7, 0.95),
                'precision': np.random.uniform(0.7, 0.95),
                'recall': np.random.uniform(0.7, 0.95),
                'f1': np.random.uniform(0.7, 0.95)
            },
            'fairness': {
                'race': {
                    'demographic_parity_diff': np.random.uniform(0.05, 0.3),
                    'equal_opportunity_diff': np.random.uniform(0.05, 0.3),
                    'disparate_impact': np.random.uniform(0.7, 1.3)
                },
                'sex': {
                    'demographic_parity_diff': np.random.uniform(0.05, 0.3),
                    'equal_opportunity_diff': np.random.uniform(0.05, 0.3),
                    'disparate_impact': np.random.uniform(0.7, 1.3)
                },
                'age': {
                    'demographic_parity_diff': np.random.uniform(0.05, 0.3),
                    'equal_opportunity_diff': np.random.uniform(0.05, 0.3),
                    'disparate_impact': np.random.uniform(0.7, 1.3)
                }
            },
            'robustness': {
                'noise': {
                    'original_accuracy': np.random.uniform(0.7, 0.95),
                    'noisy_accuracy': np.random.uniform(0.6, 0.9),
                    'noise_drop': np.random.uniform(0.05, 0.2),
                    'retained_accuracy_pct': np.random.uniform(80, 95)
                },
                'shift': {
                    'original_accuracy': np.random.uniform(0.7, 0.95),
                    'shifted_accuracy': np.random.uniform(0.6, 0.9),
                    'shift_drop': np.random.uniform(0.05, 0.2),
                    'retained_accuracy_pct': np.random.uniform(80, 95)
                },
                'adversarial': {
                    'adversarial_accuracy': np.random.uniform(0.5, 0.8)
                }
            },
            'environmental_impact': {
                'elapsed_time_seconds': np.random.uniform(10, 500),
                'cpu_memory_usage_mb': np.random.uniform(100, 1000),
                'total_energy_kwh': np.random.uniform(0.01, 0.5),
                'total_energy_kwh_per_sample': np.random.uniform(0.00001, 0.0005),
                'carbon_emissions_kg': np.random.uniform(0.005, 0.2)
            },
            'interpretability': {
                'stability': {
                    'mean_stability': np.random.uniform(0.6, 0.9),
                    'std_stability': np.random.uniform(0.05, 0.2)
                },
                'concentration': {
                    'feature_concentration_ratio': np.random.uniform(0.1, 0.5),
                    'gini_coefficient': np.random.uniform(0.3, 0.7)
                },
                'clarity': {
                    'mean_gradient_magnitude': np.random.uniform(0.1, 0.9),
                    'std_gradient_magnitude': np.random.uniform(0.05, 0.2)
                }
            },
            'category_scores': {
                'performance': np.random.uniform(0.7, 0.95),
                'fairness': np.random.uniform(0.5, 0.9),
                'robustness': np.random.uniform(0.5, 0.9),
                'environmental_impact': np.random.uniform(0.4, 0.8),
                'interpretability': np.random.uniform(0.5, 0.9)
            },
            'metrics': {
                'performance_accuracy': np.random.uniform(0.7, 0.95),
                'fairness_race_dp_diff': np.random.uniform(0.05, 0.3),
                'robustness_noise': np.random.uniform(0.7, 0.9),
                'environmental_energy': np.random.uniform(0.5, 0.9),
                'interpretability_stability': np.random.uniform(0.6, 0.9)
            },
            'overall_score': np.random.uniform(0.6, 0.9)
        }
        
        results[model] = model_results
        
        # Create context results
        context_model_results = {}
        
        for context in contexts:
            context_results_dict = {
                'overall_score': np.random.uniform(0.6, 0.9),
                'category_scores': {
                    'performance': np.random.uniform(0.7, 0.95),
                    'fairness': np.random.uniform(0.5, 0.9),
                    'robustness': np.random.uniform(0.5, 0.9),
                    'environmental_impact': np.random.uniform(0.4, 0.8),
                    'interpretability': np.random.uniform(0.5, 0.9)
                },
                'metrics': {
                    'performance_accuracy': np.random.uniform(0.7, 0.95),
                    'fairness_race_dp_diff': np.random.uniform(0.05, 0.3),
                    'robustness_noise': np.random.uniform(0.7, 0.9),
                    'environmental_energy': np.random.uniform(0.5, 0.9),
                    'interpretability_stability': np.random.uniform(0.6, 0.9)
                }
            }
            
            context_model_results[context] = context_results_dict
        
        context_results[model] = context_model_results
    
    # Generate all visualizations
    output_dir = '../test_visualizations'
    visualization_paths = generate_all_visualizations(
        results,
        context_results,
        output_dir,
        contexts
    )
    
    print("Generated visualizations:")
    for name, path in visualization_paths.items():
        print(f"  {name}: {path}")
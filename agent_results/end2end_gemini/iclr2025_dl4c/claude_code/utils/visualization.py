#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization utilities for the IETA framework.
This module provides functions to visualize experiment results and metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def set_plot_style():
    """Set the style for matplotlib plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)
    sns.set_palette("deep")

def plot_pass_rates(iterations, pass_rates, pass_k, output_path):
    """
    Plot pass@k rates over iterations.
    
    Args:
        iterations (list): List of iteration numbers
        pass_rates (list): List of pass@k rates for each iteration
        pass_k (list): List of k values
        output_path (str or Path): Path to save the plot
    """
    set_plot_style()
    plt.figure(figsize=(10, 6))
    
    for i, k in enumerate(pass_k):
        # Extract pass@k values for this k
        values = [rates[i] for rates in pass_rates]
        plt.plot(iterations, values, marker='o', linewidth=2, label=f'Pass@{k}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Pass Rate')
    plt.title('Pass@k Rates Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Pass rates plot saved to {output_path}")

def plot_execution_rates_comparison(iterations, execution_rates, method_name, output_path):
    """
    Plot execution rates over iterations.
    
    Args:
        iterations (list): List of iteration numbers
        execution_rates (list): List of execution rates for each iteration
        method_name (str): Name of the method
        output_path (str or Path): Path to save the plot
    """
    set_plot_style()
    plt.figure(figsize=(10, 6))
    
    plt.plot(iterations, execution_rates, marker='o', linewidth=2, label=method_name)
    
    # Add a constant line for baseline comparison if applicable
    if method_name != "baseline":
        plt.axhline(y=execution_rates[0], color='r', linestyle='--', label='Initial')
    
    plt.xlabel('Iteration')
    plt.ylabel('Execution Rate')
    plt.title('Execution Rate Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Execution rates plot saved to {output_path}")

def plot_error_frequency(error_frequencies, output_path):
    """
    Plot error frequencies across iterations.
    
    Args:
        error_frequencies (list): List of error frequency dictionaries for each iteration
        output_path (str or Path): Path to save the plot
    """
    set_plot_style()
    
    # Convert to DataFrame for easier plotting
    iterations = list(range(1, len(error_frequencies) + 1))
    error_types = set()
    
    for freq in error_frequencies:
        error_types.update(freq.keys())
    
    error_types = sorted(error_types)
    
    data = {
        'Iteration': [],
        'Error Type': [],
        'Frequency': []
    }
    
    for i, freq in enumerate(error_frequencies):
        for error_type in error_types:
            data['Iteration'].append(iterations[i])
            data['Error Type'].append(error_type)
            data['Frequency'].append(freq.get(error_type, 0))
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(x='Iteration', y='Frequency', hue='Error Type', 
                    data=df, marker='o', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Error Frequency')
    plt.title('Error Type Frequencies Over Iterations')
    plt.legend(title='Error Type')
    plt.grid(True)
    plt.tight_layout()
    
    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Error frequency plot saved to {output_path}")

def plot_training_loss(losses, method, output_path):
    """
    Plot training losses over iterations.
    
    Args:
        losses (list): List of loss lists for each iteration
        method (str): Method name (DPO or RLAIF)
        output_path (str or Path): Path to save the plot
    """
    set_plot_style()
    plt.figure(figsize=(10, 6))
    
    for i, loss_curve in enumerate(losses):
        if loss_curve:  # Skip empty loss curves
            steps = list(range(1, len(loss_curve) + 1))
            plt.plot(steps, loss_curve, alpha=0.5, linewidth=1, label=f'Iteration {i+1}')
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title(f'{method.upper()} Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training loss plot saved to {output_path}")

def plot_method_comparison(methods, pass_rates, execution_rates, output_path):
    """
    Plot a comparison of different methods.
    
    Args:
        methods (list): List of method names
        pass_rates (list): List of final pass@k rates for each method
        execution_rates (list): List of final execution rates for each method
        output_path (str or Path): Path to save the plot
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot pass@1 rates
    pass1_rates = [rates[0] for rates in pass_rates]
    ax1.bar(methods, pass1_rates, color=sns.color_palette("deep")[:len(methods)])
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Pass@1 Rate')
    ax1.set_title('Pass@1 Rate Comparison')
    ax1.grid(axis='y')
    
    # Plot execution rates
    ax2.bar(methods, execution_rates, color=sns.color_palette("deep")[:len(methods)])
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Execution Rate')
    ax2.set_title('Execution Rate Comparison')
    ax2.grid(axis='y')
    
    plt.tight_layout()
    
    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Method comparison plot saved to {output_path}")

def plot_variable_state_influence(variable_states, outcome_distribution, output_path):
    """
    Plot the influence of variable states on execution outcomes.
    
    Args:
        variable_states (dict): Dictionary of variable states and their frequencies
        outcome_distribution (dict): Distribution of outcomes for each variable state
        output_path (str or Path): Path to save the plot
    """
    set_plot_style()
    
    # Create a DataFrame for the heatmap
    variables = list(variable_states.keys())
    outcomes = ["S_succ", "S_err", "S_fail_test", "S_timeout"]
    
    data = np.zeros((len(variables), len(outcomes)))
    
    for i, var in enumerate(variables):
        for j, outcome in enumerate(outcomes):
            data[i, j] = outcome_distribution.get(var, {}).get(outcome, 0)
    
    # Normalize the data
    data = data / data.sum(axis=1, keepdims=True)
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(data, xticklabels=outcomes, yticklabels=variables, 
                   annot=True, fmt=".2f", cmap="viridis")
    
    plt.title("Variable State Influence on Execution Outcomes")
    plt.tight_layout()
    
    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Variable state influence plot saved to {output_path}")

def plot_trace_complexity_vs_performance(trace_complexity, performance_metrics, output_path):
    """
    Plot the relationship between trace complexity and model performance.
    
    Args:
        trace_complexity (list): List of trace complexity scores
        performance_metrics (list): List of performance metrics (e.g., pass rate)
        output_path (str or Path): Path to save the plot
    """
    set_plot_style()
    plt.figure(figsize=(10, 6))
    
    plt.scatter(trace_complexity, performance_metrics, alpha=0.7)
    
    # Add a trend line
    z = np.polyfit(trace_complexity, performance_metrics, 1)
    p = np.poly1d(z)
    plt.plot(trace_complexity, p(trace_complexity), "r--")
    
    plt.xlabel('Trace Complexity')
    plt.ylabel('Performance (Pass Rate)')
    plt.title('Trace Complexity vs. Model Performance')
    plt.grid(True)
    plt.tight_layout()
    
    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Trace complexity vs. performance plot saved to {output_path}")

def plot_code_quality_metrics(metrics, methods, output_path):
    """
    Plot code quality metrics for different methods.
    
    Args:
        metrics (dict): Dictionary of code quality metrics for each method
        methods (list): List of method names
        output_path (str or Path): Path to save the plot
    """
    set_plot_style()
    
    # Get the metric names
    metric_names = list(next(iter(metrics.values())).keys())
    num_metrics = len(metric_names)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_metrics, figsize=(15, 6))
    
    for i, metric in enumerate(metric_names):
        metric_values = [metrics[method][metric] for method in methods]
        axes[i].bar(methods, metric_values, color=sns.color_palette("deep")[:len(methods)])
        axes[i].set_title(metric)
        axes[i].set_xlabel('Method')
        axes[i].set_ylabel('Score')
        axes[i].grid(axis='y')
    
    plt.tight_layout()
    
    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Code quality metrics plot saved to {output_path}")

def plot_ablation_results(ablation_configs, metrics, metric_name, output_path):
    """
    Plot results from ablation studies.
    
    Args:
        ablation_configs (list): List of ablation configuration descriptions
        metrics (list): List of metric values for each configuration
        metric_name (str): Name of the metric being plotted
        output_path (str or Path): Path to save the plot
    """
    set_plot_style()
    plt.figure(figsize=(12, 6))
    
    plt.bar(ablation_configs, metrics, color=sns.color_palette("deep")[:len(ablation_configs)])
    
    plt.xlabel('Configuration')
    plt.ylabel(metric_name)
    plt.title(f'Ablation Study: Impact on {metric_name}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    
    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Ablation study plot saved to {output_path}")

def generate_summary_dashboard(results, output_path):
    """
    Generate a comprehensive summary dashboard of all results.
    
    Args:
        results (dict): Dictionary containing all experiment results
        output_path (str or Path): Path to save the dashboard
    """
    set_plot_style()
    fig = plt.figure(figsize=(20, 15))
    
    # Define grid layout
    gs = fig.add_gridspec(3, 3)
    
    # 1. Pass@k rates plot
    ax1 = fig.add_subplot(gs[0, 0:2])
    if "iterations" in results and "pass_rates" in results:
        iterations = results["iterations"]
        pass_rates = results["pass_rates"]
        pass_k = [1, 10, 100]  # Assuming standard pass@k values
        
        for i, k in enumerate(pass_k):
            if i < len(pass_rates[0]):  # Ensure we have data for this k
                values = [rates[i] for rates in pass_rates]
                ax1.plot(iterations, values, marker='o', linewidth=2, label=f'Pass@{k}')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Pass Rate')
        ax1.set_title('Pass@k Rates Over Iterations')
        ax1.legend()
        ax1.grid(True)
    
    # 2. Execution rates plot
    ax2 = fig.add_subplot(gs[0, 2])
    if "iterations" in results and "execution_rates" in results:
        iterations = results["iterations"]
        execution_rates = results["execution_rates"]
        
        ax2.plot(iterations, execution_rates, marker='o', linewidth=2, color='blue')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Execution Rate')
        ax2.set_title('Execution Rate Over Iterations')
        ax2.grid(True)
    
    # 3. Error frequencies plot
    ax3 = fig.add_subplot(gs[1, 0:2])
    if "iterations" in results and "error_frequencies" in results:
        iterations = results["iterations"]
        error_frequencies = results["error_frequencies"]
        
        error_types = set()
        for freq in error_frequencies:
            error_types.update(freq.keys())
        error_types = sorted(error_types)
        
        for error_type in error_types:
            values = [freq.get(error_type, 0) for freq in error_frequencies]
            ax3.plot(iterations, values, marker='o', linewidth=2, label=error_type)
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Error Frequency')
        ax3.set_title('Error Type Frequencies Over Iterations')
        ax3.legend()
        ax3.grid(True)
    
    # 4. Training loss plot
    ax4 = fig.add_subplot(gs[1, 2])
    if "training_losses" in results and results["training_losses"]:
        # For simplicity, plot the last iteration's loss curve
        last_loss = results["training_losses"][-1]
        if last_loss:
            steps = list(range(1, len(last_loss) + 1))
            ax4.plot(steps, last_loss, linewidth=2, color='blue')
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Loss')
            ax4.set_title('Training Loss (Final Iteration)')
            ax4.grid(True)
    
    # 5. Summary statistics table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('tight')
    ax5.axis('off')
    
    # Create summary data
    if all(k in results for k in ["iterations", "pass_rates", "execution_rates"]):
        table_data = []
        table_data.append(["Metric", "Initial", "Final", "Improvement"])
        
        # Pass@k rates
        for i, k in enumerate([1, 10, 100]):
            if i < len(results["pass_rates"][0]):
                initial = results["pass_rates"][0][i]
                final = results["pass_rates"][-1][i]
                improvement = final - initial
                table_data.append([f"Pass@{k}", f"{initial:.4f}", f"{final:.4f}", f"{improvement:.4f}"])
        
        # Execution rate
        initial_exec = results["execution_rates"][0]
        final_exec = results["execution_rates"][-1]
        exec_improvement = final_exec - initial_exec
        table_data.append(["Execution Rate", f"{initial_exec:.4f}", f"{final_exec:.4f}", f"{exec_improvement:.4f}"])
        
        # Error rates for top errors
        if "error_frequencies" in results and results["error_frequencies"]:
            # Find the top 3 errors from the initial iteration
            initial_errors = results["error_frequencies"][0]
            top_errors = sorted(initial_errors.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for error_type, initial_freq in top_errors:
                final_freq = results["error_frequencies"][-1].get(error_type, 0)
                improvement = initial_freq - final_freq
                table_data.append([f"{error_type} Rate", f"{initial_freq:.4f}", f"{final_freq:.4f}", f"{improvement:.4f}"])
        
        # Create the table
        ax5.table(cellText=table_data, colWidths=[0.25, 0.25, 0.25, 0.25], 
                 loc='center', cellLoc='center')
        ax5.set_title('Summary Statistics', pad=20)
    
    plt.tight_layout()
    
    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Summary dashboard saved to {output_path}")

def generate_comparison_dashboard(results_dict, methods, output_path):
    """
    Generate a dashboard comparing results from different methods.
    
    Args:
        results_dict (dict): Dictionary mapping method names to their results
        methods (list): List of method names to include in the comparison
        output_path (str or Path): Path to save the dashboard
    """
    set_plot_style()
    fig = plt.figure(figsize=(20, 15))
    
    # Define grid layout
    gs = fig.add_gridspec(3, 2)
    
    # 1. Pass@1 comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Extract Pass@1 values for each method
    pass1_values = []
    for method in methods:
        if method in results_dict and "pass_rates" in results_dict[method]:
            # Get the final Pass@1 rate
            pass1_values.append(results_dict[method]["pass_rates"][-1][0])
        else:
            pass1_values.append(0)
    
    ax1.bar(methods, pass1_values, color=sns.color_palette("deep")[:len(methods)])
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Pass@1 Rate')
    ax1.set_title('Pass@1 Rate Comparison')
    ax1.grid(axis='y')
    
    # 2. Execution rate comparison
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Extract execution rate values for each method
    exec_values = []
    for method in methods:
        if method in results_dict and "execution_rates" in results_dict[method]:
            # Get the final execution rate
            exec_values.append(results_dict[method]["execution_rates"][-1])
        else:
            exec_values.append(0)
    
    ax2.bar(methods, exec_values, color=sns.color_palette("deep")[:len(methods)])
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Execution Rate')
    ax2.set_title('Execution Rate Comparison')
    ax2.grid(axis='y')
    
    # 3. Error frequency comparison
    ax3 = fig.add_subplot(gs[1, :])
    
    # Extract error frequencies for each method
    error_types = set()
    for method in methods:
        if method in results_dict and "error_frequencies" in results_dict[method]:
            for freq in results_dict[method]["error_frequencies"]:
                error_types.update(freq.keys())
    
    error_types = sorted(list(error_types))
    
    # Create data for grouped bar chart
    error_data = {method: [] for method in methods}
    
    for error_type in error_types:
        for method in methods:
            if method in results_dict and "error_frequencies" in results_dict[method]:
                # Get the final error frequency
                error_data[method].append(
                    results_dict[method]["error_frequencies"][-1].get(error_type, 0)
                )
            else:
                error_data[method].append(0)
    
    # Set up the plot
    x = np.arange(len(error_types))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        ax3.bar(x + i * width - 0.4 + width/2, error_data[method], width, label=method)
    
    ax3.set_xlabel('Error Type')
    ax3.set_ylabel('Error Frequency')
    ax3.set_title('Error Type Frequency Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(error_types, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y')
    
    # 4. Learning curves comparison
    ax4 = fig.add_subplot(gs[2, :])
    
    for method in methods:
        if method in results_dict and "iterations" in results_dict[method] and "pass_rates" in results_dict[method]:
            iterations = results_dict[method]["iterations"]
            # Extract Pass@1 values over iterations
            pass1_over_time = [rates[0] for rates in results_dict[method]["pass_rates"]]
            ax4.plot(iterations, pass1_over_time, marker='o', linewidth=2, label=method)
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Pass@1 Rate')
    ax4.set_title('Learning Curve Comparison (Pass@1 Rate)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison dashboard saved to {output_path}")
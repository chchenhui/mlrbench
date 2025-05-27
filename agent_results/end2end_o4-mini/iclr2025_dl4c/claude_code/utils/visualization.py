import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import json

def plot_results(results: Dict[str, Any], output_dir: str):
    """
    Generate visualizations from experimental results.
    
    Args:
        results: Dictionary containing experimental results
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("Set2")
    
    # Plot 1: Acceptance Rates Comparison
    plot_acceptance_rates(results, output_dir)
    
    # Plot 2: Edit Distance Comparison
    plot_edit_distances(results, output_dir)
    
    # Plot 3: Task Completion Times
    plot_completion_times(results, output_dir)
    
    # Plot 4: Code Quality Scores
    plot_code_quality(results, output_dir)
    
    # Plot 5: Overall Performance Improvements
    plot_performance_improvements(results, output_dir)
    
    # Plot 6: Individual Developer Performance
    plot_developer_performance(results, output_dir)

def plot_acceptance_rates(results: Dict[str, Any], output_dir: str):
    """Plot comparison of acceptance rates between baseline and adaptive models."""
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    baseline_rates = results['baseline']['acceptance_rate']
    adaptive_rates = results['adaptive']['acceptance_rate']
    
    # Create boxplot
    data = pd.DataFrame({
        'Baseline': baseline_rates,
        'Adaptive': adaptive_rates
    })
    
    # Plot
    ax = sns.boxplot(data=data, width=0.4)
    sns.stripplot(data=data, color=".25", alpha=0.5, jitter=True)
    
    # Add means as text
    for i, model in enumerate(['Baseline', 'Adaptive']):
        mean_val = data[model].mean()
        plt.text(i, mean_val + 0.02, f'Mean: {mean_val:.2f}', 
                 horizontalalignment='center', fontweight='bold')
    
    # Set labels and title
    plt.ylabel('Suggestion Acceptance Rate')
    plt.title('Comparison of Suggestion Acceptance Rates', fontsize=14)
    
    # Add improvement percentage
    improvement = results['summary']['improvement']['acceptance_rate']
    plt.figtext(0.5, 0.01, f'Improvement: {improvement:.1f}%', 
                horizontalalignment='center', fontsize=12, fontweight='bold')
    
    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout for figtext
    plt.savefig(os.path.join(output_dir, 'acceptance_rates.png'), dpi=300)
    plt.close()

def plot_edit_distances(results: Dict[str, Any], output_dir: str):
    """Plot comparison of edit distances between baseline and adaptive models."""
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    baseline_distances = results['baseline']['avg_edit_distance']
    adaptive_distances = results['adaptive']['avg_edit_distance']
    
    # Convert to DataFrame for seaborn
    data = pd.DataFrame({
        'Baseline': baseline_distances,
        'Adaptive': adaptive_distances
    })
    
    # Plot
    ax = sns.boxplot(data=data, width=0.4)
    sns.stripplot(data=data, color=".25", alpha=0.5, jitter=True)
    
    # Add means as text
    for i, model in enumerate(['Baseline', 'Adaptive']):
        mean_val = data[model].mean()
        plt.text(i, mean_val + 0.02, f'Mean: {mean_val:.2f}', 
                 horizontalalignment='center', fontweight='bold')
    
    # Set labels and title
    plt.ylabel('Average Edit Distance (higher is better)')
    plt.title('Comparison of Edit Distances', fontsize=14)
    
    # Add improvement percentage
    improvement = results['summary']['improvement']['edit_distance']
    plt.figtext(0.5, 0.01, f'Improvement: {improvement:.1f}%', 
                horizontalalignment='center', fontsize=12, fontweight='bold')
    
    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout for figtext
    plt.savefig(os.path.join(output_dir, 'edit_distances.png'), dpi=300)
    plt.close()

def plot_completion_times(results: Dict[str, Any], output_dir: str):
    """Plot comparison of task completion times between baseline and adaptive models."""
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    baseline_times = results['baseline']['task_completion_times']
    adaptive_times = results['adaptive']['task_completion_times']
    
    # Create histograms
    plt.hist(baseline_times, alpha=0.6, label='Baseline', bins=20)
    plt.hist(adaptive_times, alpha=0.6, label='Adaptive', bins=20)
    
    # Add mean lines
    plt.axvline(np.mean(baseline_times), color='blue', linestyle='dashed', 
                linewidth=2, label=f'Baseline Mean: {np.mean(baseline_times):.1f}s')
    plt.axvline(np.mean(adaptive_times), color='orange', linestyle='dashed', 
                linewidth=2, label=f'Adaptive Mean: {np.mean(adaptive_times):.1f}s')
    
    # Set labels and title
    plt.xlabel('Task Completion Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Task Completion Times', fontsize=14)
    plt.legend()
    
    # Add improvement percentage
    improvement = results['summary']['improvement']['task_completion_time']
    plt.figtext(0.5, 0.01, f'Time Reduction: {improvement:.1f}%', 
                horizontalalignment='center', fontsize=12, fontweight='bold')
    
    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout for figtext
    plt.savefig(os.path.join(output_dir, 'completion_times.png'), dpi=300)
    plt.close()

def plot_code_quality(results: Dict[str, Any], output_dir: str):
    """Plot comparison of code quality scores between baseline and adaptive models."""
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    baseline_quality = results['baseline']['code_quality_scores']
    adaptive_quality = results['adaptive']['code_quality_scores']
    
    # Create violin plots
    data = pd.DataFrame({
        'Model': ['Baseline'] * len(baseline_quality) + ['Adaptive'] * len(adaptive_quality),
        'Code Quality Score': baseline_quality + adaptive_quality
    })
    
    # Plot
    sns.violinplot(x='Model', y='Code Quality Score', data=data, inner='quartile')
    sns.stripplot(x='Model', y='Code Quality Score', data=data, color='black', alpha=0.3, jitter=True)
    
    # Add means as text
    baseline_mean = np.mean(baseline_quality)
    adaptive_mean = np.mean(adaptive_quality)
    plt.text(0, baseline_mean + 2, f'Mean: {baseline_mean:.1f}', 
             horizontalalignment='center', fontweight='bold')
    plt.text(1, adaptive_mean + 2, f'Mean: {adaptive_mean:.1f}', 
             horizontalalignment='center', fontweight='bold')
    
    # Set title
    plt.title('Comparison of Code Quality Scores', fontsize=14)
    
    # Add improvement percentage
    improvement = results['summary']['improvement']['code_quality']
    plt.figtext(0.5, 0.01, f'Quality Improvement: {improvement:.1f}%', 
                horizontalalignment='center', fontsize=12, fontweight='bold')
    
    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout for figtext
    plt.savefig(os.path.join(output_dir, 'code_quality.png'), dpi=300)
    plt.close()

def plot_performance_improvements(results: Dict[str, Any], output_dir: str):
    """Plot overall performance improvements of adaptive model over baseline."""
    plt.figure(figsize=(12, 7))
    
    # Prepare data for bar chart
    metrics = ['acceptance_rate', 'edit_distance', 'reward', 'task_completion_time', 'code_quality']
    metric_names = ['Acceptance Rate', 'Edit Distance', 'Overall Reward', 'Task Time Reduction', 'Code Quality']
    improvements = [results['summary']['improvement'][m] for m in metrics]
    
    # Set colors based on positive/negative values
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    # Create bar chart
    bars = plt.bar(metric_names, improvements, color=colors, alpha=0.7)
    
    # Add value labels
    for bar, value in zip(bars, improvements):
        height = bar.get_height()
        sign = '+' if height > 0 else ''
        plt.text(bar.get_x() + bar.get_width()/2., 
                 height + np.sign(height) * 2,
                 f'{sign}{value:.1f}%',
                 ha='center', va='bottom', fontweight='bold')
    
    # Set labels and title
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylabel('Improvement (%)')
    plt.title('Performance Improvements: Adaptive vs. Baseline', fontsize=16, fontweight='bold')
    plt.xticks(rotation=15)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_improvements.png'), dpi=300)
    plt.close()

def plot_developer_performance(results: Dict[str, Any], output_dir: str):
    """Plot individual developer performance improvements."""
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    developers = results['baseline']['developers']
    baseline_rewards = results['baseline']['avg_reward']
    adaptive_rewards = results['adaptive']['avg_reward']
    
    # Sort by improvement amount for better visualization
    improvements = [(i, a - b) for i, (a, b) in enumerate(zip(adaptive_rewards, baseline_rewards))]
    sorted_indices = [i for i, _ in sorted(improvements, key=lambda x: x[1], reverse=True)]
    
    sorted_developers = [developers[i] for i in sorted_indices]
    sorted_baseline = [baseline_rewards[i] for i in sorted_indices]
    sorted_adaptive = [adaptive_rewards[i] for i in sorted_indices]
    
    # Truncate to top 15 developers if there are too many
    if len(developers) > 15:
        sorted_developers = sorted_developers[:15]
        sorted_baseline = sorted_baseline[:15]
        sorted_adaptive = sorted_adaptive[:15]
    
    # Data for plot
    x = np.arange(len(sorted_developers))
    width = 0.35
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    baseline_bars = ax.bar(x - width/2, sorted_baseline, width, label='Baseline', alpha=0.7)
    adaptive_bars = ax.bar(x + width/2, sorted_adaptive, width, label='Adaptive', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Developer')
    ax.set_ylabel('Average Reward')
    ax.set_title('Performance Comparison by Developer', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_developers, rotation=45)
    ax.legend()
    
    # Add improvement percentages above bars
    for i, (b, a) in enumerate(zip(sorted_baseline, sorted_adaptive)):
        improvement = ((a - b) / b) * 100 if b > 0 else 0
        sign = '+' if improvement > 0 else ''
        plt.text(i, max(a, b) + 0.02, 
                 f'{sign}{improvement:.1f}%',
                 ha='center', va='bottom', fontsize=8)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'developer_performance.png'), dpi=300)
    plt.close()

def create_training_curves(
    training_history: Dict[str, List[float]],
    output_dir: str
):
    """
    Plot training curves from model training history.
    
    Args:
        training_history: Dictionary containing training metrics
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 8))
    
    # Plot loss curves
    if 'baseline_loss' in training_history and 'adaptive_loss' in training_history:
        plt.subplot(2, 2, 1)
        plt.plot(training_history['baseline_loss'], label='Baseline')
        plt.plot(training_history['adaptive_loss'], label='Adaptive')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    # Plot validation metrics
    if 'baseline_val_acceptance' in training_history and 'adaptive_val_acceptance' in training_history:
        plt.subplot(2, 2, 2)
        plt.plot(training_history['baseline_val_acceptance'], label='Baseline')
        plt.plot(training_history['adaptive_val_acceptance'], label='Adaptive')
        plt.title('Validation Acceptance Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Acceptance Rate')
        plt.legend()
    
    # Plot other validation metrics if available
    if 'baseline_val_edit_distance' in training_history and 'adaptive_val_edit_distance' in training_history:
        plt.subplot(2, 2, 3)
        plt.plot(training_history['baseline_val_edit_distance'], label='Baseline')
        plt.plot(training_history['adaptive_val_edit_distance'], label='Adaptive')
        plt.title('Validation Edit Distance')
        plt.xlabel('Epoch')
        plt.ylabel('Edit Distance')
        plt.legend()
    
    if 'baseline_val_reward' in training_history and 'adaptive_val_reward' in training_history:
        plt.subplot(2, 2, 4)
        plt.plot(training_history['baseline_val_reward'], label='Baseline')
        plt.plot(training_history['adaptive_val_reward'], label='Adaptive')
        plt.title('Validation Reward')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
    plt.close()

def create_tables(results: Dict[str, Any], output_dir: str):
    """
    Create tables from experimental results.
    
    Args:
        results: Dictionary containing experimental results
        output_dir: Directory to save tables
    """
    # Create tables directory
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    # Table 1: Summary of results
    create_summary_table(results, tables_dir)
    
    # Table 2: Detailed metrics by model
    create_metrics_table(results, tables_dir)
    
    # Table 3: Developer performance breakdown
    create_developer_table(results, tables_dir)

def create_summary_table(results: Dict[str, Any], output_dir: str):
    """Create a summary table of experimental results."""
    # Extract relevant metrics
    baseline = results['summary']['baseline']
    adaptive = results['summary']['adaptive']
    improvement = results['summary']['improvement']
    
    # Create table data
    table_data = {
        'Metric': [
            'Acceptance Rate',
            'Edit Distance',
            'Task Completion Time (s)',
            'Code Quality Score',
            'Overall Reward'
        ],
        'Baseline': [
            f"{baseline['avg_acceptance_rate']:.3f} ± {baseline['std_acceptance_rate']:.3f}",
            f"{baseline['avg_edit_distance']:.3f} ± {baseline['std_edit_distance']:.3f}",
            f"{baseline['avg_task_completion_time']:.1f} ± {baseline['std_task_completion_time']:.1f}",
            f"{baseline['avg_code_quality']:.1f} ± {baseline['std_code_quality']:.1f}",
            f"{baseline['avg_reward']:.3f} ± {baseline['std_reward']:.3f}"
        ],
        'Adaptive': [
            f"{adaptive['avg_acceptance_rate']:.3f} ± {adaptive['std_acceptance_rate']:.3f}",
            f"{adaptive['avg_edit_distance']:.3f} ± {adaptive['std_edit_distance']:.3f}",
            f"{adaptive['avg_task_completion_time']:.1f} ± {adaptive['std_task_completion_time']:.1f}",
            f"{adaptive['avg_code_quality']:.1f} ± {adaptive['std_code_quality']:.1f}",
            f"{adaptive['avg_reward']:.3f} ± {adaptive['std_reward']:.3f}"
        ],
        'Improvement (%)': [
            f"{improvement['acceptance_rate']:.1f}%",
            f"{improvement['edit_distance']:.1f}%",
            f"{improvement['task_completion_time']:.1f}%",
            f"{improvement['code_quality']:.1f}%",
            f"{improvement['reward']:.1f}%"
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'summary_results.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as Markdown table
    markdown_path = os.path.join(output_dir, 'summary_results.md')
    with open(markdown_path, 'w') as f:
        f.write("# Summary of Experimental Results\n\n")
        f.write(df.to_markdown(index=False))
    
    # Save as JSON
    json_path = os.path.join(output_dir, 'summary_results.json')
    df.to_json(json_path, orient='records', indent=2)

def create_metrics_table(results: Dict[str, Any], output_dir: str):
    """Create a detailed metrics table."""
    # Create DataFrame for baseline metrics
    baseline_df = pd.DataFrame({
        'Model': ['Baseline'] * len(results['baseline']['developers']),
        'Developer': results['baseline']['developers'],
        'Acceptance Rate': results['baseline']['acceptance_rate'],
        'Edit Distance': results['baseline']['avg_edit_distance'],
        'Reward': results['baseline']['avg_reward']
    })
    
    # Create DataFrame for adaptive metrics
    adaptive_df = pd.DataFrame({
        'Model': ['Adaptive'] * len(results['adaptive']['developers']),
        'Developer': results['adaptive']['developers'],
        'Acceptance Rate': results['adaptive']['acceptance_rate'],
        'Edit Distance': results['adaptive']['avg_edit_distance'],
        'Reward': results['adaptive']['avg_reward']
    })
    
    # Combine DataFrames
    combined_df = pd.concat([baseline_df, adaptive_df])
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'detailed_metrics.csv')
    combined_df.to_csv(csv_path, index=False)
    
    # Save as Markdown table
    markdown_path = os.path.join(output_dir, 'detailed_metrics.md')
    with open(markdown_path, 'w') as f:
        f.write("# Detailed Metrics by Model and Developer\n\n")
        f.write(combined_df.to_markdown(index=False))

def create_developer_table(results: Dict[str, Any], output_dir: str):
    """Create a table showing performance improvements by developer."""
    # Get data
    developers = results['baseline']['developers']
    baseline_rewards = results['baseline']['avg_reward']
    adaptive_rewards = results['adaptive']['avg_reward']
    
    # Calculate improvements
    improvements = []
    for b, a in zip(baseline_rewards, adaptive_rewards):
        if b > 0:
            imp = ((a - b) / b) * 100
        else:
            imp = 0
        improvements.append(imp)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Developer': developers,
        'Baseline Reward': baseline_rewards,
        'Adaptive Reward': adaptive_rewards,
        'Improvement (%)': [f"{imp:.1f}%" for imp in improvements]
    })
    
    # Sort by improvement
    df['Improvement_numeric'] = improvements
    df = df.sort_values('Improvement_numeric', ascending=False)
    df = df.drop(columns=['Improvement_numeric'])
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'developer_improvements.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as Markdown table
    markdown_path = os.path.join(output_dir, 'developer_improvements.md')
    with open(markdown_path, 'w') as f:
        f.write("# Performance Improvement by Developer\n\n")
        f.write(df.to_markdown(index=False))
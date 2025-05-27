#!/usr/bin/env python3
"""
Generate mock results for visualization and results.md generation.
This script creates a realistic mock results file that can be used to test
the visualization and reporting components without running the full experiment.
"""

import os
import json
import numpy as np
import random
from pathlib import Path

from utils.visualization import plot_results, create_tables

def generate_mock_results(num_developers=30, num_tasks=12):
    """Generate mock experiment results."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create developer IDs
    developers = [f"dev_{i}" for i in range(num_developers)]
    
    # Baseline metrics (slightly worse than adaptive)
    baseline_acceptance_rates = np.random.beta(5, 7, num_developers)  # Mean around 0.42
    baseline_edit_distances = np.random.beta(6, 4, num_developers)  # Mean around 0.6
    baseline_rewards = np.random.beta(5, 5, num_developers)  # Mean around 0.5
    
    # Generate task completion times for baseline (higher is worse)
    baseline_times = []
    for _ in range(num_developers * num_tasks):
        baseline_times.append(np.random.normal(360, 70))  # Mean 6 minutes
    
    # Generate code quality scores for baseline
    baseline_quality = []
    for _ in range(num_developers * num_tasks):
        baseline_quality.append(np.random.normal(65, 12))  # Mean 65/100
    
    # Adaptive metrics (better than baseline with realistic improvements)
    # 15% improvement in acceptance rate
    adaptive_acceptance_rates = baseline_acceptance_rates * 1.15
    adaptive_acceptance_rates = np.clip(adaptive_acceptance_rates, 0, 0.95)
    
    # 25% improvement in edit distance
    adaptive_edit_distances = baseline_edit_distances * 1.25
    adaptive_edit_distances = np.clip(adaptive_edit_distances, 0, 0.95)
    
    # 20% improvement in reward
    adaptive_rewards = baseline_rewards * 1.2
    adaptive_rewards = np.clip(adaptive_rewards, 0, 0.95)
    
    # 15% improvement (reduction) in task completion times
    adaptive_times = []
    for time in baseline_times:
        adaptive_times.append(time * 0.85 * random.uniform(0.85, 1.15))
    
    # 10% improvement in code quality
    adaptive_quality = []
    for quality in baseline_quality:
        adaptive_quality.append(quality * 1.1 * random.uniform(0.9, 1.1))
        adaptive_quality[-1] = min(adaptive_quality[-1], 100)  # Cap at 100
    
    # Build results dictionary
    results = {
        'baseline': {
            'developers': developers,
            'acceptance_rate': baseline_acceptance_rates.tolist(),
            'avg_edit_distance': baseline_edit_distances.tolist(),
            'avg_reward': baseline_rewards.tolist(),
            'task_completion_times': baseline_times,
            'code_quality_scores': baseline_quality
        },
        'adaptive': {
            'developers': developers,
            'acceptance_rate': adaptive_acceptance_rates.tolist(),
            'avg_edit_distance': adaptive_edit_distances.tolist(),
            'avg_reward': adaptive_rewards.tolist(),
            'task_completion_times': adaptive_times,
            'code_quality_scores': adaptive_quality
        },
        'summary': {
            'baseline': {
                'avg_acceptance_rate': float(np.mean(baseline_acceptance_rates)),
                'std_acceptance_rate': float(np.std(baseline_acceptance_rates)),
                'avg_edit_distance': float(np.mean(baseline_edit_distances)),
                'std_edit_distance': float(np.std(baseline_edit_distances)),
                'avg_reward': float(np.mean(baseline_rewards)),
                'std_reward': float(np.std(baseline_rewards)),
                'avg_task_completion_time': float(np.mean(baseline_times)),
                'std_task_completion_time': float(np.std(baseline_times)),
                'avg_code_quality': float(np.mean(baseline_quality)),
                'std_code_quality': float(np.std(baseline_quality))
            },
            'adaptive': {
                'avg_acceptance_rate': float(np.mean(adaptive_acceptance_rates)),
                'std_acceptance_rate': float(np.std(adaptive_acceptance_rates)),
                'avg_edit_distance': float(np.mean(adaptive_edit_distances)),
                'std_edit_distance': float(np.std(adaptive_edit_distances)),
                'avg_reward': float(np.mean(adaptive_rewards)),
                'std_reward': float(np.std(adaptive_rewards)),
                'avg_task_completion_time': float(np.mean(adaptive_times)),
                'std_task_completion_time': float(np.std(adaptive_times)),
                'avg_code_quality': float(np.mean(adaptive_quality)),
                'std_code_quality': float(np.std(adaptive_quality))
            }
        }
    }
    
    # Calculate improvement percentages
    baseline_summary = results['summary']['baseline']
    adaptive_summary = results['summary']['adaptive']
    
    improvement = {
        'acceptance_rate': (
            (adaptive_summary['avg_acceptance_rate'] - baseline_summary['avg_acceptance_rate']) / 
            baseline_summary['avg_acceptance_rate'] * 100
        ),
        'edit_distance': (
            (adaptive_summary['avg_edit_distance'] - baseline_summary['avg_edit_distance']) / 
            baseline_summary['avg_edit_distance'] * 100
        ),
        'reward': (
            (adaptive_summary['avg_reward'] - baseline_summary['avg_reward']) / 
            baseline_summary['avg_reward'] * 100
        ),
        'task_completion_time': (
            (baseline_summary['avg_task_completion_time'] - adaptive_summary['avg_task_completion_time']) / 
            baseline_summary['avg_task_completion_time'] * 100
        ),
        'code_quality': (
            (adaptive_summary['avg_code_quality'] - baseline_summary['avg_code_quality']) / 
            baseline_summary['avg_code_quality'] * 100
        )
    }
    
    results['summary']['improvement'] = improvement
    
    return results

def main():
    # Create output directory
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate mock results
    print("Generating mock experiment results...")
    results = generate_mock_results(num_developers=30, num_tasks=12)
    
    # Save results to JSON
    results_file = os.path.join(output_dir, 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Mock results saved to {results_file}")
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_results(results, output_dir)
    
    # Generate tables
    print("Generating tables...")
    create_tables(results, output_dir)
    
    print("Done! Mock results and visualizations created successfully.")

if __name__ == "__main__":
    main()
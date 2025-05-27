#!/usr/bin/env python3
"""
Create visualizations for the MACP framework experiment results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_time_comparison(results, output_dir):
    """Plot time comparison between systems."""
    plt.figure(figsize=(10, 6))
    
    # Extract time values for each system
    systems = list(results['overall_comparison'].keys())
    times = [results['overall_comparison'][sys]['avg_time'] for sys in systems]
    
    # Create bar chart
    bars = plt.bar(systems, times, color=['blue', 'green'])
    
    # Add labels and title
    plt.xlabel('System')
    plt.ylabel('Time to Solution (seconds)')
    plt.title('Average Time to Solution Comparison')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + 0.02 * max(times),
                f'{height:.1f}s',
                ha='center', va='bottom', rotation=0)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_to_solution_comparison.png"))
    plt.close()
    
def plot_code_metrics_radar(results, output_dir):
    """Create a radar chart for code quality metrics."""
    # Metrics to include in the radar chart
    metrics = ['avg_maintainability', 'success_rate', 'avg_complexity']
    labels = ['Maintainability', 'Success Rate', 'Complexity (lower is better)']
    
    # Number of variables
    N = len(metrics)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Define colors for different systems
    colors = ['blue', 'green']
    
    # Add each system
    systems = list(results['overall_comparison'].keys())
    for i, system in enumerate(systems):
        # Extract values
        values = []
        for metric in metrics:
            if metric == 'avg_complexity':
                # For complexity, lower is better, so normalize and invert
                all_complexities = [results['overall_comparison'][s]['avg_complexity'] for s in systems]
                max_complexity = max(all_complexities)
                # Invert so lower complexity gets higher score on radar
                values.append(1 - (results['overall_comparison'][system]['avg_complexity'] / max_complexity))
            else:
                # For other metrics, normalize to 0-1
                if metric == 'avg_maintainability':
                    # Normalize maintainability (typically 0-100) to 0-1
                    values.append(results['overall_comparison'][system][metric] / 100)
                else:
                    # Other metrics already in 0-1 range
                    values.append(results['overall_comparison'][system][metric])
        
        # Close the loop
        values += values[:1]
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=system, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Set title
    plt.title('Code Quality Metrics Comparison')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_chart_comparison.png"))
    plt.close()

def plot_message_flow(results, output_dir):
    """Create a visualization of message flow between agents."""
    # This is a simplified visualization since we don't have the actual message timestamps
    if 'collaboration_analysis' not in results or not results['collaboration_analysis']:
        return
    
    # Get the first task's message data
    task_id = list(results['collaboration_analysis'].keys())[0]
    msg_data = results['collaboration_analysis'][task_id]
    
    if 'messages_by_sender' not in msg_data:
        return
    
    # Extract message counts by sender
    senders = list(msg_data['messages_by_sender'].keys())
    counts = [msg_data['messages_by_sender'][sender] for sender in senders]
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(
        counts, 
        labels=senders,
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.Set3.colors[:len(senders)]
    )
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Distribution of Messages by Agent Role')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "message_flow.png"))
    plt.close()

def main():
    # Load results
    results_file = Path(__file__).parent / "results" / "experiment_results.json"
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create visualizations
    plot_time_comparison(results, output_dir)
    plot_code_metrics_radar(results, output_dir)
    plot_message_flow(results, output_dir)
    
    print(f"Visualizations created in {output_dir}")

if __name__ == "__main__":
    main()
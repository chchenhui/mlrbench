"""
Utility functions for the Multi-Agent Collaborative Programming (MACP) framework.
"""

import os
import json
import time
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

# Configure logging
def setup_logging(log_file_path: str) -> logging.Logger:
    """Set up logging for the experiment."""
    logger = logging.getLogger("macp_experiment")
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Task utilities
def load_tasks(tasks_file_path: str) -> List[Dict[str, Any]]:
    """Load programming tasks from a JSON file."""
    with open(tasks_file_path, 'r') as f:
        tasks_data = json.load(f)
    return tasks_data.get('tasks', [])

# Metric calculation utilities
def calculate_metrics(
    solution: str, 
    task: Dict[str, Any],
    execution_time: float,
    messages_count: int
) -> Dict[str, Any]:
    """Calculate evaluation metrics for a solution."""
    metrics = {
        'time_to_solution': execution_time,
        'messages_count': messages_count,
    }
    
    # Code metrics
    metrics.update(calculate_code_metrics(solution))
    
    return metrics

def calculate_code_metrics(code: str) -> Dict[str, float]:
    """Calculate code quality metrics."""
    metrics = {}
    
    # Lines of code
    metrics['lines_of_code'] = len(code.strip().split('\n'))
    
    # Function count
    metrics['function_count'] = len(re.findall(r'def\s+\w+\s*\(', code))
    
    # Class count
    metrics['class_count'] = len(re.findall(r'class\s+\w+\s*(\(|:)', code))
    
    # Comment ratio
    code_lines = [line for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
    comment_lines = [line for line in code.split('\n') if line.strip() and line.strip().startswith('#')]
    metrics['comment_ratio'] = len(comment_lines) / (len(code_lines) + 1e-10)  # Avoid division by zero
    
    # Simple cyclomatic complexity (approximate)
    decision_points = (
        len(re.findall(r'\bif\b', code)) + 
        len(re.findall(r'\belif\b', code)) + 
        len(re.findall(r'\bfor\b', code)) + 
        len(re.findall(r'\bwhile\b', code)) + 
        len(re.findall(r'\bcatch\b', code))
    )
    metrics['estimated_complexity'] = decision_points
    
    return metrics

# Visualization utilities
def plot_performance_comparison(metrics_dict: Dict[str, Dict[str, Any]], output_path: str):
    """Create a bar chart comparing performance metrics across systems."""
    metrics_to_plot = ['time_to_solution', 'lines_of_code', 'estimated_complexity', 'function_count']
    
    # Prepare data
    systems = list(metrics_dict.keys())
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        values = [metrics_dict[system].get(metric, 0) for system in systems]
        
        # Create bar chart
        bars = plt.bar(systems, values)
        
        # Add labels and title
        plt.xlabel('System')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Comparison of {metric.replace("_", " ").title()} Across Systems')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + 0.02 * max(values),
                    f'{height:.2f}',
                    ha='center', va='bottom', rotation=0)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{metric}_comparison.png"))
        plt.close()

def plot_radar_chart(metrics_dict: Dict[str, Dict[str, Any]], metrics_to_plot: List[str], output_path: str):
    """Create a radar chart for multidimensional performance visualization."""
    # Number of variables
    N = len(metrics_to_plot)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add each system
    for i, (system, metrics) in enumerate(metrics_dict.items()):
        # Extract values and normalize to 0-1 range for fair comparison
        values = []
        for metric in metrics_to_plot:
            all_values = [m.get(metric, 0) for m in metrics_dict.values()]
            max_val = max(all_values) if max(all_values) > 0 else 1
            normalized_val = metrics.get(metric, 0) / max_val
            # Invert metrics where lower is better
            if metric in ['time_to_solution', 'estimated_complexity']:
                normalized_val = 1 - normalized_val
            values.append(normalized_val)
        
        # Close the loop
        values += values[:1]
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=system)
        ax.fill(angles, values, alpha=0.1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "radar_chart_comparison.png"))
    plt.close()

def plot_message_flow(messages: List[Dict[str, Any]], output_path: str):
    """Create a visualization of message flow between agents over time."""
    if not messages:
        return
    
    # Extract sender, receiver, timestamp
    senders = [m.get('sender', 'Unknown') for m in messages]
    receivers = [m.get('receiver', 'Unknown') for m in messages]
    timestamps = [m.get('timestamp', 0) for m in messages]
    
    # Convert to relative time
    start_time = min(timestamps)
    rel_times = [(t - start_time) / 60 for t in timestamps]  # Minutes
    
    # Get unique agents
    agents = sorted(list(set(senders + receivers)))
    agent_indices = {agent: i for i, agent in enumerate(agents)}
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot each message as an arrow
    for i in range(len(messages)):
        sender_idx = agent_indices[senders[i]]
        receiver_idx = agent_indices[receivers[i]]
        plt.arrow(rel_times[i], sender_idx, 0, receiver_idx - sender_idx, 
                 length_includes_head=True, head_width=0.1, head_length=0.1,
                 fc='blue', ec='blue', alpha=0.6)
    
    # Set labels
    plt.yticks(range(len(agents)), agents)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Agent')
    plt.title('Message Flow Between Agents')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "message_flow.png"))
    plt.close()

# Results management
def save_results(results: Dict[str, Any], output_path: str):
    """Save experiment results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def generate_results_markdown(results: Dict[str, Any], figures_path: str, output_path: str):
    """Generate a Markdown file summarizing experiment results."""
    with open(output_path, 'w') as f:
        f.write("# MACP Framework Experimental Results\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write("This document presents the results of experiments evaluating the Multi-Agent Collaborative ")
        f.write("Programming (MACP) framework against baseline approaches for solving programming tasks.\n\n")
        
        # Experimental Setup
        f.write("## Experimental Setup\n\n")
        f.write("The experiments compared the following systems:\n\n")
        for system in results.get('systems', []):
            f.write(f"- **{system}**: {results.get('system_descriptions', {}).get(system, '')}\n")
        f.write("\n")
        
        f.write("Tasks of varying complexity were used for evaluation:\n\n")
        for task in results.get('task_descriptions', []):
            f.write(f"- **{task['name']}** ({task['complexity']}): {task['description']}\n")
        f.write("\n")
        
        # Performance Results
        f.write("## Performance Results\n\n")
        
        # Add task-specific results
        for task_id, task_results in results.get('task_results', {}).items():
            task_name = next((t['name'] for t in results.get('task_descriptions', []) if t['id'] == task_id), task_id)
            f.write(f"### Task: {task_name}\n\n")
            
            # Performance metrics table
            f.write("#### Performance Metrics\n\n")
            f.write("| System | Time (s) | Lines of Code | Estimated Complexity | Function Count | Class Count |\n")
            f.write("|--------|----------|---------------|----------------------|----------------|------------|\n")
            
            for system, metrics in task_results.get('metrics', {}).items():
                f.write(f"| {system} | {metrics.get('time_to_solution', 'N/A'):.2f} | ")
                f.write(f"{metrics.get('lines_of_code', 'N/A')} | ")
                f.write(f"{metrics.get('estimated_complexity', 'N/A')} | ")
                f.write(f"{metrics.get('function_count', 'N/A')} | ")
                f.write(f"{metrics.get('class_count', 'N/A')} |\n")
            f.write("\n")
            
            # Add figures for this task
            f.write("#### Visualizations\n\n")
            
            # Time to solution comparison
            f.write("![Time to Solution Comparison](time_to_solution_comparison.png)\n\n")
            
            # Code metrics comparison
            f.write("![Code Metrics Comparison](radar_chart_comparison.png)\n\n")
            
            if task_id in results.get('collaboration_analysis', {}):
                # Message flow visualization (MACP only)
                f.write("#### Collaboration Analysis (MACP)\n\n")
                f.write("![Message Flow](message_flow.png)\n\n")
                
                # Communication statistics
                comm_stats = results.get('collaboration_analysis', {}).get(task_id, {})
                f.write("**Communication Statistics:**\n\n")
                f.write(f"- Total messages: {comm_stats.get('total_messages', 'N/A')}\n")
                f.write(f"- Messages per phase: {comm_stats.get('messages_per_phase', 'N/A')}\n")
                f.write(f"- Most active agent: {comm_stats.get('most_active_agent', 'N/A')}\n\n")
        
        # Overall Comparison
        f.write("## Overall Comparison\n\n")
        
        # Overall metrics table
        f.write("### Average Performance Across All Tasks\n\n")
        f.write("| System | Avg. Time (s) | Avg. Lines of Code | Avg. Complexity | Success Rate |\n")
        f.write("|--------|---------------|--------------------|-----------------|--------------|\n")
        
        overall = results.get('overall_comparison', {})
        for system, metrics in overall.items():
            f.write(f"| {system} | {metrics.get('avg_time', 'N/A'):.2f} | ")
            f.write(f"{metrics.get('avg_loc', 'N/A'):.1f} | ")
            f.write(f"{metrics.get('avg_complexity', 'N/A'):.1f} | ")
            f.write(f"{metrics.get('success_rate', 'N/A')*100:.1f}% |\n")
        f.write("\n")
        
        # Qualitative Analysis
        f.write("## Qualitative Analysis\n\n")
        f.write("### Strengths and Weaknesses\n\n")
        
        for system, analysis in results.get('qualitative_analysis', {}).items():
            f.write(f"#### {system}\n\n")
            
            f.write("**Strengths:**\n\n")
            for strength in analysis.get('strengths', []):
                f.write(f"- {strength}\n")
            f.write("\n")
            
            f.write("**Weaknesses:**\n\n")
            for weakness in analysis.get('weaknesses', []):
                f.write(f"- {weakness}\n")
            f.write("\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write(results.get('conclusion', "The experiments demonstrate the comparative performance of the MACP framework and baseline approaches."))
        f.write("\n\n")
        
        # Limitations
        f.write("## Limitations and Future Work\n\n")
        for limitation in results.get('limitations', []):
            f.write(f"- {limitation}\n")
        f.write("\n")
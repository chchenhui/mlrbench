"""
Evaluation metrics and assessment tools for the MACP framework.
"""

import os
import re
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
import networkx as nx
from collections import Counter

class CodeEvaluator:
    """Class for evaluating code quality metrics."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the code evaluator.
        
        Args:
            logger: Logger instance for tracking events
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def evaluate_code(self, code: str) -> Dict[str, float]:
        """
        Evaluate code quality metrics.
        
        Args:
            code: The code to evaluate
            
        Returns:
            Dictionary of code quality metrics
        """
        metrics = {}
        
        # Lines of code
        metrics['lines_of_code'] = self._count_lines_of_code(code)
        
        # Function count
        metrics['function_count'] = self._count_functions(code)
        
        # Class count
        metrics['class_count'] = self._count_classes(code)
        
        # Comment ratio
        metrics['comment_ratio'] = self._calculate_comment_ratio(code)
        
        # Complexity metrics
        metrics['cyclomatic_complexity'] = self._estimate_cyclomatic_complexity(code)
        
        # Maintainability metrics
        metrics['estimated_maintainability'] = self._estimate_maintainability(metrics)
        
        return metrics
    
    def _count_lines_of_code(self, code: str) -> int:
        """Count non-empty lines of code."""
        return len([line for line in code.strip().split('\n') if line.strip()])
    
    def _count_functions(self, code: str) -> int:
        """Count function definitions in the code."""
        return len(re.findall(r'def\s+\w+\s*\(', code))
    
    def _count_classes(self, code: str) -> int:
        """Count class definitions in the code."""
        return len(re.findall(r'class\s+\w+\s*(\(|:)', code))
    
    def _calculate_comment_ratio(self, code: str) -> float:
        """Calculate the ratio of comments to code lines."""
        lines = code.strip().split('\n')
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        comment_lines = [line for line in lines if line.strip() and line.strip().startswith('#')]
        
        if not code_lines:
            return 0.0
            
        return len(comment_lines) / len(code_lines)
    
    def _estimate_cyclomatic_complexity(self, code: str) -> float:
        """Estimate cyclomatic complexity based on decision points."""
        # Count decision points (if, elif, for, while, except)
        decision_points = (
            len(re.findall(r'\bif\b', code)) + 
            len(re.findall(r'\belif\b', code)) + 
            len(re.findall(r'\bfor\b', code)) + 
            len(re.findall(r'\bwhile\b', code)) + 
            len(re.findall(r'\bexcept\b', code))
        )
        
        # Base complexity is 1 for each function plus decision points
        base_complexity = self._count_functions(code) + decision_points
        
        return base_complexity
    
    def _estimate_maintainability(self, metrics: Dict[str, float]) -> float:
        """
        Estimate maintainability index based on other metrics.
        
        The maintainability index is an approximate measure of how maintainable the code is.
        Higher values indicate better maintainability (0-100 scale).
        """
        # Simplified approximation based on code metrics
        # Start with a base value
        maintainability = 100
        
        # Reduce based on complexity (higher complexity reduces maintainability)
        complexity_factor = metrics.get('cyclomatic_complexity', 0) * 0.2
        maintainability -= complexity_factor
        
        # Increase based on comment ratio (more comments improves maintainability)
        comment_bonus = metrics.get('comment_ratio', 0) * 20
        maintainability += comment_bonus
        
        # Penalize for large code size
        size_penalty = min(metrics.get('lines_of_code', 0) * 0.1, 30)
        maintainability -= size_penalty
        
        # Ensure the value stays within 0-100 range
        maintainability = max(0, min(100, maintainability))
        
        return maintainability


class CollaborationAnalyzer:
    """Class for analyzing agent collaboration patterns."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the collaboration analyzer.
        
        Args:
            logger: Logger instance for tracking events
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze collaboration patterns from messages.
        
        Args:
            messages: List of messages exchanged between agents
            
        Returns:
            Dictionary of collaboration metrics and patterns
        """
        analysis = {}
        
        # Basic message statistics
        analysis['total_messages'] = len(messages)
        
        # Message count by sender
        senders = [msg.get('sender', 'unknown') for msg in messages]
        analysis['messages_by_sender'] = dict(Counter(senders))
        
        # Message count by receiver
        receivers = [msg.get('receiver', 'unknown') for msg in messages]
        analysis['messages_by_receiver'] = dict(Counter(receivers))
        
        # Message count by type
        message_types = [msg.get('message_type', 'unknown') for msg in messages]
        analysis['messages_by_type'] = dict(Counter(message_types))
        
        # Find most active agent
        sender_counts = Counter(senders)
        analysis['most_active_agent'] = sender_counts.most_common(1)[0][0] if sender_counts else None
        
        # Analyze communication flow
        analysis['communication_flow'] = self._analyze_communication_flow(messages)
        
        # Analyze message timeline
        analysis['timeline_analysis'] = self._analyze_message_timeline(messages)
        
        return analysis
    
    def _analyze_communication_flow(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the flow of messages between agents."""
        flow_analysis = {}
        
        # Create directed graph of message flow
        G = nx.DiGraph()
        
        # Add edges for each message (sender -> receiver)
        for msg in messages:
            sender = msg.get('sender', 'unknown')
            receiver = msg.get('receiver', 'unknown')
            
            if receiver != 'all':  # Skip broadcast messages
                if G.has_edge(sender, receiver):
                    G[sender][receiver]['weight'] += 1
                else:
                    G.add_edge(sender, receiver, weight=1)
        
        # Most common communication paths
        edge_weights = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
        sorted_edges = sorted(edge_weights, key=lambda x: x[2], reverse=True)
        flow_analysis['most_common_paths'] = sorted_edges[:5]  # Top 5 paths
        
        # Communication centrality (who is most central in communication)
        if G.nodes():
            try:
                # Calculate centrality measures
                degree_centrality = nx.degree_centrality(G)
                flow_analysis['degree_centrality'] = degree_centrality
                
                betweenness_centrality = nx.betweenness_centrality(G)
                flow_analysis['betweenness_centrality'] = betweenness_centrality
                
                # Most central agent
                flow_analysis['most_central_agent'] = max(degree_centrality.items(), key=lambda x: x[1])[0]
            except Exception as e:
                self.logger.warning(f"Error calculating centrality: {str(e)}")
        
        return flow_analysis
    
    def _analyze_message_timeline(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the timeline of messages."""
        timeline_analysis = {}
        
        # Extract timestamps
        timestamps = [msg.get('timestamp', 0) for msg in messages]
        
        if not timestamps:
            return timeline_analysis
            
        # Sort messages by timestamp
        sorted_msgs = sorted(zip(timestamps, messages), key=lambda x: x[0])
        
        # Find start and end times
        start_time = sorted_msgs[0][0]
        end_time = sorted_msgs[-1][0]
        
        if start_time == end_time:
            return timeline_analysis
            
        total_duration = end_time - start_time
        timeline_analysis['total_duration'] = total_duration
        
        # Divide timeline into phases
        phase_duration = total_duration / 4  # Divide into 4 phases
        
        phases = [
            [msg for t, msg in sorted_msgs if start_time <= t < start_time + phase_duration],
            [msg for t, msg in sorted_msgs if start_time + phase_duration <= t < start_time + 2*phase_duration],
            [msg for t, msg in sorted_msgs if start_time + 2*phase_duration <= t < start_time + 3*phase_duration],
            [msg for t, msg in sorted_msgs if start_time + 3*phase_duration <= t <= end_time]
        ]
        
        # Messages per phase
        timeline_analysis['messages_per_phase'] = [len(phase) for phase in phases]
        
        # Activity by role in each phase
        activity_by_phase = []
        for phase in phases:
            senders = [msg.get('sender', 'unknown') for msg in phase]
            activity_by_phase.append(dict(Counter(senders)))
        
        timeline_analysis['activity_by_phase'] = activity_by_phase
        
        return timeline_analysis


class SolutionEvaluator:
    """Class for evaluating overall solution quality."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the solution evaluator.
        
        Args:
            logger: Logger instance for tracking events
        """
        self.logger = logger or logging.getLogger(__name__)
        self.code_evaluator = CodeEvaluator(logger)
        self.collab_analyzer = CollaborationAnalyzer(logger)
    
    def evaluate_solution(
        self, 
        solution: str, 
        task: Dict[str, Any],
        execution_time: float,
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a solution comprehensively.
        
        Args:
            solution: The solution code
            task: The programming task
            execution_time: Time taken to produce the solution
            messages: Messages exchanged during solution development (for collaborative systems)
            
        Returns:
            Dictionary of comprehensive evaluation metrics
        """
        evaluation = {}
        
        # Basic metrics
        evaluation['time_to_solution'] = execution_time
        evaluation['task_id'] = task['id']
        evaluation['task_name'] = task['name']
        evaluation['task_complexity'] = task['complexity']
        
        # Code quality metrics
        code_metrics = self.code_evaluator.evaluate_code(solution)
        evaluation.update(code_metrics)
        
        # Collaboration metrics (if applicable)
        if messages:
            collaboration_metrics = self.collab_analyzer.analyze_messages(messages)
            evaluation['collaboration_metrics'] = collaboration_metrics
        
        return evaluation
    
    def compare_solutions(
        self, 
        baseline_result: Dict[str, Any],
        macp_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare baseline and MACP solutions.
        
        Args:
            baseline_result: Results from baseline evaluation
            macp_result: Results from MACP evaluation
            
        Returns:
            Comparative analysis dictionary
        """
        comparison = {
            'time_comparison': {
                'baseline': baseline_result.get('time_to_solution', 0),
                'macp': macp_result.get('time_to_solution', 0),
                'difference': baseline_result.get('time_to_solution', 0) - macp_result.get('time_to_solution', 0),
                'percentage': ((baseline_result.get('time_to_solution', 0) - macp_result.get('time_to_solution', 0)) / 
                              baseline_result.get('time_to_solution', 1)) * 100 if baseline_result.get('time_to_solution', 0) > 0 else 0
            },
            'code_quality_comparison': {
                'baseline': {
                    'lines_of_code': baseline_result.get('lines_of_code', 0),
                    'cyclomatic_complexity': baseline_result.get('cyclomatic_complexity', 0),
                    'comment_ratio': baseline_result.get('comment_ratio', 0),
                    'estimated_maintainability': baseline_result.get('estimated_maintainability', 0)
                },
                'macp': {
                    'lines_of_code': macp_result.get('lines_of_code', 0),
                    'cyclomatic_complexity': macp_result.get('cyclomatic_complexity', 0),
                    'comment_ratio': macp_result.get('comment_ratio', 0),
                    'estimated_maintainability': macp_result.get('estimated_maintainability', 0)
                }
            }
        }
        
        # Overall quality score (simple weighted average of normalized metrics)
        baseline_score = (
            baseline_result.get('estimated_maintainability', 0) * 0.4 +
            (1 - baseline_result.get('cyclomatic_complexity', 0) / 
             max(baseline_result.get('cyclomatic_complexity', 1), 
                 macp_result.get('cyclomatic_complexity', 1))) * 30 +
            baseline_result.get('comment_ratio', 0) * 20 +
            (1 - baseline_result.get('time_to_solution', 0) / 
             max(baseline_result.get('time_to_solution', 1), 
                 macp_result.get('time_to_solution', 1))) * 10
        )
        
        macp_score = (
            macp_result.get('estimated_maintainability', 0) * 0.4 +
            (1 - macp_result.get('cyclomatic_complexity', 0) / 
             max(baseline_result.get('cyclomatic_complexity', 1), 
                 macp_result.get('cyclomatic_complexity', 1))) * 30 +
            macp_result.get('comment_ratio', 0) * 20 +
            (1 - macp_result.get('time_to_solution', 0) / 
             max(baseline_result.get('time_to_solution', 1), 
                 macp_result.get('time_to_solution', 1))) * 10
        )
        
        comparison['overall_scores'] = {
            'baseline': baseline_score,
            'macp': macp_score,
            'difference': macp_score - baseline_score
        }
        
        return comparison


class Visualizer:
    """Class for creating visualizations of experimental results."""
    
    def __init__(self, output_dir: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            logger: Logger instance for tracking events
        """
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_time_comparison(self, results: Dict[str, Dict[str, Any]], task_id: str):
        """
        Plot time comparison between systems.
        
        Args:
            results: Dictionary mapping system names to result dictionaries
            task_id: ID of the task being visualized
        """
        plt.figure(figsize=(10, 6))
        
        # Extract time values
        systems = list(results.keys())
        times = [results[sys].get('time_to_solution', 0) for sys in systems]
        
        # Create bar chart
        bars = plt.bar(systems, times, color=['blue', 'green'])
        
        # Add labels and title
        plt.xlabel('System')
        plt.ylabel('Time to Solution (seconds)')
        plt.title(f'Time to Solution Comparison - Task {task_id}')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + 0.02 * max(times),
                    f'{height:.1f}s',
                    ha='center', va='bottom', rotation=0)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"time_to_solution_comparison.png"))
        plt.close()
        
        self.logger.info(f"Time comparison plot saved for task {task_id}")
    
    def plot_code_metrics_radar(self, results: Dict[str, Dict[str, Any]], task_id: str):
        """
        Create a radar chart for code quality metrics.
        
        Args:
            results: Dictionary mapping system names to result dictionaries
            task_id: ID of the task being visualized
        """
        # Metrics to include in the radar chart
        metrics = ['estimated_maintainability', 'comment_ratio', 'function_count', 'lines_of_code', 'cyclomatic_complexity']
        
        # Number of variables
        N = len(metrics)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Define colors for different systems
        colors = ['blue', 'green']
        
        # Normalize function to scale values between 0 and 1
        def normalize(values, metric):
            """Normalize values to 0-1 scale with direction consideration."""
            if not values:
                return []
                
            max_val = max(values)
            
            # For these metrics, higher is better
            if metric in ['estimated_maintainability', 'comment_ratio', 'function_count']:
                return [v / max_val if max_val > 0 else 0 for v in values]
            # For these metrics, lower is better, so invert
            else:
                return [1 - (v / max_val) if max_val > 0 else 0 for v in values]
        
        # Add each system
        for i, (system, metrics_dict) in enumerate(results.items()):
            # Extract values
            values = [metrics_dict.get(metric, 0) for metric in metrics]
            
            # For visualization purposes, get values from all systems for normalization
            all_values = {metric: [results[sys].get(metric, 0) for sys in results.keys()] 
                         for metric in metrics}
            
            # Normalize each metric
            normalized_values = []
            for j, metric in enumerate(metrics):
                norm_vals = normalize(all_values[metric], metric)
                normalized_values.append(norm_vals[i] if norm_vals else 0)
            
            # Close the loop
            normalized_values += normalized_values[:1]
            
            # Plot values
            ax.plot(angles, normalized_values, linewidth=2, linestyle='solid', label=system, color=colors[i % len(colors)])
            ax.fill(angles, normalized_values, alpha=0.1, color=colors[i % len(colors)])
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Set title
        plt.title(f'Code Quality Metrics Comparison - Task {task_id}')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"radar_chart_comparison.png"))
        plt.close()
        
        self.logger.info(f"Code metrics radar chart saved for task {task_id}")
    
    def plot_message_flow(self, messages: List[Dict[str, Any]], task_id: str):
        """
        Create a visualization of message flow between agents.
        
        Args:
            messages: List of messages exchanged during solution development
            task_id: ID of the task being visualized
        """
        if not messages:
            self.logger.warning(f"No messages to visualize for task {task_id}")
            return
        
        # Extract sender, receiver, timestamp
        data = []
        for m in messages:
            if 'sender' in m and 'receiver' in m and 'timestamp' in m:
                data.append((m['sender'], m['receiver'], m['timestamp']))
        
        if not data:
            self.logger.warning(f"No valid message data to visualize for task {task_id}")
            return
            
        senders, receivers, timestamps = zip(*data)
        
        # Convert to relative time
        start_time = min(timestamps)
        rel_times = [(t - start_time) / 60 for t in timestamps]  # Minutes
        
        # Get unique agents
        agents = sorted(list(set(senders + receivers)))
        agent_indices = {agent: i for i, agent in enumerate(agents)}
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot each message as an arrow
        for i in range(len(data)):
            sender_idx = agent_indices[senders[i]]
            
            # Handle 'all' receiver
            if receivers[i] == 'all':
                # Draw multiple arrows to each agent
                for receiver, idx in agent_indices.items():
                    if receiver != senders[i]:  # Skip self
                        plt.arrow(rel_times[i], sender_idx, 0, idx - sender_idx, 
                                 length_includes_head=True, head_width=0.1, head_length=0.1,
                                 fc='blue', ec='blue', alpha=0.3)
            else:
                receiver_idx = agent_indices[receivers[i]]
                plt.arrow(rel_times[i], sender_idx, 0, receiver_idx - sender_idx, 
                         length_includes_head=True, head_width=0.1, head_length=0.1,
                         fc='blue', ec='blue', alpha=0.6)
        
        # Set labels
        plt.yticks(range(len(agents)), agents)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Agent')
        plt.title(f'Message Flow Between Agents - Task {task_id}')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "message_flow.png"))
        plt.close()
        
        self.logger.info(f"Message flow visualization saved for task {task_id}")
    
    def plot_message_types_pie(self, messages: List[Dict[str, Any]], task_id: str):
        """
        Create a pie chart of message types.
        
        Args:
            messages: List of messages exchanged during solution development
            task_id: ID of the task being visualized
        """
        if not messages:
            self.logger.warning(f"No messages to visualize for task {task_id}")
            return
        
        # Extract message types
        message_types = [m.get('message_type', 'unknown') for m in messages]
        
        # Count occurrences of each type
        type_counts = Counter(message_types)
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(
            type_counts.values(), 
            labels=[t.replace('_', ' ').title() for t in type_counts.keys()],
            autopct='%1.1f%%',
            startangle=90
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title(f'Distribution of Message Types - Task {task_id}')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "message_types_pie.png"))
        plt.close()
        
        self.logger.info(f"Message types pie chart saved for task {task_id}")
    
    def plot_overall_comparison(self, all_task_results: Dict[str, Dict[str, Dict[str, Any]]]):
        """
        Create a comprehensive comparison of all tasks and systems.
        
        Args:
            all_task_results: Nested dictionary mapping task IDs to systems to results
        """
        # Metrics to compare
        metrics = ['time_to_solution', 'estimated_maintainability', 'lines_of_code', 'cyclomatic_complexity']
        
        # Systems to compare
        systems = set()
        for task_results in all_task_results.values():
            systems.update(task_results.keys())
        systems = sorted(list(systems))
        
        # Calculate average metrics across all tasks
        avg_metrics = {system: {metric: 0 for metric in metrics} for system in systems}
        task_count = len(all_task_results)
        
        for task_id, task_results in all_task_results.items():
            for system in systems:
                if system in task_results:
                    for metric in metrics:
                        avg_metrics[system][metric] += task_results[system].get(metric, 0) / task_count
        
        # Plot average metrics
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            # Extract values
            values = [avg_metrics[system][metric] for system in systems]
            
            # Create bar chart
            bars = plt.bar(systems, values, color=['blue', 'green'])
            
            # Add labels and title
            plt.xlabel('System')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'Average {metric.replace("_", " ").title()} Across All Tasks')
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., 
                        height + 0.02 * max(values),
                        f'{height:.1f}',
                        ha='center', va='bottom', rotation=0)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"avg_{metric}_comparison.png"))
            plt.close()
            
        self.logger.info("Overall comparison plots saved")
    
    def generate_summary_table(self, all_task_results: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
        """
        Generate a markdown summary table of results.
        
        Args:
            all_task_results: Nested dictionary mapping task IDs to systems to results
            
        Returns:
            Markdown formatted table string
        """
        # Systems to compare
        systems = set()
        for task_results in all_task_results.values():
            systems.update(task_results.keys())
        systems = sorted(list(systems))
        
        # Generate table header
        table = "| Task | " + " | ".join(systems) + " |\n"
        table += "|------|" + "|".join(["---" for _ in systems]) + "|\n"
        
        # Generate table rows
        metrics = ['Time (s)', 'Maintainability', 'LOC', 'Complexity']
        
        for task_id in sorted(all_task_results.keys()):
            # Time row
            table += f"| {task_id} - Time (s) |"
            for system in systems:
                if system in all_task_results[task_id]:
                    value = all_task_results[task_id][system].get('time_to_solution', 'N/A')
                    table += f" {value:.1f} |"
                else:
                    table += " N/A |"
            table += "\n"
            
            # Maintainability row
            table += f"| {task_id} - Maintainability |"
            for system in systems:
                if system in all_task_results[task_id]:
                    value = all_task_results[task_id][system].get('estimated_maintainability', 'N/A')
                    table += f" {value:.1f} |"
                else:
                    table += " N/A |"
            table += "\n"
            
            # LOC row
            table += f"| {task_id} - LOC |"
            for system in systems:
                if system in all_task_results[task_id]:
                    value = all_task_results[task_id][system].get('lines_of_code', 'N/A')
                    table += f" {value} |"
                else:
                    table += " N/A |"
            table += "\n"
            
            # Complexity row
            table += f"| {task_id} - Complexity |"
            for system in systems:
                if system in all_task_results[task_id]:
                    value = all_task_results[task_id][system].get('cyclomatic_complexity', 'N/A')
                    table += f" {value:.1f} |"
                else:
                    table += " N/A |"
            table += "\n"
        
        return table
"""
Visualization Module for TrustPath Experiment Results.

This module provides functions to visualize the results of the TrustPath
experiment, including performance metrics, comparative analyses, and
visual examples of error detection and correction.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from config import VIZ_CONFIG, RESULTS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = VIZ_CONFIG["figsize"]
plt.rcParams['font.size'] = VIZ_CONFIG["font_size"]

class TrustPathVisualizer:
    """
    Visualizes the results of the TrustPath experiment.
    
    This class provides functions to create various visualizations of
    the experiment results, including performance metrics, comparative
    analyses, and visual examples of error detection and correction.
    """
    
    def __init__(self):
        """
        Initialize the visualizer.
        """
        # Create results directory if it doesn't exist
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set up color maps
        self.confidence_colors = VIZ_CONFIG["confidence_colors"]
        self.method_colors = {
            "TrustPath": "#4285F4",  # Google blue
            "simple_fact_checking": "#EA4335",  # Google red
            "uncertainty_estimation": "#FBBC05",  # Google yellow
            "standard_correction": "#34A853"  # Google green
        }
        
        # Set up line styles
        self.line_width = VIZ_CONFIG["line_width"]
        
        logger.info("Initialized TrustPathVisualizer")
    
    def visualize_error_detection_performance(self, evaluation_results: Dict[str, Any], filename: str = "error_detection_performance.png") -> str:
        """
        Visualize error detection performance metrics.
        
        Args:
            evaluation_results: Dictionary with evaluation results for all methods
            filename: Output filename
            
        Returns:
            Path to the saved figure
        """
        logger.info("Visualizing error detection performance...")
        
        # Extract data
        methods = []
        precision = []
        recall = []
        f1 = []
        
        for method_name, results in evaluation_results.items():
            error_detection = results.get("error_detection", {})
            
            methods.append(method_name)
            precision.append(error_detection.get("precision", 0.0))
            recall.append(error_detection.get("recall", 0.0))
            f1.append(error_detection.get("f1", 0.0))
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize"])
        
        # Set width of bars
        bar_width = 0.25
        
        # Set position of bars on x axis
        r1 = np.arange(len(methods))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Create bars
        ax.bar(r1, precision, width=bar_width, label='Precision', color='#4285F4')
        ax.bar(r2, recall, width=bar_width, label='Recall', color='#34A853')
        ax.bar(r3, f1, width=bar_width, label='F1 Score', color='#EA4335')
        
        # Add labels and title
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('Error Detection Performance')
        ax.set_xticks([r + bar_width for r in range(len(methods))])
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save the figure
        file_path = RESULTS_DIR / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Error detection performance visualization saved to {file_path}")
        return str(file_path)
    
    def visualize_correction_quality(self, evaluation_results: Dict[str, Any], filename: str = "correction_quality.png") -> str:
        """
        Visualize correction quality metrics.
        
        Args:
            evaluation_results: Dictionary with evaluation results for all methods
            filename: Output filename
            
        Returns:
            Path to the saved figure
        """
        logger.info("Visualizing correction quality...")
        
        # Extract data
        methods = []
        bleu = []
        rouge1 = []
        rouge2 = []
        rougeL = []
        exact_match = []
        
        for method_name, results in evaluation_results.items():
            correction_quality = results.get("correction_quality", {})
            
            methods.append(method_name)
            bleu.append(correction_quality.get("bleu", 0.0))
            rouge1.append(correction_quality.get("rouge1_f", 0.0))
            rouge2.append(correction_quality.get("rouge2_f", 0.0))
            rougeL.append(correction_quality.get("rougeL_f", 0.0))
            exact_match.append(correction_quality.get("exact_match_ratio", 0.0))
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize"])
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            'Method': methods,
            'BLEU': bleu,
            'ROUGE-1': rouge1,
            'ROUGE-2': rouge2,
            'ROUGE-L': rougeL,
            'Exact Match': exact_match
        })
        
        # Melt the DataFrame for seaborn
        df_melt = pd.melt(df, id_vars=['Method'], var_name='Metric', value_name='Score')
        
        # Create the grouped bar chart
        sns.barplot(x='Method', y='Score', hue='Metric', data=df_melt, ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('Correction Quality')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title='Metric')
        
        plt.tight_layout()
        
        # Save the figure
        file_path = RESULTS_DIR / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Correction quality visualization saved to {file_path}")
        return str(file_path)
    
    def visualize_system_efficiency(self, evaluation_results: Dict[str, Any], filename: str = "system_efficiency.png") -> str:
        """
        Visualize system efficiency metrics.
        
        Args:
            evaluation_results: Dictionary with evaluation results for all methods
            filename: Output filename
            
        Returns:
            Path to the saved figure
        """
        logger.info("Visualizing system efficiency...")
        
        # Extract data
        methods = []
        avg_processing_time = []
        avg_detection_time = []
        avg_correction_time = []
        
        for method_name, results in evaluation_results.items():
            system_efficiency = results.get("system_efficiency", {})
            
            methods.append(method_name)
            avg_processing_time.append(system_efficiency.get("average_processing_time", 0.0))
            avg_detection_time.append(system_efficiency.get("average_detection_time", 0.0))
            avg_correction_time.append(system_efficiency.get("average_correction_time", 0.0))
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize"])
        
        # Create stacked bars
        bottom = np.zeros(len(methods))
        
        # Only include non-zero detection and correction times
        if any(avg_detection_time) and any(avg_correction_time):
            p1 = ax.bar(methods, avg_detection_time, label='Detection Time')
            bottom = np.array(avg_detection_time)
            p2 = ax.bar(methods, avg_correction_time, bottom=bottom, label='Correction Time')
        else:
            # If no breakdown, just show total processing time
            p1 = ax.bar(methods, avg_processing_time, label='Processing Time')
        
        # Add labels and title
        ax.set_xlabel('Method')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('System Efficiency')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save the figure
        file_path = RESULTS_DIR / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"System efficiency visualization saved to {file_path}")
        return str(file_path)
    
    def visualize_trust_metrics(self, evaluation_results: Dict[str, Any], filename: str = "trust_metrics.png") -> str:
        """
        Visualize trust-related metrics.
        
        Args:
            evaluation_results: Dictionary with evaluation results for all methods
            filename: Output filename
            
        Returns:
            Path to the saved figure
        """
        logger.info("Visualizing trust metrics...")
        
        # Extract data
        methods = []
        trust_calibration = []
        explanation_satisfaction = []
        transparency_score = []
        
        for method_name, results in evaluation_results.items():
            trust_metrics = results.get("trust_metrics", {})
            
            methods.append(method_name)
            trust_calibration.append(trust_metrics.get("trust_calibration", 0.0))
            explanation_satisfaction.append(trust_metrics.get("explanation_satisfaction", 0.0))
            transparency_score.append(trust_metrics.get("transparency_score", 0.0))
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize"])
        
        # Set width of bars
        bar_width = 0.25
        
        # Set position of bars on x axis
        r1 = np.arange(len(methods))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Create bars
        ax.bar(r1, trust_calibration, width=bar_width, label='Trust Calibration', color='#4285F4')
        ax.bar(r2, explanation_satisfaction, width=bar_width, label='Explanation Satisfaction', color='#FBBC05')
        ax.bar(r3, transparency_score, width=bar_width, label='Transparency Score', color='#34A853')
        
        # Add labels and title
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('Trust Metrics')
        ax.set_xticks([r + bar_width for r in range(len(methods))])
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save the figure
        file_path = RESULTS_DIR / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Trust metrics visualization saved to {file_path}")
        return str(file_path)
    
    def visualize_overall_performance(self, evaluation_results: Dict[str, Any], filename: str = "overall_performance.png") -> str:
        """
        Visualize overall performance comparison.
        
        Args:
            evaluation_results: Dictionary with evaluation results for all methods
            filename: Output filename
            
        Returns:
            Path to the saved figure
        """
        logger.info("Visualizing overall performance...")
        
        # Extract data
        methods = []
        overall_scores = []
        
        for method_name, results in evaluation_results.items():
            methods.append(method_name)
            overall_scores.append(results.get("overall_score", 0.0))
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize"])
        
        # Create the bar chart
        bars = ax.bar(methods, overall_scores, color=[self.method_colors.get(m, '#4285F4') for m in methods])
        
        # Add labels and title
        ax.set_xlabel('Method')
        ax.set_ylabel('Overall Score')
        ax.set_title('Overall Performance Comparison')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Set y-axis limit
        ax.set_ylim(0, max(overall_scores) * 1.1)
        
        plt.tight_layout()
        
        # Save the figure
        file_path = RESULTS_DIR / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Overall performance visualization saved to {file_path}")
        return str(file_path)
    
    def visualize_radar_chart(self, evaluation_results: Dict[str, Any], filename: str = "radar_chart.png") -> str:
        """
        Create a radar chart comparing methods across key metrics.
        
        Args:
            evaluation_results: Dictionary with evaluation results for all methods
            filename: Output filename
            
        Returns:
            Path to the saved figure
        """
        logger.info("Creating radar chart...")
        
        # Define the key metrics to include
        metrics = [
            ('Error Detection', 'error_detection', 'f1'),
            ('Correction Quality', 'correction_quality', 'rougeL_f'),
            ('Trust Calibration', 'trust_metrics', 'trust_calibration'),
            ('Transparency', 'trust_metrics', 'transparency_score'),
            ('Efficiency', 'system_efficiency', 'efficiency_score')  # Derived metric
        ]
        
        # Extract data
        methods = list(evaluation_results.keys())
        data = []
        
        for method_name, results in evaluation_results.items():
            method_data = []
            for metric_name, category, key in metrics:
                if category == 'system_efficiency' and key == 'efficiency_score':
                    # Calculate efficiency score (higher is better)
                    avg_time = results.get(category, {}).get('average_processing_time', 1.0)
                    efficiency_score = 1.0 / (1.0 + avg_time / 10.0)  # Normalize
                    method_data.append(efficiency_score)
                else:
                    method_data.append(results.get(category, {}).get(key, 0.0))
            data.append(method_data)
        
        # Set up the figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of metrics
        N = len(metrics)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Draw the chart for each method
        for i, method in enumerate(methods):
            values = data[i]
            values += values[:1]  # Close the loop
            
            # Plot data
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=self.method_colors.get(method, f'C{i}'))
            
            # Fill area
            ax.fill(angles, values, alpha=0.1, color=self.method_colors.get(method, f'C{i}'))
        
        # Set labels and title
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m[0] for m in metrics])
        ax.set_title('Method Comparison Across Key Metrics')
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        
        # Save the figure
        file_path = RESULTS_DIR / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Radar chart saved to {file_path}")
        return str(file_path)
    
    def visualize_error_detection_example(self, sample_data: Dict[str, Any], filename: str = "error_detection_example.png") -> str:
        """
        Visualize an example of error detection with TrustPath.
        
        Args:
            sample_data: Dictionary with sample data and analysis results
            filename: Output filename
            
        Returns:
            Path to the saved figure
        """
        logger.info("Visualizing error detection example...")
        
        # Extract data
        original_text = sample_data.get("original_response", "")
        spans = sample_data.get("visualization_data", {}).get("spans", [])
        
        if not original_text or not spans:
            logger.warning("Cannot create error detection example: missing data")
            return ""
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Remove axes
        ax.axis('off')
        
        # Add title
        ax.set_title('Example of Error Detection with TrustPath')
        
        # Calculate maximum line length
        max_line_length = 80
        
        # Split text into lines
        lines = []
        current_line = ""
        
        for word in original_text.split():
            if len(current_line) + len(word) + 1 <= max_line_length:
                current_line += word + " "
            else:
                lines.append(current_line)
                current_line = word + " "
        
        if current_line:
            lines.append(current_line)
        
        # Create map from original text to line and position
        pos_map = {}
        current_pos = 0
        
        for line_idx, line in enumerate(lines):
            for i in range(len(line)):
                pos_map[current_pos] = (line_idx, i)
                current_pos += 1
        
        # Draw text and highlighted spans
        for line_idx, line in enumerate(lines):
            # Draw base text
            ax.text(0.05, 0.95 - line_idx * 0.03, line, fontsize=10, family='monospace', transform=ax.transAxes)
        
        # Draw highlights
        for span in spans:
            start_pos = span.get("start", 0)
            end_pos = span.get("end", 0)
            confidence_level = span.get("confidence_level", "medium")
            color = self.confidence_colors.get(confidence_level, "#FFC107")
            
            # Find the line and position of the span
            if start_pos in pos_map and end_pos in pos_map:
                start_line, start_col = pos_map[start_pos]
                end_line, end_col = pos_map[end_pos]
                
                if start_line == end_line:
                    # Span is on a single line
                    line_text = lines[start_line]
                    highlight_text = line_text[start_col:end_col]
                    
                    # Draw highlighted text
                    highlight_x = 0.05 + start_col * 0.0085
                    highlight_y = 0.95 - start_line * 0.03
                    
                    # Draw highlight box
                    text_bbox = ax.text(highlight_x, highlight_y, highlight_text, fontsize=10, 
                                      family='monospace', transform=ax.transAxes, color='black',
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        
        # Add legend
        patches = []
        for level, color in self.confidence_colors.items():
            patches.append(mpatches.Patch(color=color, alpha=0.3, label=f'{level.capitalize()} confidence error'))
        
        ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.15))
        
        plt.tight_layout()
        
        # Save the figure
        file_path = RESULTS_DIR / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Error detection example saved to {file_path}")
        return str(file_path)
    
    def visualize_learning_curve(self, learning_data: Dict[str, List[float]], filename: str = "learning_curve.png") -> str:
        """
        Visualize a learning curve showing performance improvement with feedback.
        
        Args:
            learning_data: Dictionary with metrics at different feedback iterations
            filename: Output filename
            
        Returns:
            Path to the saved figure
        """
        logger.info("Visualizing learning curve...")
        
        # Extract data
        iterations = list(range(len(learning_data.get("f1_scores", []))))
        f1_scores = learning_data.get("f1_scores", [])
        precision_scores = learning_data.get("precision_scores", [])
        recall_scores = learning_data.get("recall_scores", [])
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize"])
        
        # Plot the learning curves
        ax.plot(iterations, f1_scores, 'o-', linewidth=self.line_width, label='F1 Score', color='#4285F4')
        
        if precision_scores and recall_scores:
            ax.plot(iterations, precision_scores, 's--', linewidth=self.line_width, label='Precision', color='#34A853')
            ax.plot(iterations, recall_scores, '^--', linewidth=self.line_width, label='Recall', color='#EA4335')
        
        # Add labels and title
        ax.set_xlabel('Feedback Iteration')
        ax.set_ylabel('Score')
        ax.set_title('Performance Improvement with Human Feedback')
        ax.legend()
        
        # Set axis limits
        ax.set_xlim(0, max(iterations))
        ax.set_ylim(0, 1.0)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the figure
        file_path = RESULTS_DIR / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Learning curve visualization saved to {file_path}")
        return str(file_path)
    
    def visualize_domain_performance(self, domain_results: Dict[str, Dict[str, float]], filename: str = "domain_performance.png") -> str:
        """
        Visualize performance across different domains.
        
        Args:
            domain_results: Dictionary with performance metrics for each domain
            filename: Output filename
            
        Returns:
            Path to the saved figure
        """
        logger.info("Visualizing domain performance...")
        
        # Extract data
        domains = list(domain_results.keys())
        methods = list(domain_results[domains[0]].keys())
        
        # Create a DataFrame for easier plotting
        data = []
        
        for domain in domains:
            for method in methods:
                data.append({
                    'Domain': domain,
                    'Method': method,
                    'F1 Score': domain_results[domain][method]
                })
        
        df = pd.DataFrame(data)
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize"])
        
        # Create grouped bar chart
        sns.barplot(x='Domain', y='F1 Score', hue='Method', data=df, ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Domain')
        ax.set_ylabel('F1 Score')
        ax.set_title('Performance Across Domains')
        ax.legend(title='Method')
        
        plt.tight_layout()
        
        # Save the figure
        file_path = RESULTS_DIR / filename
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Domain performance visualization saved to {file_path}")
        return str(file_path)
    
    def visualize_all_metrics(self, evaluation_results: Dict[str, Any], output_dir: Path = RESULTS_DIR) -> Dict[str, str]:
        """
        Generate all visualizations for the experiment results.
        
        Args:
            evaluation_results: Dictionary with evaluation results for all methods
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary with paths to all generated figures
        """
        logger.info("Generating all visualizations...")
        
        figure_paths = {}
        
        # Generate each visualization
        figure_paths["error_detection"] = self.visualize_error_detection_performance(evaluation_results)
        figure_paths["correction_quality"] = self.visualize_correction_quality(evaluation_results)
        figure_paths["system_efficiency"] = self.visualize_system_efficiency(evaluation_results)
        figure_paths["trust_metrics"] = self.visualize_trust_metrics(evaluation_results)
        figure_paths["overall_performance"] = self.visualize_overall_performance(evaluation_results)
        figure_paths["radar_chart"] = self.visualize_radar_chart(evaluation_results)
        
        logger.info(f"Generated {len(figure_paths)} visualizations")
        return figure_paths

if __name__ == "__main__":
    # Simple test of the visualizer
    print("Testing visualizer...")
    
    # Create sample evaluation results
    evaluation_results = {
        "TrustPath": {
            "error_detection": {
                "precision": 0.85,
                "recall": 0.80,
                "f1": 0.82,
                "accuracy": 0.83
            },
            "correction_quality": {
                "bleu": 0.65,
                "rouge1_f": 0.75,
                "rouge2_f": 0.60,
                "rougeL_f": 0.70,
                "exact_match_ratio": 0.30
            },
            "system_efficiency": {
                "total_time": 10.0,
                "average_processing_time": 2.0,
                "average_detection_time": 1.2,
                "average_correction_time": 0.8
            },
            "trust_metrics": {
                "trust_calibration": 0.85,
                "explanation_satisfaction": 0.90,
                "transparency_score": 0.95
            },
            "overall_score": 0.85
        },
        "simple_fact_checking": {
            "error_detection": {
                "precision": 0.70,
                "recall": 0.65,
                "f1": 0.67,
                "accuracy": 0.68
            },
            "correction_quality": {
                "bleu": 0.55,
                "rouge1_f": 0.65,
                "rouge2_f": 0.50,
                "rougeL_f": 0.60,
                "exact_match_ratio": 0.20
            },
            "system_efficiency": {
                "total_time": 5.0,
                "average_processing_time": 1.0,
                "average_detection_time": 0.6,
                "average_correction_time": 0.4
            },
            "trust_metrics": {
                "trust_calibration": 0.60,
                "explanation_satisfaction": 0.50,
                "transparency_score": 0.40
            },
            "overall_score": 0.60
        },
        "uncertainty_estimation": {
            "error_detection": {
                "precision": 0.75,
                "recall": 0.70,
                "f1": 0.72,
                "accuracy": 0.73
            },
            "correction_quality": {
                "bleu": 0.50,
                "rouge1_f": 0.60,
                "rouge2_f": 0.45,
                "rougeL_f": 0.55,
                "exact_match_ratio": 0.15
            },
            "system_efficiency": {
                "total_time": 6.0,
                "average_processing_time": 1.2,
                "average_detection_time": 0.8,
                "average_correction_time": 0.4
            },
            "trust_metrics": {
                "trust_calibration": 0.70,
                "explanation_satisfaction": 0.60,
                "transparency_score": 0.50
            },
            "overall_score": 0.65
        },
        "standard_correction": {
            "error_detection": {
                "precision": 0.65,
                "recall": 0.75,
                "f1": 0.70,
                "accuracy": 0.68
            },
            "correction_quality": {
                "bleu": 0.60,
                "rouge1_f": 0.70,
                "rouge2_f": 0.55,
                "rougeL_f": 0.65,
                "exact_match_ratio": 0.25
            },
            "system_efficiency": {
                "total_time": 7.0,
                "average_processing_time": 1.4,
                "average_detection_time": 0.7,
                "average_correction_time": 0.7
            },
            "trust_metrics": {
                "trust_calibration": 0.65,
                "explanation_satisfaction": 0.55,
                "transparency_score": 0.45
            },
            "overall_score": 0.62
        }
    }
    
    visualizer = TrustPathVisualizer()
    figure_paths = visualizer.visualize_all_metrics(evaluation_results)
    
    for name, path in figure_paths.items():
        print(f"{name}: {path}")
"""
Visualization Module

This module generates visualizations of the experimental results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Any, Optional
import logging

class ExperimentVisualizer:
    """
    Generates visualizations of the experiment results.
    """
    
    def __init__(self, results: Dict[str, Any], config: Dict[str, Any], save_dir: str):
        """
        Initialize the visualizer.
        
        Args:
            results: Results dictionary from the experiment
            config: Configuration dictionary
            save_dir: Directory to save visualizations
        """
        self.results = results
        self.config = config
        self.save_dir = save_dir
        self.logger = logging.getLogger(__name__)
        
        # Create visualization directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up Seaborn style
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
        # Convert data to pandas DataFrames
        self.trial_df = pd.DataFrame(results["trial_data"])
        self.participant_df = pd.DataFrame(results["participant_data"])
        self.summary_metrics = results["summary_metrics"]
        
        self.logger.info(f"Initialized visualizer with {len(self.trial_df)} trial records and {len(self.participant_df)} participants")
    
    def generate_all_visualizations(self) -> List[str]:
        """
        Generate all visualizations for the experiment.
        
        Returns:
            List of paths to the generated visualizations
        """
        self.logger.info("Generating all visualizations")
        
        visualization_paths = []
        
        # Generate mental model accuracy visualization
        mma_path = self.visualize_mental_model_accuracy()
        if mma_path:
            visualization_paths.append(mma_path)
        
        # Generate diagnostic performance visualization
        perf_path = self.visualize_diagnostic_performance()
        if perf_path:
            visualization_paths.append(perf_path)
        
        # Generate complexity comparison visualization
        complexity_path = self.visualize_complexity_comparison()
        if complexity_path:
            visualization_paths.append(complexity_path)
        
        # Generate learning curves visualization
        learning_path = self.visualize_learning_curves()
        if learning_path:
            visualization_paths.append(learning_path)
        
        # Generate intervention effectiveness visualization
        intervention_path = self.visualize_intervention_effectiveness()
        if intervention_path:
            visualization_paths.append(intervention_path)
        
        # Generate confusion level visualization
        confusion_path = self.visualize_confusion_levels()
        if confusion_path:
            visualization_paths.append(confusion_path)
        
        # Generate trust calibration visualization
        trust_path = self.visualize_trust_calibration()
        if trust_path:
            visualization_paths.append(trust_path)
        
        # Generate expertise level comparison
        expertise_path = self.visualize_expertise_comparison()
        if expertise_path:
            visualization_paths.append(expertise_path)
        
        self.logger.info(f"Generated {len(visualization_paths)} visualizations")
        return visualization_paths
    
    def visualize_mental_model_accuracy(self) -> Optional[str]:
        """
        Generate visualization of mental model accuracy comparison.
        
        Returns:
            Path to the saved visualization
        """
        if "mental_model_accuracy" not in self.summary_metrics:
            self.logger.warning("Mental model accuracy data not available for visualization")
            return None
        
        self.logger.info("Generating mental model accuracy visualization")
        
        # Extract data
        metrics = self.summary_metrics["mental_model_accuracy"]
        treatment_mma = metrics["treatment"]
        control_mma = metrics["control"]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        groups = ['Control Group', 'Treatment Group']
        values = [control_mma, treatment_mma]
        colors = ['#3498db', '#2ecc71']
        
        plt.bar(groups, values, color=colors, width=0.6)
        
        # Add values on top of bars
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
        
        # Add percentage improvement label
        percent_improvement = metrics["percent_improvement"]
        plt.annotate(f"+{percent_improvement:.1f}%", 
                   xy=(1, treatment_mma), 
                   xytext=(1.3, treatment_mma + (treatment_mma - control_mma) / 2),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   fontsize=12,
                   fontweight='bold')
        
        # Set graph properties
        plt.title('Mental Model Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Mental Model Accuracy Score (0-1)', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add description
        plt.figtext(0.5, 0.01, 
                  'Mental Model Accuracy measures how well participants understand the AI system\'s reasoning and operation.',
                  ha='center', fontsize=10, style='italic')
        
        # Save figure
        output_path = os.path.join(self.save_dir, 'mental_model_accuracy.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def visualize_diagnostic_performance(self) -> Optional[str]:
        """
        Generate visualization of diagnostic performance comparison.
        
        Returns:
            Path to the saved visualization
        """
        if "diagnostic_performance" not in self.summary_metrics:
            self.logger.warning("Diagnostic performance data not available for visualization")
            return None
        
        self.logger.info("Generating diagnostic performance visualization")
        
        # Extract data
        metrics = self.summary_metrics["diagnostic_performance"]
        treatment_acc = metrics["treatment"]
        control_acc = metrics["control"]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        groups = ['Control Group', 'Treatment Group']
        values = [control_acc, treatment_acc]
        colors = ['#3498db', '#2ecc71']
        
        plt.bar(groups, values, color=colors, width=0.6)
        
        # Add values on top of bars
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
        
        # Add percentage improvement label
        percent_improvement = metrics["percent_improvement"]
        plt.annotate(f"+{percent_improvement:.1f}%", 
                   xy=(1, treatment_acc), 
                   xytext=(1.3, treatment_acc + (treatment_acc - control_acc) / 2),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   fontsize=12,
                   fontweight='bold')
        
        # Set graph properties
        plt.title('Diagnostic Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Diagnostic Accuracy (0-1)', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add description
        plt.figtext(0.5, 0.01, 
                  'Diagnostic Accuracy measures how often participants made correct diagnostic decisions.',
                  ha='center', fontsize=10, style='italic')
        
        # Save figure
        output_path = os.path.join(self.save_dir, 'diagnostic_performance.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def visualize_complexity_comparison(self) -> Optional[str]:
        """
        Generate visualization comparing performance across different complexity levels.
        
        Returns:
            Path to the saved visualization
        """
        complexity_metrics = [
            self.summary_metrics.get("diagnostic_performance_simple", None),
            self.summary_metrics.get("diagnostic_performance_medium", None),
            self.summary_metrics.get("diagnostic_performance_complex", None)
        ]
        
        if not all(complexity_metrics):
            self.logger.warning("Complexity comparison data not available for visualization")
            return None
        
        self.logger.info("Generating complexity comparison visualization")
        
        # Extract data
        simple_metrics = self.summary_metrics["diagnostic_performance_simple"]
        medium_metrics = self.summary_metrics["diagnostic_performance_medium"]
        complex_metrics = self.summary_metrics["diagnostic_performance_complex"]
        
        # Create figure
        plt.figure(figsize=(12, 7))
        
        # Set up data for grouped bar chart
        complexity_levels = ['Simple', 'Medium', 'Complex']
        treatment_values = [simple_metrics["treatment"], medium_metrics["treatment"], complex_metrics["treatment"]]
        control_values = [simple_metrics["control"], medium_metrics["control"], complex_metrics["control"]]
        
        # Set positions and width
        x = np.arange(len(complexity_levels))
        width = 0.35
        
        # Create grouped bar chart
        plt.bar(x - width/2, control_values, width, label='Control Group', color='#3498db')
        plt.bar(x + width/2, treatment_values, width, label='Treatment Group', color='#2ecc71')
        
        # Add values on top of bars
        for i, v in enumerate(control_values):
            plt.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
        
        for i, v in enumerate(treatment_values):
            plt.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
        
        # Add percentage improvement labels
        for i, (c, t) in enumerate(zip(control_values, treatment_values)):
            if c > 0:
                percent_imp = ((t - c) / c) * 100
                plt.annotate(f"{percent_imp:+.1f}%", 
                           xy=(i, t), 
                           xytext=(i, t + 0.05),
                           ha='center',
                           fontsize=10,
                           fontweight='bold')
        
        # Set graph properties
        plt.title('Diagnostic Accuracy by Case Complexity', fontsize=16, fontweight='bold')
        plt.ylabel('Diagnostic Accuracy (0-1)', fontsize=12)
        plt.xticks(x, complexity_levels, fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Add description
        plt.figtext(0.5, 0.01, 
                  'Comparison of diagnostic accuracy across different levels of case complexity.',
                  ha='center', fontsize=10, style='italic')
        
        # Save figure
        output_path = os.path.join(self.save_dir, 'complexity_comparison.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def visualize_learning_curves(self) -> Optional[str]:
        """
        Generate visualization of learning curves over trial progression.
        
        Returns:
            Path to the saved visualization
        """
        if "learning_trends" not in self.results:
            self.logger.warning("Learning trends data not available for visualization")
            return None
        
        learning_trends = self.results["learning_trends"]
        
        if not all(k in learning_trends for k in ["trial_points", "treatment_curve", "control_curve"]):
            self.logger.warning("Learning trends data incomplete")
            return None
        
        self.logger.info("Generating learning curves visualization")
        
        # Extract data
        trial_points = learning_trends["trial_points"]
        treatment_curve = learning_trends["treatment_curve"]
        control_curve = learning_trends["control_curve"]
        
        # Create figure
        plt.figure(figsize=(12, 7))
        
        # Plot learning curves
        plt.plot(trial_points, treatment_curve, marker='o', linestyle='-', linewidth=2, 
                 label='Treatment Group (with AI Cognitive Tutor)', color='#2ecc71')
        plt.plot(trial_points, control_curve, marker='s', linestyle='-', linewidth=2, 
                 label='Control Group (without AI Cognitive Tutor)', color='#3498db')
        
        # Fill the area between the curves
        plt.fill_between(trial_points, treatment_curve, control_curve, 
                       where=(np.array(treatment_curve) > np.array(control_curve)), 
                       interpolate=True, color='#2ecc71', alpha=0.1)
        
        # Set graph properties
        plt.title('Learning Curves: Diagnostic Accuracy Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Trial Number', fontsize=12)
        plt.ylabel('Cumulative Diagnostic Accuracy (0-1)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Add annotations for key points
        if len(trial_points) > 5:
            # Find point of maximum difference
            differences = [t - c for t, c in zip(treatment_curve, control_curve)]
            max_diff_idx = differences.index(max(differences))
            
            # Annotate maximum difference
            plt.annotate(f"Maximum difference: {differences[max_diff_idx]:.2f}", 
                       xy=(trial_points[max_diff_idx], treatment_curve[max_diff_idx]),
                       xytext=(trial_points[max_diff_idx] + 2, treatment_curve[max_diff_idx] + 0.05),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                       fontsize=10)
        
        # Add description
        plt.figtext(0.5, 0.01, 
                  'Learning curves show how participants\' diagnostic accuracy improves over successive trials.',
                  ha='center', fontsize=10, style='italic')
        
        # Save figure
        output_path = os.path.join(self.save_dir, 'learning_curves.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def visualize_intervention_effectiveness(self) -> Optional[str]:
        """
        Generate visualization of intervention effectiveness by type.
        
        Returns:
            Path to the saved visualization
        """
        if "subgroup_analyses" not in self.results.get("summary_metrics", {}) or \
           "intervention_types" not in self.results["summary_metrics"]["subgroup_analyses"]:
            self.logger.warning("Intervention effectiveness data not available for visualization")
            return None
        
        intervention_data = self.results["summary_metrics"]["subgroup_analyses"]["intervention_types"]
        
        if not intervention_data:
            self.logger.warning("No intervention data available")
            return None
        
        self.logger.info("Generating intervention effectiveness visualization")
        
        # Extract data
        intervention_types = list(intervention_data.keys())
        helpfulness_scores = [intervention_data[t]["helpfulness"] for t in intervention_types]
        improvement_rates = [intervention_data[t]["improvement_rate"] for t in intervention_types]
        counts = [intervention_data[t]["count"] for t in intervention_types]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot helpfulness scores
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(intervention_types)))
        
        # Helpfulness plot
        bars1 = ax1.bar(intervention_types, helpfulness_scores, color=colors)
        
        # Add count annotations
        for i, (bar, count) in enumerate(zip(bars1, counts)):
            ax1.text(bar.get_x() + bar.get_width()/2, 0.1, 
                   f"n={count}", ha='center', va='bottom', 
                   color='white', fontweight='bold', rotation=90)
        
        ax1.set_title('Helpfulness Score by Intervention Type', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Helpfulness Score (0-10)', fontsize=12)
        ax1.set_ylim(0, 10.5)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Improvement rate plot
        bars2 = ax2.bar(intervention_types, improvement_rates, color=colors)
        
        # Add count annotations
        for i, (bar, count) in enumerate(zip(bars2, counts)):
            ax2.text(bar.get_x() + bar.get_width()/2, 0.02, 
                   f"n={count}", ha='center', va='bottom', 
                   color='white', fontweight='bold', rotation=90)
        
        ax2.set_title('Understanding Improvement Rate by Intervention Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Improvement Rate (0-1)', fontsize=12)
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, v in enumerate(helpfulness_scores):
            ax1.text(i, v + 0.2, f"{v:.2f}", ha='center')
        
        for i, v in enumerate(improvement_rates):
            ax2.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Add description
        plt.figtext(0.5, 0.01, 
                  'Comparison of different intervention types in terms of perceived helpfulness and improvement in understanding.',
                  ha='center', fontsize=10, style='italic')
        
        # Save figure
        output_path = os.path.join(self.save_dir, 'intervention_effectiveness.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def visualize_confusion_levels(self) -> Optional[str]:
        """
        Generate visualization of confusion levels comparison.
        
        Returns:
            Path to the saved visualization
        """
        if "user_ai_misalignment" not in self.summary_metrics:
            self.logger.warning("Confusion level data not available for visualization")
            return None
        
        self.logger.info("Generating confusion levels visualization")
        
        # Extract data
        metrics = self.summary_metrics["user_ai_misalignment"]
        treatment_confusion = metrics["treatment"]
        control_confusion = metrics["control"]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        groups = ['Control Group', 'Treatment Group']
        values = [control_confusion, treatment_confusion]
        colors = ['#e74c3c', '#f39c12']  # Red for control (worse), yellow for treatment (better)
        
        plt.bar(groups, values, color=colors, width=0.6)
        
        # Add values on top of bars
        for i, v in enumerate(values):
            plt.text(i, v + 0.2, f"{v:.2f}", ha='center', fontweight='bold')
        
        # Add percentage improvement label
        percent_improvement = metrics["percent_improvement"]
        plt.annotate(f"-{percent_improvement:.1f}%", 
                   xy=(1, treatment_confusion), 
                   xytext=(1.3, treatment_confusion - (control_confusion - treatment_confusion) / 2),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   fontsize=12,
                   fontweight='bold')
        
        # Set graph properties
        plt.title('User Confusion Level Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Average Confusion Level (0-10)', fontsize=12)
        plt.ylim(0, 10.5)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add description
        plt.figtext(0.5, 0.01, 
                  'Lower confusion levels indicate better understanding of the AI system\'s reasoning and outputs.',
                  ha='center', fontsize=10, style='italic')
        
        # Save figure
        output_path = os.path.join(self.save_dir, 'confusion_levels.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def visualize_trust_calibration(self) -> Optional[str]:
        """
        Generate visualization of trust calibration comparison.
        
        Returns:
            Path to the saved visualization
        """
        if "trust_calibration" not in self.summary_metrics:
            self.logger.warning("Trust calibration data not available for visualization")
            return None
        
        self.logger.info("Generating trust calibration visualization")
        
        # Extract data
        metrics = self.summary_metrics["trust_calibration"]
        treatment_trust = metrics["treatment"]
        control_trust = metrics["control"]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        groups = ['Control Group', 'Treatment Group']
        values = [control_trust, treatment_trust]
        colors = ['#3498db', '#2ecc71']
        
        plt.bar(groups, values, color=colors, width=0.6)
        
        # Add values on top of bars
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
        
        # Add percentage improvement label
        percent_improvement = metrics["percent_improvement"]
        plt.annotate(f"+{percent_improvement:.1f}%", 
                   xy=(1, treatment_trust), 
                   xytext=(1.3, treatment_trust + (treatment_trust - control_trust) / 2),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   fontsize=12,
                   fontweight='bold')
        
        # Set graph properties
        plt.title('Trust Calibration Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Trust Calibration Score (0-1)', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add description
        plt.figtext(0.5, 0.01, 
                  'Trust calibration measures how well participants\' trust aligns with the AI system\'s actual reliability.',
                  ha='center', fontsize=10, style='italic')
        
        # Save figure
        output_path = os.path.join(self.save_dir, 'trust_calibration.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def visualize_expertise_comparison(self) -> Optional[str]:
        """
        Generate visualization comparing performance across different expertise levels.
        
        Returns:
            Path to the saved visualization
        """
        if "subgroup_analyses" not in self.summary_metrics or \
           "expertise_levels" not in self.summary_metrics["subgroup_analyses"]:
            self.logger.warning("Expertise level comparison data not available for visualization")
            return None
        
        expertise_data = self.summary_metrics["subgroup_analyses"]["expertise_levels"]
        
        if not expertise_data:
            self.logger.warning("No expertise level data available")
            return None
        
        self.logger.info("Generating expertise level comparison visualization")
        
        # Extract data
        expertise_levels = list(expertise_data.keys())
        treatment_values = [expertise_data[e]["treatment_accuracy"] for e in expertise_levels]
        control_values = [expertise_data[e]["control_accuracy"] for e in expertise_levels]
        sample_sizes = [expertise_data[e]["sample_size"] for e in expertise_levels]
        
        # Create figure
        plt.figure(figsize=(12, 7))
        
        # Set up data for grouped bar chart
        x = np.arange(len(expertise_levels))
        width = 0.35
        
        # Create grouped bar chart
        plt.bar(x - width/2, control_values, width, label='Control Group', color='#3498db')
        plt.bar(x + width/2, treatment_values, width, label='Treatment Group', color='#2ecc71')
        
        # Add values on top of bars
        for i, v in enumerate(control_values):
            plt.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
        
        for i, v in enumerate(treatment_values):
            plt.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
        
        # Add sample size annotations
        for i, n in enumerate(sample_sizes):
            plt.annotate(f"n={n}", xy=(i, 0.05), ha='center', fontsize=9)
        
        # Add percentage improvement labels
        for i, (c, t) in enumerate(zip(control_values, treatment_values)):
            if c > 0:
                percent_imp = ((t - c) / c) * 100
                plt.annotate(f"{percent_imp:+.1f}%", 
                           xy=(i, max(t, c)), 
                           xytext=(i, max(t, c) + 0.05),
                           ha='center',
                           fontsize=10,
                           fontweight='bold')
        
        # Set graph properties
        plt.title('Diagnostic Accuracy by Expertise Level', fontsize=16, fontweight='bold')
        plt.ylabel('Diagnostic Accuracy (0-1)', fontsize=12)
        plt.xticks(x, [e.capitalize() for e in expertise_levels], fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Add description
        plt.figtext(0.5, 0.01, 
                  'Comparison of diagnostic accuracy across different levels of user expertise.',
                  ha='center', fontsize=10, style='italic')
        
        # Save figure
        output_path = os.path.join(self.save_dir, 'expertise_comparison.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
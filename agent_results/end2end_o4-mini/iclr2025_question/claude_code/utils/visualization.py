"""
Visualization utilities for SCEC experiments.

Implements:
- Calibration plots
- Uncertainty distribution plots
- Hallucination detection ROC curves
- Task performance comparison plots
- Diversity metrics visualization
- Ablation study plots
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

logger = logging.getLogger(__name__)

# Set up plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('colorblind')


class CalibrationPlots:
    """Generate calibration plots for uncertainty estimation."""
    
    @staticmethod
    def reliability_diagram(
        confidence_scores: List[float],
        correctness: List[bool],
        num_bins: int = 10,
        method_name: str = "Method",
        ax: Optional[plt.Axes] = None,
        show_ece: bool = True,
        bar_alpha: float = 0.8,
    ) -> plt.Figure:
        """
        Generate a reliability diagram.
        
        Args:
            confidence_scores: List of model confidence scores (0-1)
            correctness: List of boolean values indicating if predictions were correct
            num_bins: Number of bins to use
            method_name: Name of the method for the plot title
            ax: Matplotlib axes to plot on (if None, creates a new figure)
            show_ece: Whether to show ECE on the plot
            bar_alpha: Alpha value for the confidence bars
            
        Returns:
            Matplotlib figure
        """
        from utils.evaluation import CalibrationMetrics
        
        # Get bin data
        binning = CalibrationMetrics.bin_predictions(confidence_scores, correctness, num_bins)
        bins = binning["bins"]
        
        # Create or get axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        else:
            fig = ax.figure
        
        # Prepare data for plotting
        bin_midpoints = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for b in bins:
            # Calculate bin midpoint
            midpoint = (b["lower"] + b["upper"]) / 2
            bin_midpoints.append(midpoint)
            
            # Get bin statistics
            if b["count"] > 0:
                bin_accuracies.append(b["accuracy"])
                bin_confidences.append(b["avg_confidence"])
                bin_counts.append(b["count"])
            else:
                bin_accuracies.append(0)
                bin_confidences.append(midpoint)  # Use midpoint for empty bins
                bin_counts.append(0)
        
        # Plot gap between accuracy and confidence
        for i, (conf, acc) in enumerate(zip(bin_confidences, bin_accuracies)):
            if bin_counts[i] > 0:
                gap = conf - acc
                if gap > 0:
                    # Overconfident (red)
                    ax.plot([bin_midpoints[i], bin_midpoints[i]], [acc, conf], 'r-', alpha=0.5)
                else:
                    # Underconfident (blue)
                    ax.plot([bin_midpoints[i], bin_midpoints[i]], [acc, conf], 'b-', alpha=0.5)
        
        # Plot confidence bars
        ax.bar(
            bin_midpoints,
            bin_confidences,
            width=1.0/num_bins,
            alpha=bar_alpha,
            edgecolor='black',
            label='Confidence'
        )
        
        # Plot accuracy points
        ax.scatter(
            bin_midpoints,
            bin_accuracies,
            color='red',
            s=50,
            zorder=3,
            label='Accuracy'
        )
        
        # Plot diagonal (perfect calibration)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Calculate and show ECE if requested
        if show_ece:
            ece = CalibrationMetrics.expected_calibration_error(confidence_scores, correctness, num_bins)
            ax.text(
                0.05, 0.95,
                f'ECE: {ece:.4f}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        # Set labels and title
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Reliability Diagram - {method_name}')
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add legend
        ax.legend(loc='lower right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    @staticmethod
    def compare_reliability_diagrams(
        method_results: Dict[str, Tuple[List[float], List[bool]]],
        num_bins: int = 10,
        figsize: Tuple[int, int] = (12, 10),
        title: str = "Calibration Comparison",
    ) -> plt.Figure:
        """
        Compare reliability diagrams for multiple methods.
        
        Args:
            method_results: Dictionary mapping method names to tuples of (confidence_scores, correctness)
            num_bins: Number of bins to use
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Set up figure
        fig, axes = plt.subplots(
            nrows=(len(method_results) + 1) // 2,
            ncols=2,
            figsize=figsize
        )
        axes = axes.flatten()
        
        # Plot reliability diagram for each method
        for i, (method_name, (confidence_scores, correctness)) in enumerate(method_results.items()):
            if i < len(axes):
                CalibrationPlots.reliability_diagram(
                    confidence_scores,
                    correctness,
                    num_bins=num_bins,
                    method_name=method_name,
                    ax=axes[i]
                )
        
        # Hide any unused axes
        for i in range(len(method_results), len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        fig.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        return fig
    
    @staticmethod
    def ece_comparison_barplot(
        method_results: Dict[str, Dict[str, Dict[str, float]]],
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Expected Calibration Error (ECE) Comparison",
    ) -> plt.Figure:
        """
        Generate a bar plot comparing ECE across methods.
        
        Args:
            method_results: Dictionary of evaluation results for multiple methods
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Extract ECE values
        methods = []
        ece_values = []
        
        for method_name, method_result in method_results.items():
            methods.append(method_name)
            
            # Get ECE from calibration metrics
            if "calibration_metrics" in method_result and "ece" in method_result["calibration_metrics"]:
                ece_values.append(method_result["calibration_metrics"]["ece"])
            else:
                ece_values.append(0.0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bars
        bars = ax.bar(methods, ece_values, width=0.6)
        
        # Color bars based on value (lower is better)
        for i, bar in enumerate(bars):
            # Color gradient from green (low ECE) to red (high ECE)
            normalized_value = min(ece_values[i] / 0.5, 1.0)  # Normalize to [0, 1], treating ECE > 0.5 as 1.0
            color = plt.cm.RdYlGn_r(normalized_value)
            bar.set_color(color)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        # Set labels and title
        ax.set_xlabel('Method')
        ax.set_ylabel('Expected Calibration Error (ECE)')
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, linestyle='--', axis='y', alpha=0.7)
        
        # Add note that lower is better
        ax.text(
            0.02, 0.98,
            'Lower is better',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Rotate x-axis labels if there are many methods
        if len(methods) > 4:
            plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


class UncertaintyDistributionPlots:
    """Generate plots for uncertainty distributions."""
    
    @staticmethod
    def uncertainty_histogram(
        uncertainty_scores: List[float],
        true_hallucinations: Optional[List[bool]] = None,
        method_name: str = "Method",
        ax: Optional[plt.Axes] = None,
        bins: int = 20,
        density: bool = True,
    ) -> plt.Figure:
        """
        Generate a histogram of uncertainty scores.
        
        Args:
            uncertainty_scores: List of uncertainty scores
            true_hallucinations: Optional list of booleans indicating true hallucinations
            method_name: Name of the method for the plot title
            ax: Matplotlib axes to plot on (if None, creates a new figure)
            bins: Number of histogram bins
            density: Whether to normalize the histogram
            
        Returns:
            Matplotlib figure
        """
        # Create or get axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
        
        # If true hallucinations are provided, create separate histograms
        if true_hallucinations is not None:
            # Split uncertainty scores by hallucination status
            hallucination_scores = [score for score, is_halluc in zip(uncertainty_scores, true_hallucinations) if is_halluc]
            non_hallucination_scores = [score for score, is_halluc in zip(uncertainty_scores, true_hallucinations) if not is_halluc]
            
            # Plot histograms
            ax.hist(
                non_hallucination_scores,
                bins=bins,
                alpha=0.6,
                label='Non-hallucinations',
                density=density,
                color='green'
            )
            
            ax.hist(
                hallucination_scores,
                bins=bins,
                alpha=0.6,
                label='Hallucinations',
                density=density,
                color='red'
            )
            
            ax.legend()
        else:
            # Plot single histogram
            ax.hist(
                uncertainty_scores,
                bins=bins,
                alpha=0.8,
                density=density,
                color='blue'
            )
        
        # Set labels and title
        ax.set_xlabel('Uncertainty Score')
        ax.set_ylabel('Density' if density else 'Count')
        ax.set_title(f'Uncertainty Distribution - {method_name}')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    @staticmethod
    def uncertainty_boxplot(
        method_uncertainties: Dict[str, List[float]],
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Uncertainty Score Distribution by Method",
    ) -> plt.Figure:
        """
        Generate a box plot comparing uncertainty distributions across methods.
        
        Args:
            method_uncertainties: Dictionary mapping method names to lists of uncertainty scores
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for boxplot
        data = []
        labels = []
        
        for method_name, uncertainties in method_uncertainties.items():
            data.append(uncertainties)
            labels.append(method_name)
        
        # Create box plot
        box = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            notch=True,
            vert=True,
            whis=1.5,
            bootstrap=5000,
            showfliers=True,
            showmeans=True,
            meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black'}
        )
        
        # Color boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        # Set labels and title
        ax.set_xlabel('Method')
        ax.set_ylabel('Uncertainty Score')
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, linestyle='--', axis='y', alpha=0.7)
        
        # Rotate x-axis labels if there are many methods
        if len(labels) > 4:
            plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    @staticmethod
    def component_contribution_plot(
        method_results: Dict[str, Dict[str, float]],
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Contribution of Uncertainty Components",
    ) -> plt.Figure:
        """
        Generate a stacked bar plot showing contribution of uncertainty components.
        
        Args:
            method_results: Dictionary mapping method names to dictionaries with component values
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for stacked bar plot
        methods = []
        variance_components = []
        evidence_components = []
        
        for method_name, components in method_results.items():
            methods.append(method_name)
            variance_components.append(components.get("variance_component", 0.0))
            evidence_components.append(components.get("evidence_component", 0.0))
        
        # Create stacked bar plot
        bar_width = 0.6
        
        # Bottom bars (variance component)
        ax.bar(methods, variance_components, bar_width, label='Variance Component', color='#1f77b4')
        
        # Top bars (evidence component)
        ax.bar(methods, evidence_components, bar_width, bottom=variance_components, label='Evidence Component', color='#ff7f0e')
        
        # Add total value labels
        for i, method in enumerate(methods):
            total = variance_components[i] + evidence_components[i]
            ax.text(i, total + 0.02, f'{total:.2f}', ha='center')
        
        # Set labels and title
        ax.set_xlabel('Method')
        ax.set_ylabel('Uncertainty Score')
        ax.set_title(title)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', axis='y', alpha=0.7)
        
        # Rotate x-axis labels if there are many methods
        if len(methods) > 4:
            plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


class HallucinationDetectionPlots:
    """Generate plots for hallucination detection performance."""
    
    @staticmethod
    def roc_curve_plot(
        method_results: Dict[str, Tuple[List[float], List[bool]]],
        figsize: Tuple[int, int] = (8, 8),
        title: str = "ROC Curve for Hallucination Detection",
    ) -> plt.Figure:
        """
        Generate ROC curves for hallucination detection across methods.
        
        Args:
            method_results: Dictionary mapping method names to tuples of (uncertainty_scores, true_hallucinations)
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve for each method
        for method_name, (uncertainty_scores, true_hallucinations) in method_results.items():
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(true_hallucinations, uncertainty_scores)
            roc_auc = auc(fpr, tpr)
            
            # Plot curve
            ax.plot(
                fpr, tpr,
                lw=2, 
                label=f'{method_name} (AUC = {roc_auc:.3f})'
            )
        
        # Plot random baseline
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        
        # Set labels and title
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        
        # Add legend
        ax.legend(loc='lower right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    @staticmethod
    def precision_recall_curve_plot(
        method_results: Dict[str, Tuple[List[float], List[bool]]],
        figsize: Tuple[int, int] = (8, 8),
        title: str = "Precision-Recall Curve for Hallucination Detection",
    ) -> plt.Figure:
        """
        Generate precision-recall curves for hallucination detection across methods.
        
        Args:
            method_results: Dictionary mapping method names to tuples of (uncertainty_scores, true_hallucinations)
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot precision-recall curve for each method
        for method_name, (uncertainty_scores, true_hallucinations) in method_results.items():
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(true_hallucinations, uncertainty_scores)
            
            # Calculate average precision
            avg_precision = np.mean(precision)
            
            # Plot curve
            ax.plot(
                recall, precision,
                lw=2, 
                label=f'{method_name} (AP = {avg_precision:.3f})'
            )
        
        # Set labels and title
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        
        # Add legend
        ax.legend(loc='lower left')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    @staticmethod
    def f1_threshold_plot(
        method_results: Dict[str, Tuple[List[float], List[bool]]],
        num_thresholds: int = 100,
        figsize: Tuple[int, int] = (10, 6),
        title: str = "F1 Score vs. Uncertainty Threshold",
    ) -> plt.Figure:
        """
        Generate a plot of F1 scores across different uncertainty thresholds.
        
        Args:
            method_results: Dictionary mapping method names to tuples of (uncertainty_scores, true_hallucinations)
            num_thresholds: Number of threshold values to try
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        from utils.evaluation import HallucinationMetrics
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create threshold range
        thresholds = np.linspace(0.01, 0.99, num_thresholds)
        
        # Plot F1 score curve for each method
        for method_name, (uncertainty_scores, true_hallucinations) in method_results.items():
            # Calculate F1 scores for each threshold
            f1_scores = []
            for threshold in thresholds:
                metrics = HallucinationMetrics.hallucination_detection_at_threshold(
                    uncertainty_scores, true_hallucinations, threshold
                )
                f1_scores.append(metrics["f1"])
            
            # Find best threshold and F1 score
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
            
            # Plot curve
            ax.plot(
                thresholds, f1_scores,
                lw=2, 
                label=f'{method_name} (Best F1 = {best_f1:.3f} at {best_threshold:.2f})'
            )
            
            # Mark best threshold
            ax.plot(
                best_threshold, best_f1,
                'o', 
                markersize=6, 
                markerfacecolor='white', 
                markeredgecolor='black'
            )
        
        # Set labels and title
        ax.set_xlabel('Uncertainty Threshold')
        ax.set_ylabel('F1 Score')
        ax.set_title(title)
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        
        # Add legend
        ax.legend(loc='lower center')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


class TaskPerformancePlots:
    """Generate plots for task performance comparisons."""
    
    @staticmethod
    def qa_performance_barplot(
        method_results: Dict[str, Dict[str, Dict[str, float]]],
        figsize: Tuple[int, int] = (10, 6),
        title: str = "QA Performance Comparison",
    ) -> plt.Figure:
        """
        Generate a bar plot comparing QA performance metrics across methods.
        
        Args:
            method_results: Dictionary of evaluation results for multiple methods
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Extract metrics
        methods = []
        em_scores = []
        f1_scores = []
        
        for method_name, method_result in method_results.items():
            methods.append(method_name)
            
            # Get metrics from QA metrics
            if "qa_metrics" in method_result:
                qa_metrics = method_result["qa_metrics"]
                em_scores.append(qa_metrics.get("exact_match", 0.0))
                f1_scores.append(qa_metrics.get("f1", 0.0))
            else:
                em_scores.append(0.0)
                f1_scores.append(0.0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set bar width and positions
        bar_width = 0.35
        x = np.arange(len(methods))
        
        # Plot bars
        ax.bar(x - bar_width/2, em_scores, bar_width, label='Exact Match', color='#1f77b4')
        ax.bar(x + bar_width/2, f1_scores, bar_width, label='F1 Score', color='#ff7f0e')
        
        # Add value labels on bars
        for i, (em, f1) in enumerate(zip(em_scores, f1_scores)):
            ax.text(x[i] - bar_width/2, em + 0.02, f'{em:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(x[i] + bar_width/2, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Set labels and title
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title(title)
        
        # Set x-axis ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.1)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', axis='y', alpha=0.7)
        
        # Rotate x-axis labels if there are many methods
        if len(methods) > 4:
            plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    @staticmethod
    def summarization_performance_barplot(
        method_results: Dict[str, Dict[str, Dict[str, float]]],
        figsize: Tuple[int, int] = (12, 6),
        title: str = "Summarization Performance Comparison",
    ) -> plt.Figure:
        """
        Generate a bar plot comparing summarization performance metrics across methods.
        
        Args:
            method_results: Dictionary of evaluation results for multiple methods
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Extract metrics
        methods = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bertscore_f1_scores = []
        
        for method_name, method_result in method_results.items():
            methods.append(method_name)
            
            # Get metrics from summarization metrics
            if "summarization_metrics" in method_result:
                metrics = method_result["summarization_metrics"]
                rouge1_scores.append(metrics.get("rouge1_fmeasure", 0.0))
                rouge2_scores.append(metrics.get("rouge2_fmeasure", 0.0))
                rougeL_scores.append(metrics.get("rougeL_fmeasure", 0.0))
                bertscore_f1_scores.append(metrics.get("bertscore_f1", 0.0))
            else:
                rouge1_scores.append(0.0)
                rouge2_scores.append(0.0)
                rougeL_scores.append(0.0)
                bertscore_f1_scores.append(0.0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set bar width and positions
        bar_width = 0.2
        x = np.arange(len(methods))
        
        # Plot bars
        ax.bar(x - 1.5*bar_width, rouge1_scores, bar_width, label='ROUGE-1', color='#1f77b4')
        ax.bar(x - 0.5*bar_width, rouge2_scores, bar_width, label='ROUGE-2', color='#ff7f0e')
        ax.bar(x + 0.5*bar_width, rougeL_scores, bar_width, label='ROUGE-L', color='#2ca02c')
        ax.bar(x + 1.5*bar_width, bertscore_f1_scores, bar_width, label='BERTScore F1', color='#d62728')
        
        # Set labels and title
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title(title)
        
        # Set x-axis ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.1)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', axis='y', alpha=0.7)
        
        # Rotate x-axis labels if there are many methods
        if len(methods) > 4:
            plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    @staticmethod
    def performance_vs_uncertainty_scatter(
        method_results: Dict[str, Tuple[List[float], List[float]]],
        figsize: Tuple[int, int] = (8, 8),
        title: str = "Performance vs. Uncertainty",
        xlabel: str = "Uncertainty Score",
        ylabel: str = "Performance Score",
    ) -> plt.Figure:
        """
        Generate a scatter plot of performance vs. uncertainty.
        
        Args:
            method_results: Dictionary mapping method names to tuples of (uncertainty_scores, performance_scores)
            figsize: Figure size
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate colors for methods
        colors = plt.cm.tab10(np.linspace(0, 1, len(method_results)))
        
        # Plot scatter plot for each method
        for (method_name, (uncertainty_scores, performance_scores)), color in zip(method_results.items(), colors):
            # Plot points
            ax.scatter(
                uncertainty_scores,
                performance_scores,
                alpha=0.7,
                label=method_name,
                color=color,
                edgecolors='black',
                s=50
            )
            
            # Calculate and plot trend line
            if len(uncertainty_scores) > 1:
                z = np.polyfit(uncertainty_scores, performance_scores, 1)
                p = np.poly1d(z)
                ax.plot(
                    sorted(uncertainty_scores),
                    p(sorted(uncertainty_scores)),
                    '--',
                    color=color,
                    alpha=0.8
                )
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


class DiversityMetricsPlots:
    """Generate plots for diversity metrics."""
    
    @staticmethod
    def diversity_barplot(
        method_results: Dict[str, Dict[str, Dict[str, float]]],
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Diversity Metrics Comparison",
    ) -> plt.Figure:
        """
        Generate a bar plot comparing diversity metrics across methods.
        
        Args:
            method_results: Dictionary of evaluation results for multiple methods
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Extract metrics
        methods = []
        distinct1_scores = []
        distinct2_scores = []
        distinct3_scores = []
        self_bleu_scores = []
        
        for method_name, method_result in method_results.items():
            methods.append(method_name)
            
            # Get metrics from diversity metrics
            if "diversity_metrics" in method_result:
                metrics = method_result["diversity_metrics"]
                distinct1_scores.append(metrics.get("distinct_1", 0.0))
                distinct2_scores.append(metrics.get("distinct_2", 0.0))
                distinct3_scores.append(metrics.get("distinct_3", 0.0))
                self_bleu_scores.append(metrics.get("self_bleu", 0.0))
            else:
                distinct1_scores.append(0.0)
                distinct2_scores.append(0.0)
                distinct3_scores.append(0.0)
                self_bleu_scores.append(0.0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set bar width and positions
        bar_width = 0.2
        x = np.arange(len(methods))
        
        # Plot bars
        ax.bar(x - 1.5*bar_width, distinct1_scores, bar_width, label='Distinct-1', color='#1f77b4')
        ax.bar(x - 0.5*bar_width, distinct2_scores, bar_width, label='Distinct-2', color='#ff7f0e')
        ax.bar(x + 0.5*bar_width, distinct3_scores, bar_width, label='Distinct-3', color='#2ca02c')
        ax.bar(x + 1.5*bar_width, self_bleu_scores, bar_width, label='Self-BLEU', color='#d62728')
        
        # Set labels and title
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title(title)
        
        # Set x-axis ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.1)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', axis='y', alpha=0.7)
        
        # Rotate x-axis labels if there are many methods
        if len(methods) > 4:
            plt.xticks(rotation=45, ha='right')
        
        # Add note about metrics
        note = (
            "Distinct-n: Higher is more diverse\n"
            "Self-BLEU: Lower is more diverse"
        )
        ax.text(
            0.02, 0.02,
            note,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


class AblationStudyPlots:
    """Generate plots for ablation studies."""
    
    @staticmethod
    def alpha_ablation_plot(
        alpha_results: Dict[float, Dict[str, float]],
        metrics: List[str],
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Performance Across Different Alpha Values",
    ) -> plt.Figure:
        """
        Generate a line plot showing performance across different alpha values.
        
        Args:
            alpha_results: Dictionary mapping alpha values to dictionaries of metrics
            metrics: List of metrics to plot
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract alpha values and sort them
        alphas = sorted(alpha_results.keys())
        
        # Plot line for each metric
        for i, metric in enumerate(metrics):
            # Extract metric values for each alpha
            metric_values = [alpha_results[alpha].get(metric, 0.0) for alpha in alphas]
            
            # Plot line
            ax.plot(
                alphas,
                metric_values,
                'o-',
                lw=2,
                label=metric,
                color=plt.cm.tab10(i),
                markersize=6
            )
        
        # Set labels and title
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        
        # Set x-axis limits and ticks
        ax.set_xlim(0, 1)
        ax.set_xticks(alphas)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add alpha explanation
        explanation = "Alpha balances variance (α) and evidence alignment (1-α)"
        ax.text(
            0.5, 0.02,
            explanation,
            transform=ax.transAxes,
            fontsize=10,
            ha='center',
            va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    @staticmethod
    def beta_ablation_plot(
        beta_results: Dict[float, Dict[str, float]],
        metrics: List[str],
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Performance Across Different Beta Values",
    ) -> plt.Figure:
        """
        Generate a line plot showing performance across different beta values.
        
        Args:
            beta_results: Dictionary mapping beta values to dictionaries of metrics
            metrics: List of metrics to plot
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract beta values and sort them
        betas = sorted(beta_results.keys())
        
        # Plot line for each metric
        for i, metric in enumerate(metrics):
            # Extract metric values for each beta
            metric_values = [beta_results[beta].get(metric, 0.0) for beta in betas]
            
            # Plot line
            ax.plot(
                betas,
                metric_values,
                'o-',
                lw=2,
                label=metric,
                color=plt.cm.tab10(i),
                markersize=6
            )
        
        # Set labels and title
        ax.set_xlabel('Beta')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add beta explanation
        explanation = "Beta controls the strength of the hallucination penalty"
        ax.text(
            0.5, 0.02,
            explanation,
            transform=ax.transAxes,
            fontsize=10,
            ha='center',
            va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    @staticmethod
    def k_samples_ablation_plot(
        k_results: Dict[int, Dict[str, float]],
        metrics: List[str],
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Performance Across Different Sample Counts (k)",
    ) -> plt.Figure:
        """
        Generate a line plot showing performance across different sample counts.
        
        Args:
            k_results: Dictionary mapping k values to dictionaries of metrics
            metrics: List of metrics to plot
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract k values and sort them
        k_values = sorted(k_results.keys())
        
        # Plot line for each metric
        for i, metric in enumerate(metrics):
            # Extract metric values for each k
            metric_values = [k_results[k].get(metric, 0.0) for k in k_values]
            
            # Plot line
            ax.plot(
                k_values,
                metric_values,
                'o-',
                lw=2,
                label=metric,
                color=plt.cm.tab10(i),
                markersize=6
            )
        
        # Set labels and title
        ax.set_xlabel('Number of Samples (k)')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        
        # Set x-axis ticks to k values
        ax.set_xticks(k_values)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add runtime information (if available)
        if "runtime" in metrics:
            ax_right = ax.twinx()
            runtime_values = [k_results[k].get("runtime", 0.0) for k in k_values]
            ax_right.plot(
                k_values,
                runtime_values,
                's--',
                color='red',
                label='Runtime (s)',
                alpha=0.7
            )
            ax_right.set_ylabel('Runtime (seconds)')
            
            # Combine legends
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_right.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig


class VisualizationManager:
    """Manager class for generating and saving visualizations."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the visualization manager.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_figure(self, fig: plt.Figure, filename: str, **kwargs) -> str:
        """
        Save a figure to file.
        
        Args:
            fig: Figure to save
            filename: Filename (without directory)
            **kwargs: Additional arguments to pass to savefig
            
        Returns:
            Path to saved figure
        """
        # Ensure filename has extension
        if not filename.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
            filename += '.png'
        
        # Create full path
        filepath = os.path.join(self.output_dir, filename)
        
        # Save figure
        fig.savefig(filepath, **kwargs)
        
        logger.info(f"Saved figure to {filepath}")
        
        return filepath
    
    def create_all_visualizations(
        self,
        eval_results: Dict[str, Dict[str, Dict[str, Any]]],
        task_type: str,
        uncertainty_scores: Optional[Dict[str, List[float]]] = None,
        true_hallucinations: Optional[List[bool]] = None,
        performance_scores: Optional[Dict[str, List[float]]] = None,
        ablation_results: Optional[Dict[str, Dict[Any, Dict[str, float]]]] = None,
    ) -> Dict[str, str]:
        """
        Create all visualizations for experiment results.
        
        Args:
            eval_results: Evaluation results dictionary
            task_type: Type of task ('qa' or 'summarization')
            uncertainty_scores: Dictionary mapping method names to lists of uncertainty scores
            true_hallucinations: Optional list of booleans indicating true hallucinations
            performance_scores: Dictionary mapping method names to lists of performance scores
            ablation_results: Dictionary of ablation study results
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        # Dictionary to store paths to saved figures
        figure_paths = {}
        
        # Create calibration plots if available
        if any("calibration_metrics" in method_result for method_result in eval_results.values()):
            # Extract calibration data
            method_calibration = {}
            for method_name, method_result in eval_results.items():
                if "calibration_metrics" in method_result:
                    # We need confidence scores and correctness, which aren't in the eval results
                    # In a real implementation, these would be provided or extracted elsewhere
                    pass
            
            # Skip for now since we don't have the raw calibration data
            logger.warning("Skipping calibration plots due to missing raw data")
        
        # Create task performance plots
        if task_type == "qa":
            # QA performance barplot
            qa_fig = TaskPerformancePlots.qa_performance_barplot(
                eval_results,
                title="QA Performance Comparison"
            )
            qa_path = self.save_figure(qa_fig, "qa_performance_comparison.png")
            figure_paths["qa_performance"] = qa_path
            plt.close(qa_fig)
        
        elif task_type == "summarization":
            # Summarization performance barplot
            summary_fig = TaskPerformancePlots.summarization_performance_barplot(
                eval_results,
                title="Summarization Performance Comparison"
            )
            summary_path = self.save_figure(summary_fig, "summarization_performance_comparison.png")
            figure_paths["summarization_performance"] = summary_path
            plt.close(summary_fig)
        
        # Create diversity metrics plots if available
        if any("diversity_metrics" in method_result for method_result in eval_results.values()):
            diversity_fig = DiversityMetricsPlots.diversity_barplot(
                eval_results,
                title="Diversity Metrics Comparison"
            )
            diversity_path = self.save_figure(diversity_fig, "diversity_metrics_comparison.png")
            figure_paths["diversity_metrics"] = diversity_path
            plt.close(diversity_fig)
        
        # Create uncertainty distribution plots if uncertainty scores are provided
        if uncertainty_scores:
            for method_name, scores in uncertainty_scores.items():
                hist_fig = UncertaintyDistributionPlots.uncertainty_histogram(
                    scores,
                    true_hallucinations=true_hallucinations,
                    method_name=method_name
                )
                hist_path = self.save_figure(hist_fig, f"uncertainty_histogram_{method_name}.png")
                figure_paths[f"uncertainty_histogram_{method_name}"] = hist_path
                plt.close(hist_fig)
            
            # Create boxplot for all methods
            box_fig = UncertaintyDistributionPlots.uncertainty_boxplot(
                uncertainty_scores,
                title="Uncertainty Score Distribution by Method"
            )
            box_path = self.save_figure(box_fig, "uncertainty_boxplot.png")
            figure_paths["uncertainty_boxplot"] = box_path
            plt.close(box_fig)
        
        # Create hallucination detection plots if true hallucinations are provided
        if uncertainty_scores and true_hallucinations:
            # Prepare data for ROC and PR curves
            method_results_for_curves = {
                method_name: (scores, true_hallucinations)
                for method_name, scores in uncertainty_scores.items()
            }
            
            # ROC curve
            roc_fig = HallucinationDetectionPlots.roc_curve_plot(
                method_results_for_curves,
                title="ROC Curve for Hallucination Detection"
            )
            roc_path = self.save_figure(roc_fig, "hallucination_roc_curve.png")
            figure_paths["hallucination_roc_curve"] = roc_path
            plt.close(roc_fig)
            
            # Precision-Recall curve
            pr_fig = HallucinationDetectionPlots.precision_recall_curve_plot(
                method_results_for_curves,
                title="Precision-Recall Curve for Hallucination Detection"
            )
            pr_path = self.save_figure(pr_fig, "hallucination_pr_curve.png")
            figure_paths["hallucination_pr_curve"] = pr_path
            plt.close(pr_fig)
            
            # F1 vs threshold plot
            f1_fig = HallucinationDetectionPlots.f1_threshold_plot(
                method_results_for_curves,
                title="F1 Score vs. Uncertainty Threshold"
            )
            f1_path = self.save_figure(f1_fig, "hallucination_f1_threshold.png")
            figure_paths["hallucination_f1_threshold"] = f1_path
            plt.close(f1_fig)
        
        # Create performance vs uncertainty scatter if both are provided
        if uncertainty_scores and performance_scores:
            # Ensure methods match
            common_methods = set(uncertainty_scores.keys()).intersection(performance_scores.keys())
            
            if common_methods:
                # Prepare data for scatter plot
                scatter_data = {
                    method_name: (uncertainty_scores[method_name], performance_scores[method_name])
                    for method_name in common_methods
                }
                
                # Create scatter plot
                scatter_fig = TaskPerformancePlots.performance_vs_uncertainty_scatter(
                    scatter_data,
                    title="Performance vs. Uncertainty",
                    xlabel="Uncertainty Score",
                    ylabel=f"{'F1 Score' if task_type == 'qa' else 'ROUGE-L F1'}"
                )
                scatter_path = self.save_figure(scatter_fig, "performance_vs_uncertainty.png")
                figure_paths["performance_vs_uncertainty"] = scatter_path
                plt.close(scatter_fig)
        
        # Create ablation study plots if results are provided
        if ablation_results:
            # Alpha ablation
            if "alpha" in ablation_results:
                metrics_to_plot = ["f1", "precision", "recall"] if task_type == "qa" else ["rougeL_fmeasure", "rouge1_fmeasure"]
                
                alpha_fig = AblationStudyPlots.alpha_ablation_plot(
                    ablation_results["alpha"],
                    metrics_to_plot,
                    title="Performance Across Different Alpha Values"
                )
                alpha_path = self.save_figure(alpha_fig, "alpha_ablation.png")
                figure_paths["alpha_ablation"] = alpha_path
                plt.close(alpha_fig)
            
            # Beta ablation
            if "beta" in ablation_results:
                metrics_to_plot = ["f1", "precision", "recall"] if task_type == "qa" else ["rougeL_fmeasure", "rouge1_fmeasure"]
                
                beta_fig = AblationStudyPlots.beta_ablation_plot(
                    ablation_results["beta"],
                    metrics_to_plot,
                    title="Performance Across Different Beta Values"
                )
                beta_path = self.save_figure(beta_fig, "beta_ablation.png")
                figure_paths["beta_ablation"] = beta_path
                plt.close(beta_fig)
            
            # K samples ablation
            if "k" in ablation_results:
                metrics_to_plot = ["f1", "ece", "runtime"] if task_type == "qa" else ["rougeL_fmeasure", "ece", "runtime"]
                
                k_fig = AblationStudyPlots.k_samples_ablation_plot(
                    ablation_results["k"],
                    metrics_to_plot,
                    title="Performance Across Different Sample Counts (k)"
                )
                k_path = self.save_figure(k_fig, "k_samples_ablation.png")
                figure_paths["k_samples_ablation"] = k_path
                plt.close(k_fig)
        
        return figure_paths


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage (commented out since it requires actual results)
    """
    # Create visualization manager
    viz_manager = VisualizationManager("results/figures")
    
    # Create example data
    eval_results = {
        "vanilla": {
            "qa_metrics": {"exact_match": 0.75, "f1": 0.82},
            "calibration_metrics": {"ece": 0.15, "brier_score": 0.12},
            "diversity_metrics": {"distinct_1": 0.4, "distinct_2": 0.6, "distinct_3": 0.7, "self_bleu": 0.3},
        },
        "scec": {
            "qa_metrics": {"exact_match": 0.78, "f1": 0.85},
            "calibration_metrics": {"ece": 0.08, "brier_score": 0.09},
            "diversity_metrics": {"distinct_1": 0.5, "distinct_2": 0.7, "distinct_3": 0.8, "self_bleu": 0.25},
        },
    }
    
    # Generate visualizations
    figure_paths = viz_manager.create_all_visualizations(
        eval_results,
        task_type="qa"
    )
    
    print("Generated figures at:")
    for name, path in figure_paths.items():
        print(f"- {name}: {path}")
    """
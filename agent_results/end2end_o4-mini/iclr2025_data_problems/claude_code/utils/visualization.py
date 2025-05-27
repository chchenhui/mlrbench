"""
Visualization module for the Gradient-Informed Fingerprinting (GIF) method.

This module provides functions for visualizing experimental results,
including performance metrics, latency comparisons, and ablation studies.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("tab10")


class ExperimentVisualizer:
    """Visualize experimental results for attribution methods."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def _load_data(self, data_path: str) -> Dict[str, Any]:
        """Load data from a JSON file."""
        with open(data_path, "r") as f:
            data = json.load(f)
        return data
    
    def create_performance_comparison(
        self,
        metrics_data: Dict[str, Dict[str, Any]],
        metrics_to_plot: List[str] = ["precision@1", "precision@5", "recall@10", "mrr"],
        output_filename: str = "performance_comparison.png",
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Create a comprehensive performance comparison plot.
        
        Args:
            metrics_data: Dictionary of method names to metric dictionaries
            metrics_to_plot: List of metrics to include in the plot
            output_filename: Output filename
            figsize: Figure size
        """
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Get method names
        methods = list(metrics_data.keys())
        colors = sns.color_palette("tab10", len(methods))
        
        # Plot each metric
        for i, metric in enumerate(metrics_to_plot):
            row, col = i // 2, i % 2
            ax = fig.add_subplot(gs[row, col])
            
            # Extract values for this metric
            values = []
            for method in methods:
                if metric in metrics_data[method]:
                    values.append(metrics_data[method][metric])
                else:
                    values.append(0)  # Default if metric not available
            
            # Create bar plot
            bars = ax.bar(methods, values, color=colors)
            
            # Add value labels on top of bars
            for bar, value in zip(bars, values):
                formatted_value = f"{value:.3f}" if isinstance(value, float) else str(value)
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       formatted_value, ha='center', va='bottom', fontsize=9)
            
            # Set title and labels
            ax.set_title(metric.capitalize())
            ax.set_ylim(0, max(values) * 1.15)  # Add some space above bars
            
            # Rotate x-tick labels for better readability
            ax.set_xticklabels(methods, rotation=45, ha='right')
            
            # Add grid for easier comparison
            ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add overall title
        fig.suptitle("Performance Comparison of Attribution Methods", fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
        
        # Save figure
        output_path = os.path.join(self.results_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance comparison plot to {output_path}")
        
        # Close figure
        plt.close(fig)
    
    def create_latency_analysis(
        self,
        latency_data: Dict[str, List[float]],
        output_filename: str = "latency_analysis.png",
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """
        Create a comprehensive latency analysis plot.
        
        Args:
            latency_data: Dictionary of method names to latency measurements lists
            output_filename: Output filename
            figsize: Figure size
        """
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Get method names
        methods = list(latency_data.keys())
        colors = sns.color_palette("tab10", len(methods))
        
        # Calculate statistics
        stats = {}
        for method, latencies in latency_data.items():
            stats[method] = {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "p25": np.percentile(latencies, 25),
                "p75": np.percentile(latencies, 75),
                "p95": np.percentile(latencies, 95),
                "std": np.std(latencies)
            }
        
        # Sort methods by median latency
        methods = sorted(methods, key=lambda m: stats[m]["median"])
        
        # 1. Box plot of latencies
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Prepare data for box plot
        box_data = [latency_data[method] for method in methods]
        ax1.boxplot(box_data, labels=methods, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="lightblue"))
        
        # Set labels and title
        ax1.set_title("Latency Distribution")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_yscale('log')  # Log scale for better visibility
        
        # Rotate x-tick labels for better readability
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        
        # 2. Bar plot of median latencies
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Extract median latencies
        medians = [stats[method]["median"] for method in methods]
        
        # Create bar plot
        bars = ax2.bar(methods, medians, color=colors)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, medians):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f"{value:.1f}", ha='center', va='bottom', fontsize=9)
        
        # Set title and labels
        ax2.set_title("Median Latency Comparison")
        ax2.set_ylabel("Latency (ms)")
        
        # Rotate x-tick labels for better readability
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        
        # Add grid for easier comparison
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 3. Latency breakdown for GIF (if available)
        ax3 = fig.add_subplot(gs[1, 0])
        
        if "GIF" in latency_data and "GIF_components" in latency_data:
            # Extract component latencies
            components = list(latency_data["GIF_components"].keys())
            comp_medians = [np.median(latency_data["GIF_components"][comp]) for comp in components]
            
            # Sort components by median latency
            sorted_indices = np.argsort(comp_medians)
            components = [components[i] for i in sorted_indices]
            comp_medians = [comp_medians[i] for i in sorted_indices]
            
            # Create bar plot
            bars = ax3.barh(components, comp_medians, color=sns.color_palette("viridis", len(components)))
            
            # Add value labels
            for bar, value in zip(bars, comp_medians):
                ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                       f"{value:.1f} ms", va='center', fontsize=9)
            
            # Set title and labels
            ax3.set_title("GIF Latency Breakdown")
            ax3.set_xlabel("Latency (ms)")
            
            # Add grid for easier comparison
            ax3.grid(axis='x', linestyle='--', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "GIF component data not available", 
                    ha='center', va='center', fontsize=12)
            ax3.axis('off')
        
        # 4. Scatter plot of latency vs. performance (e.g., MRR)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Check if we have performance data
        if all("mrr" in latency_data.get(method, {}) for method in methods):
            # Extract MRR values
            mrr_values = [latency_data[method].get("mrr", 0) for method in methods]
            
            # Create scatter plot
            ax4.scatter(medians, mrr_values, s=100, c=colors, alpha=0.8)
            
            # Add method labels
            for method, x, y in zip(methods, medians, mrr_values):
                ax4.annotate(method, (x, y), xytext=(5, 5), textcoords='offset points')
            
            # Set title and labels
            ax4.set_title("Latency vs. MRR")
            ax4.set_xlabel("Median Latency (ms)")
            ax4.set_ylabel("Mean Reciprocal Rank (MRR)")
            
            # Use log scale for latency
            ax4.set_xscale('log')
            
            # Add grid
            ax4.grid(True, linestyle='--', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "Performance data not available", 
                    ha='center', va='center', fontsize=12)
            ax4.axis('off')
        
        # Add overall title
        fig.suptitle("Latency Analysis of Attribution Methods", fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
        
        # Save figure
        output_path = os.path.join(self.results_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved latency analysis plot to {output_path}")
        
        # Close figure
        plt.close(fig)
    
    def create_ablation_plot(
        self,
        ablation_data: Dict[str, Dict[str, Any]],
        metrics_to_plot: List[str] = ["precision@1", "mrr"],
        output_filename: str = "ablation_study.png",
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Create a plot for ablation study results.
        
        Args:
            ablation_data: Dictionary of variant names to metric dictionaries
            metrics_to_plot: List of metrics to include in the plot
            output_filename: Output filename
            figsize: Figure size
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        # Get variant names
        variants = list(ablation_data.keys())
        colors = sns.color_palette("Set2", len(variants))
        
        # Plot each metric
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Extract values for this metric
            values = []
            for variant in variants:
                if metric in ablation_data[variant]:
                    values.append(ablation_data[variant][metric])
                else:
                    values.append(0)  # Default if metric not available
            
            # Create bar plot
            bars = ax.bar(variants, values, color=colors)
            
            # Add value labels on top of bars
            for bar, value in zip(bars, values):
                formatted_value = f"{value:.3f}" if isinstance(value, float) else str(value)
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       formatted_value, ha='center', va='bottom', fontsize=9)
            
            # Set title and labels
            ax.set_title(f"{metric.capitalize()} Comparison")
            ax.set_ylim(0, max(values) * 1.15)  # Add some space above bars
            
            # Rotate x-tick labels for better readability
            ax.set_xticklabels(variants, rotation=45, ha='right')
            
            # Add grid for easier comparison
            ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Hide unused subplots
        for j in range(len(metrics_to_plot), len(axes)):
            axes[j].axis('off')
        
        # Add overall title
        fig.suptitle("Ablation Study Results", fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
        
        # Save figure
        output_path = os.path.join(self.results_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ablation study plot to {output_path}")
        
        # Close figure
        plt.close(fig)
    
    def create_training_curves(
        self,
        training_data: Dict[str, List[float]],
        validation_data: Optional[Dict[str, List[float]]] = None,
        x_values: Optional[List[int]] = None,
        metrics: List[str] = ["loss", "accuracy"],
        output_filename: str = "training_curves.png",
        figsize: Tuple[int, int] = (12, 5)
    ) -> None:
        """
        Create plots of training curves.
        
        Args:
            training_data: Dictionary of metric names to lists of training values
            validation_data: Optional dictionary of metric names to lists of validation values
            x_values: Optional list of x-axis values (e.g., epochs)
            metrics: List of metrics to plot
            output_filename: Output filename
            figsize: Figure size
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        # Create x-axis values if not provided
        if x_values is None:
            # Use the length of the first metric's data
            first_metric = list(training_data.keys())[0]
            x_values = list(range(1, len(training_data[first_metric]) + 1))
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Check if this metric is available
            if metric in training_data:
                # Plot training data
                ax.plot(x_values, training_data[metric], 'b-', label='Training')
                
                # Plot validation data if available
                if validation_data is not None and metric in validation_data:
                    ax.plot(x_values, validation_data[metric], 'r-', label='Validation')
                
                # Set title and labels
                ax.set_title(f"{metric.capitalize()}")
                ax.set_xlabel("Epochs")
                ax.set_ylabel(metric.capitalize())
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.3)
                
                # Add legend
                ax.legend()
                
                # Use integer x-axis ticks
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            else:
                ax.text(0.5, 0.5, f"{metric.capitalize()} data not available", 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
        
        # Hide unused subplots
        for j in range(len(metrics), len(axes)):
            axes[j].axis('off')
        
        # Add overall title
        fig.suptitle("Training and Validation Curves", fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
        
        # Save figure
        output_path = os.path.join(self.results_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves plot to {output_path}")
        
        # Close figure
        plt.close(fig)
    
    def create_projection_dimension_plot(
        self,
        dimension_data: Dict[int, Dict[str, float]],
        metrics_to_plot: List[str] = ["precision@1", "mrr", "latency_ms"],
        output_filename: str = "projection_dimension.png",
        figsize: Tuple[int, int] = (15, 5)
    ) -> None:
        """
        Create plots showing the effect of projection dimension on performance.
        
        Args:
            dimension_data: Dictionary of projection dimensions to metric dictionaries
            metrics_to_plot: List of metrics to plot
            output_filename: Output filename
            figsize: Figure size
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        # Get dimensions and sort them
        dimensions = sorted(dimension_data.keys())
        
        # Plot each metric
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Extract values for this metric
            values = []
            for dim in dimensions:
                if metric in dimension_data[dim]:
                    values.append(dimension_data[dim][metric])
                else:
                    values.append(None)  # Skip if not available
            
            # Skip metrics with no data
            if all(v is None for v in values):
                ax.text(0.5, 0.5, f"{metric.capitalize()} data not available", 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue
            
            # Plot line
            ax.plot(dimensions, values, 'o-', markersize=8)
            
            # Add data labels
            for dim, val in zip(dimensions, values):
                if val is not None:
                    formatted_value = f"{val:.3f}" if isinstance(val, float) else str(val)
                    ax.text(dim, val, formatted_value, ha='center', va='bottom', fontsize=9)
            
            # Set title and labels
            ax.set_title(f"Effect of Projection Dimension on {metric.capitalize()}")
            ax.set_xlabel("Projection Dimension")
            ax.set_ylabel(metric.capitalize())
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Set x-axis to log scale for better visibility
            ax.set_xscale('log', base=2)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        
        # Hide unused subplots
        for j in range(len(metrics_to_plot), len(axes)):
            axes[j].axis('off')
        
        # Add overall title
        fig.suptitle("Effect of Projection Dimension on Performance", fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
        
        # Save figure
        output_path = os.path.join(self.results_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved projection dimension plot to {output_path}")
        
        # Close figure
        plt.close(fig)
    
    def create_dataset_size_plot(
        self,
        size_data: Dict[int, Dict[str, float]],
        metrics_to_plot: List[str] = ["precision@1", "mrr", "latency_ms"],
        output_filename: str = "dataset_size.png",
        figsize: Tuple[int, int] = (15, 5)
    ) -> None:
        """
        Create plots showing the effect of dataset size on performance.
        
        Args:
            size_data: Dictionary of dataset sizes to metric dictionaries
            metrics_to_plot: List of metrics to plot
            output_filename: Output filename
            figsize: Figure size
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        # Get sizes and sort them
        sizes = sorted(size_data.keys())
        
        # Plot each metric
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Extract values for this metric
            values = []
            for size in sizes:
                if metric in size_data[size]:
                    values.append(size_data[size][metric])
                else:
                    values.append(None)  # Skip if not available
            
            # Skip metrics with no data
            if all(v is None for v in values):
                ax.text(0.5, 0.5, f"{metric.capitalize()} data not available", 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue
            
            # Plot line
            ax.plot(sizes, values, 'o-', markersize=8)
            
            # Add data labels
            for size, val in zip(sizes, values):
                if val is not None:
                    formatted_value = f"{val:.3f}" if isinstance(val, float) else str(val)
                    ax.text(size, val, formatted_value, ha='center', va='bottom', fontsize=9)
            
            # Set title and labels
            ax.set_title(f"Effect of Dataset Size on {metric.capitalize()}")
            ax.set_xlabel("Dataset Size")
            ax.set_ylabel(metric.capitalize())
            
            # Format x-axis labels with K/M suffixes
            def format_size(size, pos):
                if size >= 1_000_000:
                    return f"{size / 1_000_000:.0f}M"
                elif size >= 1_000:
                    return f"{size / 1_000:.0f}K"
                else:
                    return str(size)
            
            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_size))
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Set x-axis to log scale for better visibility
            ax.set_xscale('log', base=10)
        
        # Hide unused subplots
        for j in range(len(metrics_to_plot), len(axes)):
            axes[j].axis('off')
        
        # Add overall title
        fig.suptitle("Effect of Dataset Size on Performance", fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
        
        # Save figure
        output_path = os.path.join(self.results_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved dataset size plot to {output_path}")
        
        # Close figure
        plt.close(fig)
    
    def create_fingerprint_visualization(
        self,
        fingerprints: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = "tsne",
        output_filename: str = "fingerprint_viz.png",
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Create visualization of fingerprint vectors.
        
        Args:
            fingerprints: Matrix of fingerprint vectors
            labels: Optional array of labels for coloring points
            method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
            output_filename: Output filename
            figsize: Figure size
        """
        # Create figure
        plt.figure(figsize=figsize)
        
        # Apply dimensionality reduction
        if method.lower() == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            embedding = reducer.fit_transform(fingerprints)
            title = "t-SNE Visualization of Fingerprints"
        elif method.lower() == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            embedding = reducer.fit_transform(fingerprints)
            title = "PCA Visualization of Fingerprints"
        elif method.lower() == "umap":
            try:
                import umap
                reducer = umap.UMAP(random_state=42)
                embedding = reducer.fit_transform(fingerprints)
                title = "UMAP Visualization of Fingerprints"
            except ImportError:
                logger.warning("UMAP not installed. Falling back to PCA.")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
                embedding = reducer.fit_transform(fingerprints)
                title = "PCA Visualization of Fingerprints"
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        # Create scatter plot
        if labels is not None:
            # Get unique labels
            unique_labels = np.unique(labels)
            
            # Create colormap
            cmap = plt.cm.get_cmap("tab10", len(unique_labels))
            
            # Plot each class separately for legend
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(embedding[mask, 0], embedding[mask, 1], c=[cmap(i)], label=f"Class {label}",
                            alpha=0.7, edgecolors='w', linewidth=0.5)
            
            # Add legend
            plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Plot all points with the same color
            plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, edgecolors='w', linewidth=0.5)
        
        # Add title and labels
        plt.title(title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.results_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved fingerprint visualization to {output_path}")
        
        # Close figure
        plt.close()
    
    def create_confusion_matrix(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = True,
        output_filename: str = "confusion_matrix.png",
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Create a confusion matrix visualization.
        
        Args:
            true_labels: Array of true labels
            predicted_labels: Array of predicted labels
            class_names: Optional list of class names
            normalize: Whether to normalize the confusion matrix
            output_filename: Output filename
            figsize: Figure size
        """
        from sklearn.metrics import confusion_matrix
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Get unique labels if class names not provided
        if class_names is None:
            unique_labels = np.unique(np.concatenate([true_labels, predicted_labels]))
            class_names = [str(label) for label in unique_labels]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create heatmap
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        # Add labels and title
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Add tick marks and labels
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.results_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")
        
        # Close figure
        plt.close()
    
    def create_attribution_example(
        self,
        query_text: str,
        candidate_texts: List[str],
        scores: List[float],
        true_index: Optional[int] = None,
        output_filename: str = "attribution_example.png",
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Create a visualization of an attribution example.
        
        Args:
            query_text: Text of the query sample
            candidate_texts: List of candidate texts
            scores: List of attribution scores
            true_index: Optional index of the true source (for highlighting)
            output_filename: Output filename
            figsize: Figure size
        """
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], figure=fig)
        
        # Query text subplot
        ax1 = fig.add_subplot(gs[0])
        ax1.text(0.5, 0.5, f"Query: {query_text}", ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="blue"))
        ax1.axis('off')
        
        # Candidates subplot
        ax2 = fig.add_subplot(gs[1])
        
        # Sort candidates by score
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        sorted_candidates = [candidate_texts[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        # Check if true index needs to be mapped to sorted index
        if true_index is not None:
            sorted_true_index = list(sorted_indices).index(true_index)
        else:
            sorted_true_index = None
        
        # Create horizontal bar chart
        bars = ax2.barh(range(len(sorted_candidates)), sorted_scores, 
                       color=['red' if i == sorted_true_index else 'skyblue' for i in range(len(sorted_candidates))])
        
        # Add texts and scores
        for i, (text, score) in enumerate(zip(sorted_candidates, sorted_scores)):
            # Truncate text if too long
            truncated_text = text[:50] + "..." if len(text) > 50 else text
            
            # Add candidate text
            ax2.text(0.01, i, truncated_text, va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
            
            # Add score
            ax2.text(score + 0.01, i, f"{score:.4f}", va='center', fontsize=9)
        
        # Set title and labels
        ax2.set_title("Attribution Candidates (Ranked by Score)")
        ax2.set_xlabel("Attribution Score")
        ax2.set_yticks(range(len(sorted_candidates)))
        ax2.set_yticklabels([f"Candidate {i+1}" for i in range(len(sorted_candidates))])
        
        # Add legend if true source is known
        if true_index is not None:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label='True Source'),
                Patch(facecolor='skyblue', label='Other Candidates')
            ]
            ax2.legend(handles=legend_elements, loc='upper right')
        
        # Add grid for easier comparison
        ax2.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.results_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attribution example to {output_path}")
        
        # Close figure
        plt.close(fig)


# Command-line interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test visualization module")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ExperimentVisualizer(results_dir=args.output_dir)
    
    # Create example data for performance comparison
    metrics_data = {
        "GIF": {
            "precision@1": 0.82, 
            "precision@5": 0.91, 
            "recall@10": 0.88, 
            "mrr": 0.87
        },
        "TRACE": {
            "precision@1": 0.70, 
            "precision@5": 0.85, 
            "recall@10": 0.82, 
            "mrr": 0.76
        },
        "TRAK": {
            "precision@1": 0.65, 
            "precision@5": 0.78, 
            "recall@10": 0.75, 
            "mrr": 0.71
        },
        "Vanilla": {
            "precision@1": 0.55, 
            "precision@5": 0.72, 
            "recall@10": 0.68, 
            "mrr": 0.61
        }
    }
    
    # Create performance comparison plot
    visualizer.create_performance_comparison(metrics_data)
    
    # Create example data for latency analysis
    latency_data = {
        "GIF": np.random.lognormal(3, 0.5, 100).tolist(),
        "TRACE": np.random.lognormal(4, 0.5, 100).tolist(),
        "TRAK": np.random.lognormal(3.5, 0.4, 100).tolist(),
        "Vanilla": np.random.lognormal(5, 0.6, 100).tolist(),
        "GIF_components": {
            "embedding": np.random.lognormal(2, 0.3, 100).tolist(),
            "fingerprinting": np.random.lognormal(2.3, 0.3, 100).tolist(),
            "ann_search": np.random.lognormal(1.5, 0.2, 100).tolist(),
            "influence_refinement": np.random.lognormal(2.7, 0.4, 100).tolist()
        }
    }
    
    # Create latency analysis plot
    visualizer.create_latency_analysis(latency_data)
    
    # Create example data for ablation study
    ablation_data = {
        "GIF (Full)": {
            "precision@1": 0.82, 
            "mrr": 0.87, 
            "latency_ms": 32.5
        },
        "Static Only": {
            "precision@1": 0.68, 
            "mrr": 0.74, 
            "latency_ms": 18.2
        },
        "Gradient Only": {
            "precision@1": 0.71, 
            "mrr": 0.78, 
            "latency_ms": 27.3
        },
        "No Influence": {
            "precision@1": 0.75, 
            "mrr": 0.81, 
            "latency_ms": 12.8
        }
    }
    
    # Create ablation study plot
    visualizer.create_ablation_plot(ablation_data)
    
    # Create example data for training curves
    training_data = {
        "loss": [0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15, 0.13, 0.11],
        "accuracy": [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87, 0.89, 0.9]
    }
    
    validation_data = {
        "loss": [0.85, 0.65, 0.45, 0.35, 0.3, 0.28, 0.26, 0.25, 0.24, 0.24],
        "accuracy": [0.48, 0.57, 0.67, 0.72, 0.75, 0.77, 0.79, 0.8, 0.8, 0.8]
    }
    
    # Create training curves plot
    visualizer.create_training_curves(training_data, validation_data)
    
    # Create example data for projection dimension effect
    dimension_data = {
        16: {"precision@1": 0.65, "mrr": 0.72, "latency_ms": 25.0},
        32: {"precision@1": 0.71, "mrr": 0.78, "latency_ms": 27.5},
        64: {"precision@1": 0.76, "mrr": 0.82, "latency_ms": 30.0},
        128: {"precision@1": 0.81, "mrr": 0.86, "latency_ms": 32.5},
        256: {"precision@1": 0.82, "mrr": 0.87, "latency_ms": 35.0},
        512: {"precision@1": 0.83, "mrr": 0.87, "latency_ms": 40.0}
    }
    
    # Create projection dimension plot
    visualizer.create_projection_dimension_plot(dimension_data)
    
    # Create example data for dataset size effect
    size_data = {
        1000: {"precision@1": 0.55, "mrr": 0.62, "latency_ms": 15.0},
        10000: {"precision@1": 0.65, "mrr": 0.72, "latency_ms": 20.0},
        100000: {"precision@1": 0.75, "mrr": 0.82, "latency_ms": 30.0},
        1000000: {"precision@1": 0.82, "mrr": 0.87, "latency_ms": 50.0},
        10000000: {"precision@1": 0.85, "mrr": 0.89, "latency_ms": 100.0}
    }
    
    # Create dataset size plot
    visualizer.create_dataset_size_plot(size_data)
    
    # Create example data for fingerprint visualization
    np.random.seed(42)
    n_samples = 500
    n_features = 64
    n_clusters = 5
    
    fingerprints = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, n_clusters, n_samples)
    
    # Create fingerprint visualization using t-SNE
    visualizer.create_fingerprint_visualization(fingerprints, labels, method="tsne")
    
    # Create example data for attribution example
    query_text = "This is an example query text that might be generated by a language model."
    candidate_texts = [
        "This is the true source text that was used during training.",
        "This is a similar but incorrect candidate text.",
        "This text is not very similar to the query.",
        "Another candidate text with some matching keywords.",
        "A completely unrelated text sample."
    ]
    scores = [0.85, 0.65, 0.32, 0.58, 0.12]
    true_index = 0
    
    # Create attribution example visualization
    visualizer.create_attribution_example(query_text, candidate_texts, scores, true_index)
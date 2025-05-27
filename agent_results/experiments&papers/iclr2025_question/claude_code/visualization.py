"""
Visualization utilities for experiment results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import os

class Visualizer:
    """Visualizer for experiment results."""
    
    def __init__(self, results_dir):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: The directory to save the visualizations.
        """
        self.results_dir = Path(results_dir)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('ggplot')
        
    def plot_metrics_comparison(self, metrics, model_names, metric_name, title=None, xlabel="Models", ylabel=None, figsize=(10, 6)):
        """
        Plot a comparison of metrics across models.
        
        Args:
            metrics: A dictionary mapping model names to metrics.
            model_names: The names of the models to include.
            metric_name: The name of the metric to plot.
            title: The title of the plot.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.
            figsize: The size of the figure.
        
        Returns:
            The path to the saved figure.
        """
        plt.figure(figsize=figsize)
        
        # Extract metric values
        values = [metrics[model][metric_name] for model in model_names]
        
        # Create bar plot
        bars = plt.bar(model_names, values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', rotation=0)
        
        # Set labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel or metric_name.replace('_', ' ').title())
        plt.title(title or f"{metric_name.replace('_', ' ').title()} Comparison")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        save_path = self.results_dir / f"{metric_name}_comparison.png"
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_metrics_radar(self, metrics, model_names, metric_names, title=None, figsize=(10, 8)):
        """
        Plot a radar chart comparing multiple metrics across models.
        
        Args:
            metrics: A dictionary mapping model names to metrics.
            model_names: The names of the models to include.
            metric_names: The names of the metrics to include.
            title: The title of the plot.
            figsize: The size of the figure.
        
        Returns:
            The path to the saved figure.
        """
        # Set up the radar chart
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)
        
        # Number of metrics
        num_metrics = len(metric_names)
        
        # Angle for each metric
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Set up the axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([name.replace('_', ' ').title() for name in metric_names])
        
        # Plot each model
        for model in model_names:
            # Extract metric values
            values = [metrics[model][metric] for metric in metric_names]
            values += values[:1]  # Close the loop
            
            # Plot the model
            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Set title and legend
        plt.title(title or "Metrics Comparison")
        plt.legend(loc='upper right')
        
        # Save the figure
        save_path = self.results_dir / "metrics_radar.png"
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_uncertainty_distribution(self, uncertainties, model_names, title=None, xlabel="Uncertainty", ylabel="Frequency", figsize=(10, 6)):
        """
        Plot the distribution of uncertainty values for different models.
        
        Args:
            uncertainties: A dictionary mapping model names to uncertainty values.
            model_names: The names of the models to include.
            title: The title of the plot.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.
            figsize: The size of the figure.
        
        Returns:
            The path to the saved figure.
        """
        plt.figure(figsize=figsize)
        
        # Plot histogram for each model
        for model in model_names:
            sns.histplot(uncertainties[model], label=model, alpha=0.5, kde=True)
        
        # Set labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title or "Uncertainty Distribution")
        
        # Add legend
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        save_path = self.results_dir / "uncertainty_distribution.png"
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_threshold_evolution(self, thresholds, title=None, xlabel="Generation Step", ylabel="Threshold Value", figsize=(10, 6)):
        """
        Plot the evolution of the uncertainty threshold during generation.
        
        Args:
            thresholds: A list of threshold values at each generation step.
            title: The title of the plot.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.
            figsize: The size of the figure.
        
        Returns:
            The path to the saved figure.
        """
        plt.figure(figsize=figsize)
        
        # Plot threshold evolution
        plt.plot(range(len(thresholds)), thresholds)
        
        # Set labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title or "Uncertainty Threshold Evolution")
        
        # Add grid
        plt.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        save_path = self.results_dir / "threshold_evolution.png"
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_uncertainty_vs_hallucination(self, uncertainties, hallucinations, model_names, title=None, xlabel="Uncertainty", ylabel="Hallucination Rate", figsize=(10, 6)):
        """
        Plot the relationship between uncertainty and hallucination rate.
        
        Args:
            uncertainties: A dictionary mapping model names to average uncertainty values.
            hallucinations: A dictionary mapping model names to hallucination rates.
            model_names: The names of the models to include.
            title: The title of the plot.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.
            figsize: The size of the figure.
        
        Returns:
            The path to the saved figure.
        """
        plt.figure(figsize=figsize)
        
        # Extract uncertainty and hallucination values
        x = [uncertainties[model] for model in model_names]
        y = [hallucinations[model] for model in model_names]
        
        # Create scatter plot
        plt.scatter(x, y, s=100)
        
        # Add labels for each point
        for i, model in enumerate(model_names):
            plt.annotate(model, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center')
        
        # Set labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title or "Uncertainty vs. Hallucination Rate")
        
        # Add best-fit line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(np.array(sorted(x)), p(np.array(sorted(x))), "r--", alpha=0.7)
        
        # Add grid
        plt.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        save_path = self.results_dir / "uncertainty_vs_hallucination.png"
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_computational_overhead(self, times, model_names, title=None, xlabel="Models", ylabel="Time (seconds)", figsize=(10, 6)):
        """
        Plot the computational overhead of different models.
        
        Args:
            times: A dictionary mapping model names to execution times.
            model_names: The names of the models to include.
            title: The title of the plot.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.
            figsize: The size of the figure.
        
        Returns:
            The path to the saved figure.
        """
        plt.figure(figsize=figsize)
        
        # Extract execution times
        values = [times[model] for model in model_names]
        
        # Create bar plot
        bars = plt.bar(model_names, values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', rotation=0)
        
        # Set labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title or "Computational Overhead")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        save_path = self.results_dir / "computational_overhead.png"
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_metrics_over_time(self, metrics_history, model_names, metric_name, title=None, xlabel="Epoch", ylabel=None, figsize=(10, 6)):
        """
        Plot a metric over time for different models.
        
        Args:
            metrics_history: A dictionary mapping model names to lists of metric values over time.
            model_names: The names of the models to include.
            metric_name: The name of the metric to plot.
            title: The title of the plot.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.
            figsize: The size of the figure.
        
        Returns:
            The path to the saved figure.
        """
        plt.figure(figsize=figsize)
        
        # Plot metric over time for each model
        for model in model_names:
            metric_values = metrics_history[model][metric_name]
            plt.plot(range(1, len(metric_values) + 1), metric_values, marker='o', label=model)
        
        # Set labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel or metric_name.replace('_', ' ').title())
        plt.title(title or f"{metric_name.replace('_', ' ').title()} Over Time")
        
        # Add grid and legend
        plt.grid(True)
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        save_path = self.results_dir / f"{metric_name}_over_time.png"
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def create_metrics_table(self, metrics, model_names, metric_names):
        """
        Create a metrics table.
        
        Args:
            metrics: A dictionary mapping model names to metrics.
            model_names: The names of the models to include.
            metric_names: The names of the metrics to include.
        
        Returns:
            A pandas DataFrame with the metrics table.
        """
        # Create a DataFrame
        data = []
        for model in model_names:
            row = [model]
            for metric in metric_names:
                value = metrics[model][metric]
                row.append(value)
            data.append(row)
        
        columns = ["Model"] + [metric.replace('_', ' ').title() for metric in metric_names]
        df = pd.DataFrame(data, columns=columns)
        
        return df
    
    def save_metrics_table(self, df, file_name="metrics_table.csv"):
        """
        Save a metrics table to a CSV file.
        
        Args:
            df: The pandas DataFrame with the metrics table.
            file_name: The name of the CSV file.
        
        Returns:
            The path to the saved file.
        """
        save_path = self.results_dir / file_name
        df.to_csv(save_path, index=False)
        
        return save_path
    
    def save_metrics_as_markdown(self, df, file_name="metrics_table.md"):
        """
        Save a metrics table to a Markdown file.
        
        Args:
            df: The pandas DataFrame with the metrics table.
            file_name: The name of the Markdown file.
        
        Returns:
            The path to the saved file.
        """
        save_path = self.results_dir / file_name
        with open(save_path, 'w') as f:
            f.write(df.to_markdown(index=False))
        
        return save_path
    
    def save_results_to_json(self, results, file_name="results.json"):
        """
        Save results to a JSON file.
        
        Args:
            results: The results to save.
            file_name: The name of the JSON file.
        
        Returns:
            The path to the saved file.
        """
        save_path = self.results_dir / file_name
        
        # Convert numpy arrays and tensors to lists
        def convert_numpy(obj):
            import torch
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().detach().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        # Create a simplified version of results that's JSON-serializable
        serializable_results = {}
        for experiment_name, experiment_data in results.items():
            serializable_results[experiment_name] = {
                "experiment_name": experiment_data.get("experiment_name", ""),
                "config": experiment_data.get("config", {}),
                "evaluation": convert_numpy(experiment_data.get("evaluation", {})),
                "execution_time": experiment_data.get("execution_time", 0),
                # Only include a sample of generated texts
                "generated_texts": experiment_data.get("generated_texts", [])[:5],
                # Skip complex data that might not be serializable
                "final_threshold": experiment_data.get("final_threshold", 0.0),
            }
        
        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        return save_path
"""
Evaluation metrics module for the Gradient-Informed Fingerprinting (GIF) method.

This module implements various metrics for evaluating the performance of attribution
methods, including precision, recall, mean reciprocal rank (MRR), and latency.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AttributionMetrics:
    """
    Metrics for evaluating attribution method performance.
    
    This class calculates precision, recall, MRR, and other metrics for
    evaluating the performance of attribution methods on test samples.
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Store results for each method
        self.results = {}
        
        # Store latency measurements
        self.latency_measurements = defaultdict(list)
    
    def compute_precision_at_k(
        self, 
        predicted_ids: List[str], 
        true_id: str, 
        k: int
    ) -> float:
        """
        Compute precision@k for a single prediction.
        
        Args:
            predicted_ids: List of predicted IDs
            true_id: True ID
            k: Number of top predictions to consider
        
        Returns:
            Precision@k (1.0 if true ID is in top k, 0.0 otherwise)
        """
        # Limit to top k predictions
        top_k_preds = predicted_ids[:k]
        
        # Check if true ID is in top k
        if true_id in top_k_preds:
            return 1.0
        else:
            return 0.0
    
    def compute_recall_at_k(
        self, 
        predicted_ids: List[str], 
        true_ids: List[str], 
        k: int
    ) -> float:
        """
        Compute recall@k for a single prediction.
        
        Args:
            predicted_ids: List of predicted IDs
            true_ids: List of true IDs (may be multiple)
            k: Number of top predictions to consider
        
        Returns:
            Recall@k (proportion of true IDs found in top k predictions)
        """
        # Limit to top k predictions
        top_k_preds = predicted_ids[:k]
        
        # Count true positives
        tp = sum(1 for true_id in true_ids if true_id in top_k_preds)
        
        # Compute recall
        if len(true_ids) > 0:
            return tp / len(true_ids)
        else:
            return 1.0  # If no true IDs, recall is perfect
    
    def compute_mrr(
        self, 
        predicted_ids: List[str], 
        true_id: str
    ) -> float:
        """
        Compute Mean Reciprocal Rank for a single prediction.
        
        Args:
            predicted_ids: List of predicted IDs
            true_id: True ID
        
        Returns:
            Mean Reciprocal Rank (1/rank of first true ID)
        """
        # Find rank of true ID
        try:
            rank = predicted_ids.index(true_id) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0  # True ID not found in predictions
    
    def evaluate_single_method(
        self, 
        method_name: str,
        predictions: List[List[str]],
        true_ids: List[Union[str, List[str]]],
        k_values: List[int] = [1, 3, 5, 10],
        latencies: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single attribution method.
        
        Args:
            method_name: Name of the method
            predictions: List of lists of predicted IDs
            true_ids: List of true IDs or lists of true IDs
            k_values: List of k values for precision@k and recall@k
            latencies: Optional list of latency measurements (ms)
        
        Returns:
            Dictionary of evaluation metrics
        """
        if len(predictions) != len(true_ids):
            raise ValueError(f"Number of predictions ({len(predictions)}) must match number of true IDs ({len(true_ids)})")
        
        # Convert single true IDs to lists for consistent processing
        true_ids_list = [
            [true_id] if isinstance(true_id, str) else true_id
            for true_id in true_ids
        ]
        
        # Calculate metrics
        metrics = {
            "method": method_name,
            "num_samples": len(predictions)
        }
        
        # Calculate precision@k
        for k in k_values:
            if k > max(len(p) for p in predictions):
                logger.warning(f"Some predictions have fewer than {k} results, precision@{k} may be underestimated")
            
            precisions = []
            for preds, trues in zip(predictions, true_ids_list):
                # If multiple true IDs, consider it correct if any match
                precision = max(self.compute_precision_at_k(preds, true, k) for true in trues)
                precisions.append(precision)
            
            metrics[f"precision@{k}"] = np.mean(precisions)
        
        # Calculate recall@k
        for k in k_values:
            recalls = []
            for preds, trues in zip(predictions, true_ids_list):
                recall = self.compute_recall_at_k(preds, trues, k)
                recalls.append(recall)
            
            metrics[f"recall@{k}"] = np.mean(recalls)
        
        # Calculate MRR
        mrrs = []
        for preds, trues in zip(predictions, true_ids_list):
            # If multiple true IDs, take the best MRR
            mrr = max(self.compute_mrr(preds, true) for true in trues)
            mrrs.append(mrr)
        
        metrics["mrr"] = np.mean(mrrs)
        
        # Process latency measurements
        if latencies:
            if len(latencies) != len(predictions):
                logger.warning(f"Number of latency measurements ({len(latencies)}) doesn't match number of predictions ({len(predictions)})")
            
            metrics["mean_latency_ms"] = np.mean(latencies)
            metrics["median_latency_ms"] = np.median(latencies)
            metrics["min_latency_ms"] = np.min(latencies)
            metrics["max_latency_ms"] = np.max(latencies)
            metrics["std_latency_ms"] = np.std(latencies)
            
            # Store latencies for later comparison
            self.latency_measurements[method_name].extend(latencies)
        
        # Store results for this method
        self.results[method_name] = metrics
        
        return metrics
    
    def compare_methods(
        self, 
        method_names: Optional[List[str]] = None,
        metrics_to_compare: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple attribution methods.
        
        Args:
            method_names: Optional list of method names to compare (default: all methods)
            metrics_to_compare: Optional list of metrics to compare (default: all metrics)
        
        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            raise ValueError("No results to compare. Run evaluate_single_method first.")
        
        # Use all methods if not specified
        if method_names is None:
            method_names = list(self.results.keys())
        
        # Filter results to requested methods
        filtered_results = {name: self.results[name] for name in method_names if name in self.results}
        
        if not filtered_results:
            raise ValueError(f"None of the specified methods {method_names} have results.")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame([result for result in filtered_results.values()])
        
        # Filter columns if requested
        if metrics_to_compare is not None:
            # Always keep the 'method' column
            columns_to_keep = ['method'] + [col for col in metrics_to_compare if col in results_df.columns]
            results_df = results_df[columns_to_keep]
        
        return results_df
    
    def plot_precision_at_k(
        self, 
        method_names: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
        title: str = "Precision@k Comparison",
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot precision@k for multiple methods.
        
        Args:
            method_names: Optional list of method names to compare (default: all methods)
            k_values: Optional list of k values to plot (default: all k values in results)
            title: Plot title
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No results to plot. Run evaluate_single_method first.")
        
        # Use all methods if not specified
        if method_names is None:
            method_names = list(self.results.keys())
        
        # Filter results to requested methods
        filtered_results = {name: self.results[name] for name in method_names if name in self.results}
        
        if not filtered_results:
            raise ValueError(f"None of the specified methods {method_names} have results.")
        
        # Find all precision@k metrics in results
        if k_values is None:
            k_values = sorted([
                int(col.split('@')[1])
                for col in self.results[list(self.results.keys())[0]].keys()
                if col.startswith('precision@')
            ])
        
        # Prepare data for plotting
        plot_data = []
        for method, result in filtered_results.items():
            for k in k_values:
                metric_name = f"precision@{k}"
                if metric_name in result:
                    plot_data.append({
                        'Method': method,
                        'k': k,
                        'Precision': result[metric_name]
                    })
        
        if not plot_data:
            raise ValueError(f"No precision@k metrics found for the specified methods and k values.")
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame(plot_data)
        
        # Create plot
        plt.figure(figsize=figsize)
        sns.barplot(x='k', y='Precision', hue='Method', data=plot_df)
        plt.title(title)
        plt.xlabel('k')
        plt.ylabel('Precision')
        plt.ylim(0, 1)
        plt.legend(title='Method')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        output_path = os.path.join(self.results_dir, "precision_at_k.png")
        plt.savefig(output_path)
        logger.info(f"Saved precision@k plot to {output_path}")
        
        # Close figure
        plt.close()
    
    def plot_recall_at_k(
        self, 
        method_names: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
        title: str = "Recall@k Comparison",
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot recall@k for multiple methods.
        
        Args:
            method_names: Optional list of method names to compare (default: all methods)
            k_values: Optional list of k values to plot (default: all k values in results)
            title: Plot title
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No results to plot. Run evaluate_single_method first.")
        
        # Use all methods if not specified
        if method_names is None:
            method_names = list(self.results.keys())
        
        # Filter results to requested methods
        filtered_results = {name: self.results[name] for name in method_names if name in self.results}
        
        if not filtered_results:
            raise ValueError(f"None of the specified methods {method_names} have results.")
        
        # Find all recall@k metrics in results
        if k_values is None:
            k_values = sorted([
                int(col.split('@')[1])
                for col in self.results[list(self.results.keys())[0]].keys()
                if col.startswith('recall@')
            ])
        
        # Prepare data for plotting
        plot_data = []
        for method, result in filtered_results.items():
            for k in k_values:
                metric_name = f"recall@{k}"
                if metric_name in result:
                    plot_data.append({
                        'Method': method,
                        'k': k,
                        'Recall': result[metric_name]
                    })
        
        if not plot_data:
            raise ValueError(f"No recall@k metrics found for the specified methods and k values.")
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame(plot_data)
        
        # Create plot
        plt.figure(figsize=figsize)
        sns.barplot(x='k', y='Recall', hue='Method', data=plot_df)
        plt.title(title)
        plt.xlabel('k')
        plt.ylabel('Recall')
        plt.ylim(0, 1)
        plt.legend(title='Method')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        output_path = os.path.join(self.results_dir, "recall_at_k.png")
        plt.savefig(output_path)
        logger.info(f"Saved recall@k plot to {output_path}")
        
        # Close figure
        plt.close()
    
    def plot_mrr_comparison(
        self, 
        method_names: Optional[List[str]] = None,
        title: str = "Mean Reciprocal Rank Comparison",
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot MRR comparison for multiple methods.
        
        Args:
            method_names: Optional list of method names to compare (default: all methods)
            title: Plot title
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No results to plot. Run evaluate_single_method first.")
        
        # Use all methods if not specified
        if method_names is None:
            method_names = list(self.results.keys())
        
        # Filter results to requested methods
        filtered_results = {name: self.results[name] for name in method_names if name in self.results}
        
        if not filtered_results:
            raise ValueError(f"None of the specified methods {method_names} have results.")
        
        # Prepare data for plotting
        plot_data = []
        for method, result in filtered_results.items():
            if "mrr" in result:
                plot_data.append({
                    'Method': method,
                    'MRR': result["mrr"]
                })
        
        if not plot_data:
            raise ValueError(f"No MRR metrics found for the specified methods.")
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame(plot_data)
        
        # Create plot
        plt.figure(figsize=figsize)
        sns.barplot(x='Method', y='MRR', data=plot_df)
        plt.title(title)
        plt.xlabel('Method')
        plt.ylabel('Mean Reciprocal Rank')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values above bars
        for i, v in enumerate(plot_df['MRR']):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Save figure
        plt.tight_layout()
        output_path = os.path.join(self.results_dir, "mrr_comparison.png")
        plt.savefig(output_path)
        logger.info(f"Saved MRR comparison plot to {output_path}")
        
        # Close figure
        plt.close()
    
    def plot_latency_comparison(
        self, 
        method_names: Optional[List[str]] = None,
        title: str = "Latency Comparison",
        figsize: Tuple[int, int] = (8, 6),
        log_scale: bool = True
    ) -> None:
        """
        Plot latency comparison for multiple methods.
        
        Args:
            method_names: Optional list of method names to compare (default: all methods)
            title: Plot title
            figsize: Figure size
            log_scale: Whether to use log scale for latency values
        """
        if not self.latency_measurements:
            raise ValueError("No latency measurements found. Run evaluate_single_method with latency data first.")
        
        # Use all methods with latency measurements if not specified
        if method_names is None:
            method_names = list(self.latency_measurements.keys())
        
        # Filter measurements to requested methods
        filtered_measurements = {
            name: self.latency_measurements[name] 
            for name in method_names 
            if name in self.latency_measurements
        }
        
        if not filtered_measurements:
            raise ValueError(f"None of the specified methods {method_names} have latency measurements.")
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Calculate median latencies for box plot ordering
        medians = {method: np.median(latencies) for method, latencies in filtered_measurements.items()}
        ordered_methods = sorted(medians.keys(), key=lambda m: medians[m])
        
        # Prepare data for box plot
        plot_data = []
        for method in ordered_methods:
            for latency in filtered_measurements[method]:
                plot_data.append({
                    'Method': method,
                    'Latency (ms)': latency
                })
        
        # Create box plot
        plot_df = pd.DataFrame(plot_data)
        sns.boxplot(x='Method', y='Latency (ms)', data=plot_df, order=ordered_methods)
        
        # Customize plot
        plt.title(title)
        plt.xlabel('Method')
        plt.ylabel('Latency (ms)')
        if log_scale:
            plt.yscale('log')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add median values above boxes
        for i, method in enumerate(ordered_methods):
            median = medians[method]
            plt.text(i, median, f'{median:.1f}', ha='center', va='bottom')
        
        # Save figure
        plt.tight_layout()
        output_path = os.path.join(self.results_dir, "latency_comparison.png")
        plt.savefig(output_path)
        logger.info(f"Saved latency comparison plot to {output_path}")
        
        # Close figure
        plt.close()
    
    def generate_summary_table(
        self,
        method_names: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate a summary table of all metrics for all methods.
        
        Args:
            method_names: Optional list of method names to include (default: all methods)
            metrics: Optional list of metrics to include (default: common metrics)
        
        Returns:
            DataFrame with summary table
        """
        if not self.results:
            raise ValueError("No results to summarize. Run evaluate_single_method first.")
        
        # Use all methods if not specified
        if method_names is None:
            method_names = list(self.results.keys())
        
        # Filter results to requested methods
        filtered_results = {name: self.results[name] for name in method_names if name in self.results}
        
        if not filtered_results:
            raise ValueError(f"None of the specified methods {method_names} have results.")
        
        # Default metrics to include
        if metrics is None:
            metrics = [
                "precision@1", "precision@5", "precision@10", 
                "recall@1", "recall@5", "recall@10", 
                "mrr", "mean_latency_ms"
            ]
        
        # Create summary table
        summary_data = []
        for method, result in filtered_results.items():
            row = {'Method': method}
            for metric in metrics:
                if metric in result:
                    # Format based on metric type
                    if metric.startswith('precision@') or metric.startswith('recall@') or metric == 'mrr':
                        row[metric] = f"{result[metric]:.3f}"
                    elif 'latency' in metric:
                        row[metric] = f"{result[metric]:.2f}"
                    else:
                        row[metric] = result[metric]
                else:
                    row[metric] = "N/A"
            summary_data.append(row)
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        output_path = os.path.join(self.results_dir, "metrics_summary.csv")
        summary_df.to_csv(output_path, index=False)
        logger.info(f"Saved metrics summary to {output_path}")
        
        return summary_df
    
    def save_results(self, filename: str = "attribution_metrics.json") -> None:
        """
        Save all results to a JSON file.
        
        Args:
            filename: Name of the output file
        """
        output_path = os.path.join(self.results_dir, filename)
        
        # Create a copy of results with numpy values converted to Python types
        serializable_results = {}
        
        for method, metrics in self.results.items():
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                elif isinstance(value, np.number):
                    serializable_metrics[key] = value.item()
                else:
                    serializable_metrics[key] = value
            serializable_results[method] = serializable_metrics
        
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved attribution metrics to {output_path}")
    
    def load_results(self, filename: str = "attribution_metrics.json") -> None:
        """
        Load results from a JSON file.
        
        Args:
            filename: Name of the input file
        """
        input_path = os.path.join(self.results_dir, filename)
        
        with open(input_path, "r") as f:
            self.results = json.load(f)
        
        logger.info(f"Loaded attribution metrics from {input_path}")


class LatencyTracker:
    """
    Track and measure latency of operations.
    
    This class provides utilities for measuring and tracking the latency
    of different operations, such as fingerprint generation, ANN search,
    and influence refinement.
    """
    
    def __init__(self):
        self.measurements = defaultdict(list)
    
    def measure(self, operation_name: str) -> Any:
        """
        Context manager for measuring operation latency.
        
        Args:
            operation_name: Name of the operation being measured
        
        Returns:
            Context manager that measures latency
        """
        class LatencyContext:
            def __init__(self, tracker, name):
                self.tracker = tracker
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                latency_ms = (end_time - self.start_time) * 1000
                self.tracker.add_measurement(self.name, latency_ms)
        
        return LatencyContext(self, operation_name)
    
    def add_measurement(self, operation_name: str, latency_ms: float) -> None:
        """
        Add a latency measurement.
        
        Args:
            operation_name: Name of the operation
            latency_ms: Latency in milliseconds
        """
        self.measurements[operation_name].append(latency_ms)
    
    def get_measurements(self, operation_name: str) -> List[float]:
        """
        Get all measurements for an operation.
        
        Args:
            operation_name: Name of the operation
        
        Returns:
            List of latency measurements in milliseconds
        """
        return self.measurements.get(operation_name, [])
    
    def get_stats(self, operation_name: str = None) -> Dict[str, Any]:
        """
        Get statistics for operation latencies.
        
        Args:
            operation_name: Optional name of the operation (default: stats for all operations)
        
        Returns:
            Dictionary of latency statistics
        """
        if operation_name is not None:
            measurements = self.get_measurements(operation_name)
            if not measurements:
                return {}
            
            return {
                "name": operation_name,
                "count": len(measurements),
                "mean_ms": np.mean(measurements),
                "median_ms": np.median(measurements),
                "min_ms": np.min(measurements),
                "max_ms": np.max(measurements),
                "std_ms": np.std(measurements)
            }
        else:
            stats = {}
            for name in self.measurements.keys():
                stats[name] = self.get_stats(name)
            return stats
    
    def plot_latency_breakdown(
        self,
        output_dir: str = "results",
        filename: str = "latency_breakdown.png",
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot latency breakdown for different operations.
        
        Args:
            output_dir: Output directory
            filename: Output filename
            figsize: Figure size
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.measurements:
            logger.warning("No latency measurements to plot.")
            return
        
        # Prepare data for box plot
        plot_data = []
        for operation, latencies in self.measurements.items():
            for latency in latencies:
                plot_data.append({
                    'Operation': operation,
                    'Latency (ms)': latency
                })
        
        # Create DataFrame
        plot_df = pd.DataFrame(plot_data)
        
        # Sort operations by median latency
        operation_medians = {}
        for operation in plot_df['Operation'].unique():
            operation_medians[operation] = plot_df[plot_df['Operation'] == operation]['Latency (ms)'].median()
        
        sorted_operations = sorted(operation_medians.keys(), key=lambda op: operation_medians[op])
        
        # Create plot
        plt.figure(figsize=figsize)
        sns.boxplot(x='Operation', y='Latency (ms)', data=plot_df, order=sorted_operations)
        
        # Customize plot
        plt.title("Latency Breakdown by Operation")
        plt.xlabel("Operation")
        plt.ylabel("Latency (ms)")
        plt.yscale('log')  # Log scale for better visibility of different magnitudes
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        
        # Add median values above boxes
        for i, operation in enumerate(sorted_operations):
            median = operation_medians[operation]
            plt.text(i, median, f'{median:.1f}', ha='center', va='bottom')
        
        # Save figure
        plt.tight_layout()
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        logger.info(f"Saved latency breakdown plot to {output_path}")
        
        # Close figure
        plt.close()
    
    def plot_cumulative_latency(
        self,
        operations: Optional[List[str]] = None,
        output_dir: str = "results",
        filename: str = "cumulative_latency.png",
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot cumulative latency for a sequence of operations.
        
        Args:
            operations: List of operations in the sequence (default: all operations)
            output_dir: Output directory
            filename: Output filename
            figsize: Figure size
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.measurements:
            logger.warning("No latency measurements to plot.")
            return
        
        # Use all operations if not specified
        if operations is None:
            operations = list(self.measurements.keys())
        
        # Filter to operations that have measurements
        operations = [op for op in operations if op in self.measurements]
        
        if not operations:
            logger.warning("No matching operations found for cumulative latency plot.")
            return
        
        # Calculate median latencies
        median_latencies = {op: np.median(self.measurements[op]) for op in operations}
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot bars
        cumulative = 0
        bars = []
        for i, operation in enumerate(operations):
            latency = median_latencies[operation]
            bar = plt.bar([0], [latency], bottom=cumulative, label=operation)
            cumulative += latency
            bars.append(bar)
        
        # Add annotations
        cumulative = 0
        for operation in operations:
            latency = median_latencies[operation]
            plt.text(0, cumulative + latency/2, f"{operation}\n{latency:.1f} ms", ha='center', va='center')
            cumulative += latency
        
        # Add total latency
        plt.text(0, cumulative + 5, f"Total: {cumulative:.1f} ms", ha='center', va='bottom', fontweight='bold')
        
        # Customize plot
        plt.title("Cumulative Operation Latency")
        plt.ylabel("Latency (ms)")
        plt.xticks([])  # Hide x-axis
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, cumulative * 1.1)  # Add some space at the top
        
        # Add legend
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Save figure
        plt.tight_layout()
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        logger.info(f"Saved cumulative latency plot to {output_path}")
        
        # Close figure
        plt.close()


# Command-line interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test attribution metrics")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create some example data
    num_samples = 100
    k_values = [1, 3, 5, 10]
    
    # Generate random IDs
    true_ids = [f"sample_{i}" for i in range(num_samples)]
    
    # Simulate predictions for different methods
    methods = {
        "GIF": {"accuracy": 0.8, "latency_range": (10, 50)},
        "TRACE": {"accuracy": 0.7, "latency_range": (100, 200)},
        "TRAK": {"accuracy": 0.6, "latency_range": (50, 150)},
        "Vanilla": {"accuracy": 0.5, "latency_range": (500, 1000)}
    }
    
    # Generate predictions for each method
    predictions = {}
    latencies = {}
    
    for method_name, config in methods.items():
        method_preds = []
        method_latencies = []
        
        for i in range(num_samples):
            true_id = true_ids[i]
            
            # Determine if this prediction will be correct with the method's accuracy
            if np.random.random() < config["accuracy"]:
                # Correct prediction at some position
                position = np.random.geometric(0.5)  # Geometric distribution for position
                position = min(position - 1, 9)  # Ensure within top 10
                
                # Create prediction list with true_id at the determined position
                pred_list = [f"dummy_{j}" for j in range(10)]
                pred_list[position] = true_id
            else:
                # Incorrect prediction
                pred_list = [f"dummy_{j}" for j in range(10)]
            
            method_preds.append(pred_list)
            
            # Simulate latency
            latency = np.random.uniform(*config["latency_range"])
            method_latencies.append(latency)
        
        predictions[method_name] = method_preds
        latencies[method_name] = method_latencies
    
    # Create metrics evaluator
    metrics = AttributionMetrics(results_dir=args.output_dir)
    
    # Evaluate each method
    for method_name in methods:
        metrics.evaluate_single_method(
            method_name=method_name,
            predictions=predictions[method_name],
            true_ids=true_ids,
            k_values=k_values,
            latencies=latencies[method_name]
        )
    
    # Generate comparison and plots
    comparison_df = metrics.compare_methods()
    print("Method Comparison:")
    print(comparison_df)
    
    # Plot metrics
    metrics.plot_precision_at_k()
    metrics.plot_recall_at_k()
    metrics.plot_mrr_comparison()
    metrics.plot_latency_comparison()
    
    # Generate summary table
    summary_df = metrics.generate_summary_table()
    print("\nSummary Table:")
    print(summary_df)
    
    # Save results
    metrics.save_results()
    
    # Test latency tracker
    tracker = LatencyTracker()
    
    # Simulate some operations
    operations = ["embedding", "fingerprinting", "ann_search", "influence_refinement"]
    
    for operation in operations:
        for _ in range(50):
            # Simulate different latencies for different operations
            if operation == "embedding":
                latency_ms = np.random.uniform(10, 30)
            elif operation == "fingerprinting":
                latency_ms = np.random.uniform(30, 80)
            elif operation == "ann_search":
                latency_ms = np.random.uniform(5, 15)
            else:  # influence_refinement
                latency_ms = np.random.uniform(100, 300)
            
            tracker.add_measurement(operation, latency_ms)
    
    # Plot latency breakdown
    tracker.plot_latency_breakdown(output_dir=args.output_dir)
    
    # Plot cumulative latency
    tracker.plot_cumulative_latency(output_dir=args.output_dir)
    
    # Print statistics
    print("\nLatency Statistics:")
    stats = tracker.get_stats()
    for operation, operation_stats in stats.items():
        print(f"{operation}: {operation_stats['mean_ms']:.2f} ms (median: {operation_stats['median_ms']:.2f} ms)")
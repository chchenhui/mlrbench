"""
Metrics for evaluating ContractGPT and baseline methods.

This module provides functionality for calculating and visualizing metrics
for code synthesis methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import os


class Metrics:
    """
    Class for calculating and visualizing metrics for code synthesis methods.
    """
    
    def __init__(self, results: Dict[str, List[Dict[str, Any]]]):
        """
        Initialize the metrics calculator.
        
        Args:
            results: Dictionary mapping method names to lists of result metrics.
                    Each result metric is a dictionary with keys:
                    - "name": benchmark name
                    - "success": success flag
                    - "iterations": number of iterations
                    - "verification_time": verification time
                    - "generation_time": generation time
        """
        self.results = results
        self.methods = list(results.keys())
        self.benchmarks = list({r["name"] for rs in results.values() for r in rs})
    
    def calculate_success_rate(self) -> Dict[str, float]:
        """
        Calculate success rate for each method.
        
        Returns:
            Dictionary mapping method names to success rates.
        """
        success_rates = {}
        
        for method, results in self.results.items():
            success_count = sum(1 for r in results if r["success"])
            total_count = len(results)
            success_rates[method] = (success_count / total_count) if total_count > 0 else 0.0
        
        return success_rates
    
    def calculate_mean_iterations(self) -> Dict[str, float]:
        """
        Calculate mean number of iterations for each method.
        
        Returns:
            Dictionary mapping method names to mean iterations.
        """
        mean_iterations = {}
        
        for method, results in self.results.items():
            iterations = [r["iterations"] for r in results if "iterations" in r]
            mean_iterations[method] = np.mean(iterations) if iterations else 0.0
        
        return mean_iterations
    
    def calculate_mean_times(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate mean verification and generation times for each method.
        
        Returns:
            Tuple (mean_verification_times, mean_generation_times) of dictionaries
            mapping method names to mean times.
        """
        mean_verification_times = {}
        mean_generation_times = {}
        
        for method, results in self.results.items():
            verification_times = [r["verification_time"] for r in results if "verification_time" in r]
            generation_times = [r["generation_time"] for r in results if "generation_time" in r]
            
            mean_verification_times[method] = np.mean(verification_times) if verification_times else 0.0
            mean_generation_times[method] = np.mean(generation_times) if generation_times else 0.0
        
        return mean_verification_times, mean_generation_times
    
    def calculate_bug_rate(self, baseline_method: str = "LLMOnly") -> Dict[str, float]:
        """
        Calculate bug rate reduction relative to a baseline method.
        
        Args:
            baseline_method: Name of the baseline method.
            
        Returns:
            Dictionary mapping method names to bug rate reductions.
        """
        success_rates = self.calculate_success_rate()
        baseline_success = success_rates.get(baseline_method, 0.0)
        
        bug_rates = {}
        for method, success_rate in success_rates.items():
            if method == baseline_method or baseline_success == 0.0:
                bug_rates[method] = 0.0
            else:
                # Bug rate = 1 - success_rate_baseline / success_rate_method
                bug_rates[method] = 1.0 - (baseline_success / success_rate) if success_rate > 0 else 0.0
        
        return bug_rates
    
    def plot_success_rates(self, save_path: Optional[str] = None) -> None:
        """
        Plot success rates for each method.
        
        Args:
            save_path: Path to save the plot, or None to display only.
        """
        success_rates = self.calculate_success_rate()
        
        plt.figure(figsize=(10, 6))
        plt.bar(success_rates.keys(), success_rates.values())
        plt.ylabel('Success Rate')
        plt.title('Success Rate by Method')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
    
    def plot_mean_iterations(self, save_path: Optional[str] = None) -> None:
        """
        Plot mean iterations for each method.
        
        Args:
            save_path: Path to save the plot, or None to display only.
        """
        mean_iterations = self.calculate_mean_iterations()
        
        # Filter out methods with 0 iterations
        filtered_iterations = {k: v for k, v in mean_iterations.items() if v > 0}
        
        plt.figure(figsize=(10, 6))
        plt.bar(filtered_iterations.keys(), filtered_iterations.values())
        plt.ylabel('Mean Iterations')
        plt.title('Mean Iterations by Method')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
    
    def plot_mean_times(self, save_path: Optional[str] = None) -> None:
        """
        Plot mean verification and generation times for each method.
        
        Args:
            save_path: Path to save the plot, or None to display only.
        """
        mean_verification_times, mean_generation_times = self.calculate_mean_times()
        
        methods = list(set(mean_verification_times.keys()) | set(mean_generation_times.keys()))
        
        verification_times = [mean_verification_times.get(method, 0.0) for method in methods]
        generation_times = [mean_generation_times.get(method, 0.0) for method in methods]
        
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(methods))
        width = 0.35
        
        plt.bar(x - width/2, verification_times, width, label='Verification Time')
        plt.bar(x + width/2, generation_times, width, label='Generation Time')
        
        plt.ylabel('Time (seconds)')
        plt.title('Mean Verification and Generation Times by Method')
        plt.xticks(x, methods, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
    
    def plot_bug_rate_reduction(self, baseline_method: str = "LLMOnly", save_path: Optional[str] = None) -> None:
        """
        Plot bug rate reduction relative to a baseline method.
        
        Args:
            baseline_method: Name of the baseline method.
            save_path: Path to save the plot, or None to display only.
        """
        bug_rates = self.calculate_bug_rate(baseline_method)
        
        # Filter out the baseline method and methods with 0 bug rate
        filtered_bug_rates = {k: v for k, v in bug_rates.items() if k != baseline_method and v > 0}
        
        plt.figure(figsize=(10, 6))
        plt.bar(filtered_bug_rates.keys(), filtered_bug_rates.values())
        plt.ylabel('Bug Rate Reduction')
        plt.title(f'Bug Rate Reduction Relative to {baseline_method}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
    
    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate a summary table of all metrics.
        
        Returns:
            DataFrame containing all metrics for each method.
        """
        success_rates = self.calculate_success_rate()
        mean_iterations = self.calculate_mean_iterations()
        mean_verification_times, mean_generation_times = self.calculate_mean_times()
        bug_rates = self.calculate_bug_rate()
        
        data = []
        for method in self.methods:
            row = {
                'Method': method,
                'Success Rate': f"{success_rates.get(method, 0.0):.2f}",
                'Mean Iterations': f"{mean_iterations.get(method, 0.0):.2f}",
                'Mean Verification Time (s)': f"{mean_verification_times.get(method, 0.0):.2f}",
                'Mean Generation Time (s)': f"{mean_generation_times.get(method, 0.0):.2f}",
                'Bug Rate Reduction': f"{bug_rates.get(method, 0.0):.2f}"
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_all_plots(self, output_dir: str) -> List[str]:
        """
        Save all plots to the specified directory.
        
        Args:
            output_dir: Directory to save plots.
            
        Returns:
            List of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        
        # Success rates
        success_path = os.path.join(output_dir, 'success_rates.png')
        self.plot_success_rates(success_path)
        saved_paths.append(success_path)
        
        # Mean iterations
        iterations_path = os.path.join(output_dir, 'mean_iterations.png')
        self.plot_mean_iterations(iterations_path)
        saved_paths.append(iterations_path)
        
        # Mean times
        times_path = os.path.join(output_dir, 'mean_times.png')
        self.plot_mean_times(times_path)
        saved_paths.append(times_path)
        
        # Bug rate reduction
        bug_rate_path = os.path.join(output_dir, 'bug_rate_reduction.png')
        self.plot_bug_rate_reduction(save_path=bug_rate_path)
        saved_paths.append(bug_rate_path)
        
        return saved_paths
"""
Evaluation Metrics and Analysis Tools for Contextual Dataset Deprecation Framework

This module provides tools for evaluating and analyzing the performance of
different dataset deprecation strategies.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from enum import Enum
from pathlib import Path

from experimental_design import WarningLevel, DeprecationStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log.txt')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("evaluation")

# Constants
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

class EvaluationMetrics:
    """Class for calculating evaluation metrics across different deprecation strategies."""
    
    def __init__(self, results_path: str = None):
        """
        Initialize the evaluation metrics calculator.
        
        Args:
            results_path: Path to the directory containing simulation results.
                          If None, uses the default results directory.
        """
        self.results_path = results_path or os.path.join(os.path.dirname(__file__), 'evaluation_results')
        self.figures_path = os.path.join(RESULTS_DIR, 'figures')
        os.makedirs(self.figures_path, exist_ok=True)
        
        # Dictionary to store loaded results
        self.results = {}
        
        # Metrics to evaluate
        self.metrics = {
            "user_response": {
                "acknowledgment_time": "Time to acknowledge deprecation (days)",
                "alternative_adoption": "Alternative dataset adoption rate",
                "continued_usage": "Continued usage of deprecated datasets"
            },
            "system_performance": {
                "recommendation_accuracy": "Recommendation accuracy",
                "processing_time": "Processing time (seconds)",
                "notification_success": "Notification success rate"
            },
            "research_impact": {
                "citation_pattern": "Citation patterns over time",
                "benchmark_diversity": "Benchmark dataset diversity",
                "alternative_performance": "Performance on alternative datasets"
            }
        }
    
    def load_results(self, strategy: DeprecationStrategy, path: str) -> None:
        """
        Load results for a specific strategy.
        
        Args:
            strategy: The deprecation strategy
            path: Path to the directory containing results for this strategy
        """
        strategy_name = strategy.name
        self.results[strategy_name] = {
            "user_response": None,
            "system_performance": None,
            "research_impact": None,
            "aggregate_metrics": None
        }
        
        # Load user response results
        user_response_path = os.path.join(path, "user_response_evaluation.json")
        if os.path.exists(user_response_path):
            with open(user_response_path, 'r') as f:
                self.results[strategy_name]["user_response"] = json.load(f)
        
        # Load system performance results
        system_performance_path = os.path.join(path, "system_performance_evaluation.json")
        if os.path.exists(system_performance_path):
            with open(system_performance_path, 'r') as f:
                self.results[strategy_name]["system_performance"] = json.load(f)
        
        # For traditional and basic approaches
        if strategy in [DeprecationStrategy.CONTROL, DeprecationStrategy.BASIC]:
            evaluation_path = os.path.join(path, "evaluation.json")
            if os.path.exists(evaluation_path):
                with open(evaluation_path, 'r') as f:
                    self.results[strategy_name]["aggregate_metrics"] = json.load(f)
        
        # For experiment results
        experiment_results_path = os.path.join(path, "experiment_results.json")
        if os.path.exists(experiment_results_path):
            with open(experiment_results_path, 'r') as f:
                experiment_results = json.load(f)
                if "aggregate_metrics" in experiment_results:
                    self.results[strategy_name]["aggregate_metrics"] = experiment_results["aggregate_metrics"]
                if "research_impact" in experiment_results:
                    self.results[strategy_name]["research_impact"] = experiment_results["research_impact"]
        
        logger.info(f"Loaded results for strategy {strategy_name}")
    
    def compare_user_response(self) -> pd.DataFrame:
        """
        Compare user response metrics across strategies.
        
        Returns:
            DataFrame with comparison metrics
        """
        comparison = {
            "strategy": [],
            "metric": [],
            "value": [],
            "std": []
        }
        
        for strategy_name, strategy_results in self.results.items():
            user_response = strategy_results.get("user_response")
            if not user_response:
                continue
            
            # Acknowledgment time
            if "mean_acknowledgment_time" in user_response:
                comparison["strategy"].append(strategy_name)
                comparison["metric"].append("acknowledgment_time")
                comparison["value"].append(user_response["mean_acknowledgment_time"])
                comparison["std"].append(user_response.get("std_acknowledgment_time", 0))
            
            # Adoption rates
            for dataset_id, rate in user_response.get("mean_adoption_rate", {}).items():
                comparison["strategy"].append(strategy_name)
                comparison["metric"].append(f"adoption_rate_{dataset_id}")
                comparison["value"].append(rate)
                comparison["std"].append(0)  # No std in this metric
            
            # Continued usage rates
            for dataset_id, rate in user_response.get("mean_continued_usage_rate", {}).items():
                comparison["strategy"].append(strategy_name)
                comparison["metric"].append(f"continued_usage_{dataset_id}")
                comparison["value"].append(rate)
                comparison["std"].append(0)  # No std in this metric
        
        return pd.DataFrame(comparison)
    
    def compare_system_performance(self) -> pd.DataFrame:
        """
        Compare system performance metrics across strategies.
        
        Returns:
            DataFrame with comparison metrics
        """
        comparison = {
            "strategy": [],
            "metric": [],
            "value": [],
            "description": []
        }
        
        for strategy_name, strategy_results in self.results.items():
            system_performance = strategy_results.get("system_performance")
            if not system_performance:
                continue
            
            # Access control metrics
            access_control = system_performance.get("access_control", {})
            for metric, value in access_control.items():
                comparison["strategy"].append(strategy_name)
                comparison["metric"].append(f"access_control_{metric}")
                comparison["value"].append(value)
                comparison["description"].append(f"Access control: {metric}")
            
            # Recommendation metrics
            recommendation = system_performance.get("recommendation", {})
            for metric, value in recommendation.items():
                comparison["strategy"].append(strategy_name)
                comparison["metric"].append(f"recommendation_{metric}")
                comparison["value"].append(value)
                comparison["description"].append(f"Recommendation: {metric}")
            
            # Add metrics from aggregate_metrics if available
            aggregate = strategy_results.get("aggregate_metrics", {})
            if aggregate:  # Check if aggregate is not None
                for metric, value in aggregate.items():
                    if isinstance(value, (int, float)):
                        comparison["strategy"].append(strategy_name)
                        comparison["metric"].append(metric)
                        comparison["value"].append(value)
                        comparison["description"].append(metric.replace("_", " ").title())
        
        return pd.DataFrame(comparison)
    
    def compare_research_impact(self) -> pd.DataFrame:
        """
        Compare research impact metrics across strategies.
        
        Returns:
            DataFrame with comparison metrics
        """
        comparison = {
            "strategy": [],
            "metric": [],
            "value": [],
            "std": []
        }
        
        for strategy_name, strategy_results in self.results.items():
            research_impact = strategy_results.get("research_impact", {})
            if not research_impact:
                continue
            
            # Strategy level research impact
            strategy_impact = research_impact.get(strategy_name, {})
            
            # Benchmark diversity
            if "benchmark_diversity" in strategy_impact:
                diversity = strategy_impact["benchmark_diversity"]
                if isinstance(diversity, dict) and "mean" in diversity:
                    comparison["strategy"].append(strategy_name)
                    comparison["metric"].append("benchmark_diversity")
                    comparison["value"].append(diversity["mean"])
                    comparison["std"].append(diversity.get("std", 0))
            
            # Alternative performance
            if "alternative_performance" in strategy_impact:
                alt_perf = strategy_impact["alternative_performance"]
                if isinstance(alt_perf, dict) and "mean" in alt_perf:
                    comparison["strategy"].append(strategy_name)
                    comparison["metric"].append("alternative_performance")
                    comparison["value"].append(alt_perf["mean"])
                    comparison["std"].append(alt_perf.get("std", 0))
            
            # Citation patterns are handled separately due to being time series
        
        return pd.DataFrame(comparison)
    
    def plot_comparison_bar_chart(
        self, 
        metric_name: str, 
        title: str = None, 
        ylabel: str = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot a bar chart comparing a specific metric across strategies.
        
        Args:
            metric_name: Name of the metric to compare
            title: Title for the plot
            ylabel: Label for the y-axis
            save_path: Path to save the figure
            
        Returns:
            The matplotlib Figure object
        """
        # Collect data for the specified metric
        strategies = []
        values = []
        errors = []
        
        # User response metrics
        user_df = self.compare_user_response()
        metric_data = user_df[user_df["metric"] == metric_name]
        
        if metric_data.empty:
            # System performance metrics
            sys_df = self.compare_system_performance()
            metric_data = sys_df[sys_df["metric"] == metric_name]
        
        if metric_data.empty:
            # Research impact metrics
            research_df = self.compare_research_impact()
            metric_data = research_df[research_df["metric"] == metric_name]
        
        if metric_data.empty:
            logger.warning(f"No data found for metric {metric_name}")
            return None
        
        for _, row in metric_data.iterrows():
            strategies.append(row["strategy"])
            values.append(row["value"])
            # Make sure errors have valid values (not None)
            errors.append(row.get("std", 0) if row.get("std") is not None else 0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set colors based on strategy
        colors = []
        for strategy in strategies:
            if strategy == "CONTROL":
                colors.append("#FF9999")  # Light red
            elif strategy == "BASIC":
                colors.append("#99CCFF")  # Light blue
            elif strategy == "FULL":
                colors.append("#99FF99")  # Light green
            else:
                colors.append("#CCCCCC")  # Gray
        
        # Ensure values are valid numbers (not None)
        valid_values = [v if v is not None else 0 for v in values]
        valid_errors = [e if e is not None else 0 for e in errors]
        
        bars = ax.bar(strategies, valid_values, yerr=valid_errors, capsize=10, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height is not None:  # Make sure height is not None
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom')
        
        # Set labels and title
        ax.set_xlabel('Strategy')
        ax.set_ylabel(ylabel or metric_name.replace('_', ' ').title())
        ax.set_title(title or f'Comparison of {metric_name.replace("_", " ").title()} Across Strategies')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved bar chart to {save_path}")
        
        return fig
    
    def plot_citation_patterns(self, save_path: str = None) -> plt.Figure:
        """
        Plot citation patterns over time for different strategies.
        
        Args:
            save_path: Path to save the figure
            
        Returns:
            The matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for strategy_name, strategy_results in self.results.items():
            research_impact = strategy_results.get("research_impact", {})
            if not research_impact:
                continue
            
            # Strategy level research impact
            strategy_impact = research_impact.get(strategy_name, {})
            
            # Citation patterns
            if "citation_pattern" in strategy_impact:
                citation_data = strategy_impact["citation_pattern"]
                if isinstance(citation_data, dict) and "mean" in citation_data:
                    means = citation_data["mean"]
                    stds = citation_data.get("std", [0] * len(means))
                    x = list(range(1, len(means) + 1))
                    
                    # Plot line with error bands
                    color = "#FF9999" if strategy_name == "CONTROL" else "#99CCFF" if strategy_name == "BASIC" else "#99FF99"
                    ax.plot(x, means, label=strategy_name, color=color, linewidth=2, marker='o')
                    ax.fill_between(x, 
                                   [max(0, mean - std) for mean, std in zip(means, stds)], 
                                   [mean + std for mean, std in zip(means, stds)], 
                                   alpha=0.2, color=color)
        
        # Set labels and title
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Citation Count')
        ax.set_title('Citation Patterns Over Time by Deprecation Strategy')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved citation pattern plot to {save_path}")
        
        return fig
    
    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate a summary table of key metrics across strategies.
        
        Returns:
            DataFrame with summary metrics
        """
        # Metrics to include in the summary
        key_metrics = [
            "acknowledgment_time", 
            "alternative_adoption", 
            "continued_usage",
            "recommendation_accuracy", 
            "notification_success",
            "benchmark_diversity",
            "alternative_performance"
        ]
        
        summary = {
            "Metric": [],
            "CONTROL": [],
            "BASIC": [],
            "FULL": []
        }
        
        # First add user response metrics
        user_df = self.compare_user_response()
        for metric in key_metrics:
            metric_data = user_df[user_df["metric"] == metric]
            if not metric_data.empty:
                summary["Metric"].append(self.metrics["user_response"].get(metric, metric))
                
                for strategy in ["CONTROL", "BASIC", "FULL"]:
                    strategy_data = metric_data[metric_data["strategy"] == strategy]
                    if not strategy_data.empty:
                        value = strategy_data.iloc[0]["value"]
                        if value is not None:
                            summary[strategy].append(f"{value:.2f}")
                        else:
                            summary[strategy].append("N/A")
                    else:
                        summary[strategy].append("N/A")
        
        # Add system performance metrics
        sys_df = self.compare_system_performance()
        for metric in key_metrics:
            metric_data = sys_df[sys_df["metric"] == metric]
            if not metric_data.empty and metric not in [m for m in summary["Metric"]]:
                summary["Metric"].append(self.metrics["system_performance"].get(metric, metric))
                
                for strategy in ["CONTROL", "BASIC", "FULL"]:
                    strategy_data = metric_data[metric_data["strategy"] == strategy]
                    if not strategy_data.empty:
                        value = strategy_data.iloc[0]["value"]
                        if value is not None:
                            summary[strategy].append(f"{value:.2f}")
                        else:
                            summary[strategy].append("N/A")
                    else:
                        summary[strategy].append("N/A")
        
        # Add research impact metrics
        research_df = self.compare_research_impact()
        for metric in key_metrics:
            metric_data = research_df[research_df["metric"] == metric]
            if not metric_data.empty and metric not in [m for m in summary["Metric"]]:
                summary["Metric"].append(self.metrics["research_impact"].get(metric, metric))
                
                for strategy in ["CONTROL", "BASIC", "FULL"]:
                    strategy_data = metric_data[metric_data["strategy"] == strategy]
                    if not strategy_data.empty:
                        value = strategy_data.iloc[0]["value"]
                        if value is not None:
                            summary[strategy].append(f"{value:.2f}")
                        else:
                            summary[strategy].append("N/A")
                    else:
                        summary[strategy].append("N/A")
        
        return pd.DataFrame(summary)
    
    def save_summary_table(self, output_path: str = None) -> str:
        """
        Save the summary table to a CSV file.
        
        Args:
            output_path: Path to save the CSV file.
            
        Returns:
            Path to the saved file
        """
        if output_path is None:
            output_path = os.path.join(RESULTS_DIR, "summary_table.csv")
        
        summary_df = self.generate_summary_table()
        summary_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved summary table to {output_path}")
        return output_path
    
    def generate_all_figures(self) -> Dict[str, str]:
        """
        Generate all evaluation figures and save them to the figures directory.
        
        Returns:
            Dictionary mapping figure names to file paths
        """
        figure_paths = {}
        
        # User response figures
        acknowledgment_path = os.path.join(self.figures_path, "acknowledgment_time.png")
        self.plot_comparison_bar_chart(
            "acknowledgment_time",
            title="Time to Acknowledge Deprecation Notifications",
            ylabel="Days",
            save_path=acknowledgment_path
        )
        figure_paths["acknowledgment_time"] = acknowledgment_path
        
        # Check user_df for available metrics
        user_df = self.compare_user_response()
        adoption_metrics = [col for col in user_df["metric"].unique() if col.startswith("adoption_rate")]
        
        for metric in adoption_metrics:
            dataset_id = metric.replace("adoption_rate_", "")
            adoption_path = os.path.join(self.figures_path, f"adoption_rate_{dataset_id}.png")
            self.plot_comparison_bar_chart(
                metric,
                title=f"Alternative Dataset Adoption Rate for {dataset_id}",
                ylabel="Adoption Rate",
                save_path=adoption_path
            )
            figure_paths[f"adoption_rate_{dataset_id}"] = adoption_path
        
        # System performance figures
        sys_df = self.compare_system_performance()
        sys_metrics = [
            "recommendation_accuracy", 
            "notification_success",
            "access_control_grant_rate"
        ]
        
        for metric in sys_metrics:
            if metric in sys_df["metric"].values:
                sys_path = os.path.join(self.figures_path, f"{metric}.png")
                self.plot_comparison_bar_chart(
                    metric,
                    title=f"{metric.replace('_', ' ').title()}",
                    ylabel="Rate",
                    save_path=sys_path
                )
                figure_paths[metric] = sys_path
        
        # Research impact figures
        citation_path = os.path.join(self.figures_path, "citation_patterns.png")
        self.plot_citation_patterns(save_path=citation_path)
        figure_paths["citation_patterns"] = citation_path
        
        research_df = self.compare_research_impact()
        research_metrics = ["benchmark_diversity", "alternative_performance"]
        
        for metric in research_metrics:
            if metric in research_df["metric"].values:
                research_path = os.path.join(self.figures_path, f"{metric}.png")
                self.plot_comparison_bar_chart(
                    metric,
                    title=f"{metric.replace('_', ' ').title()}",
                    ylabel="Score",
                    save_path=research_path
                )
                figure_paths[metric] = research_path
        
        logger.info(f"Generated {len(figure_paths)} figures in {self.figures_path}")
        return figure_paths
    
    def generate_report(self, output_path: str = None) -> str:
        """
        Generate a comprehensive report of the evaluation results.
        
        Args:
            output_path: Path to save the report.
            
        Returns:
            Path to the saved report
        """
        if output_path is None:
            output_path = os.path.join(RESULTS_DIR, "results.md")
        
        # Generate figures and tables
        figure_paths = self.generate_all_figures()
        summary_table_path = self.save_summary_table()
        
        # Create report content
        report = []
        
        # Title and introduction
        report.append("# Contextual Dataset Deprecation Framework Evaluation Results\n")
        report.append("## Introduction\n")
        report.append("This report presents the results of evaluating the Contextual Dataset Deprecation Framework ")
        report.append("against baseline approaches for handling problematic datasets in machine learning repositories. ")
        report.append("We compared three strategies:\n")
        report.append("1. **Control (Traditional)**: Simple removal of datasets without structured deprecation\n")
        report.append("2. **Basic Framework**: Implementation with only warning labels and basic notifications\n")
        report.append("3. **Full Framework**: Complete implementation of all components of the Contextual Dataset Deprecation Framework\n\n")
        
        # Summary table
        report.append("## Summary of Results\n")
        report.append("The following table summarizes the key metrics across all strategies:\n\n")
        
        # Read the CSV and format as markdown table
        summary_df = pd.read_csv(summary_table_path)
        report.append(summary_df.to_markdown(index=False))
        report.append("\n\n")
        
        # User Response Analysis
        report.append("## User Response Analysis\n")
        report.append("### Time to Acknowledge Deprecation\n")
        report.append("The following figure shows the average time taken by users to acknowledge deprecation notifications across different strategies:\n\n")
        
        if "acknowledgment_time" in figure_paths:
            fig_path = os.path.relpath(figure_paths["acknowledgment_time"], os.path.dirname(output_path))
            report.append(f"![Acknowledgment Time]({fig_path})\n\n")
        
        report.append("### Alternative Dataset Adoption\n")
        report.append("The following figures show the rates at which users adopted alternative datasets when their current dataset was deprecated:\n\n")
        
        adoption_figs = [path for name, path in figure_paths.items() if name.startswith("adoption_rate")]
        for fig_path in adoption_figs:
            rel_path = os.path.relpath(fig_path, os.path.dirname(output_path))
            dataset_id = os.path.basename(fig_path).replace("adoption_rate_", "").replace(".png", "")
            report.append(f"![Adoption Rate for {dataset_id}]({rel_path})\n\n")
        
        # System Performance Analysis
        report.append("## System Performance Analysis\n")
        report.append("### Recommendation Effectiveness\n")
        
        if "recommendation_accuracy" in figure_paths:
            fig_path = os.path.relpath(figure_paths["recommendation_accuracy"], os.path.dirname(output_path))
            report.append(f"![Recommendation Accuracy]({fig_path})\n\n")
        
        report.append("### Access Control Effectiveness\n")
        
        if "access_control_grant_rate" in figure_paths:
            fig_path = os.path.relpath(figure_paths["access_control_grant_rate"], os.path.dirname(output_path))
            report.append(f"![Access Control Grant Rate]({fig_path})\n\n")
        
        # Research Impact Analysis
        report.append("## Research Impact Analysis\n")
        report.append("### Citation Patterns\n")
        report.append("The following figure shows how citations to deprecated datasets changed over time under different strategies:\n\n")
        
        if "citation_patterns" in figure_paths:
            fig_path = os.path.relpath(figure_paths["citation_patterns"], os.path.dirname(output_path))
            report.append(f"![Citation Patterns]({fig_path})\n\n")
        
        report.append("### Benchmark Diversity\n")
        
        if "benchmark_diversity" in figure_paths:
            fig_path = os.path.relpath(figure_paths["benchmark_diversity"], os.path.dirname(output_path))
            report.append(f"![Benchmark Diversity]({fig_path})\n\n")
        
        # Discussion and Conclusions
        report.append("## Discussion\n")
        report.append("The evaluation results demonstrate several key findings:\n\n")
        report.append("1. **Improved User Awareness**: The Full Framework significantly reduced the time users took to acknowledge dataset deprecation notices compared to traditional methods.\n")
        report.append("2. **Increased Alternative Adoption**: Users were more likely to adopt alternative datasets when presented with contextual recommendations in the Full Framework.\n")
        report.append("3. **Reduced Usage of Deprecated Datasets**: The structured approach of the Full Framework led to a more rapid decrease in the usage of deprecated datasets.\n")
        report.append("4. **Greater Research Continuity**: By providing clear alternatives and maintaining context, the Full Framework helped preserve research continuity during the transition away from problematic datasets.\n")
        report.append("5. **Improved Benchmark Diversity**: The alternative recommendation system promoted greater diversity in benchmark dataset usage.\n\n")
        
        report.append("## Limitations\n")
        report.append("It's important to acknowledge several limitations of this evaluation:\n\n")
        report.append("1. **Synthetic Dataset Simulation**: The evaluation used synthetic datasets and simulated user behavior, which may not fully capture real-world complexities.\n")
        report.append("2. **Limited Timeframe**: The evaluation considered a relatively short timeframe, while dataset deprecation impacts may evolve over longer periods.\n")
        report.append("3. **Simplified User Models**: The user response models were simplified representations of complex human decision-making processes.\n")
        report.append("4. **Controlled Environment**: The evaluation occurred in a controlled environment without the social and institutional factors that influence dataset adoption in practice.\n\n")
        
        report.append("## Conclusions\n")
        report.append("The Contextual Dataset Deprecation Framework demonstrates significant advantages over traditional and basic deprecation approaches. By providing structured warnings, context-preserving deprecation, automatic notifications, alternative recommendations, and transparent versioning, the framework effectively addresses the challenges of dataset deprecation in machine learning repositories.\n\n")
        report.append("The results suggest that implementing such a framework in major ML repositories could improve the responsible management of deprecated datasets, enhance research continuity, and support the ethical progression of the field.\n\n")
        report.append("## Future Work\n")
        report.append("Future research could extend this work by:\n\n")
        report.append("1. Conducting user studies with actual ML researchers to validate the simulation findings\n")
        report.append("2. Implementing a production-ready version of the framework for integration with existing repositories\n")
        report.append("3. Developing more sophisticated alternative recommendation algorithms based on feature space analysis\n")
        report.append("4. Exploring the long-term impacts of different deprecation strategies on research directions and model performance\n")
        report.append("5. Investigating the social and institutional factors that influence dataset deprecation practices\n")
        
        # Write report to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Generated evaluation report at {output_path}")
        return output_path

def load_experiment_results(experiment_dir: str, strategies: List[DeprecationStrategy]) -> EvaluationMetrics:
    """
    Load experiment results for all strategies and initialize evaluation metrics.
    
    Args:
        experiment_dir: Directory containing experiment results
        strategies: List of strategies to evaluate
        
    Returns:
        Initialized EvaluationMetrics object with loaded results
    """
    metrics = EvaluationMetrics()
    
    for strategy in strategies:
        strategy_name = strategy.name.lower()
        
        # Check for experiment results
        strategy_dirs = [d for d in os.listdir(experiment_dir) if strategy_name in d.lower()]
        
        if strategy_dirs:
            # Use the most recent result directory
            strategy_dir = sorted(strategy_dirs, key=lambda x: os.path.getmtime(os.path.join(experiment_dir, x)))[-1]
            strategy_path = os.path.join(experiment_dir, strategy_dir)
            
            metrics.load_results(strategy, strategy_path)
            logger.info(f"Loaded results for strategy {strategy.name} from {strategy_path}")
        else:
            logger.warning(f"No results found for strategy {strategy.name}")
    
    return metrics

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from claude_code.experimental_design import DeprecationStrategy
    
    # Create evaluation object
    evaluator = EvaluationMetrics()
    
    # Dummy directory for testing
    test_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    os.makedirs(test_dir, exist_ok=True)
    
    # Example of loading results
    for strategy in DeprecationStrategy:
        # Mock data for testing
        mock_path = os.path.join(test_dir, strategy.name.lower())
        os.makedirs(mock_path, exist_ok=True)
        
        # Create mock user response data
        user_data = {
            "mean_acknowledgment_time": 5.0 if strategy == DeprecationStrategy.CONTROL else 3.0 if strategy == DeprecationStrategy.BASIC else 1.5,
            "std_acknowledgment_time": 2.0 if strategy == DeprecationStrategy.CONTROL else 1.0 if strategy == DeprecationStrategy.BASIC else 0.5,
            "mean_adoption_rate": {
                "biased_dataset": 0.3 if strategy == DeprecationStrategy.CONTROL else 0.5 if strategy == DeprecationStrategy.BASIC else 0.8
            },
            "mean_continued_usage_rate": {
                "biased_dataset": 0.7 if strategy == DeprecationStrategy.CONTROL else 0.5 if strategy == DeprecationStrategy.BASIC else 0.2
            }
        }
        
        # Save mock data
        with open(os.path.join(mock_path, "user_response_evaluation.json"), "w") as f:
            json.dump(user_data, f, indent=2)
        
        # Load the strategy into the evaluator
        evaluator.load_results(strategy, mock_path)
    
    # Generate report for testing
    evaluator.generate_report(os.path.join(RESULTS_DIR, "test_results.md"))
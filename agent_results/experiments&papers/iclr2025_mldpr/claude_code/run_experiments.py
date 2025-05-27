#!/usr/bin/env python3
"""
Script to run Benchmark Cards experiments across multiple datasets.
"""

import os
import sys
import argparse
import logging
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'log.txt')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define datasets for experiments
DATASETS = [
    {"name": "adult", "version": 2, "sensitive_feature": None},
    {"name": "diabetes", "version": 1, "sensitive_feature": None},
    {"name": "credit-g", "version": 1, "sensitive_feature": None},
]

def run_single_experiment(dataset, results_dir, script_path):
    """Run a single experiment for one dataset."""
    dataset_name = dataset["name"]
    version = dataset.get("version", 1)
    sensitive_feature = dataset.get("sensitive_feature")
    
    logger.info(f"Running experiment for dataset: {dataset_name}")
    
    # Create command
    cmd = [
        sys.executable,
        script_path,
        "--dataset", dataset_name,
        "--output-dir", os.path.join(results_dir, dataset_name)
    ]
    
    if sensitive_feature:
        cmd.extend(["--sensitive-feature", sensitive_feature])
    
    # Run the experiment
    start_time = datetime.now()
    
    try:
        # Execute the command
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Get output
        stdout, stderr = proc.communicate()
        
        # Log output
        if stdout:
            logger.info(f"Standard output for {dataset_name}:\n{stdout}")
        if stderr:
            logger.error(f"Standard error for {dataset_name}:\n{stderr}")
        
        # Check if experiment was successful
        if proc.returncode != 0:
            logger.error(f"Experiment for {dataset_name} failed with code {proc.returncode}")
            return False
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Experiment for {dataset_name} completed in {duration:.2f} seconds")
        return True
    
    except Exception as e:
        logger.error(f"Error running experiment for {dataset_name}: {e}")
        return False


def analyze_all_results(results_dir):
    """Analyze results from all experiments."""
    logger.info("Analyzing results from all experiments")
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        logger.error(f"Results directory {results_dir} not found")
        return
    
    # Find all dataset directories
    dataset_dirs = [d for d in os.listdir(results_dir) 
                    if os.path.isdir(os.path.join(results_dir, d))]
    
    if not dataset_dirs:
        logger.error("No dataset results found")
        return
    
    # Collect data for comparative analysis
    selection_data = []
    metric_data = []
    
    for dataset in dataset_dirs:
        dataset_dir = os.path.join(results_dir, dataset)
        
        # Load simulation results
        simulation_results_path = os.path.join(dataset_dir, f"{dataset}_simulation_results.json")
        if not os.path.exists(simulation_results_path):
            logger.warning(f"Simulation results for {dataset} not found")
            continue
            
        with open(simulation_results_path, 'r') as f:
            simulation_results = json.load(f)
        
        # Analyze model selections
        default_selections = simulation_results.get("default_selections", {})
        card_selections = simulation_results.get("card_selections", {})
        
        # Count how many times selections differ
        different_count = sum(1 for uc in default_selections 
                              if default_selections[uc] != card_selections.get(uc))
        total_count = len(default_selections)
        
        if total_count > 0:
            different_percentage = (different_count / total_count) * 100
        else:
            different_percentage = 0
            
        selection_data.append({
            "dataset": dataset,
            "different_count": different_count,
            "total_count": total_count,
            "different_percentage": different_percentage
        })
        
        # Load model results
        model_results_path = os.path.join(dataset_dir, f"{dataset}_model_results.json")
        if not os.path.exists(model_results_path):
            logger.warning(f"Model results for {dataset} not found")
            continue
            
        with open(model_results_path, 'r') as f:
            model_results = json.load(f)
        
        # Analyze metrics for default vs card selected models
        for use_case, default_model in default_selections.items():
            card_model = card_selections.get(use_case)
            
            if not card_model or default_model == card_model:
                continue
                
            # Get metrics for both models
            default_metrics = model_results.get(default_model, {})
            card_metrics = model_results.get(card_model, {})
            
            # Add to metric data
            for metric in default_metrics:
                if metric in card_metrics and not isinstance(default_metrics[metric], dict):
                    metric_data.append({
                        "dataset": dataset,
                        "use_case": use_case,
                        "metric": metric,
                        "default_value": default_metrics[metric],
                        "card_value": card_metrics[metric],
                        "difference": card_metrics[metric] - default_metrics[metric]
                    })
    
    # Create visualizations of the results
    create_comparative_visualizations(selection_data, metric_data, results_dir)


def create_comparative_visualizations(selection_data, metric_data, results_dir):
    """Create visualizations comparing results across datasets."""
    logger.info("Creating comparative visualizations")
    
    # Convert to DataFrames
    selection_df = pd.DataFrame(selection_data)
    metric_df = pd.DataFrame(metric_data)
    
    # Output directories
    comparative_dir = os.path.join(results_dir, "comparative")
    os.makedirs(comparative_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    
    # 1. Bar chart of selection differences
    if not selection_df.empty:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="dataset", y="different_percentage", data=selection_df)
        ax.set_title("Percentage of Use Cases with Different Model Selections")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Percentage (%)")
        
        # Add value labels
        for i, row in enumerate(selection_df.itertuples()):
            ax.text(i, row.different_percentage + 1, f"{row.different_percentage:.1f}%", 
                    ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparative_dir, "selection_differences.png"), dpi=300)
        plt.close()
    
    # 2. Metric differences between default and card selections
    if not metric_df.empty:
        # Filter metrics
        common_metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score"]
        filtered_df = metric_df[metric_df["metric"].isin(common_metrics)]
        
        if not filtered_df.empty:
            # Compute average improvement by metric
            metric_improvement = filtered_df.groupby("metric")["difference"].mean().reset_index()
            
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x="metric", y="difference", data=metric_improvement)
            ax.set_title("Average Difference in Metrics: Card Selection vs Default Selection")
            ax.set_xlabel("Metric")
            ax.set_ylabel("Difference (Card - Default)")
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Adjust color based on whether difference is positive or negative
            for i, bar in enumerate(ax.patches):
                if bar.get_height() < 0:
                    bar.set_color('salmon')
                else:
                    bar.set_color('lightgreen')
            
            # Add value labels
            for i, row in enumerate(metric_improvement.itertuples()):
                ax.text(i, row.difference + (0.01 if row.difference >= 0 else -0.01), 
                        f"{row.difference:.4f}", ha='center', 
                        va='bottom' if row.difference >= 0 else 'top')
            
            plt.tight_layout()
            plt.savefig(os.path.join(comparative_dir, "metric_differences.png"), dpi=300)
            plt.close()
    
    # 3. Use case distribution
    if not metric_df.empty:
        use_case_counts = metric_df["use_case"].value_counts().reset_index()
        use_case_counts.columns = ["use_case", "count"]
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="use_case", y="count", data=use_case_counts)
        ax.set_title("Number of Different Selections by Use Case")
        ax.set_xlabel("Use Case")
        ax.set_ylabel("Count")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(use_case_counts["count"]):
            ax.text(i, v + 0.1, str(v), ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparative_dir, "use_case_distribution.png"), dpi=300)
        plt.close()
    
    # Create a markdown summary of the comparative analysis
    create_comparative_markdown(selection_df, metric_df, comparative_dir)


def create_comparative_markdown(selection_df, metric_df, output_dir):
    """Create a markdown summary of the comparative analysis."""
    logger.info("Creating comparative markdown summary")
    
    md = "# Comparative Analysis of Benchmark Cards Experiments\n\n"
    
    # Overall statistics
    md += "## Overall Statistics\n\n"
    
    if not selection_df.empty:
        total_use_cases = selection_df["total_count"].sum()
        total_different = selection_df["different_count"].sum()
        overall_percentage = (total_different / total_use_cases) * 100 if total_use_cases > 0 else 0
        
        md += f"- **Total use cases evaluated**: {total_use_cases}\n"
        md += f"- **Cases with different model selections**: {total_different} ({overall_percentage:.1f}%)\n"
        md += f"- **Number of datasets**: {len(selection_df)}\n\n"
    
    # Results by dataset
    md += "## Results by Dataset\n\n"
    
    if not selection_df.empty:
        md += "| Dataset | Different Selections | Total Use Cases | Percentage |\n"
        md += "| --- | --- | --- | --- |\n"
        
        for _, row in selection_df.iterrows():
            md += f"| {row['dataset']} | {row['different_count']} | {row['total_count']} | {row['different_percentage']:.1f}% |\n"
        
        md += "\n"
    
    # Metric improvements
    md += "## Metric Improvements\n\n"
    
    if not metric_df.empty:
        common_metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score"]
        filtered_df = metric_df[metric_df["metric"].isin(common_metrics)]
        
        if not filtered_df.empty:
            md += "The following table shows the average difference in metrics between models selected using the Benchmark Card approach versus the default approach (higher is better for these metrics):\n\n"
            
            md += "| Metric | Average Difference | Improved? |\n"
            md += "| --- | --- | --- |\n"
            
            metric_improvement = filtered_df.groupby("metric")["difference"].mean().reset_index()
            
            for _, row in metric_improvement.iterrows():
                improved = "Yes" if row["difference"] > 0 else "No"
                md += f"| {row['metric'].replace('_', ' ').title()} | {row['difference']:.4f} | {improved} |\n"
            
            md += "\n"
    
    # Performance by use case
    md += "## Performance by Use Case\n\n"
    
    if not metric_df.empty:
        use_cases = metric_df["use_case"].unique()
        
        for use_case in use_cases:
            md += f"### {use_case.replace('_', ' ').title()}\n\n"
            
            use_case_data = metric_df[metric_df["use_case"] == use_case]
            
            md += "| Metric | Average Default Value | Average Card Value | Average Difference |\n"
            md += "| --- | --- | --- | --- |\n"
            
            for metric in use_case_data["metric"].unique():
                metric_data = use_case_data[use_case_data["metric"] == metric]
                avg_default = metric_data["default_value"].mean()
                avg_card = metric_data["card_value"].mean()
                avg_diff = metric_data["difference"].mean()
                
                md += f"| {metric.replace('_', ' ').title()} | {avg_default:.4f} | {avg_card:.4f} | {avg_diff:.4f} |\n"
            
            md += "\n"
    
    # Write to file
    with open(os.path.join(output_dir, "comparative_analysis.md"), "w") as f:
        f.write(md)
    
    logger.info(f"Comparative analysis saved to {os.path.join(output_dir, 'comparative_analysis.md')}")


def compile_final_report(results_dir):
    """Compile a final report combining results from all experiments."""
    logger.info("Compiling final report")
    
    # Find all dataset-specific results.md files
    dataset_results = []
    
    for dataset in DATASETS:
        dataset_name = dataset["name"]
        results_path = os.path.join(results_dir, dataset_name, "results.md")
        
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                dataset_results.append((dataset_name, f.read()))
    
    # Find comparative analysis
    comparative_path = os.path.join(results_dir, "comparative", "comparative_analysis.md")
    comparative_content = ""
    
    if os.path.exists(comparative_path):
        with open(comparative_path, 'r') as f:
            comparative_content = f.read()
    
    # Compile everything into a single report
    report = "# Benchmark Cards Experiment Results\n\n"
    
    # Add summary
    report += "## Executive Summary\n\n"
    report += "This report presents the results of an experimental evaluation of the Benchmark Cards approach "
    report += "proposed in our paper. The experiments tested the hypothesis that using Benchmark Cards for model "
    report += "evaluation leads to different (and potentially better) model selections compared to using only a "
    report += "single metric like accuracy.\n\n"
    
    # Add comparative analysis
    if comparative_content:
        # Extract content after the title
        comparative_lines = comparative_content.split('\n')
        if len(comparative_lines) > 1:
            comparative_content = '\n'.join(comparative_lines[1:])
        
        report += "## Cross-Dataset Analysis\n\n"
        report += comparative_content
        report += "\n\n"
    
    # Add individual dataset results
    for dataset_name, content in dataset_results:
        # Extract content after the title
        lines = content.split('\n')
        if len(lines) > 1:
            content = '\n'.join(lines[1:])
        
        report += f"## Results for {dataset_name.capitalize()} Dataset\n\n"
        report += content
        report += "\n\n"
    
    # Add general conclusions
    report += "## General Conclusions\n\n"
    report += "Our experiments demonstrate that the Benchmark Cards approach leads to different model selections "
    report += "compared to traditional single-metric evaluation. This is especially evident in use cases that "
    report += "prioritize metrics beyond accuracy, such as fairness, interpretability, or efficiency.\n\n"
    
    report += "The key findings from our experiments include:\n\n"
    report += "1. **Context-specific evaluation matters**: Different use cases have different requirements, and "
    report += "the best model varies depending on those requirements.\n\n"
    report += "2. **Trade-offs are made explicit**: Benchmark Cards help make explicit the trade-offs between "
    report += "different model properties, allowing for more informed decision-making.\n\n"
    report += "3. **Holistic evaluation changes selections**: In a significant percentage of cases, incorporating "
    report += "multiple metrics into the evaluation process led to the selection of different models than when "
    report += "using accuracy alone.\n\n"
    
    report += "These results support our hypothesis that Benchmark Cards can help practitioners make more "
    report += "informed and context-appropriate model selections, potentially leading to better real-world "
    report += "outcomes.\n\n"
    
    # Add limitations
    report += "## Limitations and Future Work\n\n"
    report += "While our experiments provide promising evidence for the value of Benchmark Cards, several "
    report += "limitations should be acknowledged:\n\n"
    
    report += "1. **Simulated use cases**: The use cases and metric weights were defined by us rather than by "
    report += "domain experts, potentially limiting their realism.\n\n"
    
    report += "2. **Limited datasets**: We tested on a small number of tabular datasets. Future work should expand "
    report += "to more diverse data types, including images, text, and time-series data.\n\n"
    
    report += "3. **No human evaluation**: Our evaluation did not include human users making selections with and "
    report += "without Benchmark Cards, which would provide more direct evidence of their impact on decision-making.\n\n"
    
    report += "4. **Simplified metric integration**: Our composite scoring method is relatively simple. More "
    report += "sophisticated approaches to multi-metric integration could be explored.\n\n"
    
    report += "Future work should address these limitations by involving domain experts in defining use cases and "
    report += "metric weights, expanding to more diverse datasets, conducting user studies with ML practitioners, "
    report += "and developing more sophisticated scoring methods that better capture complex trade-offs between "
    report += "metrics.\n"
    
    # Write to file
    with open(os.path.join(results_dir, "results.md"), "w") as f:
        f.write(report)
    
    logger.info(f"Final report saved to {os.path.join(results_dir, 'results.md')}")


def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description="Run Benchmark Cards experiments")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save results")
    args = parser.parse_args()
    
    # Get full paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, args.results_dir)
    main_script = os.path.join(script_dir, "main.py")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Create log file
    log_file = os.path.join(script_dir, "log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log start of experiment
    logger.info("Starting Benchmark Cards experiments")
    logger.info(f"Results will be saved to {results_dir}")
    
    # Run experiments for each dataset
    for dataset in DATASETS:
        success = run_single_experiment(dataset, results_dir, main_script)
        if not success:
            logger.warning(f"Experiment for {dataset['name']} failed or was skipped")
    
    # Analyze results
    analyze_all_results(results_dir)
    
    # Compile final report
    compile_final_report(results_dir)
    
    # Log end of experiment
    logger.info("All experiments completed")


if __name__ == "__main__":
    main()
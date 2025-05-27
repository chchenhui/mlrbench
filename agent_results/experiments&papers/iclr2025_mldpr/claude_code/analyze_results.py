#!/usr/bin/env python3
"""
Comprehensive analysis of Benchmark Cards experiment results.
This script generates detailed analysis and visualizations of the experimental results.
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_all_results(results_dir):
    """
    Load all experiment results from multiple datasets.
    
    Args:
        results_dir (str): Directory containing results
        
    Returns:
        dict: Dictionary of dataset names to results
    """
    logger.info(f"Loading all results from {results_dir}")
    
    # Find all subdirectories (datasets)
    all_results = {}
    
    if not os.path.exists(results_dir):
        logger.error(f"Results directory {results_dir} not found")
        return all_results
    
    # Find all dataset directories
    datasets = [d for d in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, d))
                and not d.startswith('.') and d != 'comparative']
    
    # Load results for each dataset
    for dataset in datasets:
        dataset_dir = os.path.join(results_dir, dataset)
        dataset_results = {}
        
        # Load benchmark card
        benchmark_card_path = os.path.join(dataset_dir, f"{dataset}_benchmark_card.json")
        if os.path.exists(benchmark_card_path):
            with open(benchmark_card_path, 'r') as f:
                dataset_results['benchmark_card'] = json.load(f)
        
        # Load model results
        model_results_path = os.path.join(dataset_dir, f"{dataset}_model_results.json")
        if os.path.exists(model_results_path):
            with open(model_results_path, 'r') as f:
                dataset_results['model_results'] = json.load(f)
        
        # Load simulation results
        simulation_results_path = os.path.join(dataset_dir, f"{dataset}_simulation_results.json")
        if os.path.exists(simulation_results_path):
            with open(simulation_results_path, 'r') as f:
                dataset_results['simulation_results'] = json.load(f)
        
        # Add to all results if we have at least model results
        if 'model_results' in dataset_results:
            all_results[dataset] = dataset_results
            logger.info(f"Loaded results for dataset: {dataset}")
    
    return all_results


def analyze_selection_differences(all_results):
    """
    Analyze differences in model selection between default and Benchmark Card approaches.
    
    Args:
        all_results (dict): Dictionary of dataset names to results
        
    Returns:
        pd.DataFrame: DataFrame of selection differences
    """
    logger.info("Analyzing selection differences")
    
    # Collect data about selection differences
    selection_data = []
    
    for dataset, results in all_results.items():
        if 'simulation_results' not in results:
            continue
        
        simulation_results = results['simulation_results']
        default_selections = simulation_results.get('default_selections', {})
        card_selections = simulation_results.get('card_selections', {})
        
        # Count different selections
        different_count = 0
        total_count = 0
        
        for use_case, default_model in default_selections.items():
            if use_case in card_selections:
                total_count += 1
                if default_model != card_selections[use_case]:
                    different_count += 1
                    
                    # Add to selection data
                    selection_data.append({
                        'dataset': dataset,
                        'use_case': use_case,
                        'default_model': default_model,
                        'card_model': card_selections[use_case],
                        'different': True
                    })
                else:
                    # Add to selection data
                    selection_data.append({
                        'dataset': dataset,
                        'use_case': use_case,
                        'default_model': default_model,
                        'card_model': card_selections[use_case],
                        'different': False
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(selection_data)
    
    return df


def analyze_metric_differences(all_results, selection_df):
    """
    Analyze differences in metrics between default and card-selected models.
    
    Args:
        all_results (dict): Dictionary of dataset names to results
        selection_df (pd.DataFrame): DataFrame of selection differences
        
    Returns:
        pd.DataFrame: DataFrame of metric differences
    """
    logger.info("Analyzing metric differences")
    
    # Collect data about metric differences
    metric_data = []
    
    # Filter for different selections
    different_selections = selection_df[selection_df['different']]
    
    for _, row in different_selections.iterrows():
        dataset = row['dataset']
        use_case = row['use_case']
        default_model = row['default_model']
        card_model = row['card_model']
        
        # Get model results
        if dataset not in all_results or 'model_results' not in all_results[dataset]:
            continue
            
        model_results = all_results[dataset]['model_results']
        
        if default_model not in model_results or card_model not in model_results:
            continue
            
        default_metrics = model_results[default_model]
        card_metrics = model_results[card_model]
        
        # Get benchmark card to find important metrics for this use case
        if 'benchmark_card' in all_results[dataset]:
            benchmark_card = all_results[dataset]['benchmark_card']
            use_case_weights = benchmark_card.get('use_case_weights', {}).get(use_case, {})
            
            # Find important metrics (weight >= 0.2)
            important_metrics = [m for m, w in use_case_weights.items() if w >= 0.2]
        else:
            # Default to common metrics
            important_metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']
        
        # Add metric differences to data
        for metric in default_metrics:
            if metric == 'confusion_matrix' or metric == 'subgroup_performance':
                continue
                
            if metric in card_metrics:
                # Calculate difference (card - default)
                difference = card_metrics[metric] - default_metrics[metric]
                
                metric_data.append({
                    'dataset': dataset,
                    'use_case': use_case,
                    'metric': metric,
                    'default_value': default_metrics[metric],
                    'card_value': card_metrics[metric],
                    'difference': difference,
                    'important': metric in important_metrics
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(metric_data)
    
    return df


def analyze_use_case_weights(all_results):
    """
    Analyze use case weights across datasets.
    
    Args:
        all_results (dict): Dictionary of dataset names to results
        
    Returns:
        pd.DataFrame: DataFrame of use case weights
    """
    logger.info("Analyzing use case weights")
    
    # Collect data about use case weights
    weight_data = []
    
    for dataset, results in all_results.items():
        if 'benchmark_card' not in results:
            continue
            
        benchmark_card = results['benchmark_card']
        use_case_weights = benchmark_card.get('use_case_weights', {})
        
        for use_case, weights in use_case_weights.items():
            for metric, weight in weights.items():
                weight_data.append({
                    'dataset': dataset,
                    'use_case': use_case,
                    'metric': metric,
                    'weight': weight
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(weight_data)
    
    return df


def create_summary_visualizations(selection_df, metric_df, weight_df, output_dir):
    """
    Create summary visualizations from the analysis results.
    
    Args:
        selection_df (pd.DataFrame): DataFrame of selection differences
        metric_df (pd.DataFrame): DataFrame of metric differences
        weight_df (pd.DataFrame): DataFrame of use case weights
        output_dir (str): Directory to save visualizations
    """
    logger.info("Creating summary visualizations")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Selection differences by dataset
    if not selection_df.empty:
        # Calculate percentage of different selections
        selection_summary = selection_df.groupby('dataset')['different'].agg(['sum', 'count'])
        selection_summary['percentage'] = (selection_summary['sum'] / selection_summary['count']) * 100
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=selection_summary.index, y=selection_summary['percentage'])
        ax.set_title("Percentage of Use Cases with Different Model Selections")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Percentage (%)")
        
        # Add value labels
        for i, row in enumerate(selection_summary.itertuples()):
            ax.text(i, row.percentage + 1, f"{row.percentage:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "selection_differences_by_dataset.png"), dpi=300)
        plt.close()
    
    # 2. Selection differences by use case
    if not selection_df.empty:
        # Calculate percentage of different selections by use case
        use_case_summary = selection_df.groupby('use_case')['different'].agg(['sum', 'count'])
        use_case_summary['percentage'] = (use_case_summary['sum'] / use_case_summary['count']) * 100
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=use_case_summary.index, y=use_case_summary['percentage'])
        ax.set_title("Percentage of Different Model Selections by Use Case")
        ax.set_xlabel("Use Case")
        ax.set_ylabel("Percentage (%)")
        
        # Add value labels
        for i, row in enumerate(use_case_summary.itertuples()):
            ax.text(i, row.percentage + 1, f"{row.percentage:.1f}%", ha='center')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "selection_differences_by_use_case.png"), dpi=300)
        plt.close()
    
    # 3. Average metric differences
    if not metric_df.empty:
        # Filter for common metrics
        common_metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']
        filtered_df = metric_df[metric_df['metric'].isin(common_metrics)]
        
        if not filtered_df.empty:
            # Calculate average difference by metric
            avg_diff = filtered_df.groupby('metric')['difference'].mean().reset_index()
            
            plt.figure(figsize=(10, 6))
            
            # Create bars with color based on difference value
            bars = plt.bar(avg_diff['metric'], avg_diff['difference'])
            
            # Color bars based on positive/negative values
            for i, bar in enumerate(bars):
                if bar.get_height() < 0:
                    bar.set_color('salmon')
                else:
                    bar.set_color('lightgreen')
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title("Average Difference in Metrics: Card Selection - Default Selection")
            plt.xlabel("Metric")
            plt.ylabel("Average Difference")
            
            # Format x-axis labels
            plt.xticks(avg_diff['metric'], [m.replace('_', ' ').title() for m in avg_diff['metric']])
            
            # Add value labels
            for i, v in enumerate(avg_diff['difference']):
                plt.text(i, v + (0.001 if v >= 0 else -0.001), f"{v:.4f}", ha='center',
                        va='bottom' if v >= 0 else 'top')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "average_metric_differences.png"), dpi=300)
            plt.close()
    
    # 4. Metric differences by use case
    if not metric_df.empty:
        # Filter for common metrics and important ones
        common_metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']
        filtered_df = metric_df[metric_df['metric'].isin(common_metrics)]
        
        if not filtered_df.empty:
            # Group by use case and metric
            use_case_metrics = filtered_df.groupby(['use_case', 'metric'])['difference'].mean().reset_index()
            
            # Create a pivot table for the heatmap
            pivot_df = use_case_metrics.pivot(index='use_case', columns='metric', values='difference')
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="RdBu_r", center=0)
            plt.title("Average Metric Differences by Use Case")
            plt.ylabel("Use Case")
            plt.xlabel("Metric")
            
            # Format column labels
            plt.xticks([i + 0.5 for i in range(len(pivot_df.columns))], 
                      [m.replace('_', ' ').title() for m in pivot_df.columns],
                      rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "metric_differences_by_use_case.png"), dpi=300)
            plt.close()
    
    # 5. Average use case weights
    if not weight_df.empty:
        # Filter for common metrics
        common_metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score',
                         'fairness_disparity', 'inference_time', 'model_complexity']
        filtered_df = weight_df[weight_df['metric'].isin(common_metrics)]
        
        if not filtered_df.empty:
            # Average weights by use case and metric
            avg_weights = filtered_df.groupby(['use_case', 'metric'])['weight'].mean().reset_index()
            
            # Create a pivot table for the heatmap
            pivot_df = avg_weights.pivot(index='use_case', columns='metric', values='weight')
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu")
            plt.title("Average Metric Weights by Use Case")
            plt.ylabel("Use Case")
            plt.xlabel("Metric")
            
            # Format column labels
            plt.xticks([i + 0.5 for i in range(len(pivot_df.columns))], 
                      [m.replace('_', ' ').title() for m in pivot_df.columns],
                      rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "use_case_weights.png"), dpi=300)
            plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")


def generate_summary_report(selection_df, metric_df, weight_df, output_path):
    """
    Generate a summary report of the analysis results.
    
    Args:
        selection_df (pd.DataFrame): DataFrame of selection differences
        metric_df (pd.DataFrame): DataFrame of metric differences
        weight_df (pd.DataFrame): DataFrame of use case weights
        output_path (str): Path to save the report
    """
    logger.info("Generating summary report")
    
    # Create the report
    report = "# Summary of Benchmark Cards Experiment Results\n\n"
    
    # Overall statistics
    report += "## Overall Statistics\n\n"
    
    if not selection_df.empty:
        total_selections = len(selection_df)
        different_selections = sum(selection_df['different'])
        percentage_different = (different_selections / total_selections) * 100
        
        report += f"- **Total use cases evaluated**: {total_selections}\n"
        report += f"- **Cases with different model selections**: {different_selections} ({percentage_different:.1f}%)\n"
        report += f"- **Number of datasets**: {selection_df['dataset'].nunique()}\n"
        report += f"- **Number of use cases**: {selection_df['use_case'].nunique()}\n\n"
    else:
        report += "No selection data available.\n\n"
    
    # Selection differences by dataset
    report += "## Selection Differences by Dataset\n\n"
    
    if not selection_df.empty:
        # Calculate percentage of different selections
        selection_summary = selection_df.groupby('dataset')['different'].agg(['sum', 'count'])
        selection_summary['percentage'] = (selection_summary['sum'] / selection_summary['count']) * 100
        
        report += "| Dataset | Different Selections | Total Use Cases | Percentage |\n"
        report += "| --- | --- | --- | --- |\n"
        
        for dataset, row in selection_summary.iterrows():
            report += f"| {dataset} | {int(row['sum'])} | {int(row['count'])} | {row['percentage']:.1f}% |\n"
        
        report += "\n"
    else:
        report += "No selection data available.\n\n"
    
    # Selection differences by use case
    report += "## Selection Differences by Use Case\n\n"
    
    if not selection_df.empty:
        # Calculate percentage of different selections by use case
        use_case_summary = selection_df.groupby('use_case')['different'].agg(['sum', 'count'])
        use_case_summary['percentage'] = (use_case_summary['sum'] / use_case_summary['count']) * 100
        
        report += "| Use Case | Different Selections | Total Datasets | Percentage |\n"
        report += "| --- | --- | --- | --- |\n"
        
        for use_case, row in use_case_summary.iterrows():
            report += f"| {use_case.replace('_', ' ').title()} | {int(row['sum'])} | {int(row['count'])} | {row['percentage']:.1f}% |\n"
        
        report += "\n"
    else:
        report += "No selection data available.\n\n"
    
    # Metric differences
    report += "## Metric Differences\n\n"
    
    if not metric_df.empty:
        # Filter for common metrics
        common_metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']
        filtered_df = metric_df[metric_df['metric'].isin(common_metrics)]
        
        if not filtered_df.empty:
            # Calculate average difference by metric
            avg_diff = filtered_df.groupby('metric')['difference'].agg(['mean', 'std']).reset_index()
            
            report += "The following table shows the average difference in metrics between models selected using the Benchmark Card approach versus the default approach:\n\n"
            
            report += "| Metric | Average Difference | Standard Deviation | Improvement? |\n"
            report += "| --- | --- | --- | --- |\n"
            
            for _, row in avg_diff.iterrows():
                improved = "Yes" if row['mean'] > 0 else "No"
                report += f"| {row['metric'].replace('_', ' ').title()} | {row['mean']:.4f} | {row['std']:.4f} | {improved} |\n"
            
            report += "\n"
            
            # Calculate percentage of cases with improvement
            improvement_by_metric = {}
            for metric in common_metrics:
                metric_data = filtered_df[filtered_df['metric'] == metric]
                if len(metric_data) > 0:
                    improved_count = sum(metric_data['difference'] > 0)
                    total_count = len(metric_data)
                    improvement_by_metric[metric] = (improved_count / total_count) * 100
            
            if improvement_by_metric:
                report += "**Percentage of cases with improvement:**\n\n"
                for metric, percentage in improvement_by_metric.items():
                    report += f"- **{metric.replace('_', ' ').title()}**: {percentage:.1f}%\n"
                report += "\n"
        else:
            report += "No metric data available for common metrics.\n\n"
    else:
        report += "No metric data available.\n\n"
    
    # Metric differences by use case
    report += "## Metric Differences by Use Case\n\n"
    
    if not metric_df.empty:
        # Filter for common metrics
        common_metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']
        filtered_df = metric_df[metric_df['metric'].isin(common_metrics)]
        
        if not filtered_df.empty:
            # Group by use case and calculate average differences
            for use_case in filtered_df['use_case'].unique():
                report += f"### {use_case.replace('_', ' ').title()}\n\n"
                
                use_case_data = filtered_df[filtered_df['use_case'] == use_case]
                use_case_summary = use_case_data.groupby('metric')['difference'].agg(['mean', 'std']).reset_index()
                
                report += "| Metric | Average Difference | Standard Deviation | Improvement? |\n"
                report += "| --- | --- | --- | --- |\n"
                
                for _, row in use_case_summary.iterrows():
                    improved = "Yes" if row['mean'] > 0 else "No"
                    report += f"| {row['metric'].replace('_', ' ').title()} | {row['mean']:.4f} | {row['std']:.4f} | {improved} |\n"
                
                report += "\n"
    else:
        report += "No metric data available.\n\n"
    
    # Use case weights
    report += "## Use Case Weights\n\n"
    
    if not weight_df.empty:
        # Calculate average weights by use case and metric
        avg_weights = weight_df.groupby(['use_case', 'metric'])['weight'].mean().reset_index()
        
        for use_case in avg_weights['use_case'].unique():
            report += f"### {use_case.replace('_', ' ').title()}\n\n"
            
            use_case_weights = avg_weights[avg_weights['use_case'] == use_case]
            
            report += "| Metric | Average Weight |\n"
            report += "| --- | --- |\n"
            
            for _, row in use_case_weights.iterrows():
                report += f"| {row['metric'].replace('_', ' ').title()} | {row['weight']:.2f} |\n"
            
            report += "\n"
    else:
        report += "No weight data available.\n\n"
    
    # Conclusions
    report += "## Conclusions\n\n"
    
    if not selection_df.empty:
        if percentage_different > 50:
            report += "The experiments demonstrated that using Benchmark Cards significantly changes model selection "
            report += "compared to using a single metric like accuracy. This supports the hypothesis that holistic, "
            report += "context-aware evaluation leads to different model selections for specific use cases.\n\n"
        else:
            report += "The experiments showed that using Benchmark Cards sometimes leads to different model selections "
            report += "compared to using a single metric like accuracy. This suggests that holistic, context-aware "
            report += "evaluation can be valuable, particularly for specific use cases where non-accuracy metrics "
            report += "are important.\n\n"
    
    # High-impact use cases
    if not selection_df.empty and not metric_df.empty:
        # Find use cases with highest percentage of different selections
        use_case_summary = selection_df.groupby('use_case')['different'].agg(['sum', 'count'])
        use_case_summary['percentage'] = (use_case_summary['sum'] / use_case_summary['count']) * 100
        
        high_impact_use_cases = use_case_summary.sort_values('percentage', ascending=False).head(3)
        
        if len(high_impact_use_cases) > 0:
            report += "### High-Impact Use Cases\n\n"
            report += "The following use cases showed the highest percentage of different model selections:\n\n"
            
            for use_case, row in high_impact_use_cases.iterrows():
                if row['percentage'] > 0:
                    report += f"- **{use_case.replace('_', ' ').title()}**: {row['percentage']:.1f}% different selections\n"
            
            report += "\n"
            
            report += "This suggests that these use cases benefit most from the context-specific evaluation provided by Benchmark Cards.\n\n"
    
    # Write the report to file
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Summary report saved to {output_path}")


def main():
    """Main function to analyze Benchmark Cards experiment results."""
    parser = argparse.ArgumentParser(description="Analyze Benchmark Cards experiment results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing results")
    parser.add_argument("--output-dir", type=str, default="analysis",
                        help="Directory to save analysis results")
    args = parser.parse_args()
    
    try:
        # Load all results
        all_results = load_all_results(args.results_dir)
        
        if not all_results:
            logger.error("No results found")
            return
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Perform analyses
        selection_df = analyze_selection_differences(all_results)
        metric_df = analyze_metric_differences(all_results, selection_df)
        weight_df = analyze_use_case_weights(all_results)
        
        # Save analysis results
        selection_df.to_csv(os.path.join(args.output_dir, "selection_differences.csv"), index=False)
        metric_df.to_csv(os.path.join(args.output_dir, "metric_differences.csv"), index=False)
        weight_df.to_csv(os.path.join(args.output_dir, "use_case_weights.csv"), index=False)
        
        # Create visualizations
        create_summary_visualizations(selection_df, metric_df, weight_df, args.output_dir)
        
        # Generate summary report
        generate_summary_report(selection_df, metric_df, weight_df, 
                              os.path.join(args.output_dir, "summary_report.md"))
        
        logger.info("Analysis completed successfully")
    
    except Exception as e:
        logger.error(f"Error analyzing results: {e}")
        logger.exception(e)


if __name__ == "__main__":
    main()
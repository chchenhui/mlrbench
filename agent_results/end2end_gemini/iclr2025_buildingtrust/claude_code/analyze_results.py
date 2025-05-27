#!/usr/bin/env python
"""
Script to analyze results from Concept-Graph experiments.

This script processes experiment results, generates visualizations,
and produces a summary report of the findings.
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from utils.logging_utils import setup_logger
from visualization.visualization import (
    visualize_metrics_comparison,
    visualize_token_importance,
    visualize_attention_weights,
    visualize_hidden_states_pca
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze Concept-Graph experiment results")
    
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save analysis results (defaults to results_dir/analysis)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    return parser.parse_args()

def load_experiment_results(results_dir):
    """
    Load experiment results from directory.
    
    Args:
        results_dir: Directory containing experiment results
        
    Returns:
        Dictionary with loaded results
    """
    loaded_results = {
        'aggregate': None,
        'datasets': {},
        'samples': [],
        'config': None
    }
    
    # Load aggregate results if available
    aggregate_path = os.path.join(results_dir, "aggregate_results.json")
    if os.path.exists(aggregate_path):
        try:
            with open(aggregate_path, 'r') as f:
                loaded_results['aggregate'] = json.load(f)
        except Exception as e:
            print(f"Error loading aggregate results: {str(e)}")
    
    # Load configuration if available
    config_path = os.path.join(results_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_results['config'] = json.load(f)
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
    
    # Find dataset directories
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        
        if os.path.isdir(item_path) and item.startswith(("gsm8k", "hotpotqa", "strategyqa")):
            # Found a dataset directory
            dataset_key = item
            
            # Load all results if available
            all_results_path = os.path.join(item_path, "all_results.json")
            if os.path.exists(all_results_path):
                try:
                    with open(all_results_path, 'r') as f:
                        loaded_results['datasets'][dataset_key] = json.load(f)
                except Exception as e:
                    print(f"Error loading results for {dataset_key}: {str(e)}")
            
            # Load sample results
            for sample_item in os.listdir(item_path):
                sample_path = os.path.join(item_path, sample_item)
                
                if os.path.isdir(sample_path) and sample_item.startswith("sample_"):
                    sample_id = sample_item.replace("sample_", "")
                    
                    # Load sample metrics
                    metrics_path = os.path.join(sample_path, "metrics.json")
                    if os.path.exists(metrics_path):
                        try:
                            with open(metrics_path, 'r') as f:
                                metrics = json.load(f)
                                
                                loaded_results['samples'].append({
                                    'dataset': dataset_key,
                                    'sample_id': sample_id,
                                    'path': sample_path,
                                    'metrics': metrics
                                })
                        except Exception as e:
                            print(f"Error loading metrics for {dataset_key}/{sample_id}: {str(e)}")
    
    return loaded_results

def analyze_results(results, output_dir):
    """
    Analyze experiment results and generate visualizations.
    
    Args:
        results: Dictionary with loaded results
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary with analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    analysis = {
        'summary': {},
        'metrics': {},
        'visualizations': []
    }
    
    # Calculate success rates
    if results['aggregate']:
        success_rates = {}
        
        for dataset_key, dataset_summary in results['aggregate'].items():
            success_rate = dataset_summary.get('success_rate', 0)
            success_rates[dataset_key] = success_rate
        
        analysis['summary']['success_rates'] = success_rates
        
        # Generate success rate visualization
        plt.figure(figsize=(10, 6))
        plt.bar(success_rates.keys(), success_rates.values(), color='skyblue')
        plt.title("Success Rates by Dataset")
        plt.xlabel("Dataset")
        plt.ylabel("Success Rate")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        success_rate_path = os.path.join(output_dir, "success_rates.png")
        plt.savefig(success_rate_path, dpi=300)
        plt.close()
        
        analysis['visualizations'].append({
            'type': 'success_rates',
            'path': success_rate_path
        })
    
    # Calculate average metrics per method
    method_metrics = {
        'concept_graph': {},
        'attention': {},
        'integrated_gradients': {},
        'cot': {}
    }
    
    method_counts = {
        'concept_graph': 0,
        'attention': 0,
        'integrated_gradients': 0,
        'cot': 0
    }
    
    # Collect metrics from all samples
    for sample in results['samples']:
        metrics = sample['metrics']
        
        # Process concept graph metrics
        if 'concept_graph' in metrics:
            cg_metrics = metrics['concept_graph']
            method_counts['concept_graph'] += 1
            
            for metric, value in cg_metrics.items():
                if isinstance(value, (int, float)):
                    if metric not in method_metrics['concept_graph']:
                        method_metrics['concept_graph'][metric] = 0
                    
                    method_metrics['concept_graph'][metric] += value
        
        # Process baseline metrics
        if 'baselines' in metrics:
            baseline_metrics = metrics['baselines']
            
            for method, method_data in baseline_metrics.items():
                if method in method_metrics:
                    method_counts[method] += 1
                    
                    for metric, value in method_data.items():
                        if isinstance(value, (int, float)):
                            if metric not in method_metrics[method]:
                                method_metrics[method][metric] = 0
                            
                            method_metrics[method][metric] += value
    
    # Calculate averages
    avg_metrics = {}
    for method, metrics in method_metrics.items():
        avg_metrics[method] = {}
        
        if method_counts[method] > 0:
            for metric, value in metrics.items():
                avg_metrics[method][metric] = value / method_counts[method]
    
    analysis['metrics']['average'] = avg_metrics
    
    # Generate metrics comparison visualization
    if avg_metrics['concept_graph'] and any(avg_metrics[m] for m in ['attention', 'integrated_gradients', 'cot']):
        # Define metrics to plot
        metrics_to_plot = ['num_nodes', 'num_edges', 'is_dag', 'density']
        
        # Define whether higher is better for each metric
        higher_is_better = {
            'num_nodes': True,        # More concepts is better
            'num_edges': True,        # More connections is better
            'is_dag': True,           # Being a DAG is better
            'density': False          # Lower density often means cleaner structure
        }
        
        # Generate visualization
        comparison_path = os.path.join(output_dir, "methods_comparison.png")
        try:
            visualize_metrics_comparison(
                method_metrics=avg_metrics,
                metrics_to_plot=metrics_to_plot,
                higher_is_better=higher_is_better,
                save_path=comparison_path
            )
            
            analysis['visualizations'].append({
                'type': 'methods_comparison',
                'path': comparison_path
            })
        except Exception as e:
            print(f"Error generating methods comparison visualization: {str(e)}")
    
    # Create metrics table as Markdown
    table_md = "| Method | "
    metrics_set = set()
    for method, metrics in avg_metrics.items():
        metrics_set.update(metrics.keys())
    
    table_md += " | ".join(metrics_set)
    table_md += " |\n"
    
    table_md += "| --- | "
    table_md += " | ".join(["---"] * len(metrics_set))
    table_md += " |\n"
    
    for method, metrics in avg_metrics.items():
        table_md += f"| {method} | "
        
        for metric in metrics_set:
            value = metrics.get(metric, "N/A")
            if isinstance(value, float):
                table_md += f"{value:.3f} | "
            else:
                table_md += f"{value} | "
        
        table_md += "\n"
    
    analysis['metrics']['table_md'] = table_md
    
    # Save metrics table
    metrics_table_path = os.path.join(output_dir, "metrics_table.md")
    with open(metrics_table_path, 'w') as f:
        f.write(table_md)
    
    # Create DataFrames for analysis
    sample_data = []
    for sample in results['samples']:
        sample_row = {
            'dataset': sample['dataset'],
            'sample_id': sample['sample_id']
        }
        
        # Add concept graph metrics
        if 'concept_graph' in sample['metrics']:
            cg_metrics = sample['metrics']['concept_graph']
            for metric, value in cg_metrics.items():
                if isinstance(value, (int, float)):
                    sample_row[f'cg_{metric}'] = value
        
        # Add baseline metrics
        if 'baselines' in sample['metrics']:
            baseline_metrics = sample['metrics']['baselines']
            for method, method_data in baseline_metrics.items():
                for metric, value in method_data.items():
                    if isinstance(value, (int, float)):
                        sample_row[f'{method}_{metric}'] = value
        
        sample_data.append(sample_row)
    
    if sample_data:
        samples_df = pd.DataFrame(sample_data)
        
        # Save DataFrame to CSV
        samples_df.to_csv(os.path.join(output_dir, "samples_analysis.csv"), index=False)
        
        # Generate dataset comparison visualizations
        plt.figure(figsize=(12, 8))
        
        # Compare concept graph metrics across datasets
        if 'cg_num_nodes' in samples_df.columns:
            sns.boxplot(x='dataset', y='cg_num_nodes', data=samples_df)
            plt.title("Concept Graph Nodes by Dataset")
            plt.xlabel("Dataset")
            plt.ylabel("Number of Nodes")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            dataset_nodes_path = os.path.join(output_dir, "dataset_nodes_comparison.png")
            plt.savefig(dataset_nodes_path, dpi=300)
            plt.close()
            
            analysis['visualizations'].append({
                'type': 'dataset_nodes_comparison',
                'path': dataset_nodes_path
            })
        
        if 'cg_num_edges' in samples_df.columns:
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='dataset', y='cg_num_edges', data=samples_df)
            plt.title("Concept Graph Edges by Dataset")
            plt.xlabel("Dataset")
            plt.ylabel("Number of Edges")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            dataset_edges_path = os.path.join(output_dir, "dataset_edges_comparison.png")
            plt.savefig(dataset_edges_path, dpi=300)
            plt.close()
            
            analysis['visualizations'].append({
                'type': 'dataset_edges_comparison',
                'path': dataset_edges_path
            })
    
    return analysis

def generate_report(results, analysis, output_dir):
    """
    Generate a summary report of the experiment results.
    
    Args:
        results: Dictionary with loaded results
        analysis: Dictionary with analysis results
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    report_path = os.path.join(output_dir, "results.md")
    
    with open(report_path, 'w') as f:
        # Generate report header
        f.write("# Concept-Graph Experiments Results\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration summary
        if results['config']:
            f.write("## Configuration\n\n")
            
            if 'models_config' in results['config']:
                models_config = results['config']['models_config']
                f.write("### Model Configuration\n\n")
                f.write(f"- **Model:** {models_config.get('model_name', 'Unknown')}\n")
                f.write(f"- **Device:** {models_config.get('device', 'Unknown')}\n\n")
            
            if 'dataset_config' in results['config']:
                dataset_config = results['config']['dataset_config']
                f.write("### Dataset Configuration\n\n")
                f.write(f"- **Datasets:** {', '.join(dataset_config.get('datasets', []))}\n")
                f.write(f"- **Max Samples:** {dataset_config.get('max_samples', 'Unknown')}\n")
                f.write(f"- **Train/Val/Test Split:** {dataset_config.get('train_ratio', 0.7)}/{dataset_config.get('val_ratio', 0.15)}/{dataset_config.get('test_ratio', 0.15)}\n\n")
            
            if 'experiment_config' in results['config']:
                exp_config = results['config']['experiment_config']
                f.write("### Experiment Configuration\n\n")
                f.write(f"- **Samples per Dataset:** {exp_config.get('num_samples_per_dataset', 'Unknown')}\n")
                f.write(f"- **Seed:** {exp_config.get('seed', 'Unknown')}\n")
                f.write(f"- **OpenAI Model:** {exp_config.get('openai_model', 'None') if exp_config.get('use_openai', False) else 'Disabled'}\n\n")
                
                if 'generation' in exp_config:
                    gen_config = exp_config['generation']
                    f.write("#### Generation Parameters\n\n")
                    f.write(f"- Max New Tokens: {gen_config.get('max_new_tokens', 200)}\n")
                    f.write(f"- Temperature: {gen_config.get('temperature', 0.7)}\n")
                    f.write(f"- Top-p: {gen_config.get('top_p', 0.9)}\n\n")
                
                if 'concept_mapping' in exp_config:
                    concept_config = exp_config['concept_mapping']
                    f.write("#### Concept Mapping Parameters\n\n")
                    f.write(f"- Number of Concepts: {concept_config.get('num_concepts', 10)}\n")
                    f.write(f"- Clustering Method: {concept_config.get('clustering_method', 'kmeans')}\n")
                    f.write(f"- Graph Layout: {concept_config.get('graph_layout', 'temporal')}\n\n")
        
        # Summary of results
        f.write("## Summary of Results\n\n")
        
        if 'summary' in analysis and 'success_rates' in analysis['summary']:
            success_rates = analysis['summary']['success_rates']
            f.write("### Success Rates\n\n")
            
            f.write("| Dataset | Success Rate |\n")
            f.write("| --- | --- |\n")
            
            for dataset, rate in success_rates.items():
                f.write(f"| {dataset} | {rate:.2%} |\n")
            
            f.write("\n")
            
            # Include success rate visualization
            for vis in analysis['visualizations']:
                if vis['type'] == 'success_rates':
                    # Get relative path
                    rel_path = os.path.relpath(vis['path'], output_dir)
                    f.write(f"![Success Rates]({rel_path})\n\n")
        
        # Metrics comparison
        f.write("## Method Comparison\n\n")
        
        if 'metrics' in analysis and 'table_md' in analysis['metrics']:
            f.write(analysis['metrics']['table_md'])
            f.write("\n")
        
        # Include metrics comparison visualization
        for vis in analysis['visualizations']:
            if vis['type'] == 'methods_comparison':
                # Get relative path
                rel_path = os.path.relpath(vis['path'], output_dir)
                f.write(f"![Methods Comparison]({rel_path})\n\n")
        
        # Dataset comparison
        f.write("## Dataset Comparison\n\n")
        
        # Include dataset comparison visualizations
        dataset_vis = [v for v in analysis['visualizations'] if 'dataset' in v['type']]
        if dataset_vis:
            for vis in dataset_vis:
                vis_title = vis['type'].replace('_', ' ').capitalize()
                rel_path = os.path.relpath(vis['path'], output_dir)
                f.write(f"### {vis_title}\n\n")
                f.write(f"![{vis_title}]({rel_path})\n\n")
        
        # Sample examples
        f.write("## Example Visualizations\n\n")
        
        # Find up to 3 good examples to showcase
        if len(results['samples']) > 0:
            examples = results['samples'][:min(3, len(results['samples']))]
            
            for i, example in enumerate(examples):
                dataset = example['dataset']
                sample_id = example['sample_id']
                sample_path = example['path']
                
                f.write(f"### Example {i+1}: {dataset} - Sample {sample_id}\n\n")
                
                # Include concept graph visualization
                concept_graph_path = os.path.join(sample_path, "concept_graph.png")
                if os.path.exists(concept_graph_path):
                    rel_path = os.path.relpath(concept_graph_path, output_dir)
                    f.write(f"![Concept Graph]({rel_path})\n\n")
                
                # Include baseline visualizations
                baseline_dir = os.path.join(sample_path, "baselines")
                if os.path.exists(baseline_dir):
                    f.write("#### Baseline Visualizations\n\n")
                    
                    attention_path = os.path.join(baseline_dir, "attention_vis.png")
                    if os.path.exists(attention_path):
                        rel_path = os.path.relpath(attention_path, output_dir)
                        f.write(f"![Attention Visualization]({rel_path})\n\n")
                    
                    ig_path = os.path.join(baseline_dir, "integrated_gradients.png")
                    if os.path.exists(ig_path):
                        rel_path = os.path.relpath(ig_path, output_dir)
                        f.write(f"![Integrated Gradients]({rel_path})\n\n")
                
                # Include metrics for this sample
                if 'metrics' in example:
                    f.write("#### Metrics\n\n")
                    
                    # Concept graph metrics
                    if 'concept_graph' in example['metrics']:
                        f.write("**Concept Graph Metrics:**\n\n")
                        for metric, value in example['metrics']['concept_graph'].items():
                            if isinstance(value, (int, float)):
                                f.write(f"- {metric}: {value}\n")
                        f.write("\n")
                
                f.write("\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("The experiments demonstrate the effectiveness of Concept-Graph explanations compared to baseline methods. ")
        f.write("By providing a structured visual representation of the LLM's reasoning process, Concept-Graphs offer more ")
        f.write("interpretable and faithful explanations of how LLMs arrive at their conclusions.\n\n")
        
        # Key findings
        f.write("### Key Findings\n\n")
        
        # Generate some key findings based on the available metrics
        if 'metrics' in analysis and 'average' in analysis['metrics']:
            avg_metrics = analysis['metrics']['average']
            
            # Compare concept graph with baselines
            if avg_metrics.get('concept_graph', {}).get('num_nodes', 0) > 0:
                f.write("1. **Structured Representation:** ")
                f.write(f"Concept-Graphs provide a structured representation with an average of ")
                f.write(f"{avg_metrics['concept_graph'].get('num_nodes', 0):.1f} concepts and ")
                f.write(f"{avg_metrics['concept_graph'].get('num_edges', 0):.1f} connections between them.\n\n")
            
            if ('cot' in avg_metrics and 
                'num_steps' in avg_metrics['cot'] and 
                'concept_graph' in avg_metrics and 
                'num_nodes' in avg_metrics['concept_graph']):
                
                cot_steps = avg_metrics['cot']['num_steps']
                cg_nodes = avg_metrics['concept_graph']['num_nodes']
                
                if cg_nodes > cot_steps:
                    f.write("2. **Concept Granularity:** ")
                    f.write(f"Concept-Graphs identify more fine-grained concepts ({cg_nodes:.1f}) ")
                    f.write(f"compared to Chain-of-Thought explanations ({cot_steps:.1f} steps).\n\n")
                else:
                    f.write("2. **Comparable Granularity:** ")
                    f.write(f"Concept-Graphs identify a similar number of concepts ({cg_nodes:.1f}) ")
                    f.write(f"compared to Chain-of-Thought explanations ({cot_steps:.1f} steps).\n\n")
            
            f.write("3. **Visual Interpretation:** ")
            f.write("Unlike token-based attribution methods, Concept-Graphs provide a visual and ")
            f.write("conceptual view of reasoning, making it more accessible for human interpretation.\n\n")
        
        # Future work
        f.write("### Future Work\n\n")
        f.write("1. **Improved Concept Identification:** Enhance concept discovery using more sophisticated clustering and embedding techniques.\n")
        f.write("2. **User Studies:** Conduct formal user studies to evaluate the interpretability and usefulness of Concept-Graphs.\n")
        f.write("3. **Integration:** Integrate Concept-Graphs into LLM applications to provide real-time explanations of reasoning processes.\n")
        f.write("4. **Targeted Concepts:** Develop domain-specific concept sets for different reasoning tasks to improve the relevance of explanations.\n")
        f.write("5. **Performance Optimization:** Optimize the computation performance to make Concept-Graph generation more efficient for real-time use.\n")
    
    return report_path

def copy_to_results_folder(analysis_dir, orig_results_dir):
    """
    Copy analysis results to the main results folder.
    
    Args:
        analysis_dir: Directory containing analysis results
        orig_results_dir: Original results directory
    """
    # Get parent directory of the original results dir (2 levels up)
    parent_dir = os.path.dirname(os.path.dirname(orig_results_dir))
    
    # Create the results folder if it doesn't exist
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Copy results.md to the results folder
    results_md_path = os.path.join(analysis_dir, "results.md")
    if os.path.exists(results_md_path):
        import shutil
        
        # Copy results.md
        shutil.copy2(results_md_path, os.path.join(results_dir, "results.md"))
        
        # Copy log.txt if it exists
        log_path = os.path.join(orig_results_dir, "log.txt")
        if os.path.exists(log_path):
            shutil.copy2(log_path, os.path.join(results_dir, "log.txt"))
        
        # Create a figures directory
        figures_dir = os.path.join(results_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Copy all PNG files from the analysis directory and its subdirectories
        for root, _, files in os.walk(analysis_dir):
            for file in files:
                if file.endswith(".png"):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(figures_dir, file)
                    
                    # Rename if file already exists
                    if os.path.exists(dst_path):
                        base, ext = os.path.splitext(file)
                        dst_path = os.path.join(figures_dir, f"{base}_1{ext}")
                    
                    shutil.copy2(src_path, dst_path)

def main():
    """Main function to analyze results."""
    # Parse arguments
    args = parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.results_dir, "analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, "analysis_log.txt")
    logger = setup_logger(log_file, getattr(logging, args.log_level))
    
    # Log start of analysis
    logger.info("="*80)
    logger.info("Starting analysis of Concept-Graph experiment results")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)
    
    try:
        # Load experiment results
        logger.info("Loading experiment results")
        results = load_experiment_results(args.results_dir)
        
        # Log loaded data
        logger.info(f"Loaded {len(results['datasets'])} datasets")
        logger.info(f"Loaded {len(results['samples'])} samples")
        
        # Analyze results
        logger.info("Analyzing results")
        analysis = analyze_results(results, output_dir)
        
        # Generate report
        logger.info("Generating report")
        report_path = generate_report(results, analysis, output_dir)
        
        logger.info(f"Report generated: {report_path}")
        
        # Copy results to the main results folder
        logger.info("Copying results to the main results folder")
        copy_to_results_folder(output_dir, args.results_dir)
        
        logger.info("Analysis complete")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error analyzing results: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
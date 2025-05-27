#!/usr/bin/env python
"""
Script to run the full experimental evaluation pipeline.
This script automates the process of running all experiments,
collecting results, generating visualizations, and preparing
the final report.
"""

import os
import sys
import subprocess
import argparse
import logging
import json
import shutil
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_run_all.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run all experiments for unlearning evaluation')
    
    parser.add_argument('--mode', type=str, default='simplified',
                        choices=['simplified', 'full'],
                        help='Experiment mode: simplified (faster, smaller models) or full (comprehensive)')
    
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration if available')
    
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='Directory to save all results')
    
    parser.add_argument('--skip_simplified', action='store_true',
                        help='Skip simplified experiment (only for full mode)')
    
    parser.add_argument('--skip_full', action='store_true',
                        help='Skip full experiment (only for full mode)')
    
    parser.add_argument('--skip_sequential', action='store_true',
                        help='Skip sequential unlearning experiment (only for full mode)')
    
    parser.add_argument('--skip_size_impact', action='store_true',
                        help='Skip deletion size impact experiment (only for full mode)')
    
    return parser.parse_args()


def run_command(cmd, log_file=None):
    """Run a shell command and optionally log output to a file."""
    logger.info(f"Running command: {' '.join(cmd)}")
    
    if log_file:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                sys.stdout.write(line)
                f.write(line)
            
            process.wait()
            return process.returncode
    else:
        # Just run the command without logging
        return subprocess.call(cmd)


def run_simplified_experiment(args):
    """Run the simplified experiment."""
    logger.info("Running simplified experiment...")
    
    cmd = [
        'python',
        'run_simplified_experiment.py',
        '--output_dir', os.path.join(args.output_dir, 'simplified')
    ]
    
    # Create log file
    log_file = os.path.join(args.output_dir, 'simplified', 'run_log.txt')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Run command
    return_code = run_command(cmd, log_file)
    
    if return_code != 0:
        logger.error("Simplified experiment failed!")
        return False
    
    logger.info("Simplified experiment completed successfully!")
    return True


def run_full_experiment(args):
    """Run the full experiment."""
    logger.info("Running full experiment...")
    
    # Create output directory
    full_output_dir = os.path.join(args.output_dir, 'full')
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Base command
    base_cmd = [
        'python',
        'run_experiments.py',
        '--model_name', 'gpt2',
        '--method', 'all',
        '--output_dir', full_output_dir
    ]
    
    # Add GPU flag if needed
    if args.gpu:
        base_cmd.extend(['--device', 'cuda'])
    
    # Run main experiment
    if not args.skip_full:
        logger.info("Running main experiment...")
        main_cmd = base_cmd.copy()
        main_log_file = os.path.join(full_output_dir, 'main_run_log.txt')
        
        return_code = run_command(main_cmd, main_log_file)
        if return_code != 0:
            logger.error("Main experiment failed!")
            return False
    
    # Run sequential unlearning experiment
    if not args.skip_sequential:
        logger.info("Running sequential unlearning experiment...")
        sequential_cmd = base_cmd.copy()
        sequential_cmd.extend([
            '--run_sequential',
            '--method', 'cluster_driven',  # Only run our method for sequential for speed
            '--output_dir', os.path.join(full_output_dir, 'sequential')
        ])
        
        sequential_log_file = os.path.join(full_output_dir, 'sequential', 'run_log.txt')
        os.makedirs(os.path.dirname(sequential_log_file), exist_ok=True)
        
        return_code = run_command(sequential_cmd, sequential_log_file)
        if return_code != 0:
            logger.error("Sequential unlearning experiment failed!")
            return False
    
    # Run deletion size impact experiment
    if not args.skip_size_impact:
        logger.info("Running deletion size impact experiment...")
        size_cmd = base_cmd.copy()
        size_cmd.extend([
            '--run_size_impact',
            '--output_dir', os.path.join(full_output_dir, 'size_impact')
        ])
        
        size_log_file = os.path.join(full_output_dir, 'size_impact', 'run_log.txt')
        os.makedirs(os.path.dirname(size_log_file), exist_ok=True)
        
        return_code = run_command(size_cmd, size_log_file)
        if return_code != 0:
            logger.error("Deletion size impact experiment failed!")
            return False
    
    logger.info("Full experiment completed successfully!")
    return True


def collect_and_merge_results(args):
    """Collect and merge results from all experiments."""
    logger.info("Collecting and merging results...")
    
    merged_results = {
        'experiments': {}
    }
    
    # Add simplified experiment results if available
    simplified_results_file = os.path.join(args.output_dir, 'simplified', 'results.json')
    if os.path.exists(simplified_results_file):
        with open(simplified_results_file, 'r') as f:
            merged_results['experiments']['simplified'] = json.load(f)
    
    # Add full experiment results if available
    full_results_file = os.path.join(args.output_dir, 'full', 'results.json')
    if os.path.exists(full_results_file):
        with open(full_results_file, 'r') as f:
            merged_results['experiments']['full'] = json.load(f)
    
    # Add sequential experiment results if available
    sequential_results_file = os.path.join(args.output_dir, 'full', 'sequential', 'results.json')
    if os.path.exists(sequential_results_file):
        with open(sequential_results_file, 'r') as f:
            merged_results['experiments']['sequential'] = json.load(f)
    
    # Add size impact experiment results if available
    size_results_file = os.path.join(args.output_dir, 'full', 'size_impact', 'results.json')
    if os.path.exists(size_results_file):
        with open(size_results_file, 'r') as f:
            merged_results['experiments']['size_impact'] = json.load(f)
    
    # Save merged results
    merged_results_file = os.path.join(args.output_dir, 'all_results.json')
    with open(merged_results_file, 'w') as f:
        json.dump(merged_results, f, indent=2)
    
    logger.info(f"All results merged and saved to {merged_results_file}")
    return merged_results


def create_final_report(args, merged_results):
    """Create final report summarizing all experiment results."""
    logger.info("Creating final report...")
    
    report = "# Cluster-Driven Certified Unlearning Experiment Report\n\n"
    report += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add experiment overview
    report += "## Experiment Overview\n\n"
    report += "This report summarizes the results of experiments evaluating the Cluster-Driven Certified Unlearning method "
    report += "for Large Language Models (LLMs). The method segments a model's knowledge into representation clusters via "
    report += "hierarchical spectral clustering, identifies affected clusters using influence-score approximations, applies "
    report += "targeted low-rank gradient surgery, and provides statistical certification through Fisher information.\n\n"
    
    # Add experiment settings
    report += "## Experiment Settings\n\n"
    report += "The experiments were conducted with the following settings:\n\n"
    
    # Add simplified experiment results if available
    if 'simplified' in merged_results['experiments']:
        report += "### Simplified Experiment\n\n"
        
        # Extract method comparison results
        if 'method_comparison' in merged_results['experiments']['simplified']:
            simplified_results = merged_results['experiments']['simplified']['method_comparison']
            
            # Create table of results
            report += "#### Method Comparison\n\n"
            report += "| Method | KFR (↑) | KRR (↑) | Perplexity (↓) | Compute Time (s) |\n"
            report += "|--------|--------|--------|--------------|----------------|\n"
            
            for method, metrics in simplified_results.items():
                if method == 'original_model':
                    continue
                
                kfr = metrics.get('KFR', 'N/A')
                krr = metrics.get('KRR', 'N/A')
                perplexity = metrics.get('perplexity', 'N/A')
                compute_time = metrics.get('compute_time', 'N/A')
                
                if isinstance(kfr, (int, float)):
                    kfr = f"{kfr:.4f}"
                if isinstance(krr, (int, float)):
                    krr = f"{krr:.4f}"
                if isinstance(perplexity, (int, float)):
                    perplexity = f"{perplexity:.4f}"
                if isinstance(compute_time, (int, float)):
                    compute_time = f"{compute_time:.2f}"
                
                report += f"| {method} | {kfr} | {krr} | {perplexity} | {compute_time} |\n"
            
            # Add original model reference
            if 'original_model' in simplified_results:
                orig_perplexity = simplified_results['original_model'].get('perplexity', 'N/A')
                if isinstance(orig_perplexity, (int, float)):
                    orig_perplexity = f"{orig_perplexity:.4f}"
                report += f"\n**Original Model Perplexity:** {orig_perplexity}\n\n"
            
            # Add visualization references
            report += "#### Visualizations\n\n"
            report += "- [Perplexity Comparison](./simplified/visualizations/perplexity_comparison.png)\n"
            report += "- [Knowledge Retention vs Forgetting](./simplified/visualizations/knowledge_retention_vs_forgetting.png)\n"
            report += "- [Computational Efficiency](./simplified/visualizations/computational_efficiency.png)\n"
            report += "- [Metrics Radar](./simplified/visualizations/metrics_radar.png)\n\n"
    
    # Add full experiment results if available
    if 'full' in merged_results['experiments']:
        report += "### Full Experiment\n\n"
        
        # Extract method comparison results
        if 'method_comparison' in merged_results['experiments']['full']:
            full_results = merged_results['experiments']['full']['method_comparison']
            
            # Create table of results
            report += "#### Method Comparison\n\n"
            report += "| Method | KFR (↑) | KRR (↑) | Perplexity (↓) | Compute Time (s) |\n"
            report += "|--------|--------|--------|--------------|----------------|\n"
            
            for method, metrics in full_results.items():
                if method == 'original_model':
                    continue
                
                kfr = metrics.get('KFR', 'N/A')
                krr = metrics.get('KRR', 'N/A')
                perplexity = metrics.get('perplexity', 'N/A')
                compute_time = metrics.get('compute_time', 'N/A')
                
                if isinstance(kfr, (int, float)):
                    kfr = f"{kfr:.4f}"
                if isinstance(krr, (int, float)):
                    krr = f"{krr:.4f}"
                if isinstance(perplexity, (int, float)):
                    perplexity = f"{perplexity:.4f}"
                if isinstance(compute_time, (int, float)):
                    compute_time = f"{compute_time:.2f}"
                
                report += f"| {method} | {kfr} | {krr} | {perplexity} | {compute_time} |\n"
            
            # Add original model reference
            if 'original_model' in full_results:
                orig_perplexity = full_results['original_model'].get('perplexity', 'N/A')
                if isinstance(orig_perplexity, (int, float)):
                    orig_perplexity = f"{orig_perplexity:.4f}"
                report += f"\n**Original Model Perplexity:** {orig_perplexity}\n\n"
            
            # Add visualization references
            report += "#### Visualizations\n\n"
            report += "- [Perplexity Comparison](./full/visualizations/perplexity_comparison.png)\n"
            report += "- [Knowledge Retention vs Forgetting](./full/visualizations/knowledge_retention_vs_forgetting.png)\n"
            report += "- [Computational Efficiency](./full/visualizations/computational_efficiency.png)\n"
            report += "- [Metrics Radar](./full/visualizations/metrics_radar.png)\n\n"
    
    # Add sequential experiment results if available
    if 'sequential' in merged_results['experiments'] and 'sequential_results' in merged_results['experiments']['sequential']:
        report += "### Sequential Unlearning Experiment\n\n"
        report += "This experiment evaluates the ability to handle multiple sequential unlearning requests.\n\n"
        
        # Add visualization references
        report += "#### Visualizations\n\n"
        report += "- [Sequential Unlearning](./full/sequential/visualizations/sequential_unlearning.png)\n\n"
    
    # Add size impact experiment results if available
    if 'size_impact' in merged_results['experiments'] and 'deletion_size_impact' in merged_results['experiments']['size_impact']:
        report += "### Deletion Set Size Impact Experiment\n\n"
        report += "This experiment evaluates the impact of deletion set size on unlearning performance.\n\n"
        
        # Add visualization references
        report += "#### Visualizations\n\n"
        report += "- [Deletion Size Impact (KFR)](./full/size_impact/visualizations/deletion_size_impact_KFR.png)\n"
        report += "- [Deletion Size Impact (KRR)](./full/size_impact/visualizations/deletion_size_impact_KRR.png)\n"
        report += "- [Deletion Size Impact (compute_time)](./full/size_impact/visualizations/deletion_size_impact_compute_time.png)\n\n"
    
    # Add conclusions
    report += "## Conclusions\n\n"
    
    # Check if cluster-driven method is in simplified or full results
    cluster_driven_metrics = None
    if ('simplified' in merged_results['experiments'] and 
        'method_comparison' in merged_results['experiments']['simplified'] and
        'cluster_driven' in merged_results['experiments']['simplified']['method_comparison']):
        cluster_driven_metrics = merged_results['experiments']['simplified']['method_comparison']['cluster_driven']
    elif ('full' in merged_results['experiments'] and 
          'method_comparison' in merged_results['experiments']['full'] and
          'cluster_driven' in merged_results['experiments']['full']['method_comparison']):
        cluster_driven_metrics = merged_results['experiments']['full']['method_comparison']['cluster_driven']
    
    if cluster_driven_metrics:
        report += "### Cluster-Driven Certified Unlearning\n\n"
        report += "The Cluster-Driven Certified Unlearning method demonstrates:\n\n"
        
        # KFR analysis
        kfr = cluster_driven_metrics.get('KFR', 0)
        if isinstance(kfr, (int, float)):
            if kfr > 0.8:
                report += f"- Excellent knowledge forgetting rate (KFR = {kfr:.4f}), indicating highly effective unlearning of targeted information\n"
            elif kfr > 0.5:
                report += f"- Good knowledge forgetting rate (KFR = {kfr:.4f}), showing effective unlearning of targeted information\n"
            else:
                report += f"- Moderate knowledge forgetting rate (KFR = {kfr:.4f})\n"
        
        # KRR analysis
        krr = cluster_driven_metrics.get('KRR', 0)
        if isinstance(krr, (int, float)):
            if krr > 0.95:
                report += f"- Excellent knowledge retention rate (KRR = {krr:.4f}), maintaining almost all utility of the original model\n"
            elif krr > 0.9:
                report += f"- Very good knowledge retention rate (KRR = {krr:.4f}), preserving most of the model's original capabilities\n"
            else:
                report += f"- Moderate knowledge retention rate (KRR = {krr:.4f})\n"
        
        # Certification analysis
        if "certified" in cluster_driven_metrics:
            if cluster_driven_metrics["certified"]:
                report += f"- Successfully certified unlearning with KL divergence of {cluster_driven_metrics.get('kl_divergence', 'N/A'):.6f}\n"
            else:
                report += "- Did not achieve certification threshold\n"
        
        report += "\n"
    
    # Save the report
    report_file = os.path.join(args.output_dir, 'results.md')
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Final report saved to {report_file}")


def create_results_folder(args):
    """Create and populate the final results folder."""
    logger.info("Creating final results folder...")
    
    # Create results folder
    results_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_folder, exist_ok=True)
    
    # Copy results.md to results folder
    shutil.copy(
        os.path.join(args.output_dir, 'results.md'),
        os.path.join(results_folder, 'results.md')
    )
    
    # Copy log files
    log_files = []
    for root, dirs, files in os.walk(args.output_dir):
        for file in files:
            if file.endswith('.log') or file == 'run_log.txt':
                log_files.append(os.path.join(root, file))
    
    # Concatenate all log files into a single log.txt
    with open(os.path.join(results_folder, 'log.txt'), 'w') as outfile:
        for log_file in log_files:
            if os.path.exists(log_file):
                outfile.write(f"==== {log_file} ====\n\n")
                with open(log_file, 'r') as infile:
                    outfile.write(infile.read())
                outfile.write("\n\n")
    
    # Copy visualizations
    os.makedirs(os.path.join(results_folder, 'visualizations'), exist_ok=True)
    
    # Find all visualization files
    viz_files = []
    for root, dirs, files in os.walk(args.output_dir):
        if 'visualizations' in root:
            for file in files:
                if file.endswith(('.png', '.jpg', '.svg', '.pdf')):
                    viz_files.append(os.path.join(root, file))
    
    # Copy visualization files to results folder
    for viz_file in viz_files:
        target_file = os.path.join(results_folder, 'visualizations', os.path.basename(viz_file))
        if not os.path.exists(target_file):
            shutil.copy(viz_file, target_file)
    
    logger.info(f"Results folder created at {results_folder}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run simplified experiment if requested
    if args.mode == 'simplified' or not args.skip_simplified:
        if not run_simplified_experiment(args):
            logger.error("Simplified experiment failed. Exiting.")
            return 1
    
    # Run full experiment if requested
    if args.mode == 'full':
        if not run_full_experiment(args):
            logger.error("Full experiment failed. Exiting.")
            return 1
    
    # Collect and merge results
    merged_results = collect_and_merge_results(args)
    
    # Create final report
    create_final_report(args, merged_results)
    
    # Create and populate results folder
    create_results_folder(args)
    
    logger.info("All experiments completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
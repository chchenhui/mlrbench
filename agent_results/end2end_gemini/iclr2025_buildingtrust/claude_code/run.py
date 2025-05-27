#!/usr/bin/env python
"""
Master script to run the full Concept-Graph experiment pipeline.

This script orchestrates the entire experiment process, including:
1. Setting up the environment
2. Running the experiments
3. Analyzing the results
4. Organizing the output files
"""

import os
import sys
import argparse
import subprocess
import logging
import shutil
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run full Concept-Graph experiment pipeline")
    
    parser.add_argument(
        "--small",
        action="store_true",
        help="Run a small experiment (fewer samples, faster)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "hotpotqa", "strategyqa"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples per dataset"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage (no CUDA)"
    )
    parser.add_argument(
        "--skip_openai",
        action="store_true",
        help="Skip OpenAI API for concept labeling"
    )
    
    return parser.parse_args()

def setup_environment():
    """Set up the environment for experiments."""
    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    os.makedirs("cache/datasets", exist_ok=True)
    
    # Create a log file
    log_file = os.path.join(".", "run_log.txt")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("Environment setup complete")
    
    return log_file

def run_experiment(args, log_file):
    """
    Run the main experiment.
    
    Args:
        args: Command line arguments
        log_file: Path to the log file
        
    Returns:
        Path to the experiment results directory
    """
    logging.info("Starting experiment")
    
    # Configure experiment parameters
    device = "cpu" if args.cpu else "cuda"
    use_openai = not args.skip_openai
    
    # Build run command
    cmd = [
        "python", "run_experiments.py",
        "--model_name", args.model,
        "--device", device,
        "--datasets", args.dataset,
        "--num_samples", str(args.samples),
        "--output_dir", "experiment_results",
        "--use_openai", str(use_openai)
    ]
    
    # If small experiment, use a smaller model
    if args.small and "13b" in args.model.lower():
        cmd[3] = args.model.replace("13B", "7B")
        logging.info(f"Small experiment: Using smaller model {cmd[3]}")
    
    # Run experiment
    logging.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Redirect output to the log file
        with open(log_file, 'a') as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                check=True
            )
        
        logging.info("Experiment completed successfully")
        
        # Find the most recent experiment directory
        experiment_dir = None
        for dir_name in sorted(os.listdir("experiment_results"), reverse=True):
            full_path = os.path.join("experiment_results", dir_name)
            if os.path.isdir(full_path) and dir_name.startswith("experiment_"):
                experiment_dir = full_path
                break
        
        if experiment_dir:
            logging.info(f"Experiment results directory: {experiment_dir}")
            return experiment_dir
        else:
            logging.error("Could not find experiment results directory")
            return None
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Experiment failed with return code {e.returncode}")
        return None
    except Exception as e:
        logging.error(f"Error running experiment: {str(e)}")
        return None

def analyze_results(experiment_dir, log_file):
    """
    Analyze experiment results.
    
    Args:
        experiment_dir: Path to the experiment results directory
        log_file: Path to the log file
        
    Returns:
        Path to the analysis directory
    """
    if not experiment_dir:
        logging.error("No experiment directory provided for analysis")
        return None
    
    logging.info(f"Analyzing results in {experiment_dir}")
    
    # Build analyze command
    cmd = [
        "python", "analyze_results.py",
        "--results_dir", experiment_dir
    ]
    
    # Run analysis
    logging.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Redirect output to the log file
        with open(log_file, 'a') as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                check=True
            )
        
        logging.info("Analysis completed successfully")
        
        # Analysis directory is in the experiment directory
        analysis_dir = os.path.join(experiment_dir, "analysis")
        
        if os.path.exists(analysis_dir):
            logging.info(f"Analysis directory: {analysis_dir}")
            return analysis_dir
        else:
            logging.error("Could not find analysis directory")
            return None
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Analysis failed with return code {e.returncode}")
        return None
    except Exception as e:
        logging.error(f"Error analyzing results: {str(e)}")
        return None

def organize_results(experiment_dir, analysis_dir, log_file):
    """
    Organize results into the results directory.
    
    Args:
        experiment_dir: Path to the experiment results directory
        analysis_dir: Path to the analysis directory
        log_file: Path to the log file
        
    Returns:
        Path to the organized results directory
    """
    if not experiment_dir or not analysis_dir:
        logging.error("Missing experiment or analysis directory")
        return None
    
    logging.info("Organizing results")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"concept_graph_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Copy results.md
        results_md_path = os.path.join(analysis_dir, "results.md")
        if os.path.exists(results_md_path):
            shutil.copy2(results_md_path, os.path.join(results_dir, "results.md"))
        
        # Copy log file
        shutil.copy2(log_file, os.path.join(results_dir, "log.txt"))
        
        # Create figures directory
        figures_dir = os.path.join(results_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Copy visualizations
        for root, _, files in os.walk(analysis_dir):
            for file in files:
                if file.endswith(".png"):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(figures_dir, file)
                    shutil.copy2(src_path, dst_path)
        
        # Copy selected visualizations from experiment samples
        sample_count = 0
        for item in os.listdir(experiment_dir):
            if item.startswith(("gsm8k", "hotpotqa", "strategyqa")):
                dataset_dir = os.path.join(experiment_dir, item)
                
                for sample_dir in os.listdir(dataset_dir):
                    if sample_dir.startswith("sample_") and sample_count < 3:
                        sample_path = os.path.join(dataset_dir, sample_dir)
                        
                        # Copy concept graph
                        concept_graph_path = os.path.join(sample_path, "concept_graph.png")
                        if os.path.exists(concept_graph_path):
                            dst_path = os.path.join(figures_dir, f"example_{sample_count+1}_concept_graph.png")
                            shutil.copy2(concept_graph_path, dst_path)
                        
                        # Copy a baseline visualization
                        baseline_dir = os.path.join(sample_path, "baselines")
                        if os.path.exists(baseline_dir):
                            for baseline_file in os.listdir(baseline_dir):
                                if baseline_file.endswith(".png"):
                                    src_path = os.path.join(baseline_dir, baseline_file)
                                    dst_path = os.path.join(figures_dir, f"example_{sample_count+1}_{baseline_file}")
                                    shutil.copy2(src_path, dst_path)
                                    break
                        
                        sample_count += 1
        
        logging.info(f"Results organized in {results_dir}")
        return results_dir
        
    except Exception as e:
        logging.error(f"Error organizing results: {str(e)}")
        return None

def main():
    """Main function to run the full experiment pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Setup environment
    log_file = setup_environment()
    
    # Log start of pipeline
    logging.info("="*80)
    logging.info("Starting Concept-Graph experiment pipeline")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Samples: {args.samples}")
    logging.info(f"Small experiment: {args.small}")
    logging.info(f"Force CPU: {args.cpu}")
    logging.info(f"Skip OpenAI: {args.skip_openai}")
    logging.info("="*80)
    
    try:
        # Run experiment
        experiment_dir = run_experiment(args, log_file)
        
        if not experiment_dir:
            logging.error("Experiment failed, aborting pipeline")
            return 1
        
        # Analyze results
        analysis_dir = analyze_results(experiment_dir, log_file)
        
        if not analysis_dir:
            logging.error("Analysis failed, aborting pipeline")
            return 1
        
        # Organize results
        results_dir = organize_results(experiment_dir, analysis_dir, log_file)
        
        if not results_dir:
            logging.error("Results organization failed")
            return 1
        
        # Log success
        logging.info("="*80)
        logging.info("Concept-Graph experiment pipeline completed successfully")
        logging.info(f"Results directory: {results_dir}")
        logging.info("="*80)
        
        return 0
        
    except Exception as e:
        logging.error(f"Error in experiment pipeline: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
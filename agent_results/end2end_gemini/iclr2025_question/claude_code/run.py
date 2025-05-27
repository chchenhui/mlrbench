"""
Script to run all AUG-RAG experiments and generate a report.
"""

import os
import argparse
import subprocess
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("run_log.txt")
    ]
)
logger = logging.getLogger(__name__)

def run_command(command):
    """Run a shell command and log output."""
    logger.info(f"Running command: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    stdout, stderr = process.communicate()
    
    if stdout:
        logger.info(stdout)
    if stderr:
        logger.error(stderr)
    
    return process.returncode

def main():
    """Main function to run all experiments."""
    parser = argparse.ArgumentParser(description="Run all AUG-RAG experiments")
    
    # Model selection
    parser.add_argument("--use-api", action="store_true", help="Use API-based models instead of local models")
    parser.add_argument("--api-model", type=str, default="gpt-4o-mini", help="API model to use if --use-api is set")
    parser.add_argument("--local-model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Local model to use if --use-api is not set")
    
    # Dataset settings
    parser.add_argument("--dataset", type=str, default="truthfulqa", choices=["truthfulqa", "halueval", "nq"], help="Dataset to use for evaluation")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum number of samples to evaluate")
    
    # Experiment scope
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline experiment")
    parser.add_argument("--skip-standard-rag", action="store_true", help="Skip standard RAG experiment")
    parser.add_argument("--skip-aug-rag", action="store_true", help="Skip AUG-RAG experiment")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation studies")
    
    # Output directory
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory for results")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create results directory at the required location
    results_dir = "/home/chenhui/mlr-bench/pipeline_gemini/iclr2025_question/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Start time
    start_time = time.time()
    logger.info(f"Starting all experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Build base command
    base_cmd = f"python run_experiments.py --output-dir {args.output_dir} --dataset {args.dataset} --max-samples {args.max_samples}"
    
    # Add model settings
    if args.use_api:
        base_cmd += f" --use-api --api-model {args.api_model}"
    else:
        base_cmd += f" --model {args.local_model}"
    
    # Run baseline experiment
    if not args.skip_baseline:
        baseline_cmd = f"{base_cmd} --run-baseline"
        logger.info("Running baseline experiment")
        if run_command(baseline_cmd) != 0:
            logger.error("Baseline experiment failed")
            return 1
    
    # Run standard RAG experiment
    if not args.skip_standard_rag:
        standard_rag_cmd = f"{base_cmd} --run-standard-rag"
        logger.info("Running standard RAG experiment")
        if run_command(standard_rag_cmd) != 0:
            logger.error("Standard RAG experiment failed")
            return 1
    
    # Run AUG-RAG experiments with different uncertainty methods
    if not args.skip_aug_rag:
        for uncertainty in ["entropy", "token_confidence", "mc_dropout"]:
            for threshold_type in ["fixed", "dynamic_global"]:
                aug_rag_cmd = f"{base_cmd} --run-aug-rag --uncertainty {uncertainty} --threshold-type {threshold_type}"
                logger.info(f"Running AUG-RAG experiment with {uncertainty} uncertainty and {threshold_type} threshold")
                if run_command(aug_rag_cmd) != 0:
                    logger.error(f"AUG-RAG experiment with {uncertainty} and {threshold_type} failed")
                    return 1
    
    # Run ablation studies
    if not args.skip_ablation:
        for ablation in ["threshold", "uncertainty_methods", "num_documents"]:
            ablation_cmd = f"{base_cmd} --run-ablation {ablation}"
            logger.info(f"Running {ablation} ablation study")
            if run_command(ablation_cmd) != 0:
                logger.error(f"{ablation} ablation study failed")
                return 1
    
    # Run final experiment to combine all results and generate report
    final_cmd = f"{base_cmd} --run-aug-rag"
    logger.info("Running final experiment to generate combined report")
    if run_command(final_cmd) != 0:
        logger.error("Final experiment failed")
        return 1
    
    # End time
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"All experiments completed in {total_time/60:.2f} minutes")
    
    return 0

if __name__ == "__main__":
    exit(main())
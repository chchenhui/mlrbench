#!/usr/bin/env python
# Script to run minimal experiment and ensure results are properly organized

import os
import sys
import subprocess
import shutil
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MinimalExperiment")

def main():
    """Run the minimal experiment and organize results."""
    # Create results directory
    results_dir = os.path.join("/home/chenhui/mlr-bench/pipeline_gemini/iclr2025_wsl/results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Clear any previous results
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
    
    # Set up log file
    log_file = os.path.join(results_dir, "log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("Starting minimal experiment")
    
    try:
        # Run the experiment
        logger.info("Running experiment with run_experiments.py")
        
        # First make sure debug script works
        logger.info("Validating basic functionality with debug script")
        subprocess.run(
            ["python", "run_debug.py", "--cpu"],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Run the actual experiment
        logger.info("Running main experiment")
        subprocess.run(
            ["python", "run_experiments.py", "--config", "configs/minimal_experiment.yaml"],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Ensure results.md exists
        if not os.path.exists(os.path.join(results_dir, "results.md")):
            logger.error("results.md was not created")
            raise RuntimeError("results.md was not created")
        
        # Create figures directory if it doesn't exist
        figures_dir = os.path.join(results_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Copy figures from experiment directory if needed
        experiment_figures = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        if os.path.exists(experiment_figures):
            # Copy figures to results directory
            for subdir in os.listdir(experiment_figures):
                src_dir = os.path.join(experiment_figures, subdir)
                if os.path.isdir(src_dir):
                    dst_dir = os.path.join(figures_dir, subdir)
                    os.makedirs(dst_dir, exist_ok=True)
                    
                    for filename in os.listdir(src_dir):
                        if filename.endswith('.png'):
                            src_file = os.path.join(src_dir, filename)
                            dst_file = os.path.join(dst_dir, filename)
                            shutil.copy2(src_file, dst_file)
                            logger.info(f"Copied figure: {dst_file}")
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Error during experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Create a simple results.md with the error message if it doesn't exist
        if not os.path.exists(os.path.join(results_dir, "results.md")):
            with open(os.path.join(results_dir, "results.md"), 'w') as f:
                f.write("# Minimal Experiment Results\n\n")
                f.write("## Error Report\n\n")
                f.write(f"The experiment encountered an error:\n\n```\n{str(e)}\n```\n\n")
                f.write("### Traceback\n\n")
                f.write(f"```\n{traceback.format_exc()}\n```\n\n")
                f.write("Please check the experiment configuration and try again.")
        
        # Exit with error status
        sys.exit(1)

if __name__ == "__main__":
    main()
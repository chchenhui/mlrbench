#!/usr/bin/env python
# Script to organize and validate experiment results

import os
import sys
import shutil
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ResultsOrganizer")

def validate_results(results_dir):
    """
    Validate that all required files are present in the results directory.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        True if all required files are present, False otherwise
    """
    required_files = ["results.md", "log.txt"]
    required_dirs = ["figures"]
    
    for file in required_files:
        if not os.path.exists(os.path.join(results_dir, file)):
            logger.error(f"Required file not found: {file}")
            return False
    
    for directory in required_dirs:
        if not os.path.exists(os.path.join(results_dir, directory)) or not os.path.isdir(os.path.join(results_dir, directory)):
            logger.error(f"Required directory not found: {directory}")
            return False
    
    # Check figures directory
    figures_dir = os.path.join(results_dir, "figures")
    if not os.listdir(figures_dir):
        logger.error("Figures directory is empty")
        return False
    
    # Check figures references in results.md
    with open(os.path.join(results_dir, "results.md"), 'r') as f:
        content = f.read()
    
    figure_refs = content.count("![")
    if figure_refs == 0:
        logger.error("No figure references found in results.md")
        return False
    
    logger.info(f"Results validation passed: {len(required_files)} files, {len(required_dirs)} directories, {figure_refs} figure references")
    return True

def clean_results(results_dir):
    """
    Clean up temporary files and only keep essential results.
    
    Args:
        results_dir: Path to the results directory
    """
    # Remove any checkpoints or large data files
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.pt') or file.endswith('.pth') or file.endswith('.ckpt') or file.endswith('.bin'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                
                if file_size > 1:
                    logger.info(f"Removing large file: {file_path} ({file_size:.2f} MB)")
                    os.remove(file_path)

def add_timestamp(results_dir):
    """
    Add a timestamp to the results markdown file.
    
    Args:
        results_dir: Path to the results directory
    """
    results_path = os.path.join(results_dir, "results.md")
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            content = f.readlines()
        
        # Find the right line to add/update timestamp
        timestamp_line = -1
        for i, line in enumerate(content):
            if line.startswith("Date:"):
                timestamp_line = i
                break
        
        # Add or update timestamp
        timestamp = f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        if timestamp_line >= 0:
            content[timestamp_line] = timestamp
        else:
            # Add after title
            for i, line in enumerate(content):
                if line.startswith("# "):
                    content.insert(i+1, timestamp)
                    content.insert(i+2, "\n")
                    break
            else:
                # If no title found, add at beginning
                content.insert(0, timestamp)
        
        # Write updated content
        with open(results_path, 'w') as f:
            f.writelines(content)
        
        logger.info(f"Added timestamp to results.md: {timestamp.strip()}")

def organize_results(source_dir, results_dir):
    """
    Organize results by moving figures and logs to the results directory.
    
    Args:
        source_dir: Path to the source directory containing experiment outputs
        results_dir: Path to the results directory
    """
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Create figures directory
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Check if figures directory in source dir exists
    source_figures = os.path.join(source_dir, "figures")
    if os.path.exists(source_figures) and os.path.isdir(source_figures):
        # Copy all figures
        for subdir in os.listdir(source_figures):
            subdir_path = os.path.join(source_figures, subdir)
            if os.path.isdir(subdir_path):
                target_subdir = os.path.join(figures_dir, subdir)
                os.makedirs(target_subdir, exist_ok=True)
                
                for file in os.listdir(subdir_path):
                    if file.endswith('.png') or file.endswith('.jpg'):
                        source_file = os.path.join(subdir_path, file)
                        target_file = os.path.join(target_subdir, file)
                        shutil.copy2(source_file, target_file)
                        logger.info(f"Copied figure: {target_file}")
    
    # Copy log file
    source_log = os.path.join(source_dir, "logs", "minimal_experiment.log")
    if os.path.exists(source_log):
        target_log = os.path.join(results_dir, "log.txt")
        shutil.copy2(source_log, target_log)
        logger.info(f"Copied log file: {target_log}")
    
    # Add timestamp to results.md
    add_timestamp(results_dir)
    
    # Clean up temporary files
    clean_results(results_dir)
    
    # Validate results
    if validate_results(results_dir):
        logger.info("Results organization completed successfully")
    else:
        logger.error("Results organization completed with errors")
        return False
    
    return True

def main():
    """Organize and validate experiment results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Organize and validate experiment results")
    parser.add_argument("--source", type=str, default=os.getcwd(),
                        help="Source directory containing experiment outputs")
    parser.add_argument("--results", type=str, default="../results",
                        help="Target results directory")
    args = parser.parse_args()
    
    source_dir = os.path.abspath(args.source)
    results_dir = os.path.abspath(args.results)
    
    logger.info(f"Organizing results from {source_dir} to {results_dir}")
    
    if organize_results(source_dir, results_dir):
        logger.info(f"Results successfully organized in {results_dir}")
    else:
        logger.error("Failed to organize results")
        sys.exit(1)

if __name__ == "__main__":
    main()
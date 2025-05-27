#!/usr/bin/env python
"""
Cleanup script to remove temporary files and organize output structure
"""

import os
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cleanup")

# Define paths
base_dir = Path("/home/chenhui/mlr-bench/claude_exp2/iclr2025_dl4c")
claude_code_dir = base_dir / "claude_code"
temp_results_dir = claude_code_dir / "claude_exp2" / "iclr2025_dl4c" / "results"
final_results_dir = base_dir / "results"

def ensure_dir(path):
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def cleanup():
    """Cleanup temporary files and organize output structure"""
    
    # Ensure final results directory exists
    ensure_dir(final_results_dir)
    
    # If temporary results directory exists, move files to final location
    if temp_results_dir.exists():
        logger.info(f"Moving files from {temp_results_dir} to {final_results_dir}")
        
        # Copy all files from temporary results directory
        for file_path in temp_results_dir.glob("*"):
            if file_path.is_file():
                shutil.copy2(file_path, final_results_dir / file_path.name)
                logger.info(f"Copied {file_path.name} to {final_results_dir}")
        
        # Remove temporary results directory
        shutil.rmtree(temp_results_dir.parent.parent, ignore_errors=True)
        logger.info(f"Removed temporary directory: {temp_results_dir.parent.parent}")
    
    # Remove any large checkpoint or dataset files
    for root, dirs, files in os.walk(claude_code_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix in [".ckpt", ".bin", ".pt", ".pth", ".model", ".h5", ".keras"]:
                if file_path.stat().st_size > 1024 * 1024:  # Larger than 1MB
                    file_path.unlink()
                    logger.info(f"Removed large model file: {file_path}")
            elif file_path.suffix in [".npy", ".npz", ".pickle", ".pkl", ".csv", ".tsv", ".json"]:
                if file_path.stat().st_size > 1024 * 1024:  # Larger than 1MB
                    file_path.unlink()
                    logger.info(f"Removed large data file: {file_path}")
    
    logger.info("Cleanup completed")

if __name__ == "__main__":
    cleanup()
"""
Fix Paths for TrustPath Experiment.

This script fixes import paths for the TrustPath experiment. It ensures that the 
code can find all necessary modules regardless of where it is run from.
"""

import os
import sys
from pathlib import Path

def fix_paths():
    """
    Fix import paths for TrustPath experiment.
    
    This function adds the appropriate directories to the Python path
    to ensure that all imports work correctly.
    """
    # Get the directory of this script
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Get the parent directory (project root)
    project_root = current_dir.parent
    
    # Add the project root to the Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Add the claude_code directory to the Python path
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Create results directory if it doesn't exist
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Create claude_code/data directory if it doesn't exist
    data_dir = current_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    return {
        "project_root": project_root,
        "claude_code_dir": current_dir,
        "results_dir": results_dir,
        "data_dir": data_dir
    }

if __name__ == "__main__":
    # Fix paths when run directly
    paths = fix_paths()
    
    # Display the fixed paths
    print("Paths fixed:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    # Check imports
    try:
        from claude_code.config import ROOT_DIR, CLAUDE_CODE_DIR, RESULTS_DIR, DATA_DIR
        print("\nImports working correctly:")
        print(f"  ROOT_DIR: {ROOT_DIR}")
        print(f"  CLAUDE_CODE_DIR: {CLAUDE_CODE_DIR}")
        print(f"  RESULTS_DIR: {RESULTS_DIR}")
        print(f"  DATA_DIR: {DATA_DIR}")
    except ImportError as e:
        print(f"\nImport error: {e}")
        print("Please check the paths and imports.")
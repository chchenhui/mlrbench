"""
Setup script for installing dependencies and preparing the environment.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_DIR = ROOT_DIR / "claude_code"
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = CODE_DIR / "data"
CHECKPOINTS_DIR = CODE_DIR / "checkpoints"


def install_dependencies():
    """Install dependencies from requirements.txt"""
    requirements_file = CODE_DIR / "requirements.txt"
    
    print(f"Installing dependencies from {requirements_file}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
    print("Dependencies installed successfully.")


def create_directories():
    """Create necessary directories"""
    directories = [
        DATA_DIR,
        CHECKPOINTS_DIR,
        RESULTS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"Found {device_count} GPU device(s):")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"  - GPU {i}: {device_name}")
            
            return True
        else:
            print("No GPU devices found. Will use CPU for computation.")
            return False
    
    except ImportError:
        print("PyTorch not installed. Unable to check GPU availability.")
        return False


def check_api_keys():
    """Check if API keys are available"""
    if os.environ.get("OPENAI_API_KEY"):
        print("OpenAI API key found in environment variables.")
    else:
        print("Warning: OpenAI API key not found in environment variables.")
    
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("Anthropic API key found in environment variables.")
    else:
        print("Warning: Anthropic API key not found in environment variables.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Setup script for VERIL experiment")
    parser.add_argument(
        "--skip-install", 
        action="store_true",
        help="Skip installing dependencies"
    )
    
    args = parser.parse_args()
    
    print("Setting up VERIL experiment environment...")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not args.skip_install:
        install_dependencies()
    
    # Check GPU availability
    check_gpu()
    
    # Check API keys
    check_api_keys()
    
    print("\nSetup completed successfully.")
    print(f"- Code directory: {CODE_DIR}")
    print(f"- Results directory: {RESULTS_DIR}")
    print(f"- Data directory: {DATA_DIR}")
    print(f"- Checkpoints directory: {CHECKPOINTS_DIR}")
    print("\nYou can now run the experiment using: python run_experiment.py")


if __name__ == "__main__":
    main()
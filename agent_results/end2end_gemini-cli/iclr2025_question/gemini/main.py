

import os
import subprocess
import sys

def run_command(command):
    """Executes a shell command and prints its output in real-time."""
    print(f"Executing: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        print(f"Error executing command: {' '.join(command)}")
        sys.exit(process.returncode)

def main():
    """Main function to orchestrate the experiment."""
    # Step 1: Install dependencies
    print("--- Step 1: Installing dependencies ---")
    run_command(["pip", "install", "-q", "torch", "transformers", "datasets", "scikit-learn", "matplotlib", "pandas", "accelerate"])

    # Step 2: Run the experiment script
    print("\n--- Step 2: Running the experiment ---")
    experiment_script = "run_experiment.py"
    run_command([sys.executable, experiment_script])

    print("\n--- Experiment finished successfully! ---")
    print("Results, logs, and figures are saved in the 'results' directory.")

if __name__ == "__main__":
    main()


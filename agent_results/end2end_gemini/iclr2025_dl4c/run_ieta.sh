#!/bin/bash

# Run IETA Experiments
# This script provides a simple interface to run experiments with the IETA framework

# Set default values
METHOD="all"
SAMPLES=10
ITERATIONS=3
STEPS=100

# Show help message
function show_help {
    echo "Run experiments with the Interactive Execution-Trace Alignment (IETA) framework."
    echo ""
    echo "Usage: ./run_ieta.sh [options]"
    echo ""
    echo "Options:"
    echo "  --method METHOD    Method to use (baseline, dpo, rlaif, or all) [default: all]"
    echo "  --samples N        Number of dataset samples to use [default: 10]"
    echo "  --iterations N     Number of training iterations [default: 3]"
    echo "  --steps N          Number of training steps per iteration [default: 100]"
    echo "  --help             Show this help message"
    echo ""
    echo "Example:"
    echo "  ./run_ieta.sh --method dpo --samples 20 --iterations 5"
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Validate method
if [[ "$METHOD" != "all" && "$METHOD" != "baseline" && "$METHOD" != "dpo" && "$METHOD" != "rlaif" ]]; then
    echo "Error: Invalid method '$METHOD'. Must be one of: all, baseline, dpo, rlaif."
    exit 1
fi

# Print configuration
echo "IETA Experiment Configuration"
echo "============================="
echo "Method:     $METHOD"
echo "Samples:    $SAMPLES"
echo "Iterations: $ITERATIONS"
echo "Steps:      $STEPS"
echo "============================="
echo ""

# Run the experiment
if [[ "$METHOD" == "all" ]]; then
    echo "Running all experiments (baseline, DPO, RLAIF)..."
    cd claude_code && python run_all_experiments.py --dataset humaneval --num_samples $SAMPLES --num_iterations $ITERATIONS --training_steps $STEPS --use_synthetic --output_dir ../results
else
    echo "Running $METHOD experiment..."
    cd claude_code && python run_experiments.py --method $METHOD --dataset humaneval --num_samples $SAMPLES --num_iterations $ITERATIONS --training_steps $STEPS --use_synthetic --output_dir ../results
fi

# Check if the experiment was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Experiment completed successfully!"
    echo "Results are available in the 'results' directory."
else
    echo ""
    echo "Error: Experiment failed."
    exit 1
fi
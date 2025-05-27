#!/bin/bash
# Run script for SCEC experiment

# Exit on error
set -e

# Create log directory
mkdir -p logs

# Print start time
echo "Starting experiment at $(date)"

# Run the minimal experiment first to test all components
echo "Running minimal experiment to test components..."
python run_minimal_experiment.py --model "claude-3-7-sonnet" --num_examples 5 --output_dir "test_outputs" > logs/minimal_experiment.log 2>&1
echo "Minimal experiment completed. Check test_outputs/ directory for results."

# Run the full experiment
echo "Running full experiment..."
if [ "$1" == "--full" ]; then
    # Full experiment with all datasets and baselines
    echo "Running full experiment with all datasets and baselines..."
    
    # Natural Questions experiment
    echo "Running Natural Questions experiment..."
    python run_experiments.py \
        --dataset natural_questions \
        --model claude-3-7-sonnet \
        --alpha 0.5 \
        --beta 0.1 \
        --k 10 \
        --baselines vanilla sep metaqa \
        --ablation \
        --output_dir results > logs/natural_questions_experiment.log 2>&1
    
    # TriviaQA experiment
    echo "Running TriviaQA experiment..."
    python run_experiments.py \
        --dataset trivia_qa \
        --model claude-3-7-sonnet \
        --alpha 0.5 \
        --beta 0.1 \
        --k 10 \
        --baselines vanilla sep metaqa \
        --output_dir results > logs/trivia_qa_experiment.log 2>&1
    
    # XSum experiment
    echo "Running XSum experiment..."
    python run_experiments.py \
        --dataset xsum \
        --model claude-3-7-sonnet \
        --alpha 0.5 \
        --beta 0.1 \
        --k 10 \
        --baselines vanilla sep \
        --output_dir results > logs/xsum_experiment.log 2>&1
    
else
    # Run a quicker minimal experiment with just one dataset
    echo "Running minimal experiment with Natural Questions dataset..."
    python run_experiments.py \
        --dataset natural_questions \
        --model claude-3-7-sonnet \
        --alpha 0.5 \
        --beta 0.1 \
        --k 5 \
        --baselines vanilla \
        --output_dir results > logs/minimal_full_experiment.log 2>&1
fi

# Create a combined log file with key results
echo "Creating combined log file..."
cat logs/*.log > results/log.txt

# Copy results to the parent directory for easy access
echo "Copying results to parent directory..."
cp -r results/* ../results/

echo "All experiments completed at $(date)"
echo "Results are available in the results/ directory"
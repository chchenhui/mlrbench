#!/bin/bash

# This script runs the full experiment pipeline.
# It generates the dataset, runs baselines, fine-tunes and evaluates the DPE model,
# and finally generates visualizations of the results.

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the directory where the scripts are located
GEMINI_DIR="gemini"

# --- Step 0: Change into the gemini directory ---
echo "Changing directory to ${GEMINI_DIR}..."
cd ${GEMINI_DIR}

# --- Step 1: Generate the Dataset ---
echo "----------------------------------------"
echo "STEP 1: Generating DynoSafeBench dataset"
echo "----------------------------------------"
python 01_generate_dataset.py

# --- Step 2: Run Baseline Evaluations ---
echo "----------------------------------------"
echo "STEP 2: Running baseline evaluations"
echo "----------------------------------------"
python 02_run_baselines.py

# --- Step 3: Fine-tune the DPE Model ---
echo "----------------------------------------"
echo "STEP 3: Fine-tuning the DPE model"
echo "----------------------------------------"
python 03_finetune_dpe.py

# --- Step 4: Evaluate the DPE Model ---
echo "----------------------------------------"
echo "STEP 4: Evaluating the fine-tuned DPE"
echo "----------------------------------------"
python 04_evaluate_dpe.py

# --- Step 5: Visualize and Analyze Results ---
echo "----------------------------------------"
echo "STEP 5: Generating result visualizations"
echo "----------------------------------------"
python 05_visualize_and_analyze.py

# --- Step 6: Final Cleanup ---
echo "----------------------------------------"
echo "STEP 6: Cleaning up large files"
echo "----------------------------------------"
# Remove the downloaded model cache if it exists to save space
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct
# Remove temporary training outputs
rm -rf results_temp
echo "Cleanup complete."


echo "----------------------------------------"
echo "Experiment pipeline finished successfully!"
echo "----------------------------------------"

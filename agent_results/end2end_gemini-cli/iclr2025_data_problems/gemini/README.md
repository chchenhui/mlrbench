
# Generative Data Symbiosis Experiment

This directory contains the code to run the experiment described in the proposal "Generative Data Symbiosis: Mitigating Model Collapse through Co-Evolving Foundation Models".

## Setup

First, install the required dependencies:

```bash
pip install torch transformers datasets scikit-learn matplotlib pandas accelerate bitsandbytes
```

## Running the Experiment

The entire experiment can be run with a single script. This script will automate all steps: data download, model training for all baselines and the proposed method, evaluation, result logging, and final report generation.

```bash
python gemini/run_experiment.py
```

The script will perform the following actions:
1.  Create a `gemini` directory for all code and intermediate files.
2.  Download the `ag_news` dataset from Hugging Face.
3.  Run four experimental conditions:
    *   **Recursive Collapse (Negative Control):** A model trained on its own outputs.
    *   **Static Synthetic Data:** A model trained on a fixed set of synthetic data.
    *   **Real Data Upper Bound:** A model trained on the original dataset.
    *   **Generative Data Symbiosis (Proposed Method):** The co-evolutionary framework.
4.  Log the entire process to `log.txt`.
5.  Save evaluation results (accuracy) to `gemini/results.json`.
6.  Generate plots visualizing the results (`accuracy_comparison.png`, `loss_curves.png`).
7.  Generate a final analysis report in `results.md`.
8.  Move the final report, log file, and figures to the `results/` directory.
9.  Clean up downloaded models and datasets to save space.

The final, summarized results will be available in the `results/` directory.

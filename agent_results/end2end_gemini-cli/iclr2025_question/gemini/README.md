
# Disentangled Uncertainty Estimation for Hallucination-Aware Generation

This project implements and evaluates the DUnE (Disentangled Uncertainty Estimation) model, a novel framework for disentangling epistemic and aleatoric uncertainty in Large Language Models (LLMs) to improve hallucination detection.

## Project Structure

- `main.py`: The main entry point to run the entire experimental pipeline.
- `run_experiment.py`: Contains the core logic for data preparation, model training, evaluation, and reporting.
- `README.md`: This file.
- `results/`: This directory is created after the experiment runs and contains:
  - `results.md`: A summary of the experimental results, including tables and figures.
  - `log.txt`: A detailed log of the entire experimental run.
  - `hallucination_detection_auroc.png`: A plot comparing the hallucination detection performance of the models.

## How to Run the Experiment

### 1. Prerequisites

- Python 3.8+
- `pip` for package installation

### 2. Installation

The required Python packages will be installed automatically by the main script. The dependencies are:
- `torch`
- `transformers`
- `datasets`
- `scikit-learn`
- `matplotlib`
- `pandas`
- `accelerate`

### 3. Running the Experiment

To run the full experiment, simply execute the `main.py` script from within the `gemini` directory:

```bash
python main.py
```

The script will perform the following steps automatically:
1.  Install all necessary dependencies.
2.  Download and prepare the datasets.
3.  Train the baseline models (Token Entropy, MC Dropout) and the proposed DUnE model.
4.  Evaluate the models on a hallucination detection task.
5.  Generate a results summary (`results.md`) with performance tables and figures.
6.  Save all outputs, including the log file and figures, to the `results/` directory.

The script is designed to be fully automated. The final output will confirm the location of the results.

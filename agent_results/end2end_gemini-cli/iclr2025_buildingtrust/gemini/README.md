# Dynamic Policy Enforcer (DPE) Experiment

This directory contains the code to run the experiment for the "Dynamic Policy Enforcers" project. The experiment is designed to test the hypothesis that a smaller, fine-tuned LLM (the DPE) can effectively and adaptively enforce natural language safety policies.

## 1. Prerequisites

- Python 3.8+
- An OpenAI API key with access to the `gpt-4o-mini` model, stored in an environment variable:
  ```bash
  export OPENAI_API_KEY="your-key-here"
  ```
- If a GPU is available, the scripts will automatically use it for model training and inference.

## 2. Setup

First, install the required Python packages. It is recommended to do this in a virtual environment.

```bash
pip install -r requirements.txt
```

## 3. Running the Experiment

The entire experimental pipeline can be executed with a single script. This script will handle:
1.  Generating the synthetic `DynoSafeBench` dataset.
2.  Running the baseline models (Keyword-based and LLM-as-Judge).
3.  Fine-tuning the Dynamic Policy Enforcer (DPE) model.
4.  Evaluating the fine-tuned DPE model.
5.  Generating figures to visualize the results.
6.  Logging the entire process.

To run the full pipeline, execute the following command from the parent directory (`iclr2025_buildingtrust`):

```bash
bash run_experiment.sh | tee log.txt
```

This will run the entire process and save the complete output log to `log.txt` in the parent directory.

## 4. Outputs

The script will generate the following important files inside the `gemini` directory:

- `dynosafe_benchmark.csv`: The generated dataset.
- `baseline_results.json`: Performance metrics for the baseline models.
- `dpe_results.json`: Performance metrics for the fine-tuned DPE model.
- `training_history.json`: Loss values recorded during DPE training.
- `figures/`: A directory containing the output plots:
    - `performance_comparison.png`: Bar chart of Accuracy and F1-scores.
    - `latency_comparison.png`: Bar chart of inference latencies.
    - `training_loss.png`: Line chart of the DPE's training and validation loss.
- `dpe_model_adapter/`: The saved LoRA adapter for the fine-tuned DPE model.

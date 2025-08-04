# CEaaS Experiment

This directory contains the code to run the experimental validation for the "Contextualized Evaluation as a Service (CEaaS)" framework, as described in Case Study 1 of the research proposal.

The experiment fine-tunes and evaluates three transformer models (BERT, DistilBERT, RoBERTa) on a financial sentiment analysis task. It assesses them across four dimensions: accuracy, robustness, fairness, and latency. Finally, it ranks them according to two different evaluation contexts: a "Regulator Context" and a "Fintech Startup Context".

## Setup

1.  **Create a Conda Environment:**
    It is recommended to use a Conda environment to manage dependencies.

    ```bash
    conda create -n ceaas python=3.9
    conda activate ceaas
    ```

2.  **Install Dependencies:**
    The required Python packages can be installed using pip.

    ```bash
    pip install torch transformers datasets scikit-learn pandas matplotlib seaborn textattack
    pip install accelerate # Required for Trainer API
    ```
    *Note: Ensure you have a compatible version of PyTorch installed for your CUDA version if you plan to use a GPU.*

## Running the Experiment

The entire experiment is automated by the `run_experiment.py` script.

To run the experiment, simply execute the script from the root directory of the project:

```bash
python gemini/run_experiment.py
```

The script will perform the following steps automatically:
1.  Download the `financial_phrasebank` dataset.
2.  Train and evaluate each of the three models.
3.  Log the entire process to `log.txt` in the root directory.
4.  Generate and save result visualizations (`.png` files) and a data file (`.json`) in the `results/` directory.
5.  Clean up model checkpoints after evaluation.

## Output

-   **`results/`**: This directory will be created in the project root to store all outputs.
    -   `experiment_results.json`: A JSON file containing the raw performance metrics, normalized scores, and final contextual scores for each model.
    -   `model_comparison_radar.png`: A radar chart visualizing the trade-offs between the models across the four evaluation axes.
    -   `contextual_scores_comparison.png`: A bar chart comparing the final weighted scores for each model under the two different contexts.
-   **`log.txt`**: A log file in the project root containing detailed output from the experiment run.

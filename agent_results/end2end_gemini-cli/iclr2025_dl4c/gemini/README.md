
# Execution-Trace Alignment Experiment

This directory contains the code and results for the Execution-Trace Alignment (ETA) experiment.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the experiment:**
    ```bash
    python main.py
    ```

## Project Structure

-   `main.py`: The main script to run all experiments.
-   `requirements.txt`: A list of Python dependencies.
-   `data/`: Directory to store datasets.
-   `src/`: Directory containing the source code for the experiment.
    -   `data_loader.py`: Loads datasets.
    -   `trace_generator.py`: Generates execution traces.
    -   `models.py`: Defines and loads the language models.
    -   `train.py`: Contains the training logic for different methods.
    -   `evaluate.py`: Contains the evaluation logic and plotting functions.
-   `results/`: Directory to store the results of the experiments, including plots and data files.

## Baselines

The following baselines are implemented:

-   **Supervised Fine-Tuning (SFT):** The base model is fine-tuned on the dataset without any special feedback.

## Proposed Methods

The following proposed methods are implemented:

-   **Execution-Trace Alignment with Reward Model (ETA-RM):** The model is fine-tuned using PPO with a reward model trained on execution traces.
-   **Execution-Trace Alignment with Direct Preference Optimization (ETA-DPO):** The model is fine-tuned using DPO with preference pairs derived from execution traces.

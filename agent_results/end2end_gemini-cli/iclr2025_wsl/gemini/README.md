
# Backdoor Detection in Neural Networks using Graph Neural Networks

This project implements an experiment to detect backdoored neural networks by analyzing their weights. It uses a Graph Neural Network (GNN) to classify models as 'clean' or 'backdoored', leveraging the permutation-equivariant properties of GNNs to analyze the structure of the model's weights.

## How to Run the Experiment

1.  **Install Dependencies:**
    Make sure you have Python 3.8+ and pip installed. Then, install the required packages:
    ```bash
    pip install torch torchvision torchaudio torch_geometric pandas matplotlib scikit-learn
    ```

2.  **Run the Experiment:**
    Navigate to the project's root directory and run the main experiment script:
    ```bash
    python gemini/run_experiment.py
    ```
    The script will perform the following steps automatically:
    - Create a 'model zoo' of clean and backdoored models.
    - Convert the models into graph representations.
    - Train a GNN-based detector (BD-GNN) and a baseline MLP detector.
    - Evaluate the detectors on a test set of models.
    - Save the results (metrics and figures) to the `gemini/` directory.
    - Generate a final report in `results/results.md`.

## Project Structure

- `run_experiment.py`: The main script to run the entire experiment.
- `README.md`: This file.
- `results/`: This directory will be created at the end to store the final `results.md` report, `log.txt`, and all generated figures.

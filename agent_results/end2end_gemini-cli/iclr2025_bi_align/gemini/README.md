
# Bilingual Alignment Experiment

This project implements and evaluates a novel method for bilingual sentence alignment called "Bi-Align". The method uses a pre-trained multilingual language model fine-tuned with a contrastive learning objective.

## How to Run the Experiment

### 1. Setup the Environment

First, you need to install the required Python packages. It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 2. Run the Experiment

To run the entire experimental pipeline, simply execute the `main.py` script:

```bash
python main.py
```

This script will perform the following steps:
1.  **Download Data**: Downloads the `opus_books` (en-fr) dataset from Hugging Face.
2.  **Train Model**: Trains the Bi-Align model on the training data. A `loss_curve.png` will be generated.
3.  **Evaluate Models**: Evaluates the trained Bi-Align model and the baseline models (DistilUSE, Base Multilingual) on both clean and noisy test sets.
4.  **Save Results**: Saves the numerical results to `experiment_results.csv` and generates a comparison plot `performance_comparison.png`.

### 3. Output Files

After running the experiment, the following files will be generated in the `gemini` directory:
- `loss_curve.png`: A plot of the training and validation loss for the Bi-Align model.
- `performance_comparison.png`: A bar chart comparing the performance of all models.
- `experiment_results.csv`: A CSV file containing the raw accuracy and F1 scores.
- `bialign_model/`: A directory containing the saved trained Bi-Align model.
- `loss_history.csv`: A CSV file with the loss history.
- `cache/`: Directory for the downloaded Hugging Face dataset.

The main script will print the final results table to the console.

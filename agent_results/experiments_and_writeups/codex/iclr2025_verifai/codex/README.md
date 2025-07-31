# Codex Experiments

This directory contains scripts to run automated experiments comparing a baseline MLP model and a dropout-enhanced MLP model on synthetic classification data.

## Requirements
- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- matplotlib

Install dependencies:
```
pip install torch scikit-learn pandas matplotlib
```

## Running the Experiment
```
python run_experiments.py --output-dir output --epochs 10 --hidden-dim 64 --lr 1e-3 --dropout 0.5
```
This will:
1. Generate a synthetic binary classification dataset.
2. Train two models (baseline and proposed) for the specified number of epochs.
3. Save results to `output/results.csv` and `output/results.json`.
4. Generate figures in `output/figures/`.

## Outputs
- `results.csv`: CSV containing metrics for each epoch and model.
- `results.json`: JSON with training history.
- `figures/val_loss.png`: Validation loss curves.
- `figures/val_acc.png`: Validation accuracy curves.
- `figures/accuracy_comparison.png`: Bar chart of final validation accuracies.
```

# Automated Experiment

This directory contains an automated experiment comparing a baseline DistilBERT model with a retrieval-augmented variant on a subset of the 20 Newsgroups dataset.

## Requirements
- Python 3.8+
- torch
- transformers
- scikit-learn
- matplotlib

You can install dependencies with:
```
pip install torch transformers scikit-learn matplotlib
```

## Running the Experiment
Execute the main script:
```
cd codex
python3 experiment.py
```

This will:
1. Load and preprocess the data.
2. Train baseline and retrieval-augmented models for 2 epochs each.
3. Evaluate and compute metrics (accuracy, F1).
4. Generate loss curves and performance bar charts.
5. Save logs (`log.txt`), results (`results/results.md`, CSVs, and figures).

## Results
- All outputs are saved under the top-level `results/` directory at `codex_experiments/iclr2025_scope/results`:
- `results.md`: summary and figures.
- `log.txt`: execution logs.
- `loss_curves.png`, `metrics.png`: visualizations.
- `results.csv`: metrics.
- `baseline_loss.csv`, `aug_loss.csv`: training histories.

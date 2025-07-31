# InfluenceSpace Prototype Experiment

This directory contains code to run a prototype experiment of the InfluenceSpace dataset curation pipeline
on the SST-2 sentiment classification task from GLUE. This is a proof-of-concept adaptation of the
proposed hierarchical influence-driven curation framework to a text classification setting.

## Requirements
- Python 3.8+
- torch
- transformers
- datasets
- scikit-learn
- pandas
- matplotlib

Install dependencies via:
```
pip install torch transformers datasets scikit-learn pandas matplotlib
```

## Running the Experiment

From this directory, run:
```
python run_experiments.py
```

This will:
1. Load a small subset (2k samples) of SST-2 train set.
2. Compute DistilBERT embeddings for sentences.
3. Cluster embeddings into K=20 clusters.
4. Split into train/validation (80/20).
5. Train a linear classifier on the full dataset (baseline).
6. Approximate cluster-level influence via validation gradient and prune low-influence clusters.
7. Retrain classifier on curated subset (proposed method).
8. Train classifiers on two baselines: random sampling and heuristic by sentence length.
9. Save metrics to `results.csv`, plots `train_loss.png`, `val_acc.png`, summary in `summary.json`, and logs in `log.txt`.

## Results

- `results.csv`: per-epoch train loss and validation accuracy for each method.
- `train_loss.png`, `val_acc.png`: training curves.
- `summary.json`: final accuracies and dataset sizes.
- `log.txt`: full execution log.

## Cleanup
Any models and intermediate checkpoints are not saved. You may delete the `__pycache__` folders if needed.

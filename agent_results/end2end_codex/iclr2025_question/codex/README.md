# Experiment Pipeline

This folder contains scripts to run experiments comparing two methods on the SST-2 sentiment classification task:

- **baseline**: full fine-tuning of `distilbert-base-uncased`.
- **head_only**: freeze the transformer encoder and fine-tune only the classification head.

Requirements:
- Python 3.7+
- PyTorch
- Transformers
- Datasets
- Matplotlib
- Pandas

Install dependencies:
```
pip install torch transformers datasets matplotlib pandas
```

Run the full experiment:
```
python run_experiments.py
```

This will:
1. Train and evaluate both methods, saving results in `results/<method>/results.json`.
2. Generate plots (`loss_curve.png`, `metrics.png`) and a summary CSV (`results.csv`) in `results/`.
3. Log output to `log.txt`.

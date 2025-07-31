# Uncertainty-Aware Decoding (UAD) Experiment

This folder contains scripts and results to evaluate the proposed Uncertainty-Aware Decoding (UAD) method against a baseline greedy decoding on the CNN/DailyMail summarization dataset.

## Requirements
- Python 3.8+
- PyTorch
- Transformers
- Datasets
- Matplotlib

Install requirements with:
```bash
pip install torch transformers datasets matplotlib evaluate
```

## Running the Experiment
Run the main script:
```bash
python run_experiments.py
```

This will:
- Generate summaries for a subset of CNN/DailyMail (20 samples) using:
  - Baseline greedy decoding
  - UAD decoding with predictive entropy and nucleus sampling
- Compute ROUGE scores for both methods
- Record generation times
- Plot and save:
  - `rouge_comparison.png`: Bar chart of ROUGE-1/2/L F1 scores
  - `entropy_curve.png`: Average token entropy over generation steps (UAD)
- Save `results.json` and `log.txt` for detailed logs and metrics.

## Outputs
- `results.json`: Summary of metrics and settings
- `rouge_comparison.png`, `entropy_curve.png`
- `log.txt`: Execution log

After completion, move `results.json`, logs, and figures to the `results/` folder for analysis.

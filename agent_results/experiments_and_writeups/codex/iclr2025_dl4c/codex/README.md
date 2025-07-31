# Automated Experiments for DL4C Proposal

This directory contains scripts to run automated experiments comparing the proposed method vs baseline.

## Requirements
- Python 3.8+
- PyTorch
- Transformers (`pip install transformers`)
- Datasets (`pip install datasets`)
- Matplotlib (`pip install matplotlib`)

## Structure
- `run_experiment.py`: Main script to run experiments, generate results, figures, and logs.

## Usage
```
cd codex_experiments/iclr2025_dl4c/codex
python3 run_experiment.py
```

Results (JSON, Markdown, figures) will be generated under this directory:
- `log.txt`: Log of execution.
- `results.json`: Raw metrics.
- `figures/`: Generated plots.
- `results.md`: Summary of results with embedded figures.

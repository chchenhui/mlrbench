# Automated Summarization Experiment

This folder contains scripts to run a comparitive summarization experiment between a baseline BART model and a proposed LED model with extended context.

## Setup
Install required packages:
```
pip install -r requirements.txt
```

## Run Experiment
```
python3 scripts/run_experiments.py
```

Results will be saved in:
- `results/results.json` and `results/results.csv`: quantitative results.
- `figures/comparison_rouge.png` and `figures/time_memory.png`: visualizations.
- `results/results.md`: markdown summary of results.
- `log.txt`: execution log.

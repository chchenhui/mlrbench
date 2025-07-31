# Self-Correction Experiment

This directory contains scripts to run a simplified self-correction experiment on the FEVER fact verification dataset.

## Requirements
Install required Python packages:
```bash
pip install -r requirements.txt
```

## Run the Experiment
```bash
cd codex
python run_experiment.py --output_dir . --num_samples 100 --threshold 0.9
```

This will produce:
- `results.csv`: Detailed per-sample records.
- `accuracy_comparison.png`: Bar chart of baseline vs proposed accuracy.
- `baseline_score_dist.png`: Histogram of baseline confidence scores.
- `results.md`: Markdown summary with figures and discussion.
- `log.txt`: Log of the execution.

## Notes
- The script uses the HuggingFace `facebook/bart-large-mnli` model for zero-shot classification.
- GPU acceleration is used if available.

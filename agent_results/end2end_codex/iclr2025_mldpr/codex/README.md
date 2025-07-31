# Automated Deprecation Scoring Experiment

This folder contains scripts to run the Adaptive Deprecation Scoring experiment on a small set of OpenML datasets.

## Setup

Requirements:
- Python 3.8+
- `openml`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`

Install dependencies:
```
pip install openml numpy pandas scikit-learn matplotlib
```

## Running the Experiment

Run the main script:
```
python3 run_experiment.py
```

This will:
- Fetch two OpenML datasets (iris, mnist_784).
- Compute a simple Deprecation Score based on dataset age.
- Run a reproducibility test via a logistic regression model.
- Save results to `results.csv`.
- Generate a bar chart `deprecation_scores.png`.
- Log progress in `log.txt`.

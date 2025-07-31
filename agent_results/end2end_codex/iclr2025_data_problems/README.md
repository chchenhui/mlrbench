# Experimental Pipeline for MRIA Prototype

This repository provides a minimal experimental pipeline to evaluate the retrieval attribution component of the MRIA framework on a small Wikipedia subset.

## Folder Structure
- codex/
  - __init__.py
  - requirements.txt
  - run_experiment.py    # Main experiment script
- run_experiment.sh      # Automated bash script to install dependencies and run the experiment
- README.md              # This file

## Requirements
- Python 3.8+
- pip

## Installation
1. Ensure Python 3 and pip are installed.
2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

## Running the Experiment
Simply execute:
```bash
bash run_experiment.sh
```

This will:
1. Install required Python packages.
2. Run the experiment, logging output to `codex/log.txt`.
3. Generate results (JSON, CSV, and figures) in `results/`.

## Results
After completion, the `results/` folder contains:
- `results.json`: Detailed per-query metrics and attribution scores.
- `metrics.csv`: Summary of Kendall τ and Spearman ρ per query.
- `attr_comp_<i>.png`: Scatter plots comparing MRIA vs LOO for each query.
- `correlations.png`: Line plot of correlations across queries.
- `log.txt`: Experiment execution log.

## Notes
- This prototype only implements the retrieval attribution stage with a trivial linear utility function. It serves as a demonstration of the automated pipeline structure.
- For full MRIA and generation attribution experiments, extend `run_experiment.py` accordingly.

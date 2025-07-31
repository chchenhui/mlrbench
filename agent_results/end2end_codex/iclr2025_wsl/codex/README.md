# Experimental Code

This directory contains the code to run the experiments for the Task-Conditioned Diffusion Models in Weight Space.

## Requirements
- Python 3.8+
- PyTorch
- numpy
- matplotlib

Install dependencies via:
```
pip install torch numpy matplotlib
```

## Running the Experiment
From the project root, execute:
```
python3 codex/experiment.py
```

This will:
- Generate synthetic classification tasks
- Train a model zoo of classifiers and extract weights
- Train a diffusion model in weight space conditioned on task descriptors
- Evaluate diffusion-based initialization vs random initialization
- Save results and figures in `codex/`

## Outputs
- `codex/log.txt`: Experiment logs
- `codex/results.json`: Raw numerical results
- `codex/loss_curve.png`: Training loss comparison plot
- `codex/bl_losses.npy`, `codex/dm_losses.npy`: Loss arrays

You can find final consolidated results in the top-level `results/` directory.

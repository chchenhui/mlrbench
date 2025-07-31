# Codex Experiments: AIFS vs ERM on Colored MNIST

This directory contains code to automatically run experiments comparing the proposed AIFS method with standard ERM on a synthetic Colored MNIST dataset to test robustness against spurious correlations.

## Setup

Dependencies (install via pip):
```
torch torchvision matplotlib tqdm
```

## Running the Experiments

From this directory, run:
```
python run_experiments.py
```

This will:
- Train ERM and AIFS for 5 epochs each (default) on Colored MNIST.
- Save training logs to `log.txt`.
- Store raw results in `results_raw/<METHOD>/` as JSON files.
- Generate figures (`loss_curve.png`, `acc_curve.png`) in `figures/`.

## Results

- Check `figures/` for loss and accuracy plots.
- Detailed training history is in `results_raw/`.

Adjust hyperparameters (epochs, batch size, learning rate, etc.) by editing `train.py` or passing arguments in `run_experiments.py`.

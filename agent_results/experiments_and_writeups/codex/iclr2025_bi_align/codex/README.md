# Automated Experiments for Dynamic Human-AI Co-Adaptation

This folder contains scripts to run automated experiments on the CartPole-v1 environment, comparing:
- Baseline DQN agent
- Hybrid agent: DQN + Behavioral Cloning (BC) from expert data

The experiments collect expert trajectories via a heuristic policy, train both agents, evaluate performance, and generate plots.

## Requirements
- Python 3.7+
- gymnasium
- numpy
- torch
- matplotlib

Install dependencies via:
```
pip install gymnasium numpy torch matplotlib
```

## Running the experiments
```
cd codex_experiments/iclr2025_bi_align/codex
python run_experiments.py
```
Optional arguments:
- `--max_steps`: total RL training steps (default 20000)
- `--expert_episodes`: episodes to collect expert data (default 50)
- See `python run_experiments.py --help` for all options.

## Outputs
- `results`: contains `results.json`, training reward/loss curves as PNGs
- `log.txt`: JSON log of final evaluation

After running, move the `results` folder contents and `log.txt` to `codex_experiments/iclr2025_bi_align/results` for summary.

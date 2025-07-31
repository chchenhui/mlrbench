# Experimental Pipeline for UDCA Framework

This project implements a simulation-based experiment to evaluate the Uncertainty-Driven Co-Adaptation (UDCA) framework for bidirectional human-AI alignment on a recommendation benchmark.

## Setup
Ensure you have Python 3.7+ and the following packages installed:
```
pip install numpy matplotlib
```

## Running the Experiment
```
python3 codex/experiment.py
```
This will:
- Simulate user preferences and run three methods: static, passive, and UDCA.
- Save results (`results.json`) and figures (`decision_quality.png`, `pce.png`) in the `codex/` folder.
- Log the process to `codex/log.txt`.

## Results
After running, check:
- `codex/results.json` for structured metrics.
- `codex/decision_quality.png` for decision quality curves.
- `codex/pce.png` for final preference calibration error.
- `codex/log.txt` for execution logs.

# Uncertainty-Driven Reciprocal Alignment (UDRA) Experiment

This repository contains the implementation of the UDRA framework for bidirectional human-AI alignment as described in the proposal. The experiment aims to demonstrate how a Bayesian user modeling approach combined with uncertainty estimation can improve human-AI collaboration in decision-making tasks.

## Project Structure

- `environments/`: Contains the simulated environments for testing UDRA
  - `resource_allocation.py`: Resource allocation task simulation
  - `safety_critical.py`: Safety-critical scenario simulation
- `models/`: Contains the implementation of baseline and UDRA algorithms
  - `baseline.py`: Standard RL with static alignment (RLHF)
  - `udra.py`: Uncertainty-Driven Reciprocal Alignment implementation
  - `bayesian_user_model.py`: Bayesian inference for user preference modeling
- `utils/`: Utility functions and helpers
  - `metrics.py`: Evaluation metrics implementation
  - `visualization.py`: Functions for visualizing results
  - `simulated_human.py`: Simulated human feedback mechanisms
- `visualizations/`: Directory for saved visualizations
- `run_experiments.py`: Main script to run all experiments
- `analyze_results.py`: Script to analyze results and generate visualizations

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- SciPy
- Matplotlib
- Pandas
- Gym (OpenAI Gym)

## How to Run the Experiment

1. Make sure all dependencies are installed.
2. Run the experiments:
   ```
   python run_experiments.py
   ```
3. Analyze the results:
   ```
   python analyze_results.py
   ```

## Experiment Overview

The experiment compares two approaches:
1. **Baseline**: Standard Reinforcement Learning with Human Feedback (RLHF) using static alignment
2. **UDRA**: Uncertainty-Driven Reciprocal Alignment with Bayesian user modeling

We evaluate these approaches on two environments:
- Resource allocation task (e.g., supply chain routing)
- Safety-critical scenario (e.g., autonomous driving intersection)

## Evaluation Metrics

- Alignment Error: Average distance between AI actions and human corrections
- Task Efficiency: Average cumulative task reward per episode
- Trust Calibration: Spearman's correlation between AI confidence and observed accuracy
- Simulated User Satisfaction: Based on control, transparency, and trust metrics
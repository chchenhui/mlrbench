# Adaptive Code Assistant Experiment

This repository contains the experimental implementation for testing the hypothesis that AI code assistants can be significantly more effective when they continuously adapt to individual developer workflows, preferences, and coding habits.

## Overview

The experiment simulates interactions between developers and code assistants to evaluate the effectiveness of different adaptation approaches. It compares:

**Baseline Methods:**
1. Static LLM (no adaptation)
2. Fine-tuned LLM (general coding patterns, no personalization)
3. Rule-based Personalization (manual rules for personalization)

**Proposed Adaptive Methods:**
1. Online Learning (continuous adaptation using SGD)
2. MAML-based Adaptation (model-agnostic meta-learning)
3. Hybrid Approach (combining online learning with MAML)

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Required Dependencies

The code requires the following Python packages:
- numpy
- matplotlib
- torch
- datasets
- transformers
- logging

## Experiment Structure

The experiment consists of the following components:

- `utils.py`: Utility functions for file operations, visualization, etc.
- `data.py`: Developer profile simulation and task dataset management
- `models.py`: Implementation of baseline and adaptive code assistant models
- `evaluation.py`: Metrics and visualization for evaluating performance
- `simulation.py`: Simulation of developer-AI interactions
- `run_experiment.py`: Main script to run the full experiment

## Running the Experiment

To run the experiment with default settings:

```bash
python run_experiment.py
```

This will:
1. Generate developer profiles
2. Load coding tasks
3. Run simulations for all models
4. Evaluate and visualize results
5. Generate a final report

### Command Line Options

You can customize the experiment with the following options:

```
--developers N    Number of developer profiles to generate (default: 3)
--tasks N         Number of tasks per model (default: 5)
--iterations N    Maximum iterations per task (default: 3)
--small-models    Use small models for faster experimentation (default: True)
```

Example with custom settings:

```bash
python run_experiment.py --developers 5 --tasks 10 --iterations 5
```

## Output Structure

The experiment creates the following output files:

- `log.txt`: Detailed experiment execution log
- `results/developer_profiles.json`: Generated developer profiles
- `results/experiment_data.json`: Raw experiment data
- `results/evaluation_results.json`: Evaluation metrics
- `results/*.png`: Visualization figures
- `results/results.md`: Final results summary and analysis

## Results Interpretation

The experiment evaluates models based on the following metrics:

- **Correctness Rate**: Percentage of code that passes test cases
- **Style Score**: Alignment with developer's coding style preferences
- **Speed Score**: Development efficiency (iterations and time)
- **Satisfaction**: Simulated developer satisfaction
- **Adaptation Gain**: Improvement in satisfaction over time
- **Adaptation Rate**: How quickly the model adapts to preferences

The final results are summarized in `results.md`, which includes:
- Quantitative comparison of methods
- Visualizations of performance across metrics
- Analysis of the effectiveness of adaptation
- Limitations and future work

## Customization

You can customize the experiment by modifying:

- `data.py`: Change developer profile generation or task selection
- `models.py`: Adjust model parameters or implementation
- `evaluation.py`: Modify evaluation metrics or visualization
- `simulation.py`: Change interaction simulation details

## Troubleshooting

Common issues:

1. **Memory errors**: Reduce the number of developers and tasks, or ensure `--small-models` is set
2. **ImportError**: Ensure all dependencies are installed
3. **Slow execution**: Use small models and reduce the number of tasks and iterations
4. **Missing visuals**: Ensure matplotlib is properly installed

## Contact

For questions or issues, please open an issue in the repository.
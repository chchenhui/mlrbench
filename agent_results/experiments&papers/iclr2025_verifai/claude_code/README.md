# LLM-TAC: LLM-Guided Tactic Autogeneration for Interactive Theorem Provers

This repository contains the implementation of LLM-TAC, a framework for automating tactic generation in interactive theorem provers (ITPs) like Coq. LLM-TAC uses large language models (LLMs) with reinforcement learning to generate and refine proof tactics.

## Overview

LLM-TAC is designed to reduce the manual effort in interactive theorem proving by automating tactic generation. The framework consists of three main components:

1. **Contextual Encoding**: Encodes the proof state, including goals, hypotheses, and library context
2. **Tactic Generation**: Generates candidate tactics using an LLM
3. **Reinforcement Learning**: Improves tactic generation through feedback from the theorem prover

## Requirements

- Python 3.8+
- PyTorch
- transformers
- matplotlib
- seaborn
- numpy
- scikit-learn

## Project Structure

```
claude_code/
├── main.py                    # Main script to run experiments
├── data_processing.py         # Data processing utilities
├── evaluation.py              # Evaluation metrics and utilities
├── visualization.py           # Visualization functions
├── utils.py                   # Utility functions
├── models/
│   ├── contextual_encoding.py # Contextual encoding component
│   ├── tactic_generator.py    # Tactic generation component
│   ├── reinforcement_learner.py # Reinforcement learning component
│   └── baselines.py           # Baseline models for comparison
├── data/                      # Directory for datasets
└── results/                   # Directory for experiment results
```

## Running the Experiments

To run the full LLM-TAC experiment:

```bash
python main.py --data_dir data --output_dir results --model_name Llama-3.1-8B --num_epochs 5 --rl_iterations 10 --use_gpu --run_baselines
```

### Command-line Options

- `--data_dir`: Directory containing the Coq proof dataset
- `--output_dir`: Directory to save results and outputs
- `--model_name`: Base LLM model to use
- `--num_epochs`: Number of training epochs for supervised fine-tuning
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for fine-tuning
- `--rl_iterations`: Number of reinforcement learning iterations
- `--seed`: Random seed for reproducibility
- `--use_gpu`: Use GPU for training if available
- `--run_baselines`: Run baseline models for comparison
- `--ablation_studies`: Run ablation studies on LLM-TAC components

## Experiments

Our experiments evaluate LLM-TAC against several baselines:

1. **LLM-TAC**: Our full framework with contextual encoding and reinforcement learning
2. **Naive LLM**: An LLM without specialized fine-tuning for theorem proving
3. **In-Context Learning (ICL)**: LLM with few-shot examples but no fine-tuning
4. **Traditional Automated Tactics**: Coq's built-in automated tactics

We evaluate these methods on the following metrics:

- **Tactic Generation Accuracy**: The percentage of generated tactics that are syntactically correct and semantically meaningful
- **Proof Completion Rate**: The percentage of theorems successfully proven
- **Reduction in Manual Tactic Writing**: The percentage reduction in manual tactic writing required
- **Proof Completion Time**: The time taken to complete proofs

## Results

After running the experiments, results will be saved in the specified output directory:

- `training_curve.png`: Learning curves from supervised fine-tuning
- `rl_progression.png`: Performance progression during reinforcement learning
- `metrics_comparison.png`: Comparison of metrics across different methods
- `metrics_comparison_time.png`: Comparison of proof completion times
- `results.json`: Detailed results in JSON format
- `log.txt`: Experiment execution log

## Acknowledgements

This work builds upon prior research in applying LLMs to interactive theorem proving, including:

- LeanDojo
- LLMSTEP
- COPRA
- Lean Copilot
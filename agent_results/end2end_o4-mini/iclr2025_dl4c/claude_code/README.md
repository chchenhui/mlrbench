# Adaptive Code Assistant via Implicit Developer Feedback

This repository contains the implementation of an Adaptive Code Assistant that uses reinforcement learning to adapt to developer preferences based on implicit feedback signals.

## Overview

The Adaptive Code Assistant experiment is designed to test whether incorporating implicit developer feedback can improve code suggestion quality and developer productivity. The system:

1. Captures implicit feedback signals (edit distance, acceptance rates, etc.)
2. Represents developer preferences via a learned embedding
3. Uses Proximal Policy Optimization (PPO) to adapt a pre-trained transformer model

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
claude_code/
├── data/                  # Dataset directory
├── logs/                  # Log files
├── models/                # Trained model checkpoints
│   ├── baseline/          # Static CodeT5+ model
│   └── adaptive/          # Adaptive model with RL
├── results/               # Experiment results
│   ├── figures/           # Generated visualizations
│   └── tables/            # Result tables
└── utils/                 # Utility modules
```

## Running Experiments

### 1. Download Model (Optional)

Download and cache the CodeT5+ model locally:

```bash
python download_model.py --model_name "Salesforce/codet5p-220m-py" --output_dir "./models/cached"
```

### 2. Run Full Experiment

Run the complete experiment with default parameters:

```bash
python run_experiments.py --gpu
```

### 3. Run Simplified Experiment (Quick Test)

For a quicker test of the pipeline, run the simplified experiment:

```bash
python run_simplified.py --synthetic_samples 20 --num_developers 5 --num_tasks 3 --epochs 2
```

### 4. Visualize Existing Results

To regenerate visualizations from existing results:

```bash
python run_experiments.py --visualize_only
```

## Command Line Arguments

The experiment scripts accept the following arguments:

- `--output_dir`: Directory to save results (default: './results')
- `--log_dir`: Directory to save logs (default: './logs')
- `--data_dir`: Directory containing datasets (default: './data')
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 3e-5)
- `--ppo_epochs`: Number of PPO epochs (default: 4)
- `--seed`: Random seed (default: 42)
- `--gpu`: Use GPU if available (flag)
- `--num_developers`: Number of simulated developers (default: 30)
- `--num_tasks`: Number of coding tasks per developer (default: 12)
- `--eval_only`: Run only evaluation, no training (flag)
- `--visualize_only`: Only generate visualizations from existing results (flag)

## Experiment Outputs

The experiment generates:

1. **Model Checkpoints**: Saved in the `models/` directory
2. **Evaluation Results**: JSON file with raw metrics in `results/experiment_results.json`
3. **Visualizations**: PNG files showing performance comparisons in `results/`
4. **Tables**: CSV and Markdown tables summarizing results in `results/tables/`
5. **Logs**: Detailed logs of the experiment in `logs/experiments.log`

## Implementation Details

### Models

- **Baseline**: Static CodeT5+ model without adaptation
- **Adaptive**: CodeT5+ with a policy network trained using PPO on implicit feedback signals

### Evaluation Metrics

- **Acceptance Rate**: Percentage of suggestions kept without major edits
- **Edit Distance**: Average Levenshtein distance between suggestions and final code
- **Task Completion Time**: Simulated time to complete tasks
- **Code Quality**: Measured through static analysis
- **Overall Reward**: Combined metric incorporating all feedback signals

## Cite This Work

If you use this implementation in your research, please cite:

```
@article{adaptive_code_assistant_2025,
  title={Adaptive Code Assistant via Implicit Developer Feedback and Reinforcement Learning},
  author={Author, A.},
  journal={ICLR Workshop on Deep Learning for Code},
  year={2025}
}
```
# Dynamic Human-AI Co-Adaptation Experiment

This repository contains the implementation of the Dynamic Human-AI Co-Adaptation via Real-Time Feedback-Driven Alignment framework, as described in the proposal. The framework combines online reinforcement learning (RL) with interpretable human feedback loops to enable continuous co-adaptation between humans and AI systems.

## Overview

The experiment simulates a recommendation system environment with evolving user preferences. Three different agents are compared:

1. **Dynamic Alignment Agent**: Implements the proposed framework combining online RL with imitation learning and explanation generation.
2. **Static RLHF Agent**: A baseline agent implementing standard Reinforcement Learning from Human Feedback without adaptation mechanisms.
3. **Direct RLAIF Agent**: A baseline agent implementing direct Reinforcement Learning from AI Feedback.

The experiment evaluates each agent's performance on multiple metrics:
- Reward: The immediate feedback from the environment
- Alignment: How well the agent's recommendations align with user preferences
- Trust: Consistency and reliability of recommendations
- Adaptability: How well the agent adapts to changing user preferences

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- Seaborn
- Pandas
- tqdm

You can install all dependencies with:

```bash
pip install torch numpy matplotlib seaborn pandas tqdm
```

## File Structure

- `main.py`: Main script for running the experiment
- `environment.py`: Implements the recommendation environment with dynamic user preferences
- `models.py`: Implements the agent architectures
- `explanation.py`: Implements the explanation generation mechanisms
- `utils.py`: Utility functions for data processing, visualization, and evaluation

## Running the Experiment

To run the experiment with default parameters:

```bash
python main.py
```

The script will automatically:
1. Initialize the environment and agents
2. Run the training and evaluation loop
3. Generate plots and tables of the results
4. Save all outputs to the `../results/` directory

## Experiment Parameters

The main configuration parameters can be modified in the `CONFIG` dictionary in `main.py`:

- `seed`: Random seed for reproducibility
- `n_users`: Number of users in the recommendation environment
- `n_items`: Number of items available for recommendation
- `n_features`: Dimensionality of item and preference features
- `n_episodes`: Number of training episodes
- `preference_shift_interval`: Number of episodes between preference shifts
- `preference_shift_magnitude`: Magnitude of preference shifts
- `learning_rate`: Learning rate for all neural networks
- `discount_factor`: Discount factor for RL
- `imitation_weight`: Weight of imitation learning component
- `explanation_threshold`: Threshold for determining significant feature contributions
- `batch_size`: Batch size for network updates
- `eval_interval`: Interval for evaluating alignment, trust, and adaptability

## Results

After running the experiment, the following outputs will be available in the `../results/` directory:

- `experiment_results.json`: Raw results data
- `log.txt`: Detailed log of the experiment
- `config.json`: Configuration parameters used
- `comparison_table.csv`: Summary of performance metrics
- `reward_curve.png`: Plot of rewards over time
- `alignment_curve.png`: Plot of alignment scores
- `trust_curve.png`: Plot of trust scores
- `adaptability_curve.png`: Plot of adaptability scores
- `reward_comparison.png`: Bar plot comparing average rewards
- `alignment_comparison.png`: Bar plot comparing average alignment scores
- `trust_comparison.png`: Bar plot comparing average trust scores
- `adaptability_comparison.png`: Bar plot comparing average adaptability scores
- `radar_comparison.png`: Radar chart comparing all metrics
- `preference_shifts.png`: Visualization of preference shifts
- `results.md`: Comprehensive report of the experiment results

## Extending the Experiment

To extend or modify the experiment:

1. **Custom Environments**: Modify `environment.py` to implement different interaction environments
2. **New Agents**: Add new agent classes to `models.py`
3. **Additional Metrics**: Implement new evaluation metrics in the environment or main script
4. **Alternative Explanations**: Modify `explanation.py` to generate different types of explanations

## Citation

If you use this code in your research, please cite:

```
@article{dynamic_alignment_2025,
  title={Dynamic Human-AI Co-Adaptation via Real-Time Feedback-Driven Alignment},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```
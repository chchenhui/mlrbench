# MeLPA: Meta-Learned Personalized Adapters

This repository contains the implementation of MeLPA (Meta-Learned Personalized Adapters), a novel framework for efficient continual adaptation of foundation models using meta-learning to optimize adapter modules.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Project Structure

- `models/`: Implementation of adapters and meta-learning components
  - `adapters.py`: Adapter modules and model integration
  - `meta_learning.py`: Meta-learning framework (MeLPA)
- `data/`: Data handling and task simulation
  - `task_datasets.py`: Text classification task generators
  - `mock_data.py`: Mock datasets for testing
- `baselines/`: Baseline methods for comparison
  - `ewc.py`: Elastic Weight Consolidation implementation
  - `lwf.py`: Learning without Forgetting implementation
- `utils/`: Utility functions
  - `training.py`: Trainers for meta-learning and continual learning
  - `evaluation.py`: Evaluation metrics and visualization
- `run_experiments.py`: Main script for running full experiments
- `run.py`: Simplified script for experiment execution
- `generate_results.py`: Script to generate mock results for demonstration

## Running the Experiments

### Full Experiment

To run the full experiment with all components (meta-learning, baselines, MeLPA variants, and analysis):

```bash
python run.py
```

This will:
1. Run meta-learning phase to learn initialization and update rules
2. Run baseline methods (Standard Adapter, EWC, LwF)
3. Run MeLPA with different configurations
4. Analyze and compare results
5. Generate visualizations and comprehensive report

### Quick Mode

For a faster run with reduced scale (useful for testing):

```bash
python run.py --quick
```

### Generating Mock Results

To generate mock results without running the actual experiments:

```bash
python generate_results.py
```

This creates all visualizations and results files for demonstration purposes.

## Experiment Configuration

You can customize various aspects of the experiments:

- `--seed`: Random seed for reproducibility
- `--model_name`: Base foundation model to use
- `--adapter_type`: Type of adapter ("pfeiffer" or "lora")
- `--bottleneck_dim`: Bottleneck dimension for Pfeiffer adapters
- `--lora_rank`: Rank for LoRA adapters
- `--n_meta_epochs`: Number of meta-training epochs
- `--n_tasks`: Number of tasks in continual learning sequence
- `--n_epochs_per_task`: Number of epochs per task

## Results

After running the experiments, results are saved in the `../results/` directory:

- `figures/`: Visualizations of results
- `results.md`: Comprehensive summary of experiments and findings
- `run_log.txt`: Detailed log of the experiment execution

## Methodology

MeLPA combines three key components:

1. **Frozen Foundation Model**: A pre-trained model whose core weights remain fixed
2. **Personalized Adapter Modules**: Lightweight neural networks inserted into the frozen model
3. **Meta-Learner**: Optimizes initialization and update dynamics of the adapters

The framework uses meta-learning to learn optimal initialization strategies and update rules for adapters, enabling faster adaptation, improved knowledge retention, and superior personalization capabilities compared to standard adapter tuning and other continual learning methods.
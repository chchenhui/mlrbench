# VERIL: Verification-Enriched Recursive Improvement Learning

This repository contains the implementation of the VERIL (Verification-Enriched Recursive Improvement Learning) framework for self-correcting code generation. VERIL bridges the gap between verification and learning by creating a closed-loop system where formal verification feedback directly informs model improvement.

## Overview

The VERIL framework consists of four core components:

1. **Comprehensive Fault Taxonomy (CFT)**: A hierarchical taxonomy of code faults that systematically categorizes the types of errors that can occur in code generation.

2. **Verification Integration Layer (VIL)**: Orchestrates the verification process by integrating multiple verification tools and standardizing their outputs.

3. **Error-to-Explanation Converter (E2EC)**: Transforms verification outcomes into natural language explanations and remediation examples.

4. **Recursive Improvement Learning (RIL)**: Implements a multi-tiered learning strategy that uses error explanations and remediation examples to improve the model's code generation capabilities.

## Directory Structure

```
claude_code/
├── README.md                # This file
├── config.py                # Configuration parameters
├── data.py                  # Data loading and processing
├── evaluation.py            # Evaluation metrics and reporting
├── model.py                 # Model implementation and training
├── requirements.txt         # Python dependencies
├── run_experiment.py        # Main experiment runner
├── setup.py                 # Setup script
├── utils.py                 # Utility functions
└── verification.py          # Verification tools
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Set up the environment:

```bash
cd claude_code
python setup.py
```

This will install the required dependencies and create the necessary directories.

## Configuration

The experiment can be configured in the `config.py` file. Key configuration parameters include:

- **Dataset configuration**: Choose the dataset to use (HumanEval, APPS, or custom)
- **Model configuration**: Configure the models to use (baseline, VERIL variants)
- **Verification tools**: Choose which verification tools to use
- **Training parameters**: Set learning rate, batch size, etc.
- **Experiment settings**: Choose which models to run, set the random seed, etc.

## Running the Experiment

To run the full experiment:

```bash
python run_experiment.py
```

This will run the experiment with the default configuration. 

You can also customize the experiment using command-line arguments:

```bash
python run_experiment.py --dataset custom --dataset_size 50 --run_baseline --run_veril_static --seed 42
```

Available command-line arguments:

- `--dataset`: Dataset to use (default: "HumanEval")
- `--dataset_size`: Number of examples to use (default: 100)
- `--run_baseline`: Run baseline model
- `--run_veril_static`: Run VERIL model with static verification
- `--run_veril_dynamic`: Run VERIL model with dynamic verification
- `--run_veril_full`: Run VERIL model with full verification
- `--seed`: Random seed (default: 42)
- `--num_trials`: Number of trials (default: 3)
- `--gpu`: Use GPU if available (default: True)

## Results

After running the experiment, the results will be saved in the `results` directory:

- `baseline_results.json`: Results for the baseline model
- `veril_static_results.json`: Results for the VERIL model with static verification
- `veril_dynamic_results.json`: Results for the VERIL model with dynamic verification
- `veril_full_results.json`: Results for the VERIL model with full verification
- `all_results.json`: Consolidated results for all models
- `results.md`: Evaluation report
- `model_comparison.png`: Bar chart comparing model performance
- `learning_curve_*.png`: Learning curves for VERIL models

## API Keys

To use the OpenAI and Anthropic APIs, you need to set the following environment variables:

```bash
export OPENAI_API_KEY=<your-openai-api-key>
export ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project was developed as part of the VerifAI: AI Verification in the Wild workshop.
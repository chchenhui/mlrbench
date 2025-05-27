# Reasoning Uncertainty Networks (RUNs)

This repository contains the implementation of "Reasoning Uncertainty Networks: Enhancing LLM Transparency Through Graph-Based Belief Propagation" for hallucination detection in Large Language Models.

## Overview

Reasoning Uncertainty Networks (RUNs) is a novel framework that represents LLM reasoning as a directed graph where uncertainty is explicitly modeled and propagated throughout the reasoning process. This approach aims to:

1. Make uncertainty an explicit, integral component of the reasoning chain rather than a post-hoc calculation
2. Provide fine-grained transparency into how confidence levels flow through complex reasoning steps
3. Enable automatic detection of potential hallucination points based on uncertainty thresholds
4. Create a computationally efficient method that operates at the semantic level rather than requiring multiple model inferences
5. Enhance explainability by allowing users to identify precisely where reasoning uncertainty originates

## Project Structure

```
claude_code/
├── config.py              # Configuration parameters
├── data.py                # Dataset loading and processing
├── model.py               # RUNs model implementation
├── uncertainty.py         # Baseline uncertainty estimation methods
├── evaluation.py          # Evaluation metrics and visualization
├── utils.py               # Utility functions
├── run_experiment.py      # Main experiment runner
├── run_test.py            # Test script for individual components
├── outputs/               # Output directory for results
└── README.md              # This file
```

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

The implementation requires Python 3.8+ and the following key dependencies:
- numpy
- pandas
- matplotlib
- scikit-learn
- torch
- transformers
- networkx
- sentence-transformers
- anthropic

## Usage

### Running a Quick Test

To run a quick test of the system components:

```bash
python run_test.py --all
```

This will run tests for data loading, RUNs components, baseline methods, and evaluation metrics.

### Running the Full Experiment

To run the full experiment with all baselines:

```bash
python run_experiment.py --dataset scientific --use_gpu
```

Options:
- `--dataset`: The dataset to use (`scientific`, `legal`, or `medical`)
- `--baselines`: List of baseline methods to include (default: all)
- `--num_examples`: Number of test examples to use
- `--use_gpu`: Use GPU acceleration if available

For example, to run only specific baselines on a smaller number of examples:

```bash
python run_experiment.py --dataset scientific --baselines selfcheckgpt hudex --num_examples 20
```

## Environment Variables

The code uses the following environment variables:
- `ANTHROPIC_API_KEY`: API key for Anthropic Claude API
- `OPENAI_API_KEY`: API key for OpenAI API (optional, used for some baselines)

Make sure to set these variables before running the experiments.

## Results

After running the experiment, results will be saved in the following locations:
- `outputs/`: Contains all result files, visualizations, and logs
- `../results/`: Contains a copy of the main results, figures, and logs

The main results file is `results.md`, which provides a comprehensive summary of the experiment results, including performance metrics, visualizations, and analysis.

## Model Components

The RUNs framework consists of four main components:

1. **Reasoning Graph Constructor**: Transforms LLM-generated reasoning into a directed graph structure
2. **Uncertainty Initializer**: Assigns initial uncertainty distributions to graph nodes
3. **Belief Propagation Engine**: Updates uncertainty values across the graph using message passing
4. **Hallucination Detection Module**: Identifies potential hallucinations based on uncertainty thresholds

## Baseline Methods

The following baseline methods are implemented for comparison:

1. **SelfCheckGPT**: A sampling-based approach that checks consistency across multiple samples
2. **Multi-dimensional UQ**: An approach that integrates semantic and knowledge-aware similarity analysis
3. **Calibration-based approaches**: Traditional methods that calibrate the model's output probabilities
4. **HuDEx**: An explanation-enhanced hallucination detection model
5. **MetaQA**: A metamorphic relation-based approach for hallucination detection

## Citation

If you use this code in your research, please cite:

```
@article{reasoning_uncertainty_networks,
  title={Reasoning Uncertainty Networks: Enhancing LLM Transparency Through Graph-Based Belief Propagation},
  author={Author},
  journal={ICLR 2025 Workshop on Uncertainty Quantification},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
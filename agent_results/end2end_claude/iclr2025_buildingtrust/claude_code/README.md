# TrustPath: Transparent Error Detection and Correction in LLMs

This code implements the TrustPath framework for transparent error detection and correction in Large Language Models (LLMs), as described in the accompanying proposal.

## Overview

The TrustPath framework consists of three main components:

1. **Self-Verification Module**: Prompts the LLM to evaluate its own outputs for potential errors or uncertainties
2. **Factual Consistency Checker**: Verifies claims made in the LLM output against trusted knowledge sources
3. **Human-in-the-Loop Feedback System**: Simulates user feedback on detected errors and suggested corrections

The experiment compares TrustPath against three baseline methods:
- Simple Fact Checking
- Uncertainty Estimation
- Standard Correction

## Project Structure

- `config.py`: Configuration settings for the experiment
- `self_verification.py`: Implementation of the self-verification module
- `factual_checker.py`: Implementation of the factual consistency checker
- `human_feedback.py`: Implementation of the human feedback simulation
- `trust_path.py`: Integration of all components into the TrustPath framework
- `baselines.py`: Implementation of baseline methods
- `data_processing.py`: Dataset creation and processing
- `evaluation.py`: Evaluation metrics and procedures
- `visualization.py`: Visualization of results
- `run_experiment.py`: Main script to run the experiment
- `fix_paths.py`: Helper script to fix import paths

## Requirements

The code requires the following Python packages:

```
anthropic
matplotlib
numpy
pandas
scikit-learn
sentence-transformers
nltk
seaborn
rouge-score
```

You can install them with pip:

```bash
pip install anthropic matplotlib numpy pandas scikit-learn sentence-transformers nltk seaborn rouge-score
```

## Running the Experiment

To run the experiment with default settings:

```bash
python run_experiment.py
```

You can customize the experiment with the following command-line arguments:

- `--samples <n>`: Number of samples to use (default: from config)
- `--force-new-dataset`: Force creation of a new dataset even if one exists
- `--api-key <key>`: API key for the Anthropic API (default: use environment variable)

For example:

```bash
python run_experiment.py --samples 20 --force-new-dataset
```

## Experiment Process

1. **Dataset Creation**: The experiment creates a dataset of LLM responses with injected errors
2. **Method Execution**: TrustPath and baseline methods are applied to the dataset
3. **Evaluation**: Results are evaluated using precision, recall, F1, and other metrics
4. **Visualization**: Visualizations are generated to compare the methods
5. **Results**: A summary of results is generated in markdown format

## Output

The experiment generates the following outputs in the `results` directory:

- `results.md`: A summary of the experiment results
- `log.txt`: A log of the experiment execution
- Various `.png` files with visualizations of the results

## GPU Acceleration

If available, this code will automatically utilize GPU acceleration through the libraries it uses (e.g., sentence-transformers uses PyTorch which will detect and use available GPUs).

## Notes

- The code simulates human feedback for the purpose of the experiment
- The factual consistency checker simulates document retrieval using the LLM
- In a real-world implementation, these would be connected to actual user interactions and knowledge bases

## License

This code is provided for research purposes only.
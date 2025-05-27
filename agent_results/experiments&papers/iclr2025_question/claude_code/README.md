# Uncertainty-Aware Decoding (UAD) for Mitigating Hallucinations in LLMs

This repository contains the implementation of the Uncertainty-Aware Decoding (UAD) mechanism for mitigating hallucinations in large language models. The UAD mechanism monitors token-level uncertainty metrics at each decoding step and intervenes when uncertainty surpasses a dynamically adjusted threshold, suggesting a high risk of hallucination.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running the Experiment](#running-the-experiment)
- [Configuration](#configuration)
- [Results](#results)
- [License](#license)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/uncertainty-aware-decoding.git
cd uncertainty-aware-decoding
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

The project is organized as follows:

- `config.py`: Configuration settings for experiments
- `data.py`: Data loading and preprocessing utilities
- `uncertainty.py`: Uncertainty estimation methods
- `decoding.py`: Decoding strategies, including UAD
- `evaluation.py`: Evaluation metrics
- `visualization.py`: Visualization utilities
- `experiment.py`: Experiment runner
- `main.py`: Main script to run experiments

## Running the Experiment

To run the default experiment configuration:

```bash
python main.py
```

### Command-line Arguments

You can customize the experiment using the following command-line arguments:

- `--model`: Model configuration to use (default: "small")
- `--dataset`: Dataset to use (default: "squad")
- `--experiments`: Specific experiments to run (default: all)
- `--results_dir`: Directory to save results (default: "results")
- `--organize_results`: Flag to organize results into a separate directory

### Examples

Run experiments with a specific model and dataset:

```bash
python main.py --model medium --dataset xsum
```

Run only selected experiments:

```bash
python main.py --experiments baseline uad_entropy
```

## Configuration

The experiment configuration is defined in `config.py`. You can modify the following settings:

### Model Configurations

```python
MODEL_CONFIGS = {
    "default": {
        "name": "facebook/opt-350m",
        "cache_dir": str(MODELS_DIR),
    },
    "small": {
        "name": "distilgpt2",
        "cache_dir": str(MODELS_DIR),
    },
    "medium": {
        "name": "facebook/opt-1.3b",
        "cache_dir": str(MODELS_DIR),
    },
}
```

### Dataset Configurations

```python
DATASET_CONFIGS = {
    "squad": {
        "name": "squad_v2",
        "split": "validation[:1000]",
        "cache_dir": str(DATA_DIR),
    },
    "xsum": {
        "name": "xsum",
        "split": "test[:500]",
        "cache_dir": str(DATA_DIR),
    },
}
```

### Experiment Configurations

```python
EXPERIMENT_CONFIGS = {
    "baseline": {
        "decoding_method": "greedy",
        # ...
    },
    "beam_search": {
        "decoding_method": "beam_search",
        # ...
    },
    "uad_entropy": {
        "decoding_method": "uad",
        "uncertainty_method": "entropy",
        # ...
    },
    # ...
}
```

## Results

The experiment results are saved in the `results` directory (or the directory specified by `--results_dir`). The following files are generated:

- `results.json`: Raw experiment results
- `results.md`: Markdown report of the experiment results
- `metrics_table.csv`: Table of evaluation metrics
- Various visualization figures (PNG files)

If the `--organize_results` flag is used, the results are organized into the `claude_exp2/iclr2025_question/results` directory, including:

- `results.md`: Markdown report
- `log.txt`: Experiment log
- Visualization figures (PNG files)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
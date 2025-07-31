# Self-Correcting Language Model (SCLM) Experiment

This repository contains the implementation of the Self-Correcting Language Model (SCLM) framework as proposed in the paper "Self-Correcting Language Models: Automated Error Detection and Correction for Enhanced Trustworthiness".

## Overview

SCLM is a framework that enables language models to iteratively detect and correct errors in their outputs. It consists of two main components:

1. **Internal Confidence Scorer**: Identifies low-confidence spans in generated text using self-attention patterns and uncertainty quantification.
2. **Retrieval-Augmented Corrector**: Queries knowledge bases to refine errors and improve factual accuracy.

The model operates in an iterative self-correction loop:
- Generate an initial response
- Identify low-confidence spans
- Correct these spans using retrieved information
- Repeat until confidence thresholds are met or maximum iterations are reached

## Repository Structure

```
claude_code/
├── config.py           # Configuration parameters
├── data_loader.py      # Dataset loading utilities
├── models.py           # SCLM implementation
├── baseline.py         # Baseline model implementations
├── evaluation.py       # Evaluation metrics
├── experiment.py       # Main experiment script
├── utils.py            # Utility functions
└── README.md           # This file
```

## Requirements

The experiment requires the following Python packages:

```
torch
transformers
datasets
nltk
rouge-score
scikit-learn
matplotlib
seaborn
tqdm
openai
anthropic
```

## Running the Experiment

To run the full experiment with default settings:

```bash
python experiment.py
```

### Command-Line Arguments

- `--max_samples`: Maximum number of samples to evaluate (default: 100)
- `--seed`: Random seed for reproducibility (default: 42)
- `--use_api`: Flag to use API models instead of local models
- `--model`: Base model to use

Example:

```bash
python experiment.py --max_samples 50 --use_api --model claude-3.7-sonnet
```

## Experiment Details

The experiment evaluates the SCLM framework against multiple baselines:

### Models
1. **SCLM**: Our Self-Correcting Language Model implementation
2. **Zero-Shot Baseline**: Base model without correction
3. **Retrieval-Augmented Baseline**: Base model with retrieval but no self-correction loop
4. **Rule-Based Correction**: Model using simple rules for correction

### Datasets
1. **TruthfulQA**: Dataset testing factual accuracy
2. **FEVER**: Fact Extraction and Verification dataset

### Metrics
1. **Factuality**: Accuracy, F1 score, hallucination rate
2. **Efficiency**: Latency, average iterations
3. **Quality**: BLEU, ROUGE scores

## Results

After running the experiment, results will be saved in the `results` directory:
- JSON files with detailed metrics for each model and dataset
- Visualizations in the `figures` subdirectory
- A comprehensive `results.md` summary

## Notes on Implementation

- API usage: The experiment can use either local models from Hugging Face or API-based models from OpenAI and Anthropic. API keys should be set in the environment variables `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.
- Retrieval simulation: For simplicity, we simulate retrieval by asking the model to generate factual information. A production implementation would use real knowledge bases.
- Confidence scoring: For API models, we estimate confidence by asking the model directly, while for local models we use self-attention patterns.

## Customization

To customize the experiment:
- Modify model configurations in `config.py`
- Add new datasets by extending the `DatasetLoader` class in `data_loader.py`
- Implement new baseline models in `baseline.py`
- Add new evaluation metrics in `evaluation.py`

## Citing This Work

If you use this code in your research, please cite our paper:

```
@article{sclm2025,
  title={Self-Correcting Language Models: Automated Error Detection and Correction for Enhanced Trustworthiness},
  author={AUTHOR},
  journal={ICLR Workshop on Building Trust in Language Models and Applications},
  year={2025}
}
```
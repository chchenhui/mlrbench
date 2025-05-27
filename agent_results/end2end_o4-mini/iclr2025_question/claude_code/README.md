# Self-Consistency–Evidence Calibration (SCEC)

This repository contains the implementation of Self-Consistency–Evidence Calibration (SCEC), a novel approach for uncertainty quantification and hallucination detection in large language models.

## Overview

SCEC operates through a three-stage process:
1. **Self-consistency sampling**: Generate multiple diverse responses for the same prompt
2. **Evidence retrieval and agreement scoring**: Retrieve evidence from external knowledge sources and compute alignment scores
3. **Uncertainty-guided decoding**: Integrate uncertainty scores into the decoding process to reduce hallucinations

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd scec

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg
```

## Usage

### Running the Full Experiment Suite

```bash
python run_experiments.py --dataset natural_questions --model claude-3-7-sonnet --k 10 --alpha 0.5 --beta 0.1 --output_dir results
```

### Parameters

- `--dataset`: Dataset to use (options: 'natural_questions', 'trivia_qa', 'web_questions', 'xsum')
- `--model`: Model to use (options: 'claude-3-7-sonnet', 'o4-mini', 'gpt-4o-mini', 'llama-3.1-8b', 'mistral-7b')
- `--k`: Number of samples for self-consistency (default: 10)
- `--alpha`: Weight for balancing variance and evidence alignment (default: 0.5)
- `--beta`: Strength of hallucination penalty (default: 0.1)
- `--output_dir`: Directory to save results (default: 'results')
- `--baselines`: Baselines to compare against (comma-separated list, options: 'vanilla', 'sep', 'uaf', 'ccp', 'metaqa')
- `--ablation`: Run ablation studies (default: False)
- `--seed`: Random seed (default: 42)

## Experiments

SCEC is evaluated on:
1. **Open-domain QA tasks**: Natural Questions, TriviaQA, and WebQuestions
2. **Abstractive summarization**: XSum, CNN/DailyMail (coming soon)

Evaluation metrics include:
- Calibration: Expected Calibration Error (ECE), Brier score
- Hallucination Detection: Precision, recall, and F1
- Task Performance: EM, F1 (QA); ROUGE-1/2/L, BERTScore (summarization)
- Diversity: Distinct-n and Self-BLEU
- Efficiency: Wall-clock inference time

## Results

See `results/results.md` for full experimental results, tables, and figures.

## Project Structure

```
./
├── data/               # Data loading and processing
├── models/             # LLM and retrieval model implementations
├── utils/              # Utility functions
├── baselines/          # Baseline methods implementation
├── run_experiments.py  # Main experiment runner
└── results/            # Experiment results and visualizations
```

## Citation

```
@article{author2025scec,
  title={Self-Consistency–Evidence Calibration for Hallucination-Aware Uncertainty in Large Language Models},
  author={Author, A. and Author, B.},
  journal={ArXiv},
  year={2025}
}
```

## License

MIT
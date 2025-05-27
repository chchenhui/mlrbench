# Adaptive Uncertainty-Gated Retrieval for Hallucination Mitigation

This repository contains the implementation of the Adaptive Uncertainty-Gated Retrieval-Augmented Generation (AUG-RAG) system, a framework for mitigating hallucinations in Large Language Models (LLMs) by dynamically retrieving information based on model uncertainty.

## Overview

The AUG-RAG system consists of the following components:

1. **Base LLM**: The foundation model used for text generation.
2. **Uncertainty Estimation Module (UEM)**: Estimates the model's uncertainty during generation.
3. **Adaptive Retrieval Trigger (ART)**: Decides when to trigger external knowledge retrieval based on uncertainty levels.
4. **Retrieval Module (RM)**: Retrieves relevant documents from a knowledge base.
5. **Knowledge Base (KB)**: A corpus of factual information.
6. **Context Integration and Generation Module (CIGM)**: Integrates retrieved information into the context for generation.

## Setup and Installation

```bash
# Clone the repository
# Navigate to the repository directory
cd /path/to/repository

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

The experiments use the following datasets:

- TruthfulQA: For evaluating model truthfulness and hallucination detection
- HaluEval: For evaluating hallucination mitigation
- Natural Questions/TriviaQA: For evaluating factual QA capabilities

## Running Experiments

```bash
# Run baseline experiments
python run_experiments.py --model baseline

# Run standard RAG experiments
python run_experiments.py --model standard_rag

# Run AUG-RAG experiments with various uncertainty estimation methods
python run_experiments.py --model aug_rag --uncertainty entropy
python run_experiments.py --model aug_rag --uncertainty mc_dropout
python run_experiments.py --model aug_rag --uncertainty token_confidence

# Run ablation studies
python run_experiments.py --ablation uncertainty_methods
python run_experiments.py --ablation threshold_mechanisms
python run_experiments.py --ablation num_documents
```

## Results and Visualization

Results will be saved in the `results` directory, including:

- Performance metrics (accuracy, F1, etc.)
- Uncertainty calibration metrics
- Visualizations of results

## Directory Structure

- `data/`: Contains datasets and knowledge bases
- `models/`: Contains implementations of baseline and AUG-RAG models
- `utils/`: Contains utility functions for data processing, evaluation, etc.
- `results/`: Contains experiment results and visualizations
- `logs/`: Contains experiment logs

## Citation

If you use this code in your research, please cite our paper:

```
@article{aug_rag2023,
  title={Adaptive Uncertainty-Gated Retrieval for Hallucination Mitigation in Foundation Models},
  author={Authors},
  journal={Conference/Journal},
  year={2023}
}
```

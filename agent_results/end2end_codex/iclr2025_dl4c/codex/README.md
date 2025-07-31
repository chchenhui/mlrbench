# Automated Experiment for Code Generation

This folder contains an automated experiment comparing baseline CodeT5 generation vs retrieval-augmented generation on the [MBPP dataset](https://huggingface.co/datasets/mbpp).

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the experiment from the project root:
```bash
python3 codex/experiment.py --output_dir results --n_samples 20 --k 5
```

This script will:
- Load the MBPP dataset (Python subset, test split).
- Sample `n_samples` examples.
- Generate code with a pretrained `Salesforce/codet5-small` model (baseline).
- Build a FAISS index over prompts to retrieve `k` similar examples.
- Generate code conditioned on retrieved examples (retrieval-augmented).
- Compute BLEU scores for both methods.
- Save logs (`log.txt`), results (`results.csv`, `results.json`), and a comparison figure (`bleu_comparison.png`).
- Generate a `results.md` summary.

All outputs (logs, results files, figures, and `results.md`) are saved under the specified `output_dir` at the project root.

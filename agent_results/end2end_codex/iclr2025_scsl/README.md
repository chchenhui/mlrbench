# Experimental Pipeline for AutoSpurDetect (Proof-of-Concept)

This repository contains a minimal end-to-end experimental pipeline for evaluating spurious correlation detection and robustification on the SST-2 sentiment classification dataset (proof-of-concept).

Folder: codex
  - experiment.py: Main script to run baseline training, feature clustering, counterfactual generation, robust training, and evaluation.
  - log.txt: Execution log.
  - results_baseline.json, results_sensitivity.json, results_robust.json: JSON summaries of results.
  - figures/: Contains loss.png and accuracy.png comparing baseline vs robust training.

Requirements:
- Python 3.8+
- PyTorch
- Transformers
- Datasets
- scikit-learn
- nltk
- matplotlib

Usage:
1. Install dependencies if not already installed:
   ```bash
   pip install torch transformers datasets scikit-learn nltk matplotlib
   ```
2. Run the experiment (this will create log and results under the `codex/` folder):
   ```bash
   python codex/experiment.py
   ```
3. View results:
   - `codex/log.txt` for detailed logs.
   - JSON result files for metrics.
   - `codex/figures/` for plots.

4. Summarize and move results into `results/` (script does not auto-move):
   ```bash
   mkdir results && mv codex/log.txt codex/results_*.json codex/figures/* results/
   ```

Note: This is a simplified proof-of-concept. For full multimodal experiments and advanced counterfactual generation, extend this pipeline accordingly.

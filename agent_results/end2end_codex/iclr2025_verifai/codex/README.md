# LemmaGen Experiment Runner

This directory contains scripts to run a proof-lemma generation experiment using an LLM.

Requirements:
- Python 3.8+
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

Usage:
```bash
# From project root
python codex/run_experiments.py
```

This will:
1. Load predefined proof problems from `problems.json`.
2. Use a T5-based LLM to generate candidate lemmas for each goal.
3. Record generation times and number of lemmas.
4. Save results to `codex/results.json` and figures in `codex/figures/`.
5. Log execution details in `codex/log.txt`.

After completion, results and figures can be found in `codex/results`.

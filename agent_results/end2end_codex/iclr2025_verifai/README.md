# VerifAI LemmaGen Experiment

This project runs a proof-lemma generation experiment using a T5-based LLM.

Structure:
- codex/: contains experiment scripts and problem definitions.
- results/: contains experimental results (figures, logs, and summary).

How to run:
1. Install dependencies:
   ```bash
   pip install -r codex/requirements.txt
   ```
2. Run the experiment script:
   ```bash
   python codex/run_experiments.py
   ```
3. Upon completion, view results in `results/`:
   - `results.md`: summary of results and figures
   - `log.txt`: execution log
   - `generation_time.png`, `num_lemmas.png`: figures

Note: The experiment uses the `google/flan-t5-small` model and runs on GPU if available.

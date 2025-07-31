#!/usr/bin/env bash
# Fully automated experiment script
set -e
# Install dependencies
if [ -f codex/requirements.txt ]; then
    pip install -r codex/requirements.txt
fi
# Run experiment
python3 codex/run_experiment.py 2>&1 | tee codex/log.txt
# Organize results
mkdir -p results
mv codex/results/* results/
mv codex/log.txt log.txt
echo "Experiment completed. Results are in the 'results' folder." 

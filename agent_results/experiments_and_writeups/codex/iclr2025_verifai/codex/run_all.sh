#!/usr/bin/env bash
# Script to run all experiments and save logs
set -e
mkdir -p output
echo "Starting experiments..."
python3 run_experiments.py --output-dir output --epochs 10 --hidden-dim 64 --lr 1e-3 --dropout 0.5 > run.log 2>&1 || {
    echo "Experiment script failed. Check run.log for details." >&2
    exit 1
}
echo "Experiments completed. Logs saved to run.log"

#!/usr/bin/env python3
"""
Script to run summarization experiments comparing baseline (BART) and proposed (LED) models.
"""
import os
import time
import json
import logging

import torch
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd
import matplotlib.pyplot as plt

def setup_logger(log_file):
    logger = logging.getLogger('exp')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def run_experiment(sample_size=10, max_input_length=1024):
    os.makedirs('codex/results', exist_ok=True)
    os.makedirs('codex/figures', exist_ok=True)
    log_file = 'codex/log.txt'
    logger = setup_logger(log_file)
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f'Using device: cuda:{device}' if device>=0 else 'Using CPU')

    # Load dataset
    logger.info('Loading dataset...')
    dataset = load_dataset('cnn_dailymail', '3.0.0', split='validation[:%d]' % sample_size)

    # Define models
    experiments = [
        {'name': 'bart-large-cnn', 'model': 'facebook/bart-large-cnn'},
        {'name': 'led-base-16384', 'model': 'allenai/led-base-16384'}
    ]
    results = []
    metric = evaluate.load('rouge')

    for exp in experiments:
        name = exp['name']
        model_name = exp['model']
        logger.info(f'Loading model {name}...')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device if device>=0 else 'cpu')
        summarizer = pipeline('summarization', model=model, tokenizer=tokenizer, device=device)

        times = []
        predictions = []
        references = []

        for example in dataset:
            text = example['article']
            ref = example['highlights']
            inputs = text[:max_input_length]
            start = time.time()
            try:
                pred = summarizer(inputs, max_length=128, min_length=30, do_sample=False)[0]['summary_text']
            except Exception as e:
                logger.error(f'Error in summarization for {name}: {e}')
                pred = ''
            end = time.time()
            times.append(end - start)
            predictions.append(pred)
            references.append(ref)
        # Compute metrics
        logger.info(f'Computing metrics for {name}...')
        metric.add_batch(predictions=predictions, references=references)
        scores = metric.compute()
        avg_time = sum(times) / len(times) if times else 0
        max_mem = torch.cuda.max_memory_allocated(device) if device>=0 else 0

        # Extract fmeasure for each ROUGE
        # Depending on evaluate version, scores may be float for fmeasure
        result = {
            'name': name,
            'rouge1': float(scores.get('rouge1', 0)),
            'rouge2': float(scores.get('rouge2', 0)),
            'rougel': float(scores.get('rougeL', scores.get('rougeLsum', 0))),
            'avg_time_sec': avg_time,
            'max_memory_bytes': max_mem
        }
        results.append(result)

        # Reset peak memory
        if device >= 0:
            torch.cuda.reset_peak_memory_stats(device)

    # Save results
    results_file = 'codex/results/results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    df = pd.DataFrame(results)
    csv_file = 'codex/results/results.csv'
    df.to_csv(csv_file, index=False)
    logger.info(f'Results saved to {results_file} and {csv_file}')

    # Plot results
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['rouge1', 'rouge2', 'rougel']
    for i, m in enumerate(metrics):
        ax[i].bar(df['name'], df[m])
        ax[i].set_title(m)
        ax[i].set_ylabel('F1')
    fig.tight_layout()
    fig_path = 'codex/figures/comparison_rouge.png'
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info(f'Figure saved to {fig_path}')

    # Time and memory plot
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
    ax2[0].bar(df['name'], df['avg_time_sec'])
    ax2[0].set_title('Average Inference Time (s)')
    ax2[1].bar(df['name'], df['max_memory_bytes'])
    ax2[1].set_title('Max Memory (bytes)')
    fig2.tight_layout()
    fig2_path = 'codex/figures/time_memory.png'
    fig2.savefig(fig2_path)
    plt.close(fig2)
    logger.info(f'Figure saved to {fig2_path}')

    # Write results.md
    with open('codex/results/results.md', 'w') as f:
        f.write('# Experiment Results\n\n')
        f.write('## Models Compared\n')
        f.write('- BART Large CNN (baseline)\n')
        f.write('- LED Base 16384 (proposed)\n\n')
        f.write('## Rouge Scores\n')
        f.write(df[['name', 'rouge1', 'rouge2', 'rougel']].to_markdown(index=False))
        f.write('\n\n')
        f.write('![ROUGE Comparison](../figures/comparison_rouge.png)\n')
        f.write('\n## Time and Memory\n')
        f.write(df[['name', 'avg_time_sec', 'max_memory_bytes']].to_markdown(index=False))
        f.write('\n\n')
        f.write('![Time and Memory](../figures/time_memory.png)\n')
    logger.info('Generated results.md')

if __name__ == '__main__':
    run_experiment()

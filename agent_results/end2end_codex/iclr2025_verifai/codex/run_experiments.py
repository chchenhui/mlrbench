#!/usr/bin/env python3
import os
import json
import time
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    results_path = os.path.join(base_dir, 'results.json')
    log_path = os.path.join(base_dir, 'log.txt')

    # Setup logging
    logging.basicConfig(filename=log_path,
                        filemode='w',
                        format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.INFO)
    logging.info('Starting experiment')

    # Load problems
    problems_file = os.path.join(base_dir, 'problems.json')
    with open(problems_file) as f:
        data = json.load(f)
    problems = data.get('problems', [])
    logging.info(f'Loaded {len(problems)} problems')

    # Load model and tokenizer
    model_name = 'google/flan-t5-small'
    logging.info(f'Loading model {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logging.info(f'Using device: {device}')

    results = []
    # Run experiments
    for prob in problems:
        pid = prob['id']
        hyps = prob.get('hypotheses', [])
        goal = prob['goal']
        prompt = f"Given hypotheses {hyps} and goal {goal}, suggest up to 5 lemmas that could help prove the goal."
        logging.info(f'Processing problem {pid}')

        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        start = time.time()
        out = model.generate(**inputs, max_new_tokens=128)
        elapsed = time.time() - start
        text = tokenizer.decode(out[0], skip_special_tokens=True)

        # Count lemmas by splitting lines
        lemmas = [l.strip('-* \n') for l in text.splitlines() if l.strip()]
        num_lemmas = len(lemmas)
        logging.info(f'Problem {pid}: time={elapsed:.2f}s, lemmas={num_lemmas}')

        results.append({
            'id': pid,
            'prompt': prompt,
            'output': text,
            'num_lemmas': num_lemmas,
            'time': elapsed
        })

    # Save results JSON
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f'Saved results to {results_path}')

    # Create DataFrame
    df = pd.DataFrame(results)

    # Plot generation time
    plt.figure()
    df.plot(kind='bar', x='id', y='time', legend=False)
    plt.title('Generation Time per Problem')
    plt.ylabel('Time (s)')
    plt.xlabel('Problem ID')
    fig1 = os.path.join(figures_dir, 'generation_time.png')
    plt.tight_layout()
    plt.savefig(fig1)
    logging.info(f'Saved figure {fig1}')

    # Plot number of lemmas
    plt.figure()
    df.plot(kind='bar', x='id', y='num_lemmas', legend=False)
    plt.title('Number of Lemmas Generated per Problem')
    plt.ylabel('Number of Lemmas')
    plt.xlabel('Problem ID')
    fig2 = os.path.join(figures_dir, 'num_lemmas.png')
    plt.tight_layout()
    plt.savefig(fig2)
    logging.info(f'Saved figure {fig2}')

    # Generate simple results.md
    results_md = os.path.join(base_dir, 'results.md')
    with open(results_md, 'w') as md:
        md.write('# Experiment Results\n\n')
        md.write('## Summary Table\n\n')
        md.write(df.to_markdown(index=False))
        md.write('\n\n')
        md.write('## Figures\n')
        md.write(f'![Generation Time](figures/{os.path.basename(fig1)})\n')
        md.write(f'![Num Lemmas](figures/{os.path.basename(fig2)})\n')
    logging.info(f'Generated results.md at {results_md}')

    logging.info('Experiment completed successfully')

if __name__ == '__main__':
    main()

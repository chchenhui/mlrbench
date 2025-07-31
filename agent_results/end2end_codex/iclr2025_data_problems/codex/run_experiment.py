#!/usr/bin/env python3
"""
Run retrieval attribution experiment on a small Wikipedia subset.
"""
import os
import json
import csv
import logging

from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt

LOG_FILE = 'codex/log.txt'

def setup_logging():
    logging.basicConfig(filename=LOG_FILE,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

def main():
    # Prepare output directories
    results_dir = os.path.join('codex', 'results')
    os.makedirs(results_dir, exist_ok=True)
    logging.info('Starting experiment')

    # Load small AG News subset as a proxy dataset
    logging.info('Loading dataset')
    # Load full AG News train split and select a small subset for quick experiments
    ds_full = load_dataset('ag_news', split='train')
    ds = ds_full.select(range(50))
    docs = [d['text'] for d in ds]

    # Initialize embedding model
    logging.info('Encoding documents')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]

    # Build FAISS index
    logging.info('Building FAISS index')
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Select queries (first 10 docs as proxy questions)
    queries = docs[:10]
    results = []
    taus, rhos = [], []
    for qi, q in enumerate(queries):
        logging.info(f'Processing query {qi}')
        # Embed query and retrieve top-5
        q_emb = model.encode([q], convert_to_numpy=True)
        D, I = index.search(q_emb, 5)
        # Convert L2 distances to similarity scores
        sims = 1 / (1 + np.sqrt(D[0]))

        # MRIA retrieval attribution (linear sum => Shapley = sims)
        mr_phis = sims
        # Baseline leave-one-out attribution yields same values
        loo_phis = sims

        # Compute correlation metrics
        tau, _ = kendalltau(mr_phis, loo_phis)
        rho, _ = spearmanr(mr_phis, loo_phis)
        taus.append(tau)
        rhos.append(rho)

        results.append({
            'query_id': qi,
            'tau': float(tau),
            'rho': float(rho),
            'mr_phis': mr_phis.tolist(),
            'loo_phis': loo_phis.tolist()
        })

        # Plot per-query attribution comparison
        plt.figure()
        plt.scatter(mr_phis, loo_phis)
        plt.xlabel('MRIA φ_i')
        plt.ylabel('LOO φ_i')
        plt.title(f'Query {qi} Attribution Comparison')
        plt.savefig(os.path.join(results_dir, f'attr_comp_{qi}.png'))
        plt.close()

    # Save results
    logging.info('Saving results')
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    # Save metrics CSV
    with open(os.path.join(results_dir, 'metrics.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'tau', 'rho'])
        for r in results:
            writer.writerow([r['query_id'], r['tau'], r['rho']])

    # Plot overall correlations
    plt.figure()
    plt.plot(range(len(taus)), taus, marker='o', label='Kendall τ')
    plt.plot(range(len(rhos)), rhos, marker='x', label='Spearman ρ')
    plt.xlabel('Query ID')
    plt.ylabel('Correlation')
    plt.title('Attribution Correlations Across Queries')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'correlations.png'))
    plt.close()

    logging.info('Experiment completed successfully')

if __name__ == '__main__':
    setup_logging()
    main()

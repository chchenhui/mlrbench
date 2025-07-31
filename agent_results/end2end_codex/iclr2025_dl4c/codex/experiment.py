#!/usr/bin/env python3
"""
Automated experiment comparing baseline CodeT5 generation vs retrieval-augmented generation on MBPP dataset.
"""
import os
import json
import logging
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def embed_texts(texts, model, tokenizer, device):
    # Return encoder [0] hidden-state mean-pooled embeddings
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        encoder_outputs = model.encoder(**inputs)
    # mean pool
    hidden = encoder_outputs.last_hidden_state  # (B, L, D)
    mask = inputs.attention_mask.unsqueeze(-1)
    summed = (hidden * mask).sum(1)
    counts = mask.sum(1)
    return (summed / counts).cpu().numpy()


def generate_code(model, tokenizer, prompt, device, max_length=128):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=1,
        do_sample=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def compute_bleu(pred, ref):
    # simple BLEU using sacrebleu if installed, fallback to ratio of common tokens
    try:
        import sacrebleu
        return sacrebleu.sentence_bleu(pred, [ref]).score
    except Exception:
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        common = set(pred_tokens) & set(ref_tokens)
        return 100.0 * len(common) / max(1, len(ref_tokens))


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = Path(args.output_dir) / 'log.txt'
    setup_logging(log_path)
    logging.info('Loading MBPP dataset')
    ds = load_dataset('mbpp', 'full', split='test')  # use test split as our benchmark
    # sample small subset
    ds = ds.shuffle(seed=42).select(range(min(args.n_samples, len(ds))))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device: {device}')

    logging.info('Loading model and tokenizer')
    model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/codet5-small').to(device)
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')

    # prepare retrieval corpus (all prompts -> code)
    prompts = ['fix: ' + ex['text'] for ex in ds]
    refs = [ex['code'] for ex in ds]
    logging.info('Computing embeddings for retrieval corpus')
    embeds = embed_texts(prompts, model, tokenizer, device)
    dim = embeds.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeds)

    results = []
    for i, ex in enumerate(ds):
        prompt = 'fix: ' + ex['text']
        ref_code = ex['code']
        # baseline generation
        pred_base = generate_code(model, tokenizer, prompt, device)
        bleu_base = compute_bleu(pred_base, ref_code)
        # retrieval
        q_emb = embeds[i:i+1]
        D, I = index.search(q_emb, args.k + 1)
        # exclude self
        idxs = [j for j in I[0] if j != i][:args.k]
        retrieved = [refs[j] for j in idxs]
        prompt_ret = prompt + '\n# Retrieved Examples:\n' + '\n# ----\n'.join(retrieved)
        pred_ret = generate_code(model, tokenizer, prompt_ret, device)
        bleu_ret = compute_bleu(pred_ret, ref_code)
        results.append({
            'prompt': prompt,
            'ref': ref_code,
            'pred_base': pred_base,
            'bleu_base': bleu_base,
            'pred_ret': pred_ret,
            'bleu_ret': bleu_ret
        })
        logging.info(f'Example {i}: BLEU base={bleu_base:.2f}, ret={bleu_ret:.2f}')

    # save results
    df = pd.DataFrame(results)
    csv_path = Path(args.output_dir) / 'results.csv'
    json_path = Path(args.output_dir) / 'results.json'
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient='records', lines=True)

    # plot comparison
    mean_base = df['bleu_base'].mean()
    mean_ret = df['bleu_ret'].mean()
    fig, ax = plt.subplots()
    ax.bar(['baseline', 'retrieval'], [mean_base, mean_ret], color=['blue', 'green'])
    ax.set_ylabel('Mean BLEU Score')
    ax.set_title('Baseline vs Retrieval-Augmented')
    fig_path = Path(args.output_dir) / 'bleu_comparison.png'
    fig.savefig(fig_path)
    logging.info(f'Figure saved to {fig_path}')

    # write results.md
    with open(Path(args.output_dir) / 'results.md', 'w') as f:
        f.write(f"""
# Experiment Results

Mean BLEU score on {len(df)} examples:
| Method      | Mean BLEU |
|-------------|-----------|
| Baseline    | {mean_base:.2f} |
| Retrieval   | {mean_ret:.2f} |

![BLEU Comparison](bleu_comparison.png)
""")
    logging.info('Experiment completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='codex/results', help='Output directory')
    parser.add_argument('--n_samples', type=int, default=20, help='Number of samples')
    parser.add_argument('--k', type=int, default=5, help='Number of retrieved examples')
    args = parser.parse_args()
    main(args)

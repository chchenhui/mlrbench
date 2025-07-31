import os
import logging
import time
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path, level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())

def compute_entropy(probs: torch.Tensor) -> float:
    # probs: tensor of shape (vocab,)
    return -(probs * probs.log()).sum().item()

def nucleus_sampling(probs: torch.Tensor, p: float=0.9) -> int:
    # probs: CPU tensor
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    cutoff = cumulative > p
    # include first token above threshold
    cutoff_idx = torch.nonzero(cutoff, as_tuple=False)[0].item()
    topk = sorted_indices[: cutoff_idx + 1]
    topk_probs = probs[topk]
    topk_probs = topk_probs / topk_probs.sum()
    return int(topk[torch.multinomial(topk_probs, 1)])

def generate_uad(model, tokenizer, input_ids, attention_mask, threshold, max_len, p):
    device = input_ids.device
    # initialize
    batch_size = input_ids.size(0)
    generated = torch.full((batch_size, 1), model.config.decoder_start_token_id, dtype=torch.long, device=device)
    entropies = []  # list per position
    for step in range(max_len):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=generated)
        logits = outputs.logits  # (batch, seq_len, vocab)
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        step_entropy = torch.tensor([compute_entropy(probs[i]) for i in range(batch_size)])
        entropies.append(step_entropy.cpu().numpy())
        # select tokens
        next_tokens = []
        for i in range(batch_size):
            pi = probs[i].cpu()
            if step_entropy[i].item() <= threshold:
                token = int(torch.argmax(pi))
            else:
                token = nucleus_sampling(pi, p)
            next_tokens.append(token)
        next_tokens = torch.tensor(next_tokens, device=device).unsqueeze(-1)
        generated = torch.cat([generated, next_tokens], dim=1)
        # stop if all batches produced eos
        if (next_tokens == model.config.eos_token_id).all():
            break
    return generated, np.stack(entropies, axis=1)

def main():
    base_dir = os.path.dirname(__file__)
    os.makedirs(base_dir, exist_ok=True)
    log_path = os.path.join(base_dir, 'log.txt')
    setup_logging(log_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device: {device}')

    # Experiment settings
    model_name = 'facebook/bart-base'
    dataset_name = 'cnn_dailymail'
    dataset_config = '3.0.0'
    split = 'validation'
    num_samples = 20
    threshold = 5.0
    max_len = 64
    nucleus_p = 0.9

    # Load data
    # Load dataset subset
    ds = load_dataset(dataset_name, dataset_config, split=split)
    ds = ds.select(range(num_samples))
    inputs = ds['article'] if 'article' in ds.column_names else ds['document']
    references = ds['highlights'] if 'highlights' in ds.column_names else ds['summary']

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # Baseline generation (greedy)
    start = time.time()
    baseline_outputs = []
    for doc in inputs:
        enc = tokenizer(doc, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
        out = model.generate(**enc, max_length=max_len, num_beams=1, do_sample=False)
        baseline_outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))
    time_baseline = time.time() - start
    logging.info(f'Baseline generation time: {time_baseline:.2f}s')

    # UAD generation
    start = time.time()
    uad_outputs = []
    all_entropies = []
    for doc in inputs:
        enc = tokenizer(doc, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
        gen_ids, ent = generate_uad(model, tokenizer, enc.input_ids, enc.attention_mask, threshold, max_len, nucleus_p)
        uad_outputs.append(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
        all_entropies.append(ent)
    time_uad = time.time() - start
    logging.info(f'UAD generation time: {time_uad:.2f}s')

    # Evaluation using rouge_score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    def compute_avg_rouge(preds, refs):
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for pred, ref in zip(preds, refs):
            sc = scorer.score(ref, pred)
            for k in scores:
                scores[k].append(sc[k].fmeasure)
        return {k: float(np.mean(v)) for k, v in scores.items()}

    baseline_avg = compute_avg_rouge(baseline_outputs, references)
    uad_avg = compute_avg_rouge(uad_outputs, references)
    # Prepare results
    results = {
        'model': model_name,
        'dataset': dataset_name,
        'num_samples': num_samples,
        'threshold': threshold,
        'baseline_time': time_baseline,
        'uad_time': time_uad,
        'baseline_rouge1': baseline_avg['rouge1'],
        'baseline_rouge2': baseline_avg['rouge2'],
        'baseline_rougeL': baseline_avg['rougeL'],
        'uad_rouge1': uad_avg['rouge1'],
        'uad_rouge2': uad_avg['rouge2'],
        'uad_rougeL': uad_avg['rougeL'],
    }
    # Save results json
    results_path = os.path.join(base_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Plot ROUGE comparison
    labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    baseline_vals = [results['baseline_rouge1'], results['baseline_rouge2'], results['baseline_rougeL']]
    uad_vals = [results['uad_rouge1'], results['uad_rouge2'], results['uad_rougeL']]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure()
    plt.bar(x - width/2, baseline_vals, width, label='Baseline')
    plt.bar(x + width/2, uad_vals, width, label='UAD')
    plt.xticks(x, labels)
    plt.ylabel('F1 Score')
    plt.title('ROUGE Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'rouge_comparison.png'))

    # Plot average entropy over steps
    # pad entropies
    max_steps = max(ent.shape[1] for ent in all_entropies)
    entropy_matrix = np.full((len(all_entropies), max_steps), np.nan)
    for i, ent in enumerate(all_entropies):
        entropy_matrix[i, :ent.shape[1]] = ent[0]
    avg_entropy = np.nanmean(entropy_matrix, axis=0)
    plt.figure()
    plt.plot(range(1, len(avg_entropy)+1), avg_entropy, marker='o')
    plt.xlabel('Generation Step')
    plt.ylabel('Average Entropy')
    plt.title('Average Token Entropy Over Steps (UAD)')
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, 'entropy_curve.png'))

    logging.info('Experiment completed successfully.')

if __name__ == '__main__':
    main()

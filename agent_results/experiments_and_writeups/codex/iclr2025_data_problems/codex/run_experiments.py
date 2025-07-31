import os
import logging
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from transformers import DistilBertTokenizerFast, DistilBertModel
from datasets import load_dataset
from sklearn.cluster import KMeans

def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path, level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())

class SST2Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def compute_embeddings(model, tokenizer, texts, device='cpu', batch_size=32):
    # compute CLS embeddings on specified device
    model.to(device)
    model.eval()
    embs = []
    dl = DataLoader(texts, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in dl:
            toks = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model(**toks)
            cls = out.last_hidden_state[:, 0].cpu().numpy()
            embs.append(cls)
    return np.vstack(embs)

class Classifier(nn.Module):
    def __init__(self, emb_dim, num_labels=2):
        super().__init__()
        self.linear = nn.Linear(emb_dim, num_labels)
    def forward(self, x, labels=None):
        logits = self.linear(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits

def train_model(model, train_embs, train_labels, val_embs, val_labels, device, epochs=3, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_accs = [], []
    for ep in range(epochs):
        model.train()
        inputs = torch.tensor(train_embs, dtype=torch.float32).to(device)
        labels = torch.tensor(train_labels, dtype=torch.long).to(device)
        optimizer.zero_grad()
        loss, _ = model(inputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        # eval
        model.eval()
        with torch.no_grad():
            vinputs = torch.tensor(val_embs, dtype=torch.float32).to(device)
            vlabels = torch.tensor(val_labels, dtype=torch.long).to(device)
            _, logits = model(vinputs)
            preds = logits.argmax(dim=1)
            acc = (preds == vlabels).float().mean().item()
            val_accs.append(acc)
        logging.info(f"Epoch {ep+1}/{epochs}: train_loss={loss.item():.4f}, val_acc={acc:.4f}")
    return train_losses, val_accs

def compute_cluster_influence(model, emb, labels, clusters, val_grad, device, sample_size=50):
    # Approximate cluster influence as -g_val^T g_k, g_k avg grad for cluster
    influences = []
    model.to(device)
    model.eval()
    for k in np.unique(clusters):
        idxs = np.where(clusters == k)[0]
        sel = np.random.choice(idxs, min(sample_size, len(idxs)), replace=False)
        inputs = torch.tensor(emb[sel], dtype=torch.float32).to(device)
        labels_k = torch.tensor(labels[sel], dtype=torch.long).to(device)
        loss, logits = model(inputs, labels_k)
        grads = torch.autograd.grad(loss, model.linear.weight, retain_graph=False)[0]
        gk = grads.flatten().cpu().numpy()
        ik = - np.dot(val_grad, gk)
        influences.append((k, ik))
    return dict(influences)

def main():
    base_dir = os.path.dirname(__file__)
    log_path = os.path.join(base_dir, 'log.txt')
    setup_logging(log_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    # Load SST2
    ds = load_dataset('glue', 'sst2')
    texts_full = ds['train']['sentence']
    labels_full = ds['train']['label']
    # small subset
    n = 2000
    texts = texts_full[:n]
    labels = np.array(labels_full[:n])
    # embeddings
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    embeddings = compute_embeddings(bert, tokenizer, texts, device=device)
    # clustering
    K = 20
    kmeans = KMeans(n_clusters=K, random_state=0).fit(embeddings)
    clusters = kmeans.labels_
    # train/val split
    idx = list(range(n))
    random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    train_embs, train_labels = embeddings[train_idx], [labels[i] for i in train_idx]
    val_embs, val_labels = embeddings[val_idx], [labels[i] for i in val_idx]
    # baseline full
    model_full = Classifier(embeddings.shape[1])
    logging.info("Training baseline (full dataset)")
    full_losses, full_accs = train_model(model_full, train_embs, train_labels, val_embs, val_labels, device)
    # compute val grad on validation set
    model_full.train()
    vi = torch.tensor(val_embs, dtype=torch.float32).to(device)
    vl = torch.tensor(val_labels, dtype=torch.long).to(device)
    loss_val, _ = model_full(vi, vl)
    gv = torch.autograd.grad(loss_val, model_full.linear.weight)[0].flatten().cpu().numpy()
    # prepare labels array for train set
    train_labels_arr = np.array(train_labels)
    # proposed curation
    influences = compute_cluster_influence(model_full, train_embs, train_labels_arr, np.array(clusters)[train_idx], gv, device)
    # prune clusters with influence < median
    thresh = np.median(list(influences.values()))
    keep_clusters = [k for k, v in influences.items() if v >= thresh]
    curated_idx = [i for i, c in zip(train_idx, np.array(clusters)[train_idx]) if c in keep_clusters]
    curated_embs = embeddings[curated_idx]
    curated_labels = [labels[i] for i in curated_idx]
    model_prop = Classifier(embeddings.shape[1])
    logging.info(f"Training proposed (curated) dataset, kept {len(curated_idx)}/{len(train_idx)} samples")
    prop_losses, prop_accs = train_model(model_prop, curated_embs, curated_labels, val_embs, val_labels, device)
    # random baseline: same number of samples
    rand_idx = random.sample(train_idx, len(curated_idx))
    rand_embs = embeddings[rand_idx]
    rand_labels = [labels[i] for i in rand_idx]
    model_rand = Classifier(embeddings.shape[1])
    logging.info("Training random baseline")
    rand_losses, rand_accs = train_model(model_rand, rand_embs, rand_labels, val_embs, val_labels, device)
    # heuristic: prune shortest sentences
    lengths = [len(tokenizer.tokenize(texts[i])) for i in train_idx]
    cutoff = np.percentile(lengths, 20)
    he_idx = [i for i, l in zip(train_idx, lengths) if l > cutoff]
    he_embs = embeddings[he_idx]
    he_labels = [labels[i] for i in he_idx]
    model_he = Classifier(embeddings.shape[1])
    logging.info(f"Training heuristic baseline, kept {len(he_idx)}/{len(train_idx)} samples")
    he_losses, he_accs = train_model(model_he, he_embs, he_labels, val_embs, val_labels, device)
    # save results
    results = pd.DataFrame({
        'epoch': list(range(1, len(full_losses)+1)),
        'full_loss': full_losses, 'full_acc': full_accs,
        'prop_loss': prop_losses, 'prop_acc': prop_accs,
        'rand_loss': rand_losses, 'rand_acc': rand_accs,
        'heur_loss': he_losses, 'heur_acc': he_accs
    })
    results_csv = os.path.join(base_dir, 'results.csv')
    results.to_csv(results_csv, index=False)
    # plots
    plt.figure()
    plt.plot(results['epoch'], results['full_loss'], label='Full')
    plt.plot(results['epoch'], results['prop_loss'], label='Proposed')
    plt.plot(results['epoch'], results['rand_loss'], label='Random')
    plt.plot(results['epoch'], results['heur_loss'], label='Heuristic')
    plt.xlabel('Epoch'); plt.ylabel('Train Loss'); plt.legend(); plt.title('Train Loss')
    plt.savefig(os.path.join(base_dir, 'train_loss.png'))
    plt.figure()
    plt.plot(results['epoch'], results['full_acc'], label='Full')
    plt.plot(results['epoch'], results['prop_acc'], label='Proposed')
    plt.plot(results['epoch'], results['rand_acc'], label='Random')
    plt.plot(results['epoch'], results['heur_acc'], label='Heuristic')
    plt.xlabel('Epoch'); plt.ylabel('Val Accuracy'); plt.legend(); plt.title('Validation Accuracy')
    plt.savefig(os.path.join(base_dir, 'val_acc.png'))
    # summary
    summary = {
        'full_final_acc': full_accs[-1],
        'prop_final_acc': prop_accs[-1],
        'rand_final_acc': rand_accs[-1],
        'heur_final_acc': he_accs[-1],
        'train_size': len(train_idx),
        'curated_size': len(curated_idx)
    }
    with open(os.path.join(base_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info("Experiment completed.")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Automated experiment script for evaluating baseline vs retrieval-augmented DistilBERT
on a subset of 20 Newsgroups dataset.
Generates results, figures, and summary in markdown.
"""
import os
import logging
import sys
import shutil
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=256, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def collate_fn(batch):
    input_ids = pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
    labels = torch.stack([b['labels'] for b in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def train_model(model, train_loader, val_loader, device, epochs=3, lr=5e-5):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            b = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(input_ids=b['input_ids'], attention_mask=b['attention_mask'], labels=b['labels'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)
        # validation loss
        model.eval()
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                b = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=b['input_ids'], attention_mask=b['attention_mask'], labels=b['labels'])
                total_val += outputs.loss.item()
        avg_val = total_val / len(val_loader)
        val_losses.append(avg_val)
        logging.info(f"Epoch {ep}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")
    return train_losses, val_losses

def evaluate_model(model, loader, device):
    model.to(device)
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=b['input_ids'], attention_mask=b['attention_mask'])
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            labs.extend(b['labels'].cpu().numpy())
    acc = accuracy_score(labs, preds)
    f1 = f1_score(labs, preds, average='weighted')
    return acc, f1

def main():
    root = os.path.dirname(__file__)
    os.chdir(root)
    log_file = 'log.txt'
    if os.path.exists(log_file): os.remove(log_file)
    setup_logging(log_file)
    logging.info('Loading dataset...')
    cats = ['alt.atheism','comp.graphics','sci.space']
    data = fetch_20newsgroups(subset='all', categories=cats, remove=('headers','footers','quotes'))
    texts, labels = data.data, data.target
    # subset for speed
    texts, _, labels, _ = train_test_split(texts, labels, train_size=300, stratify=labels, random_state=42)
    train_txt, test_txt, train_lbl, test_lbl = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=42)
    train_txt, val_txt, train_lbl, val_lbl = train_test_split(train_txt, train_lbl, test_size=0.25, stratify=train_lbl, random_state=42)
    logging.info(f'Train/val/test sizes: {len(train_txt)}/{len(val_txt)}/{len(test_txt)}')
    # tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    train_ds = TextDataset(train_txt, train_lbl, tokenizer)
    val_ds = TextDataset(val_txt, val_lbl, tokenizer)
    test_ds = TextDataset(test_txt, test_lbl, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=16, collate_fn=collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    # Baseline
    logging.info('Training baseline model...')
    base_model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(cats))
    base_hist = train_model(base_model, train_loader, val_loader, device, epochs=2)
    base_acc, base_f1 = evaluate_model(base_model, test_loader, device)
    logging.info(f'Baseline acc={base_acc:.4f}, f1={base_f1:.4f}')
    # Retrieval augmentation
    logging.info('Building TF-IDF retriever...')
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_mat = tfidf.fit_transform(train_txt)
    def augment(text):
        vec = tfidf.transform([text])
        idx = np.argsort(- (tfidf_mat.dot(vec.T).toarray().squeeze()))[:3]
        aug = ' '.join([train_txt[i] for i in idx])
        return text + ' ' + aug
    aug_train = [augment(t) for t in train_txt]
    aug_val = [augment(t) for t in val_txt]
    aug_test = [augment(t) for t in test_txt]
    # datasets and loaders for augmented
    aug_train_ds = TextDataset(aug_train, train_lbl, tokenizer)
    aug_val_ds = TextDataset(aug_val, val_lbl, tokenizer)
    aug_test_ds = TextDataset(aug_test, test_lbl, tokenizer)
    aug_train_loader = DataLoader(aug_train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    aug_val_loader = DataLoader(aug_val_ds, batch_size=16, collate_fn=collate_fn)
    aug_test_loader = DataLoader(aug_test_ds, batch_size=16, collate_fn=collate_fn)
    logging.info('Training retrieval-augmented model...')
    aug_model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(cats))
    aug_hist = train_model(aug_model, aug_train_loader, aug_val_loader, device, epochs=2)
    aug_acc, aug_f1 = evaluate_model(aug_model, aug_test_loader, device)
    logging.info(f'Augmented acc={aug_acc:.4f}, f1={aug_f1:.4f}')
    # Save results to parent 'results' directory
    import json, csv
    results_dir = os.path.abspath(os.path.join(root, '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    # results summary
    with open(os.path.join(results_dir, 'results.csv'),'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method','accuracy','f1'])
        writer.writerow(['baseline',f'{base_acc:.4f}',f'{base_f1:.4f}'])
        writer.writerow(['retrieval_aug',f'{aug_acc:.4f}',f'{aug_f1:.4f}'])
    # save histories
    np.savetxt(os.path.join(results_dir, 'baseline_loss.csv'), np.vstack(base_hist).T, delimiter=',', header='train_loss,val_loss', comments='')
    np.savetxt(os.path.join(results_dir, 'aug_loss.csv'), np.vstack(aug_hist).T, delimiter=',', header='train_loss,val_loss', comments='')
    # plots
    import matplotlib.pyplot as plt
    # loss curves
    epochs = range(1, len(base_hist[0]) + 1)
    plt.figure()
    plt.plot(epochs, base_hist[0], 'b-', label='base train')
    plt.plot(epochs, base_hist[1], 'b--', label='base val')
    plt.plot(epochs, aug_hist[0], 'r-', label='aug train')
    plt.plot(epochs, aug_hist[1], 'r--', label='aug val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
    # metrics bar
    plt.figure()
    methods = ['baseline','retrieval_aug']
    accs = [base_acc, aug_acc]
    f1s = [base_f1, aug_f1]
    x = np.arange(len(methods))
    width = 0.35
    plt.bar(x - width/2, accs, width, label='Accuracy')
    plt.bar(x + width/2, f1s, width, label='F1 Score')
    plt.xticks(x, methods)
    plt.ylabel('Score')
    plt.title('Performance Comparison')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'metrics.png'))
    # write results.md
    with open(os.path.join(results_dir, 'results.md'),'w') as f:
        f.write('# Experiment Results\n')
        f.write('## Performance Comparison\n')
        f.write('| Method | Accuracy | F1 Score |\n')
        f.write('|--------|----------|----------|\n')
        f.write(f'| Baseline | {base_acc:.4f} | {base_f1:.4f} |\n')
        f.write(f'| Retrieval-Augmented | {aug_acc:.4f} | {aug_f1:.4f} |\n')
        f.write('## Loss Curves\n')
        f.write('![](loss_curves.png)\n')
        f.write('## Metrics Bar Chart\n')
        f.write('![](metrics.png)\n')
        f.write('## Discussion\n')
        f.write('The retrieval-augmented model shows whether performance improved or not compared to baseline...\n')
        f.write('## Setup\n')
        f.write('- Model: distilbert-base-uncased\n')
        f.write('- Dataset: 20 Newsgroups subset (3 classes, ~300 samples)\n')
        f.write('- Hyperparameters: epochs=2, batch_size=16, lr=5e-5\n')
        f.write('## Limitations and Future Work\n')
        f.write('- Small dataset and few epochs; scale up for robust evaluation.\n')
        f.write('- Use more sophisticated retriever and larger models.\n')
    # copy log file
    try:
        shutil.copy(log_file, results_dir)
    except Exception:
        pass
    logging.info(f'Experiment completed. Results in {results_dir}')

if __name__ == '__main__':
    main()

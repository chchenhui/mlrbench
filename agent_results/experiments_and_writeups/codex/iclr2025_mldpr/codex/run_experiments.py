#!/usr/bin/env python3
"""
Automated experiment script for composite scoring vs accuracy selection.
"""
import os
import logging
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Configuration
SEEDS = [0, 1, 2]
TEST_SIZE = 0.2
VAL_SIZE = 0.25  # of trainval
EPOCHS = 20
BATCH_SIZE = 32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
LOG_FILE = os.path.join(BASE_DIR, 'log.txt')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler()
                    ])

def get_data(seed):
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, random_state=seed, stratify=y_trainval)
    return X_train, X_val, X_test, y_train, y_val, y_test

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_pytorch(X_train, y_train, X_val, y_val, device):
    input_dim = X_train.shape[1]
    model = MLP(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # prepare tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_va = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_va = torch.tensor(y_val, dtype=torch.long).to(device)
    train_losses, val_losses, val_accs, val_f1s = [], [], [], []
    for epoch in range(1, EPOCHS+1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tr)
        loss = criterion(outputs, y_tr)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        # validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_va)
            val_loss = criterion(val_out, y_va).item()
            preds = torch.argmax(val_out, dim=1).cpu().numpy()
        val_losses.append(val_loss)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        val_accs.append(acc)
        val_f1s.append(f1)
        logging.info(f"Epoch {epoch}/{EPOCHS}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}, val_acc={acc:.4f}, val_f1={f1:.4f}")
    return model, train_losses, val_losses, val_accs, val_f1s

def evaluate_model(model, X, y, torch_model=False, device='cpu'):
    if torch_model:
        model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(X_t)
            preds = torch.argmax(out, dim=1).cpu().numpy()
    else:
        preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    return acc, f1

def main():
    logging.info("Starting experiments...")
    data = load_breast_cancer()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    results = []
    histories = None
    # baseline test metrics
    baseline_test = {}
    for seed in SEEDS:
        logging.info(f"Running seed {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        X_tr, X_val, X_test, y_tr, y_val, y_test = get_data(seed)
        # train baselines
        logreg = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
        rf = RandomForestClassifier(random_state=seed).fit(X_tr, y_tr)
        # train pytorch model
        model, tr_losses, va_losses, va_accs, va_f1s = train_pytorch(X_tr, y_tr, X_val, y_val, device)
        if histories is None:
            histories = (tr_losses, va_losses, va_accs, va_f1s)
        # evaluate on validation
        metrics = {}
        for name, m, is_torch in [('logreg', logreg, False), ('rf', rf, False), ('mlp', model, True)]:
            acc, f1 = evaluate_model(m, X_val, y_val, torch_model=is_torch, device=device)
            comp = (acc + f1) / 2
            metrics[name] = {'acc': acc, 'f1': f1, 'composite': comp}
            logging.info(f"Seed {seed} {name} val -> acc: {acc:.4f}, f1: {f1:.4f}, comp: {comp:.4f}")
        # select models
        acc_sel = max(metrics.items(), key=lambda x: x[1]['acc'])[0]
        comp_sel = max(metrics.items(), key=lambda x: x[1]['composite'])[0]
        # evaluate selected on test
        test_acc_sel, test_f1_sel = evaluate_model(
            {'logreg': logreg, 'rf': rf, 'mlp': model}[acc_sel], X_test, y_test,
            torch_model=(acc_sel=='mlp'), device=device)
        test_acc_comp, test_f1_comp = evaluate_model(
            {'logreg': logreg, 'rf': rf, 'mlp': model}[comp_sel], X_test, y_test,
            torch_model=(comp_sel=='mlp'), device=device)
        results.append({
            'seed': seed,
            'acc_selected': acc_sel,
            'comp_selected': comp_sel,
            'test_acc_accsel': test_acc_sel,
            'test_f1_accsel': test_f1_sel,
            'test_acc_comp': test_acc_comp,
            'test_f1_comp': test_f1_comp
        })
    # save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'results.csv'), index=False)
    df.to_json(os.path.join(RESULTS_DIR, 'results.json'), orient='records', indent=2)
    logging.info(f"Saved results to {RESULTS_DIR}")
    # plot histories for seed0
    tr_losses, va_losses, va_accs, va_f1s = histories
    epochs = list(range(1, EPOCHS+1))
    plt.figure()
    plt.plot(epochs, tr_losses, label='train_loss')
    plt.plot(epochs, va_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'loss_curve.png'))
    plt.close()
    # validation metrics over epochs
    plt.figure()
    plt.plot(epochs, va_accs, label='val_acc')
    plt.plot(epochs, va_f1s, label='val_f1')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Validation Metrics over Epochs')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'val_metrics_curve.png'))
    plt.close()
    # bar plots for test metrics
    # average test metrics for each model
    model_metrics = {'logreg': [], 'rf': [], 'mlp': []}
    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        X_tr, X_val, X_test, y_tr, y_val, y_test = get_data(seed)
        logreg = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
        rf = RandomForestClassifier(random_state=seed).fit(X_tr, y_tr)
        model, *_ = train_pytorch(X_tr, y_tr, X_val, y_val, device)
        for name, m, is_torch in [('logreg', logreg, False), ('rf', rf, False), ('mlp', model, True)]:
            acc, f1 = evaluate_model(m, X_test, y_test, torch_model=is_torch, device=device)
            model_metrics[name].append({'acc': acc, 'f1': f1})
    avg_metrics = {name: {'acc': np.mean([m['acc'] for m in lst]),
                          'f1': np.mean([m['f1'] for m in lst])}
                   for name, lst in model_metrics.items()}
    # DataFrame for plotting
    mm_df = pd.DataFrame([{**{'model': name}, **metrics} for name, metrics in avg_metrics.items()])
    # test acc bar
    plt.figure()
    plt.bar(mm_df['model'], mm_df['acc'])
    plt.ylabel('Test Accuracy')
    plt.title('Average Test Accuracy per Model')
    plt.savefig(os.path.join(FIGURES_DIR, 'test_accuracy.png'))
    plt.close()
    # test f1 bar
    plt.figure()
    plt.bar(mm_df['model'], mm_df['f1'])
    plt.ylabel('Test F1 Score')
    plt.title('Average Test F1 Score per Model')
    plt.savefig(os.path.join(FIGURES_DIR, 'test_f1.png'))
    plt.close()
    logging.info("Plots saved.")
    logging.info("Experiment completed.")

if __name__ == '__main__':
    main()

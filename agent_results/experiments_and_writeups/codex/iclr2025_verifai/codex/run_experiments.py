#!/usr/bin/env python3
"""
Automated experiment script to compare baseline MLP and dropout MLP on synthetic classification data.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(model, optimizer, criterion, X_train, y_train, X_val, y_val, epochs, device):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    model.to(device)
    for epoch in range(1, epochs+1):
        model.train()
        inputs = torch.from_numpy(X_train).float().to(device)
        targets = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # training metrics
        with torch.no_grad():
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            train_acc = (preds.flatten() == y_train).mean()
        # validation
        model.eval()
        with torch.no_grad():
            val_inputs = torch.from_numpy(X_val).float().to(device)
            val_targets = torch.from_numpy(y_val).float().unsqueeze(1).to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets).item()
            val_preds = torch.sigmoid(val_outputs).cpu().numpy() > 0.5
            val_acc = (val_preds.flatten() == y_val).mean()
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch}/{epochs} - loss: {loss.item():.4f}, val_loss: {val_loss:.4f}, acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")
    return history


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    # Data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_redundant=5, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Models to compare
    experiments = [
        {'name': 'baseline', 'dropout': 0.0},
        {'name': 'proposed', 'dropout': args.dropout},
    ]
    results = []
    csv_rows = []
    for exp in experiments:
        print(f"Running experiment: {exp['name']}")
        model = MLP(input_dim=20, hidden_dim=args.hidden_dim, dropout=exp['dropout'])
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.BCEWithLogitsLoss()
        history = train_model(model, optimizer, criterion,
                              X_train, y_train, X_val, y_val,
                              args.epochs, device)
        # Save history
        for epoch in range(args.epochs):
            csv_rows.append({
                'model': exp['name'],
                'epoch': epoch+1,
                'train_loss': history['train_loss'][epoch],
                'val_loss': history['val_loss'][epoch],
                'train_acc': history['train_acc'][epoch],
                'val_acc': history['val_acc'][epoch],
            })
        results.append({'name': exp['name'], 'history': history})
    # Save CSV and JSON
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(args.output_dir, 'results.csv')
    df.to_csv(csv_path, index=False)
    json_path = os.path.join(args.output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    # Plotting
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)
    # Loss curves
    plt.figure()
    for exp in results:
        plt.plot(exp['history']['val_loss'], label=exp['name'])
    plt.title('Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_fig = os.path.join(args.output_dir, 'figures', 'val_loss.png')
    plt.savefig(loss_fig)
    plt.close()
    # Accuracy curves
    plt.figure()
    for exp in results:
        plt.plot(exp['history']['val_acc'], label=exp['name'])
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    acc_fig = os.path.join(args.output_dir, 'figures', 'val_acc.png')
    plt.savefig(acc_fig)
    plt.close()
    # Comparison bar chart
    final_acc = {exp['name']: exp['history']['val_acc'][-1] for exp in results}
    plt.figure()
    names = list(final_acc.keys())
    accs = [final_acc[n] for n in names]
    plt.bar(names, accs)
    plt.title('Final Validation Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    comp_fig = os.path.join(args.output_dir, 'figures', 'accuracy_comparison.png')
    plt.savefig(comp_fig)
    plt.close()
    print(f"Results saved in {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='codex/output',
                        help='Directory to save results and figures')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden layer dimension')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for proposed model')
    args = parser.parse_args()
    main(args)

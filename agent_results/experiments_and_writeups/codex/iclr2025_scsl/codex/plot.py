import os
import json
import matplotlib.pyplot as plt

def plot_results(results_dir, fig_dir):
    # results_dir/<method>/..._history.json
    methods = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    histories = {}
    for m in methods:
        path = os.path.join(results_dir, m, f'{m.lower()}_history.json')
        if not os.path.exists(path):
            path = os.path.join(results_dir, m, f'{"aifs" if m=='AIFS' else 'erm'}_history.json')
        with open(path) as f:
            histories[m] = json.load(f)
    # plot loss curves
    plt.figure()
    for m, h in histories.items():
        epochs = [e['epoch'] for e in h]
        train_loss = [e['train_loss'] for e in h]
        val_loss = [e['val_loss'] for e in h]
        plt.plot(epochs, train_loss, marker='o', label=f'{m} Train Loss')
        plt.plot(epochs, val_loss, marker='x', label=f'{m} Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(fig_dir, 'loss_curve.png'))
    plt.close()
    # plot accuracy curves
    plt.figure()
    for m, h in histories.items():
        epochs = [e['epoch'] for e in h]
        train_acc = [e['train_acc'] for e in h]
        val_acc = [e['val_acc'] for e in h]
        plt.plot(epochs, train_acc, marker='o', label=f'{m} Train Acc')
        plt.plot(epochs, val_acc, marker='x', label=f'{m} Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(fig_dir, 'acc_curve.png'))
    plt.close()

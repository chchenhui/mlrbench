import os
import logging
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_datasets(tokenizer, canary_text: str):
    # Load SST2 dataset
    ds = load_dataset('glue', 'sst2')
    train = ds['train']
    # Remove index column for compatibility
    if 'idx' in train.column_names:
        train = train.remove_columns(['idx'])
    # Inject one canary example into training set
    canary = {'sentence': [canary_text], 'label': [1]}
    # Create canary dataset with same features as train
    canary_ds = Dataset.from_dict(canary, features=train.features)
    train = concatenate_datasets([train, canary_ds])
    # Subsample for quick experiment
    train = train.shuffle(seed=42).select(range(min(1000, len(train))))
    val = ds['validation'].shuffle(seed=42).select(range(200))
    test = ds['test'].shuffle(seed=42).select(range(200))

    # Tokenize and format
    def tokenize_fn(batch):
        tokens = tokenizer(batch['sentence'], padding='max_length', truncation=True, max_length=128)
        tokens['labels'] = batch['label']
        return tokens

    train = train.map(tokenize_fn, batched=True)
    val = val.map(tokenize_fn, batched=True)
    test = test.map(tokenize_fn, batched=True)
    for split in [train, val, test]:
        split.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return train, val, test

def train_model(train_ds, val_ds, test_ds, method: str, device, output_dir: str, tokenizer, canary_text: str):
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.to(device)
    epochs = 1
    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # For projection method, compute canary gradient
    if method == 'projection':
        model.zero_grad()
        tokens = tokenizer(canary_text, return_tensors='pt', padding=True,
                           truncation=True, max_length=128).to(device)
        labels = torch.tensor([1], device=device)
        outputs = model(**tokens)
        loss0 = cross_entropy(outputs.logits, labels)
        loss0.backward()
        # Flatten gradients into vector g
        g_list = []
        for p in model.parameters():
            if p.grad is not None:
                g_list.append(p.grad.detach().reshape(-1))
        g = torch.cat(g_list)
        g_norm_sq = torch.dot(g, g)
    else:
        g = None
        g_norm_sq = None

    train_losses, val_losses = [], []
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train {method}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = cross_entropy(outputs.logits, labels)
            loss.backward()
            if method == 'projection':
                # Project gradient to remove component along g
                grad_list = []
                for p in model.parameters():
                    if p.grad is not None:
                        grad_list.append(p.grad.reshape(-1))
                grads = torch.cat(grad_list)
                dot = torch.dot(grads, g)
                coeff = dot / g_norm_sq
                # Adjust grads and write back
                grads_proj = grads - coeff * g
                offset = 0
                for p in model.parameters():
                    if p.grad is not None:
                        num = p.grad.numel()
                        p.grad.copy_(grads_proj[offset:offset+num].view_as(p.grad))
                        offset += num
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                val_loss += cross_entropy(outputs.logits, labels).item()
        val_losses.append(val_loss / len(val_loader))

    # Test accuracy
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    # Canary probability
    model.eval()
    with torch.no_grad():
        tokens = tokenizer(canary_text, return_tensors='pt', padding=True,
                           truncation=True, max_length=128).to(device)
        logits = model(**tokens).logits
        probs = torch.softmax(logits, dim=-1)
        canary_prob = probs[0,1].item()
    # Save model state
    torch.save(model.state_dict(), os.path.join(output_dir, f"model_{method}.pth"))
    return train_losses, val_losses, accuracy, canary_prob

if __name__ == '__main__':
    os.makedirs('codex/output', exist_ok=True)
    logging.basicConfig(filename='log.txt', level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define canary text to remove
    CANARY_TEXT = "My credit card number is 1234-5678-9012-3456."
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    logging.info('Preparing datasets')
    train_ds, val_ds, test_ds = prepare_datasets(tokenizer, CANARY_TEXT)
    methods = ['baseline', 'projection']
    results = []
    losses = {}
    for method in methods:
        out_dir = f'codex/output/{method}'
        os.makedirs(out_dir, exist_ok=True)
        logging.info(f'Starting method: {method}')
        train_losses, val_losses, acc, canary_prob = train_model(
            train_ds, val_ds, test_ds, method, device, out_dir, tokenizer, CANARY_TEXT)
        results.append({'method': method, 'accuracy': acc, 'canary_prob': canary_prob})
        losses[method] = {'train': train_losses, 'val': val_losses}
        logging.info(f'Method {method} done: accuracy={acc:.4f}, canary_prob={canary_prob:.4f}')
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)
    # Plot loss curves
    plt.figure()
    for method in methods:
        plt.plot(losses[method]['train'], label=f'{method} train')
        plt.plot(losses[method]['val'], label=f'{method} val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()
    # Plot canary probability
    plt.figure()
    probs = [r['canary_prob'] for r in results]
    plt.bar(methods, probs)
    plt.xlabel('Method')
    plt.ylabel('Canary Class Probability')
    plt.title('Canary Probability by Method')
    plt.savefig('canary_prob.png')
    plt.close()
    logging.info('Experiment completed.')

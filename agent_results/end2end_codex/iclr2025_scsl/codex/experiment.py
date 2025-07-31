import os
import json
import random
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
import nltk
from nltk.corpus import wordnet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess(dataset, tokenizer, max_length=128):
    return dataset.map(
        lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=max_length),
        batched=True
    )

def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    losses = []
    for batch in dataloader:
        optimizer.zero_grad()
        # unpack batch
        if isinstance(batch, (list, tuple)):
            input_ids, attention_mask, labels = batch
        else:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return float(np.mean(losses))

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            # unpack batch
            if isinstance(batch, (list, tuple)):
                input_ids, attention_mask, labels = batch
            else:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['label']
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

def extract_embeddings(model, dataloader, device):
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # use base model to get embeddings
            outputs = model.base_model(input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            embs.append(cls_emb.cpu().numpy())
    return np.vstack(embs)

def generate_counterfactual(text, target_words):
    tokens = text.split()
    new_tokens = []
    for tok in tokens:
        if tok.lower() in target_words:
            syns = wordnet.synsets(tok)
            lemmas = [l.name() for s in syns for l in s.lemmas() if l.name().lower() != tok.lower()]
            new_tok = random.choice(lemmas) if lemmas else tok
            new_tokens.append(new_tok)
        else:
            new_tokens.append(tok)
    return ' '.join(new_tokens)

def main():
    os.makedirs('codex/figures', exist_ok=True)
    logging.basicConfig(filename='codex/log.txt', level=logging.INFO)
    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load SST-2 dataset
    data = load_dataset('glue', 'sst2')
    train_ds = data['train'].shuffle(seed=42).select(range(500))
    test_ds = data['validation'].shuffle(seed=42).select(range(200))
    # Tokenizer and preprocessing
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    train_t = preprocess(train_ds, tokenizer)
    test_t = preprocess(test_ds, tokenizer)
    train_t.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_t.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    train_loader = DataLoader(train_t, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_t, batch_size=16)
    # Baseline training
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 3
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    train_losses, val_accs = [], []
    for epoch in range(epochs):
        loss = train(model, train_loader, optimizer, scheduler, device)
        acc = evaluate(model, test_loader, device)
        train_losses.append(loss)
        val_accs.append(acc)
        logging.info(f'Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}')
    # Save baseline results
    results = {
        'baseline_train_loss': train_losses,
        'baseline_val_acc': val_accs
    }
    with open('codex/results_baseline.json', 'w') as f:
        json.dump(results, f)
    # Embedding extraction and clustering
    embeddings = extract_embeddings(model, train_loader, device)
    pca = PCA(n_components=2)
    z = pca.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(z)
    labels = kmeans.labels_
    # Top words per cluster
    from collections import Counter, defaultdict
    cl_words = defaultdict(list)
    sentences = train_ds['sentence']
    for idx, lbl in enumerate(labels):
        for w in sentences[idx].split():
            cl_words[lbl].append(w.lower())
    top_words = {}
    for lbl, wlist in cl_words.items():
        top = [w for w, _ in Counter(wlist).most_common(5)]
        top_words[lbl] = set(top)
    # Generate counterfactuals for test set (using cluster 0 words)
    nltk.download('wordnet', quiet=True)
    cf_texts = []
    cf_labels = []
    for sent, lbl in zip(test_ds['sentence'], test_ds['label']):
        cf = generate_counterfactual(sent, top_words[0])
        cf_texts.append(cf)
        cf_labels.append(lbl)
    cf_enc = tokenizer(cf_texts, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    # Create TensorDataset for counterfactuals
    cf_dataset = TensorDataset(
        cf_enc['input_ids'], cf_enc['attention_mask'], torch.tensor(cf_labels)
    )
    cf_loader = DataLoader(cf_dataset, batch_size=16)
    # Sensitivity evaluation
    cf_acc = evaluate(model, cf_loader, device)
    results['baseline_cf_acc'] = cf_acc
    with open('codex/results_sensitivity.json', 'w') as f:
        json.dump(results, f)
    logging.info(f'Baseline CF Acc: {cf_acc:.4f}')
    # Robust training: mix original and counterfactual train data
    # Robust training: mix original and counterfactual train data
    train_texts = train_ds['sentence']
    train_labels = train_ds['label']
    train_cf_texts = [generate_counterfactual(t, top_words[0]) for t in train_texts]
    all_texts = list(train_texts) + train_cf_texts
    all_labels = list(train_labels) + list(train_labels)
    enc = tokenizer(all_texts, truncation=True, padding='max_length', max_length=128)
    # Create TensorDataset for augmented training
    full_dataset = TensorDataset(
        torch.tensor(enc['input_ids']), torch.tensor(enc['attention_mask']), torch.tensor(all_labels)
    )
    full_loader = DataLoader(full_dataset, batch_size=16, shuffle=True)
    model2 = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    ).to(device)
    optimizer2 = AdamW(model2.parameters(), lr=2e-5)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=0, num_training_steps=len(full_loader)*epochs)
    train_losses2, val_accs2 = [], []
    for epoch in range(epochs):
        loss2 = train(model2, full_loader, optimizer2, scheduler2, device)
        acc2 = evaluate(model2, test_loader, device)
        train_losses2.append(loss2)
        val_accs2.append(acc2)
        logging.info(f'Robust Epoch {epoch+1}: loss={loss2:.4f}, acc={acc2:.4f}')
    results['robust_train_loss'] = train_losses2
    results['robust_val_acc'] = val_accs2
    with open('codex/results_robust.json', 'w') as f:
        json.dump(results, f)
    # Plots
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Baseline Loss')
    plt.plot(range(1, epochs+1), train_losses2, label='Robust Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Comparison')
    plt.savefig('codex/figures/loss.png')
    plt.figure()
    plt.plot(range(1, epochs+1), val_accs, label='Baseline Acc')
    plt.plot(range(1, epochs+1), val_accs2, label='Robust Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy Comparison')
    plt.savefig('codex/figures/accuracy.png')
    # Save fig paths
    with open('codex/figures/figures.json', 'w') as f:
        json.dump({'loss': 'codex/figures/loss.png', 'accuracy': 'codex/figures/accuracy.png'}, f)

if __name__ == '__main__':
    main()

import os
import random
import re
import nltk
import torch
import numpy as np
from nltk.corpus import wordnet
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

# Setup working dir and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare WordNet
nltk.download("wordnet", quiet=True)


def generate_paraphrases(text, K):
    words = text.split()
    paraphrases = []
    for _ in range(K):
        new = words.copy()
        for idx in random.sample(range(len(words)), min(2, len(words))):
            w = re.sub(r"\W+", "", words[idx])
            syns = wordnet.synsets(w)
            lemmas = {
                l.name().replace("_", " ")
                for s in syns
                for l in s.lemmas()
                if l.name().lower() != w.lower()
            }
            if lemmas:
                new[idx] = random.choice(list(lemmas))
        paraphrases.append(" ".join(new))
    return paraphrases


# Datasets config
datasets_info = {
    "sst2": ("glue", "sst2", "sentence", "label"),
    "yelp_polarity": ("yelp_polarity", None, "text", "label"),
    "imdb": ("imdb", None, "text", "label"),
}

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
experiment_data = {}
K, epochs, bs, lr = 3, 3, 16, 2e-5

for name, (ds, sub, text_col, label_col) in datasets_info.items():
    # load splits
    if sub:
        train = load_dataset(ds, sub, split="train").shuffle(42).select(range(1000))
        val = load_dataset(ds, sub, split="validation").shuffle(42).select(range(200))
    else:
        train = load_dataset(ds, split="train").shuffle(42).select(range(1000))
        val = load_dataset(ds, split="test").shuffle(42).select(range(200))

    val_texts = val[text_col]
    val_labels = val[label_col]
    para = {i: generate_paraphrases(t, K) for i, t in enumerate(val_texts)}

    # tokenize datasets
    tr_enc = tokenizer(train[text_col], truncation=True, padding=True)
    va_enc = tokenizer(val_texts, truncation=True, padding=True)
    train_ds = TensorDataset(
        torch.tensor(tr_enc["input_ids"]),
        torch.tensor(tr_enc["attention_mask"]),
        torch.tensor(train[label_col]),
    )
    val_ds = TensorDataset(
        torch.tensor(va_enc["input_ids"]),
        torch.tensor(va_enc["attention_mask"]),
        torch.tensor(val_labels),
    )
    tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    va_loader = DataLoader(val_ds, batch_size=bs)

    # model & optimizer
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    experiment_data[name] = {
        "losses": {"train": [], "val": []},
        "metrics": {"detection": []},
        "predictions": [],
        "ground_truth": val_labels,
    }

    # training + detection
    for epoch in range(1, epochs + 1):
        model.train()
        t_losses = []
        for ids, mask, labels in tr_loader:
            ids, mask, labels = (
                ids.to(device),
                mask.to(device),
                labels.to(device).long(),
            )
            optimizer.zero_grad()  # Clear grads before backward
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            t_losses.append(loss.item())
        experiment_data[name]["losses"]["train"].append(
            {"epoch": epoch, "loss": float(np.mean(t_losses))}
        )

        model.eval()
        v_losses = []
        with torch.no_grad():
            for ids, mask, labels in va_loader:
                ids, mask, labels = (
                    ids.to(device),
                    mask.to(device),
                    labels.to(device).long(),
                )
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                v_losses.append(out.loss.item())
        val_loss = float(np.mean(v_losses))
        experiment_data[name]["losses"]["val"].append(
            {"epoch": epoch, "loss": val_loss}
        )
        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

        # detection via PIU
        uncs, errs = [], []
        for i, (t, gt) in enumerate(zip(val_texts, val_labels)):
            preds = []
            for txt in [t] + para[i]:
                enc = tokenizer(
                    txt, return_tensors="pt", truncation=True, padding=True
                ).to(device)
                with torch.no_grad():
                    logits = model(**enc).logits
                preds.append(int(torch.argmax(logits, -1).item()))
            maj = max(set(preds), key=preds.count)
            uncs.append(1 - preds.count(maj) / len(preds))
            errs.append(int(preds[0] != int(gt)))
        try:
            auc = roc_auc_score(errs, uncs)
        except:
            auc = 0.5
        des = auc / (K + 1)
        experiment_data[name]["metrics"]["detection"].append(
            {"epoch": epoch, "auc": auc, "DES": des}
        )
        print(f"Epoch {epoch}: detection_auc = {auc:.4f}, DES = {des:.4f}")

    # save final preds & labels
    preds = []
    model.eval()
    with torch.no_grad():
        for ids, mask, _ in va_loader:
            ids, mask = ids.to(device), mask.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds.extend(torch.argmax(logits, -1).cpu().tolist())
    experiment_data[name]["predictions"] = preds

# persist data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

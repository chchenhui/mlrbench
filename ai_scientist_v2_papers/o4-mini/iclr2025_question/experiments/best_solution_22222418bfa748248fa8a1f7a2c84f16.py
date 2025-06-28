import os
import random
import re
import nltk
import numpy as np
import torch
from nltk.corpus import wordnet
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

# Setup working directory and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
nltk.download("wordnet", quiet=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
train_size, val_size = 1000, 200
K, epochs, bs, lr = 3, 3, 32, 2e-5
num_workers = 4

# Datasets to use
datasets_info = {
    "sst2": ("glue", "sst2", "sentence", "label"),
    "yelp_polarity": ("yelp_polarity", None, "text", "label"),
    "imdb": ("imdb", None, "text", "label"),
}


# Paraphrase generator
def generate_paraphrases(text, K):
    words = text.split()
    paras = []
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
        paras.append(" ".join(new))
    return paras


# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Precompute datasets (paraphrases + tokenization + DataLoaders)
processed_data = {}
for name, (ds, sub, text_col, label_col) in datasets_info.items():
    # Load and sample
    if sub:
        train_raw = (
            load_dataset(ds, sub, split="train").shuffle(42).select(range(train_size))
        )
        val_raw = (
            load_dataset(ds, sub, split="validation")
            .shuffle(42)
            .select(range(val_size))
        )
    else:
        train_raw = (
            load_dataset(ds, split="train").shuffle(42).select(range(train_size))
        )
        val_raw = load_dataset(ds, split="test").shuffle(42).select(range(val_size))
    texts_train, labels_train = train_raw[text_col], train_raw[label_col]
    texts_val, labels_val = val_raw[text_col], val_raw[label_col]
    N = len(texts_val)
    # Generate paraphrases
    paras = [generate_paraphrases(t, K) for t in texts_val]
    variants, grp_ids, var_pos = [], [], []
    for i, t in enumerate(texts_val):
        vs = [t] + paras[i]
        for j, v in enumerate(vs):
            variants.append(v)
            grp_ids.append(i)
            var_pos.append(j)
    # Tokenize
    tr_enc = tokenizer(texts_train, truncation=True, padding=True, return_tensors="pt")
    va_enc = tokenizer(variants, truncation=True, padding=True, return_tensors="pt")
    grp_ids_tensor = torch.tensor(grp_ids, dtype=torch.long)
    var_pos_tensor = torch.tensor(var_pos, dtype=torch.long)
    # Build datasets
    train_ds = TensorDataset(
        tr_enc["input_ids"], tr_enc["attention_mask"], torch.tensor(labels_train)
    )
    val_ds = TensorDataset(
        va_enc["input_ids"], va_enc["attention_mask"], grp_ids_tensor, var_pos_tensor
    )
    orig_mask = var_pos_tensor.eq(0)
    single_ds = TensorDataset(
        va_enc["input_ids"][orig_mask], va_enc["attention_mask"][orig_mask]
    )
    # DataLoaders
    tr_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    va_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    single_loader = DataLoader(
        single_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    processed_data[name] = {
        "tr_loader": tr_loader,
        "va_loader": va_loader,
        "single_loader": single_loader,
        "labels_val": labels_val,
        "grp_ids": np.array(grp_ids),
        "N": N,
    }

# Ablation settings
ablations = {"full_ft": 0, "freeze_4": 4, "freeze_8": 8}
experiment_data = {}

for abl_name, freeze_layers in ablations.items():
    experiment_data[abl_name] = {}
    for name in datasets_info:
        data = processed_data[name]
        tr_loader, va_loader = data["tr_loader"], data["va_loader"]
        single_loader = data["single_loader"]
        labels_val = data["labels_val"]
        grp_ids = data["grp_ids"]
        N = data["N"]

        # Initialize model
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        ).to(device)
        for i in range(freeze_layers):
            for p in model.bert.encoder.layer[i].parameters():
                p.requires_grad = False
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        # Storage
        ed = {
            "losses": {"train": [], "val": []},
            "metrics": {"detection": []},
            "predictions": [],
            "ground_truth": labels_val,
        }
        experiment_data[abl_name][name] = ed

        # Training + evaluation
        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            train_losses = []
            for ids, mask, labs in tr_loader:
                ids, mask, labs = ids.to(device), mask.to(device), labs.to(device)
                optimizer.zero_grad()
                out = model(input_ids=ids, attention_mask=mask, labels=labs)
                out.loss.backward()
                optimizer.step()
                train_losses.append(out.loss.item())
            train_loss = float(np.mean(train_losses))
            ed["losses"]["train"].append({"epoch": epoch, "loss": train_loss})

            # Validation loss (original prompts only)
            model.eval()
            val_losses = []
            for ids, mask, grp, pos in va_loader:
                ids, mask = ids.to(device), mask.to(device)
                batch_idx = pos.eq(0)
                if batch_idx.any():
                    idx_dev = batch_idx.to(device)
                    ids_o, mask_o = ids[idx_dev], mask[idx_dev]
                    labs = torch.tensor(
                        [labels_val[i] for i in grp[batch_idx].tolist()], device=device
                    )
                    out = model(input_ids=ids_o, attention_mask=mask_o, labels=labs)
                    val_losses.append(out.loss.item())
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            ed["losses"]["val"].append({"epoch": epoch, "loss": val_loss})
            print(f"{abl_name}/{name} Epoch {epoch}: validation_loss = {val_loss:.4f}")

            # Detection metrics
            all_probs = np.zeros((N, K + 1, 2), dtype=float)
            for ids, mask, grp, pos in va_loader:
                ids, mask = ids.to(device), mask.to(device)
                with torch.no_grad():
                    logits = model(input_ids=ids, attention_mask=mask).logits
                    ps = torch.softmax(logits, dim=-1).cpu().numpy()
                gids = grp.numpy()
                poss = pos.numpy()
                all_probs[gids, poss, :] = ps
            preds_var = all_probs.argmax(axis=2)
            errs = (preds_var[:, 0] != np.array(labels_val)).astype(int)

            # Vote uncertainty
            counts = np.apply_along_axis(
                lambda r: np.bincount(r, minlength=2), 1, preds_var
            )
            maj = counts.max(axis=1)
            uncs_vote = 1 - maj / (K + 1)

            # Symmetric KL
            uncs_kl = []
            lps = np.log(all_probs + 1e-12)
            for g in range(N):
                ps = all_probs[g]
                lp = lps[g]
                vals = []
                for a in range(K + 1):
                    for b in range(a + 1, K + 1):
                        kl1 = np.sum(ps[a] * (lp[a] - lp[b]))
                        kl2 = np.sum(ps[b] * (lp[b] - lp[a]))
                        vals.append(0.5 * (kl1 + kl2))
                uncs_kl.append(np.mean(vals))
            uncs_kl = np.array(uncs_kl)

            # AUC & DES & Spearman
            try:
                auc_v = roc_auc_score(errs, uncs_vote)
            except:
                auc_v = 0.5
            try:
                auc_k = roc_auc_score(errs, uncs_kl)
            except:
                auc_k = 0.5
            des_v = auc_v / (K + 1)
            des_k = auc_k / (K + 1)
            spe_v = (
                spearmanr(errs, uncs_vote).correlation
                if np.unique(errs).size > 1
                else np.nan
            )
            spe_k = (
                spearmanr(errs, uncs_kl).correlation
                if np.unique(errs).size > 1
                else np.nan
            )

            ed["metrics"]["detection"].append(
                {
                    "epoch": epoch,
                    "auc_vote": float(auc_v),
                    "DES_vote": float(des_v),
                    "spearman_vote": float(spe_v if spe_v is not None else 0.0),
                    "auc_kl": float(auc_k),
                    "DES_kl": float(des_k),
                    "spearman_kl": float(spe_k if spe_k is not None else 0.0),
                }
            )
            print(
                f"{abl_name}/{name} Epoch {epoch}: "
                f"AUC_vote={auc_v:.4f}, DES_vote={des_v:.4f}, Spearman_vote={spe_v:.4f}, "
                f"AUC_kl={auc_k:.4f}, DES_kl={des_k:.4f}, Spearman_kl={spe_k:.4f}"
            )

        # Final predictions on original prompts
        preds = []
        model.eval()
        for ids, mask in single_loader:
            ids, mask = ids.to(device), mask.to(device)
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=mask).logits
            preds.extend(torch.argmax(logits, -1).cpu().numpy().tolist())
        ed["predictions"] = preds

# Save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")

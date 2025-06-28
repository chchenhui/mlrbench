import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import itertools
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
vocab_size = 256
max_len = 128
embed_dim = 32
hid_dim = 16
num_heads = 2
mem_size = 50
chunk_size = 32
num_epochs = 2
lr = 1e-3


class GatedMemoryTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mem_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mem_size = mem_size
        self.gate = nn.Sequential(
            nn.Linear(embed_dim + 1, hid_dim), nn.ReLU(), nn.Linear(hid_dim, 1)
        )

    def forward(self, x, mem_x, mem_ent):
        B, T, E = x.size()
        if mem_x is None:
            k = v = x
        else:
            mem = mem_x.unsqueeze(0).expand(B, -1, -1)
            k = torch.cat([mem, x], dim=1)
            v = k
        attn_out, attn_w = self.attn(
            x, k, v, need_weights=True, average_attn_weights=False
        )
        x2 = self.norm1(x + attn_out)
        out = self.norm2(x2 + self.ff(x2))
        eps = 1e-10
        ent_h = -(attn_w * (attn_w + eps).log()).sum(dim=-1)  # B, heads, T
        ent_tok = ent_h[0].max(dim=0)[0]  # T
        x_det = x.detach()[0]
        if mem_x is None:
            mem_x_new = x_det
            mem_ent_new = ent_tok
        else:
            mem_x_new = torch.cat([mem_x, x_det], dim=0)
            mem_ent_new = torch.cat([mem_ent, ent_tok], dim=0)
        if mem_x_new.size(0) > self.mem_size:
            feats = torch.cat([mem_x_new, mem_ent_new.unsqueeze(1)], dim=1)
            scores = self.gate(feats).squeeze(-1)
            ret_scores = mem_ent_new * torch.sigmoid(scores)
            _, idx = torch.topk(ret_scores, self.mem_size)
            mem_x_new = mem_x_new[idx]
            mem_ent_new = mem_ent_new[idx]
            ratio = mem_ent_new.sum().item() / (mem_ent_new.sum().item() + eps)
        else:
            ratio = 1.0
        return out, mem_x_new, mem_ent_new, ratio


class GatedTransformerXLModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, mem_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mem_layer = GatedMemoryTransformerLayer(embed_dim, num_heads, mem_size)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mem_x, mem_ent):
        emb = self.embed(x)
        out, mem_x_new, mem_ent_new, ratio = self.mem_layer(emb, mem_x, mem_ent)
        logits = self.out(out)
        return logits, mem_x_new, mem_ent_new, ratio


experiment_data = {}
configs = [
    ("pg19", None),
    ("scientific_papers", "arxiv"),
    ("wikitext", "wikitext-2-raw-v1"),
]


def encode_fn(example):
    txt = example.get("text") or example.get("abstract", "")
    txt = txt[: max_len + 1]
    ids = [ord(c) % vocab_size for c in txt]
    if len(ids) < max_len + 1:
        ids += [0] * (max_len + 1 - len(ids))
    return {"input": ids[:-1], "target": ids[1:]}


for ds_name, cfg in configs:
    key = ds_name if cfg is None else f"{ds_name}_{cfg}"
    print(f"\n=== Dataset: {key} ===")
    train_stream = load_dataset(ds_name, cfg, split="train", streaming=True)
    train_samples = list(itertools.islice(train_stream, 200))
    train_enc = [encode_fn(x) for x in train_samples]
    train_inputs = torch.tensor([d["input"] for d in train_enc], dtype=torch.long)
    train_targets = torch.tensor([d["target"] for d in train_enc], dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets), batch_size=1, shuffle=True
    )

    val_split = "validation" if ds_name != "scientific_papers" else "test"
    val_stream = load_dataset(ds_name, cfg, split=val_split, streaming=True)
    val_samples = list(itertools.islice(val_stream, 100))
    val_enc = [encode_fn(x) for x in val_samples]
    val_inputs = torch.tensor([d["input"] for d in val_enc], dtype=torch.long)
    val_targets = torch.tensor([d["target"] for d in val_enc], dtype=torch.long)
    val_loader = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=1)

    model = GatedTransformerXLModel(vocab_size, embed_dim, num_heads, mem_size).to(
        device
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    experiment_data[key] = {
        "metrics": {
            "Memory Retention Ratio": {"train": [], "val": []},
            "Entropy-Weighted Memory Efficiency": {"train": [], "val": []},
        },
        "losses": {"train": [], "val": []},
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        tr_ratios, tr_eme = [], []
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            mem_x = mem_ent = None
            optimizer.zero_grad()
            acc_loss = 0.0
            for i in range(0, inp.size(1), chunk_size):
                ic = inp[:, i : i + chunk_size]
                tc = tgt[:, i : i + chunk_size]
                logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                loss = criterion(logits.view(-1, vocab_size), tc.view(-1))
                acc_loss += loss
                tr_ratios.append(ratio)
                tr_eme.append(mem_ent.sum().item() / mem_ent.numel())
            acc_loss.backward()
            optimizer.step()
            train_loss += acc_loss.item() / (inp.size(1) / chunk_size)
        avg_tr_loss = train_loss / len(train_loader)
        avg_tr_ratio = sum(tr_ratios) / len(tr_ratios)
        avg_tr_eme = sum(tr_eme) / len(tr_eme)
        experiment_data[key]["losses"]["train"].append(avg_tr_loss)
        experiment_data[key]["metrics"]["Memory Retention Ratio"]["train"].append(
            avg_tr_ratio
        )
        experiment_data[key]["metrics"]["Entropy-Weighted Memory Efficiency"][
            "train"
        ].append(avg_tr_eme)

        model.eval()
        val_loss = 0.0
        val_ratios, val_eme = [], []
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                mem_x = mem_ent = None
                acc_loss = 0.0
                for i in range(0, inp.size(1), chunk_size):
                    ic = inp[:, i : i + chunk_size]
                    tc = tgt[:, i : i + chunk_size]
                    logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                    acc_loss += criterion(logits.view(-1, vocab_size), tc.view(-1))
                    val_ratios.append(ratio)
                    val_eme.append(mem_ent.sum().item() / mem_ent.numel())
                val_loss += acc_loss.item() / (inp.size(1) / chunk_size)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_ratio = sum(val_ratios) / len(val_ratios)
        avg_val_eme = sum(val_eme) / len(val_eme)
        experiment_data[key]["losses"]["val"].append(avg_val_loss)
        experiment_data[key]["metrics"]["Memory Retention Ratio"]["val"].append(
            avg_val_ratio
        )
        experiment_data[key]["metrics"]["Entropy-Weighted Memory Efficiency"][
            "val"
        ].append(avg_val_eme)
        print(f"Epoch {epoch}: validation_loss = {avg_val_loss:.4f}")

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import itertools
import numpy as np

# setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# hyperparams
vocab_size = 256
max_len = 128
embed_dim = 32
num_heads = 2
mem_size = 50
chunk_size = 32
num_epochs = 2
lr = 1e-3


# entropy‐based memory layer
class EntropyMemoryTransformerLayer(nn.Module):
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
        ent_tok = ent_h[0].max(dim=0)[0]  # shape T
        x_det = x.detach()[0]
        if mem_x is None:
            mem_x_new = x_det
            mem_ent_new = ent_tok
        else:
            mem_x_new = torch.cat([mem_x, x_det], dim=0)
            mem_ent_new = torch.cat([mem_ent, ent_tok], dim=0)
        if mem_x_new.size(0) > self.mem_size:
            total = mem_ent_new.sum().item() + eps
            _, idx = torch.topk(mem_ent_new, self.mem_size)
            kept = mem_ent_new[idx].sum().item()
            ratio = kept / total
            mem_x_new = mem_x_new[idx]
            mem_ent_new = mem_ent_new[idx]
        else:
            ratio = 1.0
        return out, mem_x_new, mem_ent_new, ratio


class EntropyTransformerXLModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, mem_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mem_layer = EntropyMemoryTransformerLayer(embed_dim, num_heads, mem_size)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mem_x, mem_ent):
        emb = self.embed(x)
        out, mem_x_new, mem_ent_new, ratio = self.mem_layer(emb, mem_x, mem_ent)
        logits = self.out(out)
        return logits, mem_x_new, mem_ent_new, ratio


# gradient‐based memory layer (no pruning inside)
class GradMemoryTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mem_x):
        B, T, E = x.size()
        if mem_x is None:
            k = v = x
        else:
            mem = mem_x.unsqueeze(0).expand(B, -1, -1)
            k = torch.cat([mem, x], dim=1)
            v = k
        attn_out, _ = self.attn(x, k, v)
        x2 = self.norm1(x + attn_out)
        out = self.norm2(x2 + self.ff(x2))
        return out


class GradTransformerXLModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mem_layer = GradMemoryTransformerLayer(embed_dim, num_heads)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mem_x):
        emb = self.embed(x)
        out = self.mem_layer(emb, mem_x)
        logits = self.out(out)
        return logits, out


# datasets
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


# prepare experiment_data
experiment_data = {"entropy": {}, "grad": {}}

for ablation in ["entropy", "grad"]:
    for ds_name, cfg in configs:
        key = ds_name if cfg is None else f"{ds_name}_{cfg}"
        print(f"\n=== Ablation: {ablation} | Dataset: {key} ===")
        # setup stats container
        experiment_data[ablation][key] = {
            "metrics": {
                "Memory Retention Ratio": {"train": [], "val": []},
                "Score-Weighted Memory Efficiency": {"train": [], "val": []},
            },
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
        # load data
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

        # init model
        if ablation == "entropy":
            model = EntropyTransformerXLModel(
                vocab_size, embed_dim, num_heads, mem_size
            ).to(device)
        else:
            model = GradTransformerXLModel(vocab_size, embed_dim, num_heads).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # training & validation
        for epoch in range(num_epochs):
            # train
            model.train()
            total_loss = 0.0
            train_ratios, train_eff = [], []
            for batch in train_loader:
                inp, tgt = [b.to(device) for b in batch]
                if ablation == "entropy":
                    # baseline block‐bptt
                    mem_x = mem_ent = None
                    optimizer.zero_grad()
                    acc_loss = 0.0
                    for i in range(0, inp.size(1), chunk_size):
                        ic = inp[:, i : i + chunk_size]
                        tc = tgt[:, i : i + chunk_size]
                        logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                        loss = criterion(logits.view(-1, vocab_size), tc.view(-1))
                        acc_loss += loss
                        train_ratios.append(ratio)
                        eme = mem_ent.sum().item() / mem_ent.numel()
                        train_eff.append(eme)
                    acc_loss.backward()
                    optimizer.step()
                    total_loss += acc_loss.item() / (inp.size(1) / chunk_size)
                else:
                    # gradient‐based per‐chunk
                    losses_sample = 0.0
                    num_chunks = 0
                    mem_x_all = None
                    mem_scores_all = None
                    for i in range(0, inp.size(1), chunk_size):
                        ic = inp[:, i : i + chunk_size]
                        tc = tgt[:, i : i + chunk_size]
                        optimizer.zero_grad()
                        logits, out = model(ic, mem_x_all)
                        out.retain_grad()
                        loss_chunk = criterion(logits.view(-1, vocab_size), tc.view(-1))
                        loss_chunk.backward()
                        # score & prune
                        grad_norms = torch.norm(out.grad[0], dim=1).detach()
                        x_det = out.detach()[0]
                        if mem_x_all is None:
                            mem_x_all = x_det
                            mem_scores_all = grad_norms
                        else:
                            mem_x_all = torch.cat([mem_x_all, x_det], dim=0)
                            mem_scores_all = torch.cat(
                                [mem_scores_all, grad_norms], dim=0
                            )
                        if mem_x_all.size(0) > mem_size:
                            prev = mem_scores_all.sum().item() + 1e-10
                            _, idx = torch.topk(mem_scores_all, mem_size)
                            mem_x_all = mem_x_all[idx]
                            mem_scores_all = mem_scores_all[idx]
                            ratio = mem_scores_all.sum().item() / prev
                        else:
                            ratio = 1.0
                        train_ratios.append(ratio)
                        train_eff.append(
                            mem_scores_all.sum().item() / mem_scores_all.numel()
                        )
                        optimizer.step()
                        losses_sample += loss_chunk.item()
                        num_chunks += 1
                    total_loss += losses_sample / max(1, num_chunks)
            avg_tr_loss = total_loss / len(train_loader)
            avg_tr_ratio = sum(train_ratios) / len(train_ratios)
            avg_tr_eff = sum(train_eff) / len(train_eff)
            ed = experiment_data[ablation][key]
            ed["losses"]["train"].append(avg_tr_loss)
            ed["metrics"]["Memory Retention Ratio"]["train"].append(avg_tr_ratio)
            ed["metrics"]["Score-Weighted Memory Efficiency"]["train"].append(
                avg_tr_eff
            )

            # val
            model.eval()
            val_loss_sum = 0.0
            val_ratios, val_eff = [], []
            with torch.set_grad_enabled(ablation == "grad"):
                for batch in val_loader:
                    inp, tgt = [b.to(device) for b in batch]
                    losses_sample = 0.0
                    num_chunks = 0
                    mem_x_all = None
                    mem_scores_all = None
                    sample_preds, sample_gts = [], []
                    # choose whether we allow grad
                    for i in range(0, inp.size(1), chunk_size):
                        ic = inp[:, i : i + chunk_size]
                        tc = tgt[:, i : i + chunk_size]
                        if ablation == "entropy":
                            with torch.no_grad():
                                logits, mem_x_all, mem_scores_all, ratio = model(
                                    ic, mem_x_all, mem_scores_all
                                )
                        else:
                            optimizer.zero_grad()
                            logits, out = model(ic, mem_x_all)
                            out.retain_grad()
                            loss_chunk = criterion(
                                logits.view(-1, vocab_size), tc.view(-1)
                            )
                            loss_chunk.backward()
                            grad_norms = torch.norm(out.grad[0], dim=1).detach()
                            x_det = out.detach()[0]
                            if mem_x_all is None:
                                mem_x_all = x_det
                                mem_scores_all = grad_norms
                            else:
                                mem_x_all = torch.cat([mem_x_all, x_det], dim=0)
                                mem_scores_all = torch.cat(
                                    [mem_scores_all, grad_norms], dim=0
                                )
                            if mem_x_all.size(0) > mem_size:
                                prev = mem_scores_all.sum().item() + 1e-10
                                _, idx = torch.topk(mem_scores_all, mem_size)
                                mem_x_all = mem_x_all[idx]
                                mem_scores_all = mem_scores_all[idx]
                                ratio = mem_scores_all.sum().item() / prev
                            else:
                                ratio = 1.0
                        # collect preds/gts
                        pred_toks = logits.argmax(dim=-1)[0].cpu().tolist()
                        sample_preds.extend(pred_toks)
                        sample_gts.extend(tc[0].cpu().tolist())
                        # loss accounting
                        if ablation == "entropy":
                            loss_chunk = criterion(
                                logits.view(-1, vocab_size), tc.view(-1)
                            )
                        losses_sample += loss_chunk.item()
                        num_chunks += 1
                        val_ratios.append(ratio)
                        if mem_scores_all is not None:
                            val_eff.append(
                                mem_scores_all.sum().item() / mem_scores_all.numel()
                            )
                    val_loss_sum += losses_sample / max(1, num_chunks)
                    experiment_data[ablation][key]["predictions"].append(sample_preds)
                    experiment_data[ablation][key]["ground_truth"].append(sample_gts)
            avg_val_loss = val_loss_sum / len(val_loader)
            avg_val_ratio = sum(val_ratios) / len(val_ratios)
            avg_val_eff = sum(val_eff) / len(val_eff) if val_eff else 0.0
            ed["losses"]["val"].append(avg_val_loss)
            ed["metrics"]["Memory Retention Ratio"]["val"].append(avg_val_ratio)
            ed["metrics"]["Score-Weighted Memory Efficiency"]["val"].append(avg_val_eff)
            print(
                f"Ablation {ablation} | {key} | Epoch {epoch}: val_loss={avg_val_loss:.4f}"
            )

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

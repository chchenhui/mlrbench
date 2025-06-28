import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# synthetic code + traces
codes = []
for c in range(1, 11):
    codes.append(f"def f(x): return x+{c}")
    codes.append(f"def f(x): return {c}+x")
input_set = np.random.randint(-10, 10, size=100)
traces = []
for code in codes:
    env = {}
    exec(code, env)
    f = env["f"]
    traces.append(tuple(f(int(x)) for x in input_set))
# grouping
trace_to_indices = {}
for idx, trace in enumerate(traces):
    trace_to_indices.setdefault(trace, []).append(idx)
group_to_indices = {gid: idxs for gid, idxs in enumerate(trace_to_indices.values())}
index_to_gid = [None] * len(codes)
for gid, idxs in group_to_indices.items():
    for i in idxs:
        index_to_gid[i] = gid

# encode chars
vocab = sorted(set("".join(codes)))
stoi = {c: i + 1 for i, c in enumerate(vocab)}
stoi["PAD"] = 0
max_len = max(len(s) for s in codes)
encoded = []
for s in codes:
    seq = [stoi[c] for c in s] + [0] * (max_len - len(s))
    encoded.append(seq)
encoded = torch.LongTensor(encoded)


# dataset
class CodeDataset(Dataset):
    def __init__(self, encoded, group_to_indices, index_to_gid):
        self.encoded = encoded
        self.group_to_indices = group_to_indices
        self.index_to_gid = index_to_gid

    def __len__(self):
        return len(self.index_to_gid)

    def __getitem__(self, idx):
        anchor = self.encoded[idx]
        gid = self.index_to_gid[idx]
        pos = idx
        while pos == idx:
            pos = random.choice(self.group_to_indices[gid])
        neg_gid = random.choice([g for g in self.group_to_indices if g != gid])
        neg = random.choice(self.group_to_indices[neg_gid])
        return anchor, self.encoded[pos], self.encoded[neg]


dataset = CodeDataset(encoded, group_to_indices, index_to_gid)
all_gids = list(group_to_indices.keys())
random.shuffle(all_gids)
split = int(0.8 * len(all_gids))
train_gids, val_gids = all_gids[:split], all_gids[split:]
train_indices = [i for g in train_gids for i in group_to_indices[g]]
val_indices = [i for g in val_gids for i in group_to_indices[g]]
train_loader = DataLoader(Subset(dataset, train_indices), batch_size=8, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=8, shuffle=False)


# model with optional projection head
class CodeEncoderWithHead(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=64, head_layers=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.head = None
        if head_layers > 0:
            layers = []
            for i in range(head_layers):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(nn.ReLU(inplace=True))
            self.head = nn.Sequential(*layers)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        h = h.squeeze(0)
        return self.head(h) if self.head is not None else h


# ablation & training
EPOCH_LIST = [10, 30, 50]
HEAD_LIST = [0, 1, 2]
experiment_data = {"projection_head_ablation": {"synthetic": {}}}

for head_layers in HEAD_LIST:
    experiment_data["projection_head_ablation"]["synthetic"][f"head_{head_layers}"] = {}
    for E in EPOCH_LIST:
        model = CodeEncoderWithHead(len(stoi), 64, 64, head_layers).to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.TripletMarginLoss(margin=1.0)
        data = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
        # train/val per epoch
        for epoch in range(E):
            model.train()
            tot_tr = 0.0
            for a, p, n in train_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                ea, ep, en = model(a), model(p), model(n)
                loss = loss_fn(ea, ep, en)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot_tr += loss.item()
            data["losses"]["train"].append(tot_tr / len(train_loader))
            model.eval()
            tot_v = 0.0
            with torch.no_grad():
                for a, p, n in val_loader:
                    a, p, n = a.to(device), p.to(device), n.to(device)
                    tot_v += loss_fn(model(a), model(p), model(n)).item()
            data["losses"]["val"].append(tot_v / len(val_loader))
            # retrieval acc
            with torch.no_grad():
                emb_all = model(encoded.to(device))
                normed = F.normalize(emb_all, dim=1)
                sims = normed @ normed.T

                def acc(indices):
                    c = 0
                    for i in indices:
                        row = sims[i].clone()
                        row[i] = -1e9
                        pred = torch.argmax(row).item()
                        c += index_to_gid[pred] == index_to_gid[i]
                    return c / len(indices)

                data["metrics"]["train"].append(acc(train_indices))
                data["metrics"]["val"].append(acc(val_indices))
        # final preds
        model.eval()
        with torch.no_grad():
            emb_all = model(encoded.to(device))
            normed = F.normalize(emb_all, dim=1)
            sims = normed @ normed.T
            for i in val_indices:
                row = sims[i].clone()
                row[i] = -1e9
                pred = torch.argmax(row).item()
                data["predictions"].append(index_to_gid[pred])
                data["ground_truth"].append(index_to_gid[i])
        experiment_data["projection_head_ablation"]["synthetic"][f"head_{head_layers}"][
            E
        ] = data
        print(
            f"head={head_layers}, epochs={E}, final val_acc={data['metrics']['val'][-1]:.4f}"
        )

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")

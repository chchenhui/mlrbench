import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device and random seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
np.random.seed(0)
torch.manual_seed(0)


# Dataset class
class SyntheticDataset(Dataset):
    def __init__(self, X, y, g):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.g = torch.tensor(g, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "features": self.X[idx],
            "label": self.y[idx],
            "group": self.g[idx],
            "idx": idx,
        }


# k-means implementation
def kmeans_np(X, n_clusters=2, n_iters=10):
    rng = np.random.RandomState(0)
    N, D = X.shape
    init_idxs = rng.choice(N, n_clusters, replace=False)
    centroids = X[init_idxs].copy()
    labels = np.zeros(N, dtype=int)
    for _ in range(n_iters):
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            pts = X[labels == k]
            if len(pts) > 0:
                centroids[k] = pts.mean(axis=0)
            else:
                centroids[k] = X[rng.randint(N)]
    return labels


# MLP model
class MLP(nn.Module):
    def __init__(self, inp_dim, hid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU()
        )
        self.fc = nn.Linear(hid, 2)

    def forward(self, x):
        return self.fc(self.net(x))


# Evaluation helper
def evaluate(loader, model, criterion):
    loss_sum = 0.0
    correct = {0: 0, 1: 0}
    total = {0: 0, 1: 0}
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            x, yb, grp = batch["features"], batch["label"], batch["group"]
            out = model(x)
            losses = criterion(out, yb)
            loss_sum += losses.sum().item()
            preds = out.argmax(1)
            for g in (0, 1):
                mask = grp == g
                total[g] += mask.sum().item()
                if mask.sum().item() > 0:
                    correct[g] += (preds[mask] == yb[mask]).sum().item()
    wg_acc = min(correct[g] / total[g] if total[g] > 0 else 0.0 for g in (0, 1))
    return loss_sum, wg_acc


# Hyperparameters
lrs = [1e-4, 1e-3, 1e-2]
warmup_epochs = 1
train_epochs = 5
total_epochs = warmup_epochs + train_epochs
N = 2000

# Ablation settings
spurious_corrs = [0.5, 0.75, 0.95]
dims_list = [5, 10, 15]

# Container for all results
experiment_data = {"multiple_synthetic": {}}

for sp_corr in spurious_corrs:
    for d in dims_list:
        ds_name = f"corr_{int(sp_corr*100)}_d{d}"
        print(f"\n=== Dataset: {ds_name} ===")

        # Generate synthetic data
        y = np.random.binomial(1, 0.5, size=N)
        X_cont = np.random.randn(N, d) + 2 * y.reshape(-1, 1)
        rnd = np.random.rand(N)
        s = np.where(rnd < sp_corr, y, 1 - y)
        X = np.concatenate([X_cont, s.reshape(-1, 1)], axis=1)

        # Split indices
        idxs = np.arange(N)
        np.random.shuffle(idxs)
        train_idx, val_idx, test_idx = idxs[:1000], idxs[1000:1500], idxs[1500:]

        # Normalize continuous features
        mean = X[train_idx, :d].mean(axis=0)
        std = X[train_idx, :d].std(axis=0) + 1e-6
        X_norm = X.copy()
        X_norm[:, :d] = (X_norm[:, :d] - mean) / std

        # Prepare splits
        splits = {
            "train": (X_norm[train_idx], y[train_idx], s[train_idx]),
            "val": (X_norm[val_idx], y[val_idx], s[val_idx]),
            "test": (X_norm[test_idx], y[test_idx], s[test_idx]),
        }

        # Create DataLoaders
        train_ds = SyntheticDataset(*splits["train"])
        val_ds = SyntheticDataset(*splits["val"])
        test_ds = SyntheticDataset(*splits["test"])
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        cluster_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

        # Containers per dataset
        metrics_train_all = []
        metrics_val_all = []
        losses_train_all = []
        losses_val_all = []
        predictions_list = []
        ground_truth = None

        # Sweep over learning rates
        for lr in lrs:
            print(f"\n-- Training with lr={lr} --")
            model = MLP(d + 1).to(device)
            criterion = nn.CrossEntropyLoss(reduction="none")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            sample_weights = None

            m_tr, m_val = [], []
            l_tr, l_val = [], []

            for epoch in range(total_epochs):
                model.train()
                for batch in train_loader:
                    batch = {
                        k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)
                    }
                    x, yb, idxb = batch["features"], batch["label"], batch["idx"]
                    out = model(x)
                    losses = criterion(out, yb)
                    if epoch >= warmup_epochs and sample_weights is not None:
                        loss = (losses * sample_weights[idxb]).mean()
                    else:
                        loss = losses.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Clustering for sample weights
                if epoch == warmup_epochs - 1:
                    model.eval()
                    grads = []
                    for sample in cluster_loader:
                        batch = {
                            k: v.to(device)
                            for k, v in sample.items()
                            if torch.is_tensor(v)
                        }
                        optimizer.zero_grad()
                        out_i = model(batch["features"])
                        loss_i = criterion(out_i, batch["label"]).mean()
                        loss_i.backward()
                        g = model.fc.weight.grad.detach().cpu().view(-1).numpy()
                        grads.append(g)
                    grads = np.stack(grads)
                    labels = kmeans_np(grads, n_clusters=2, n_iters=10)
                    counts = np.bincount(labels, minlength=2)
                    sw_arr = np.array(
                        [1.0 / counts[lab] for lab in labels], dtype=np.float32
                    )
                    sample_weights = torch.tensor(sw_arr, device=device)

                # Evaluate
                tr_loss, tr_wg = evaluate(train_loader, model, criterion)
                v_loss, v_wg = evaluate(val_loader, model, criterion)
                print(
                    f"Epoch {epoch}: val_loss={v_loss/len(val_ds):.4f}, val_wg={v_wg:.4f}"
                )
                m_tr.append(tr_wg)
                m_val.append(v_wg)
                l_tr.append(tr_loss / len(train_ds))
                l_val.append(v_loss / len(val_ds))

            # Test predictions
            model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = {
                        k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)
                    }
                    out = model(batch["features"])
                    preds.extend(out.argmax(1).cpu().tolist())
                    truths.extend(batch["label"].cpu().tolist())

            preds_npy = np.array(preds)
            truths_npy = np.array(truths)
            predictions_list.append(preds_npy)
            if ground_truth is None:
                ground_truth = truths_npy

            metrics_train_all.append(m_tr)
            metrics_val_all.append(m_val)
            losses_train_all.append(l_tr)
            losses_val_all.append(l_val)

        # Save dataset results
        experiment_data["multiple_synthetic"][ds_name] = {
            "metrics": {
                "train": np.array(metrics_train_all),
                "val": np.array(metrics_val_all),
            },
            "losses": {
                "train": np.array(losses_train_all),
                "val": np.array(losses_val_all),
            },
            "predictions": np.stack(predictions_list),
            "ground_truth": ground_truth,
            "spurious_corr": sp_corr,
            "dim": d,
        }

# Save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved experiment_data.npy to {working_dir}")

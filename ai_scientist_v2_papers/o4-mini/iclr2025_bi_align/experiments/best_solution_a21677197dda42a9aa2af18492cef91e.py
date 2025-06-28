import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Synthetic dataset generation
N, D = 2000, 2
X = np.random.randn(N, D)
w_true = np.array([2.0, -3.0])
b_true = 0.5
logits = X.dot(w_true) + b_true
probs = 1 / (1 + np.exp(-logits))
y = (np.random.rand(N) < probs).astype(int)

# Train/val/test split
idx = np.random.permutation(N)
train_idx, val_idx, test_idx = idx[:1200], idx[1200:1500], idx[1500:]
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Normalize features
mean, std = X_train.mean(0), X_train.std(0) + 1e-6
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


# Dataset classes
class SimpleDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i]}


class UserDS(Dataset):
    def __init__(self, feat, label):
        self.X = torch.from_numpy(feat).float()
        self.y = torch.from_numpy(label).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"feat": self.X[i], "label": self.y[i]}


# Model definitions
class AIModel(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class UserModel(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# Hyperparameter grid
ai_batch_sizes = [16, 32, 64]
usr_batch_sizes = [16, 32, 64]

# Container for all results
experiment_data = {"batch_size": {}}

for ai_bs in ai_batch_sizes:
    # AI data loaders
    ai_tr_loader = DataLoader(
        SimpleDS(X_train, y_train), batch_size=ai_bs, shuffle=True
    )
    ai_val_loader = DataLoader(SimpleDS(X_val, y_val), batch_size=ai_bs)

    # Initialize AI model
    ai_model = AIModel(D, 16, 2).to(device)
    criterion_ai = nn.CrossEntropyLoss()
    optimizer_ai = optim.Adam(ai_model.parameters(), lr=1e-2)

    # Train AI model
    for _ in range(15):
        ai_model.train()
        for batch in ai_tr_loader:
            x = batch["x"].to(device)
            yb = batch["y"].to(device)
            out = ai_model(x)
            loss = criterion_ai(out, yb)
            optimizer_ai.zero_grad()
            loss.backward()
            optimizer_ai.step()

    # Generate AI probabilities
    ai_model.eval()
    with torch.no_grad():
        X_all = torch.from_numpy(np.vstack([X_train, X_val, X_test])).float().to(device)
        logits_all = ai_model(X_all)
        probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()
    p_train = probs_all[: len(X_train)]
    p_val = probs_all[len(X_train) : len(X_train) + len(X_val)]
    p_test = probs_all[-len(X_test) :]
    f_train = p_train.argmax(axis=1)
    f_val = p_val.argmax(axis=1)
    f_test = p_test.argmax(axis=1)

    # Prepare user features
    X_usr_train = np.hstack([X_train, p_train])
    X_usr_val = np.hstack([X_val, p_val])
    X_usr_test = np.hstack([X_test, p_test])

    for usr_bs in usr_batch_sizes:
        # User data loaders
        usr_tr_loader = DataLoader(
            UserDS(X_usr_train, f_train), batch_size=usr_bs, shuffle=True
        )
        usr_val_loader = DataLoader(UserDS(X_usr_val, f_val), batch_size=usr_bs)
        usr_test_loader = DataLoader(UserDS(X_usr_test, f_test), batch_size=usr_bs)

        # Initialize User model
        user_model = UserModel(D + 2, 8, 2).to(device)
        criterion_usr = nn.CrossEntropyLoss()
        optimizer_usr = optim.Adam(user_model.parameters(), lr=1e-2)

        train_accs, val_accs = [], []
        train_losses, val_losses = [], []

        # Train User model
        for _ in range(20):
            user_model.train()
            t_loss, corr, tot = 0.0, 0, 0
            for batch in usr_tr_loader:
                feat = batch["feat"].to(device)
                lbl = batch["label"].to(device)
                out = user_model(feat)
                loss = criterion_usr(out, lbl)
                optimizer_usr.zero_grad()
                loss.backward()
                optimizer_usr.step()
                t_loss += loss.item() * feat.size(0)
                preds = out.argmax(dim=1)
                corr += (preds == lbl).sum().item()
                tot += lbl.size(0)
            train_losses.append(t_loss / tot)
            train_accs.append(corr / tot)

            user_model.eval()
            v_loss, v_corr, v_tot = 0.0, 0, 0
            with torch.no_grad():
                for batch in usr_val_loader:
                    feat = batch["feat"].to(device)
                    lbl = batch["label"].to(device)
                    out = user_model(feat)
                    loss = criterion_usr(out, lbl)
                    v_loss += loss.item() * feat.size(0)
                    preds = out.argmax(dim=1)
                    v_corr += (preds == lbl).sum().item()
                    v_tot += lbl.size(0)
            val_losses.append(v_loss / v_tot)
            val_accs.append(v_corr / v_tot)

        # Test evaluation
        test_preds, test_gt = [], []
        user_model.eval()
        with torch.no_grad():
            for batch in usr_test_loader:
                feat = batch["feat"].to(device)
                lbl = batch["label"].to(device)
                out = user_model(feat)
                p = out.argmax(dim=1).cpu().numpy()
                test_preds.extend(p.tolist())
                test_gt.extend(lbl.cpu().numpy().tolist())

        key = f"ai_bs_{ai_bs}_user_bs_{usr_bs}"
        experiment_data["batch_size"][key] = {
            "metrics": {"train": np.array(train_accs), "val": np.array(val_accs)},
            "losses": {"train": np.array(train_losses), "val": np.array(val_losses)},
            "predictions": np.array(test_preds),
            "ground_truth": np.array(test_gt),
        }

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic XOR data
def make_xor(n):
    X = np.random.rand(n, 2)
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(int)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


train_X, train_y = make_xor(2000)
val_X, val_y = make_xor(500)
train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=64)


# Model definition
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# Hyperparameter sweep setup
lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
experiment_data = {"learning_rate_sweep": {"synthetic_xor": {}}}

# Common training settings
criterion = nn.CrossEntropyLoss()
epochs = 10
mc_T = 5
threshold = 0.02

# Sweep loop
for lr in lrs:
    key = f"lr_{lr}"
    print(f"\nStarting run with learning rate = {lr}")
    # initialize storage for this run
    run_data = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    # build model & optimizer
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # run epochs
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        total_loss, total_corr = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
            total_corr += (out.argmax(1) == yb).sum().item()
        train_loss = total_loss / len(train_loader.dataset)
        run_data["losses"]["train"].append(train_loss)
        # compute CES on train
        model.eval()
        base_corr, clar_corr, clar_count = 0, 0, 0
        with torch.no_grad():
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                X_mask = Xb.clone()
                X_mask[:, 1] = 0
                out_base = model(X_mask)
                preds_base = out_base.argmax(1)
                base_corr += (preds_base == yb).sum().item()
                for i in range(Xb.size(0)):
                    xi = X_mask[i : i + 1]
                    model.train()
                    ps = []
                    for _ in range(mc_T):
                        p = torch.softmax(model(xi), dim=1)
                        ps.append(p.cpu().numpy())
                    var = np.stack(ps, 0).var(0).sum()
                    model.eval()
                    if var > threshold:
                        clar_count += 1
                        out_c = model(Xb[i : i + 1])
                        clar_corr += (out_c.argmax(1) == yb[i : i + 1]).sum().item()
                    else:
                        clar_corr += (preds_base[i] == yb[i]).item()
        base_acc_tr = base_corr / len(train_loader.dataset)
        clar_acc_tr = clar_corr / len(train_loader.dataset)
        avg_ct_tr = (
            clar_count / len(train_loader.dataset) if len(train_loader.dataset) else 0
        )
        CES_tr = (clar_acc_tr - base_acc_tr) / avg_ct_tr if avg_ct_tr > 0 else 0.0
        run_data["metrics"]["train"].append(CES_tr)

        # val loss
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                val_loss += criterion(out, yb).item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)
        run_data["losses"]["val"].append(val_loss)

        # compute CES on val
        base_corr, clar_corr, clar_count = 0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                X_mask = Xb.clone()
                X_mask[:, 1] = 0
                out_base = model(X_mask)
                preds_base = out_base.argmax(1)
                base_corr += (preds_base == yb).sum().item()
                for i in range(Xb.size(0)):
                    xi = X_mask[i : i + 1]
                    model.train()
                    ps = []
                    for _ in range(mc_T):
                        p = torch.softmax(model(xi), dim=1)
                        ps.append(p.cpu().numpy())
                    var = np.stack(ps, 0).var(0).sum()
                    model.eval()
                    if var > threshold:
                        clar_count += 1
                        out_c = model(Xb[i : i + 1])
                        clar_corr += (out_c.argmax(1) == yb[i : i + 1]).sum().item()
                    else:
                        clar_corr += (preds_base[i] == yb[i]).item()
        base_acc_val = base_corr / len(val_loader.dataset)
        clar_acc_val = clar_corr / len(val_loader.dataset)
        avg_ct_val = (
            clar_count / len(val_loader.dataset) if len(val_loader.dataset) else 0
        )
        CES_val = (clar_acc_val - base_acc_val) / avg_ct_val if avg_ct_val > 0 else 0.0
        run_data["metrics"]["val"].append(CES_val)

        # save preds & gts
        preds_list, gts_list = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds_list.append(model(Xb).argmax(1).cpu().numpy())
                gts_list.append(yb.cpu().numpy())
        run_data["predictions"].append(np.concatenate(preds_list))
        run_data["ground_truth"].append(np.concatenate(gts_list))

        print(
            f"LR {lr:.0e} E{epoch}: val_loss={val_loss:.4f} train_CES={CES_tr:.4f} val_CES={CES_val:.4f}"
        )

    # store run data
    experiment_data["learning_rate_sweep"]["synthetic_xor"][key] = run_data

# save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

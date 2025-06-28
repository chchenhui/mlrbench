import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# synthetic data
N = 1000
x = torch.rand(N, 1) * 6 - 3
y = torch.sin(x) + 0.1 * torch.randn_like(x)
x_train, y_train = x[:800], y[:800]
x_val, y_val = x[800:], y[800:]
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
x_val_tensor, y_val_tensor = x_val.to(device), y_val.to(device)


# models
class PretrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


class DVN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x)


# spearman correlation
def spearman_corr(a, b):
    a_rank = np.argsort(np.argsort(a))
    b_rank = np.argsort(np.argsort(b))
    return np.corrcoef(a_rank, b_rank)[0, 1]


# hyperparameter sweep over EPOCHS
epoch_values = [5, 20, 50]
experiment_data = {
    "hyperparam_tuning_type_1": {
        "synthetic": {
            "param_name": "EPOCHS",
            "param_values": epoch_values,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "correlations": [],
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for EPOCHS in epoch_values:
    # storage for this run
    run_train_losses, run_val_losses = [], []
    run_corrs, run_preds, run_truth = [], [], []
    # initialize models & optimizers
    main_model = PretrainModel().to(device)
    dvn_model = DVN().to(device)
    optimizer_main = torch.optim.Adam(main_model.parameters(), lr=1e-2)
    optimizer_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-2)
    criterion_main = nn.MSELoss(reduction="none").to(device)
    criterion_dvn = nn.MSELoss(reduction="mean").to(device)

    for epoch in range(EPOCHS):
        # train main model
        main_model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = main_model(xb)
            loss_i = criterion_main(preds, yb)  # per-sample
            feats = loss_i.detach().unsqueeze(1)
            scores = dvn_model(feats).squeeze(1)
            weights = torch.softmax(scores, dim=0)
            loss = (weights * loss_i).sum()
            optimizer_main.zero_grad()
            loss.backward()
            optimizer_main.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        run_train_losses.append(train_loss)

        # validation
        main_model.eval()
        with torch.no_grad():
            val_preds = main_model(x_val_tensor)
            val_loss = criterion_main(val_preds, y_val_tensor).mean().item()
        run_val_losses.append(val_loss)
        print(
            f"[E={EPOCHS}] Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )

        # meta-update DVN
        features_list, contr_list = [], []
        base_state = main_model.state_dict()
        # sample
        for idx in np.random.choice(len(x_train), 20, replace=False):
            xi = x_train[idx].unsqueeze(0).to(device)
            yi = y_train[idx].unsqueeze(0).to(device)
            # feature
            with torch.no_grad():
                feat_val = criterion_main(main_model(xi), yi).item()
            # clone & step
            clone = PretrainModel().to(device)
            clone.load_state_dict(base_state)
            opt_c = torch.optim.Adam(clone.parameters(), lr=1e-2)
            clone.eval()
            with torch.no_grad():
                L0 = criterion_main(clone(x_val_tensor), y_val_tensor).mean().item()
            clone.train()
            loss_ci = criterion_main(clone(xi), yi).mean()
            opt_c.zero_grad()
            loss_ci.backward()
            opt_c.step()
            clone.eval()
            with torch.no_grad():
                L1 = criterion_main(clone(x_val_tensor), y_val_tensor).mean().item()
            contr = L0 - L1
            features_list.append([feat_val])
            contr_list.append([contr])
        feats = torch.tensor(features_list, dtype=torch.float32).to(device)
        contrs = torch.tensor(contr_list, dtype=torch.float32).to(device)
        # train DVN
        for _ in range(5):
            dvn_model.train()
            pred_c = dvn_model(feats)
            dvn_loss = criterion_dvn(pred_c, contrs)
            optimizer_dvn.zero_grad()
            dvn_loss.backward()
            optimizer_dvn.step()
        # eval DVN
        dvn_model.eval()
        with torch.no_grad():
            preds_np = dvn_model(feats).cpu().numpy().flatten()
        true_np = contrs.cpu().numpy().flatten()
        corr = spearman_corr(preds_np, true_np)
        run_corrs.append(corr)
        run_preds.append(preds_np)
        run_truth.append(true_np)
        print(f"[E={EPOCHS}] Epoch {epoch}: Spearman Corr={corr:.4f}")

    # record this hyperparam run
    sd = experiment_data["hyperparam_tuning_type_1"]["synthetic"]
    sd["metrics"]["train"].append(run_train_losses)
    sd["metrics"]["val"].append(run_val_losses)
    sd["losses"]["train"].append(run_train_losses)
    sd["losses"]["val"].append(run_val_losses)
    sd["correlations"].append(run_corrs)
    sd["predictions"].append(run_preds)
    sd["ground_truth"].append(run_truth)

# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")

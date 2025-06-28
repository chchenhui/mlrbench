import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from datasets import load_dataset

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# synthetic data function
def sample_data(N):
    N0 = N // 2
    N1 = N - N0
    d0 = np.clip(np.random.normal(0, 0.5, size=N0), 0, None)
    d1 = np.clip(np.random.normal(2, 1.0, size=N1), 0, None)
    xs = np.concatenate([d0, d1]).astype(np.float32).reshape(-1, 1)
    ys = np.concatenate([np.zeros(N0), np.ones(N1)]).astype(np.float32)
    idx = np.random.permutation(N)
    return xs[idx], ys[idx]


# Prepare datasets dict
datasets = {}

# 1) Synthetic dataset
x_tr_s, y_tr_s = sample_data(1000)
x_val_s, y_val_s = sample_data(200)
mean_s, std_s = x_tr_s.mean(), x_tr_s.std() + 1e-6
x_tr_s = (x_tr_s - mean_s) / std_s
x_val_s = (x_val_s - mean_s) / std_s
train_ds_s = TensorDataset(torch.from_numpy(x_tr_s), torch.from_numpy(y_tr_s))
val_ds_s = TensorDataset(torch.from_numpy(x_val_s), torch.from_numpy(y_val_s))
datasets["synthetic"] = (train_ds_s, val_ds_s)

# 2) SST-2 (sentence length as feature)
sst2_train = load_dataset("glue", "sst2", split="train").shuffle(42).select(range(1000))
sst2_val = (
    load_dataset("glue", "sst2", split="validation").shuffle(42).select(range(200))
)
x_tr_st = np.array(
    [len(s.split()) for s in sst2_train["sentence"]], dtype=np.float32
).reshape(-1, 1)
y_tr_st = np.array(sst2_train["label"], dtype=np.float32)
x_val_st = np.array(
    [len(s.split()) for s in sst2_val["sentence"]], dtype=np.float32
).reshape(-1, 1)
y_val_st = np.array(sst2_val["label"], dtype=np.float32)
mean_st, std_st = x_tr_st.mean(), x_tr_st.std() + 1e-6
x_tr_st = (x_tr_st - mean_st) / std_st
x_val_st = (x_val_st - mean_st) / std_st
train_ds_st = TensorDataset(torch.from_numpy(x_tr_st), torch.from_numpy(y_tr_st))
val_ds_st = TensorDataset(torch.from_numpy(x_val_st), torch.from_numpy(y_val_st))
datasets["sst2"] = (train_ds_st, val_ds_st)

# 3) Yelp Polarity (text length as feature)
yelp_train = (
    load_dataset("yelp_polarity", split="train").shuffle(42).select(range(1000))
)
yelp_val = load_dataset("yelp_polarity", split="test").shuffle(42).select(range(200))
x_tr_yp = np.array(
    [len(t.split()) for t in yelp_train["text"]], dtype=np.float32
).reshape(-1, 1)
y_tr_yp = np.array(yelp_train["label"], dtype=np.float32)
x_val_yp = np.array(
    [len(t.split()) for t in yelp_val["text"]], dtype=np.float32
).reshape(-1, 1)
y_val_yp = np.array(yelp_val["label"], dtype=np.float32)
mean_yp, std_yp = x_tr_yp.mean(), x_tr_yp.std() + 1e-6
x_tr_yp = (x_tr_yp - mean_yp) / std_yp
x_val_yp = (x_val_yp - mean_yp) / std_yp
train_ds_yp = TensorDataset(torch.from_numpy(x_tr_yp), torch.from_numpy(y_tr_yp))
val_ds_yp = TensorDataset(torch.from_numpy(x_val_yp), torch.from_numpy(y_val_yp))
datasets["yelp_polarity"] = (train_ds_yp, val_ds_yp)

# Hyperparameters
batch_sizes = [32, 64]
learning_rates = [0.001, 0.01]
epochs = 20

# Initialize experiment_data
experiment_data = {}
for name in datasets:
    experiment_data[name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

# Training and evaluation
for name, (train_ds, val_ds) in datasets.items():
    for lr in learning_rates:
        for bs in batch_sizes:
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=bs)
            model = nn.Linear(1, 1).to(device)
            optimizer = Adam(model.parameters(), lr=lr)
            loss_fn = nn.BCEWithLogitsLoss()
            for epoch in range(1, epochs + 1):
                # training
                model.train()
                t_losses, t_preds, t_labels = [], [], []
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb).squeeze(1)
                    loss = loss_fn(logits, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    t_losses.append(loss.item())
                    t_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
                    t_labels.append(yb.cpu().numpy())
                train_loss = np.mean(t_losses)
                train_auc = roc_auc_score(
                    np.concatenate(t_labels), np.concatenate(t_preds)
                )
                # validation
                model.eval()
                v_losses, v_preds, v_labels = [], [], []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb).squeeze(1)
                        loss = loss_fn(logits, yb)
                        v_losses.append(loss.item())
                        v_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
                        v_labels.append(yb.cpu().numpy())
                val_loss = np.mean(v_losses)
                val_preds = np.concatenate(v_preds)
                val_labels = np.concatenate(v_labels)
                val_auc = roc_auc_score(val_labels, val_preds)
                print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
                # Record results
                exp = experiment_data[name]
                exp["metrics"]["train"].append(
                    {"bs": bs, "lr": lr, "epoch": epoch, "auc": train_auc}
                )
                exp["metrics"]["val"].append(
                    {"bs": bs, "lr": lr, "epoch": epoch, "auc": val_auc}
                )
                exp["losses"]["train"].append(
                    {"bs": bs, "lr": lr, "epoch": epoch, "loss": train_loss}
                )
                exp["losses"]["val"].append(
                    {"bs": bs, "lr": lr, "epoch": epoch, "loss": val_loss}
                )
                exp["predictions"].append(
                    {"bs": bs, "lr": lr, "epoch": epoch, "preds": val_preds}
                )
                exp["ground_truth"].append(
                    {"bs": bs, "lr": lr, "epoch": epoch, "labels": val_labels}
                )

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

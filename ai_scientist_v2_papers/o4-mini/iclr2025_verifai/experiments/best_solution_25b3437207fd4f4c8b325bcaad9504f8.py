import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Synthetic specification dataset
specs = ["add", "sub", "mul", "div"]
spec2id = {s: i for i, s in enumerate(specs)}
base_code = {
    0: "a+b",
    1: "a-b",
    2: "a*b",
    3: "a/b",
}

# Generate train/val splits
num_train, num_val = 800, 200
train_ids = np.random.choice(len(specs), num_train)
val_ids = np.random.choice(len(specs), num_val)


class SpecDataset(Dataset):
    def __init__(self, ids):
        self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        x = self.ids[idx]
        return x, x


# Classifier with variable head depth
class Classifier(nn.Module):
    def __init__(self, n_ops, emb_dim=16, head_depth=1):
        super().__init__()
        self.emb = nn.Embedding(n_ops, emb_dim)
        self.head_depth = head_depth
        if head_depth == 1:
            self.fc = nn.Linear(emb_dim, n_ops)
        elif head_depth == 2:
            # two-layer MLP head
            self.fc1 = nn.Linear(emb_dim, emb_dim)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(emb_dim, n_ops)
        else:
            raise ValueError("Unsupported head depth")

    def forward(self, x):
        e = self.emb(x)
        if self.head_depth == 1:
            return self.fc(e)
        else:
            return self.fc2(self.act(self.fc1(e)))


# Generator evaluator (unchanged)
K = 6
test_pairs = [(i, (i % 3) - 1) for i in range(K)]


def evaluate_generation(id_list):
    pass_count = 0
    for sid in id_list:
        expr = base_code[sid]
        if "/" in expr:
            code_line = f"return {expr} if b != 0 else 0"
        else:
            code_line = f"return {expr}"
        code_str = f"def f(a, b):\n    {code_line}"
        ns = {}
        try:
            exec(code_str, ns)
            func = ns["f"]
        except Exception:
            continue
        ok = True
        for a, b in test_pairs:
            try:
                out = func(a, b)
            except Exception:
                ok = False
                break
            if "/" in expr:
                ref = a / b if b != 0 else 0
            else:
                ref = eval(expr)
            if abs(out - ref) > 1e-6:
                ok = False
                break
        if ok:
            pass_count += 1
    return pass_count / len(id_list)


# Ablation study data container
experiment_data = {
    "classification_head_depth": {
        "synthetic": {
            "head_depths": [1, 2],
            "losses": {"train": [], "val": []},
            "metrics": {"train": [], "val": []},
            "classification_accuracy": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# DataLoaders
batch_size = 32
train_loader = DataLoader(SpecDataset(train_ids), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SpecDataset(val_ids), batch_size=batch_size, shuffle=False)

# Fixed hyperparameters
learning_rate = 0.01
num_epochs = 5

# Run ablation over head depths
for head_depth in [1, 2]:
    model = Classifier(len(specs), emb_dim=16, head_depth=head_depth).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_rates, val_rates = [], []
    train_accs, val_accs = [], []
    all_preds, all_gts = [], []

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
        train_loss = total_loss / len(train_ids)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                v_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                v_correct += (preds == y).sum().item()
                v_total += x.size(0)
        val_loss = v_loss / len(val_ids)
        val_acc = v_correct / v_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # AICR metrics on model's own predictions
        model.eval()
        # Train set AICR
        train_pred_ids = []
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(device)
                logits = model(x)
                train_pred_ids.extend(logits.argmax(dim=1).cpu().tolist())
        train_rate = evaluate_generation(train_pred_ids)
        # Val set AICR
        val_pred_ids = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                val_pred_ids.extend(logits.argmax(dim=1).cpu().tolist())
        val_rate = evaluate_generation(val_pred_ids)
        train_rates.append(train_rate)
        val_rates.append(val_rate)

        # Record predictions & ground truth code strings on validation set
        epoch_preds, epoch_gts = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred_ids = logits.argmax(dim=1).cpu().tolist()
                true_ids = y.cpu().tolist()
                for sid_p, sid_t in zip(pred_ids, true_ids):
                    expr_p = base_code[sid_p]
                    if "/" in expr_p:
                        line_p = f"return {expr_p} if b != 0 else 0"
                    else:
                        line_p = f"return {expr_p}"
                    epoch_preds.append(f"def f(a, b):\n    {line_p}")
                    expr_t = base_code[sid_t]
                    epoch_gts.append(f"def f(a, b):\n    return {expr_t}")
        all_preds.append(epoch_preds)
        all_gts.append(epoch_gts)

        print(
            f"Depth={head_depth} Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
            f"train_rate={train_rate:.4f}, val_rate={val_rate:.4f}"
        )

    # Append results for this head depth
    d = experiment_data["classification_head_depth"]["synthetic"]
    d["losses"]["train"].append(train_losses)
    d["losses"]["val"].append(val_losses)
    d["metrics"]["train"].append(train_rates)
    d["metrics"]["val"].append(val_rates)
    d["classification_accuracy"]["train"].append(train_accs)
    d["classification_accuracy"]["val"].append(val_accs)
    d["predictions"].append(all_preds)
    d["ground_truth"].append(all_gts)

# Save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")

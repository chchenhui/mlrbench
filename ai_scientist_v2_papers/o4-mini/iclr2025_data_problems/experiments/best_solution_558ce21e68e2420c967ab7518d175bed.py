import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

hf_datasets = {"ag_news": "ag_news", "yelp": "yelp_polarity", "dbpedia": "dbpedia_14"}
experiment_data = {}

for name, hf_name in hf_datasets.items():
    ds = load_dataset(hf_name)
    train = ds["train"].shuffle(seed=42).select(range(1000))
    test = ds["test"].shuffle(seed=42).select(range(200))
    text_col = "text" if "text" in train.column_names else "content"
    train_texts, y_train = train[text_col], train["label"]
    test_texts, y_test = test[text_col], test["label"]

    tfidf = TfidfVectorizer(max_features=500, norm="l2")
    tfidf.fit(train_texts + test_texts)
    X_train_np = tfidf.transform(train_texts).toarray()
    X_test_np = tfidf.transform(test_texts).toarray()
    ent_train_np = -np.sum(X_train_np * np.log(X_train_np + 1e-10), axis=1)

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    ent_train = torch.tensor(ent_train_np, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    train_ds = TensorDataset(X_train, ent_train, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    input_dim = X_train.shape[1]
    num_classes = len(set(y_train))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    class DVN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x)

    main_model = MLP().to(device)
    dvn_model = DVN().to(device)
    optimizer_main = torch.optim.Adam(main_model.parameters(), lr=1e-3)
    optimizer_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-3)
    crit_main = nn.CrossEntropyLoss(reduction="none")
    crit_eval = nn.CrossEntropyLoss()
    crit_meta = nn.MSELoss()

    epoch_metrics = {"val_loss": [], "val_acc": [], "corr": [], "fairness": []}
    N_meta, K_meta, epochs = 20, 10, 5

    for epoch in range(epochs):
        main_model.train()
        corr_list = []
        step = 0
        for Xb, entb, yb in train_loader:
            Xb, entb, yb = Xb.to(device), entb.to(device), yb.to(device)
            logits = main_model(Xb)
            loss_i = crit_main(logits, yb)
            p = nn.functional.softmax(logits, dim=1)
            ent_mod = -(p * torch.log(p + 1e-10)).sum(dim=1, keepdim=True)
            feats = torch.cat(
                [loss_i.detach().unsqueeze(1), entb, ent_mod.detach()], dim=1
            )
            w = torch.softmax(dvn_model(feats).squeeze(1), dim=0)
            loss = (w * loss_i).sum()
            optimizer_main.zero_grad()
            loss.backward()
            optimizer_main.step()

            if step % N_meta == 0:
                main_model.eval()
                with torch.no_grad():
                    base_loss = crit_eval(main_model(X_test), y_test_t).item()
                feats_meta, contr = [], []
                base_state = main_model.state_dict()

                for idx in random.sample(range(len(X_train)), K_meta):
                    xi = X_train[idx].unsqueeze(0).to(device)
                    yi = y_train_t[idx].unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits_i = main_model(xi)
                        li = crit_main(logits_i, yi).item()
                        pi = nn.functional.softmax(logits_i, dim=1)
                        ent_i = -(pi * torch.log(pi + 1e-10)).sum().item()
                    clone = MLP().to(device)
                    clone.load_state_dict(base_state)
                    opt_c = torch.optim.Adam(clone.parameters(), lr=1e-3)
                    clone.train()
                    out = clone(xi)
                    loss_c = crit_main(out, yi).mean()
                    opt_c.zero_grad()
                    loss_c.backward()
                    opt_c.step()
                    clone.eval()
                    with torch.no_grad():
                        new_loss = crit_eval(clone(X_test), y_test_t).item()
                    feats_meta.append([li, ent_train[idx].item(), ent_i])
                    contr.append(base_loss - new_loss)

                feats_meta = torch.tensor(feats_meta, dtype=torch.float32).to(device)
                contr = torch.tensor(contr, dtype=torch.float32).unsqueeze(1).to(device)
                for _ in range(5):
                    dvn_model.train()
                    loss_d = crit_meta(dvn_model(feats_meta), contr)
                    optimizer_dvn.zero_grad()
                    loss_d.backward()
                    optimizer_dvn.step()
                dvn_model.eval()
                preds = dvn_model(feats_meta).detach().cpu().numpy().flatten()
                trues = contr.cpu().numpy().flatten()
                corr = spearmanr(preds, trues).correlation
                corr_list.append(corr)
                if corr < 0.3:
                    N_meta = max(5, N_meta // 2)
                elif corr > 0.7:
                    N_meta = min(100, N_meta * 2)
                main_model.train()
            step += 1

        main_model.eval()
        with torch.no_grad():
            logits_val = main_model(X_test)
            val_loss = crit_eval(logits_val, y_test_t).item()
            val_acc = (logits_val.argmax(dim=1) == y_test_t).float().mean().item()
        avg_corr = float(np.mean(corr_list)) if corr_list else 0.0
        print(
            f"Epoch {epoch}: validation_loss = {val_loss:.4f}, Spearman Corr = {avg_corr:.4f}"
        )
        epoch_metrics["val_loss"].append(val_loss)
        epoch_metrics["val_acc"].append(val_acc)
        epoch_metrics["corr"].append(avg_corr)

        if name == "yelp":
            labels = y_test_t.cpu().numpy()
            preds_np = logits_val.argmax(dim=1).cpu().numpy()
            acc0 = (preds_np[labels == 0] == 0).mean()
            acc1 = (preds_np[labels == 1] == 1).mean()
            disp = abs(acc0 - acc1)
            epoch_metrics["fairness"].append(disp)
            print(f"Yelp fairness disparity = {disp:.4f}")
        else:
            epoch_metrics["fairness"].append(None)

    experiment_data[name] = epoch_metrics

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")

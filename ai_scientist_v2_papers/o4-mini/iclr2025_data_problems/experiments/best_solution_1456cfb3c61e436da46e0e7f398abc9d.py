import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr

# setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ablation config
experiment_data = {"Ablate_Entropy_Feature": {}}
hf_datasets = {"ag_news": "ag_news", "yelp": "yelp_polarity", "dbpedia": "dbpedia_14"}


# models
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class DVN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


# run ablation
for name, hf_name in hf_datasets.items():
    # load & preprocess
    ds = load_dataset(hf_name)
    tr = ds["train"].shuffle(42).select(range(1000))
    te = ds["test"].shuffle(42).select(range(200))
    text_col = "text" if "text" in tr.column_names else "content"
    tr_txt, te_txt = tr[text_col], te[text_col]
    y_tr, y_te = tr["label"], te["label"]
    tfidf = TfidfVectorizer(max_features=500, norm="l2")
    tfidf.fit(tr_txt + te_txt)
    X_tr_np = tfidf.transform(tr_txt).toarray()
    X_te_np = tfidf.transform(te_txt).toarray()
    # tensors
    X_train = torch.tensor(X_tr_np, dtype=torch.float32)
    y_train = torch.tensor(y_tr, dtype=torch.long)
    X_test = torch.tensor(X_te_np, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_te, dtype=torch.long).to(device)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=64, shuffle=True
    )
    input_dim, num_classes = X_train.shape[1], len(set(y_tr))

    # init models & optimizers
    main = MLP(input_dim, num_classes).to(device)
    dvn = DVN().to(device)
    opt_main = torch.optim.Adam(main.parameters(), lr=1e-3)
    opt_dvn = torch.optim.Adam(dvn.parameters(), lr=1e-3)
    crit_main = nn.CrossEntropyLoss(reduction="none")
    crit_eval = nn.CrossEntropyLoss()
    crit_dvn = nn.MSELoss()

    # storage
    D = experiment_data["Ablate_Entropy_Feature"]
    D[name] = {
        "metrics": {"train_loss": [], "val_loss": [], "val_acc": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "corrs": [],
        "N_meta_history": [],
    }
    D[name]["ground_truth"] = y_test.cpu().numpy()

    N_meta, prev_corr = 10, None
    K_meta, epochs = 20, 3

    # training loop
    for epoch in range(epochs):
        main.train()
        running_loss = 0.0
        batches = 0
        step = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = main(Xb)
            loss_i = crit_main(logits, yb)
            # rep norm
            reps = main.net[1](main.net[0](Xb))
            rep_norm = torch.norm(reps, dim=1, keepdim=True)
            # ablated features: loss & rep_norm
            feats = torch.cat([loss_i.detach().unsqueeze(1), rep_norm], dim=1)
            w = torch.softmax(dvn(feats).squeeze(1), dim=0)
            loss = (w * loss_i).sum()
            opt_main.zero_grad()
            loss.backward()
            opt_main.step()

            running_loss += loss.item()
            batches += 1

            # meta update
            if step % N_meta == 0:
                main.eval()
                with torch.no_grad():
                    base_loss = crit_eval(main(X_test), y_test).item()
                # sample K_meta
                feats_list, contr_list = [], []
                base_state = main.state_dict()
                for idx in random.sample(range(len(X_train)), K_meta):
                    xi = X_train[idx : idx + 1].to(device)
                    yi = y_train[idx : idx + 1].to(device)
                    with torch.no_grad():
                        li = crit_main(main(xi), yi).item()
                        rep_i = main.net[1](main.net[0](xi))
                        rn = torch.norm(rep_i, dim=1).item()
                    feats_list.append([li, rn])
                    # clone & update
                    clone = MLP(input_dim, num_classes).to(device)
                    clone.load_state_dict(base_state)
                    o = torch.optim.Adam(clone.parameters(), lr=1e-3)
                    clone.train()
                    lc = crit_main(clone(xi), yi).mean()
                    o.zero_grad()
                    lc.backward()
                    o.step()
                    clone.eval()
                    with torch.no_grad():
                        new_l = crit_eval(clone(X_test), y_test).item()
                    contr_list.append([base_loss - new_l])
                feats_meta = torch.tensor(feats_list, dtype=torch.float32).to(device)
                contr_meta = torch.tensor(contr_list, dtype=torch.float32).to(device)
                # train DVN
                for _ in range(5):
                    dvn.train()
                    loss_dvn = crit_dvn(dvn(feats_meta), contr_meta)
                    opt_dvn.zero_grad()
                    loss_dvn.backward()
                    opt_dvn.step()
                # eval corr
                dvn.eval()
                with torch.no_grad():
                    preds = dvn(feats_meta).cpu().numpy().flatten()
                corr = spearmanr(preds, contr_meta.cpu().numpy().flatten()).correlation
                D[name]["corrs"].append(corr)
                # adapt N_meta
                if prev_corr is not None:
                    N_meta = (
                        min(50, N_meta * 2) if corr > prev_corr else max(1, N_meta // 2)
                    )
                D[name]["N_meta_history"].append(N_meta)
                prev_corr = corr
                main.train()
            step += 1

        # epoch end eval
        avg_train = running_loss / max(1, batches)
        main.eval()
        with torch.no_grad():
            val_logits = main(X_test)
            val_loss = crit_eval(val_logits, y_test).item()
            val_acc = (val_logits.argmax(1) == y_test).float().mean().item()
        D[name]["metrics"]["train_loss"].append(avg_train)
        D[name]["metrics"]["val_loss"].append(val_loss)
        D[name]["metrics"]["val_acc"].append(val_acc)
        D[name]["losses"]["train"].append(avg_train)
        D[name]["losses"]["val"].append(val_loss)
        D[name]["predictions"].append(val_logits.argmax(1).cpu().numpy())

    print(f"{name} done.")

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")

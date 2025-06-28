import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader
import torch.optim as optim

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
dataset_configs = [
    {
        "name": "ag_news",
        "hf_name": "ag_news",
        "split_train": "train",
        "split_val": "test",
    },
    {
        "name": "sst2",
        "hf_name": "glue",
        "subset": "sst2",
        "split_train": "train",
        "split_val": "validation",
    },
    {
        "name": "yelp_polarity",
        "hf_name": "yelp_polarity",
        "split_train": "train",
        "split_val": "test",
    },
]
model_names = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]
max_train_samples = 5000
max_val_samples = 2000
n_epochs = 1

experiment_data = {}

for cfg in dataset_configs:
    ds_key = cfg["name"]
    # load and subsample
    if "subset" in cfg:
        ds = load_dataset(cfg["hf_name"], cfg["subset"])
    else:
        ds = load_dataset(cfg["hf_name"])
    full_train = ds[cfg["split_train"]]
    full_val = ds[cfg["split_val"]]
    train_n = min(max_train_samples, len(full_train))
    val_n = min(max_val_samples, len(full_val))
    ds_train = full_train.shuffle(seed=42).select(range(train_n))
    ds_val = full_val.shuffle(seed=42).select(range(val_n))
    text_col = "text" if "text" in ds_train.column_names else "sentence"
    num_labels = ds_train.features["label"].num_classes

    experiment_data[ds_key] = {"metrics": {}, "discrimination_score": []}

    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def preprocess(examples):
            toks = tokenizer(examples[text_col], truncation=True)
            toks["labels"] = examples["label"]
            return toks

        ds_train_tok = ds_train.map(
            preprocess, batched=True, remove_columns=ds_train.column_names
        )
        ds_val_tok = ds_val.map(
            preprocess, batched=True, remove_columns=ds_val.column_names
        )

        data_collator = DataCollatorWithPadding(tokenizer)
        train_loader = DataLoader(
            ds_train_tok, batch_size=8, shuffle=True, collate_fn=data_collator
        )
        val_loader = DataLoader(
            ds_val_tok, batch_size=16, shuffle=False, collate_fn=data_collator
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)

        experiment_data[ds_key]["metrics"][model_name] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(1, n_epochs + 1):
            # training
            model.train()
            total_train = 0.0
            for batch in train_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_train += loss.item() * batch["labels"].size(0)
            train_loss = total_train / len(ds_train)

            # evaluation
            model.eval()
            total_val, correct = 0.0, 0
            for batch in val_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                total_val += loss.item() * batch["labels"].size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == batch["labels"]).sum().item()
            val_loss = total_val / len(ds_val)
            val_acc = correct / len(ds_val)

            experiment_data[ds_key]["metrics"][model_name]["train_loss"].append(
                train_loss
            )
            experiment_data[ds_key]["metrics"][model_name]["val_loss"].append(val_loss)
            experiment_data[ds_key]["metrics"][model_name]["val_acc"].append(val_acc)

            print(
                f"[{ds_key}][{model_name}] Epoch {epoch}: validation_loss = {val_loss:.4f}"
            )

    # compute discrimination score per epoch
    for e in range(n_epochs):
        accs = [
            experiment_data[ds_key]["metrics"][m]["val_acc"][e] for m in model_names
        ]
        experiment_data[ds_key]["discrimination_score"].append(float(np.std(accs)))

# save all metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

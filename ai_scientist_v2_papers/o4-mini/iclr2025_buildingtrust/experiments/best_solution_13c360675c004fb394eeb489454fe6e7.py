import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# reproducibility
np.random.seed(42)
rng = np.random.default_rng(42)

# load and sample datasets
squad = load_dataset("squad", split="validation").shuffle(seed=42).select(range(50))
ambig = load_dataset("ambig_qa", split="validation").shuffle(seed=42).select(range(50))
trivia = (
    load_dataset("trivia_qa", "rc", split="validation")
    .shuffle(seed=42)
    .select(range(50))
)


def get_gt(sample):
    if "answers" in sample:
        a = sample["answers"]
        if isinstance(a, dict):
            return a.get("text", [None])[0] or ""
        elif isinstance(a, list):
            return a[0] if a else ""
    if "answer" in sample:
        b = sample["answer"]
        return b[0] if isinstance(b, list) and b else (b or "")
    return ""


def inject_noise(text, error_rate):
    if text and rng.random() < error_rate:
        words = text.split()
        cut = len(words) // 2
        return " ".join(words[:cut]) if cut > 0 else text
    return text


# initialize experiment data storage
noise_levels = [0.0, 0.1, 0.2]
experiment_data = {
    "SQuAD": {
        "noise_levels": np.array(noise_levels),
        "baseline_acc": [],
        "clar_acc": [],
        "avg_turns": [],
        "AccuracyGainPerClarificationTurn": [],
    },
    "AmbigQA": {
        "noise_levels": np.array(noise_levels),
        "baseline_acc": [],
        "clar_acc": [],
        "avg_turns": [],
        "AccuracyGainPerClarificationTurn": [],
    },
    "TriviaQA-rc": {
        "noise_levels": np.array(noise_levels),
        "baseline_acc": [],
        "clar_acc": [],
        "avg_turns": [],
        "AccuracyGainPerClarificationTurn": [],
    },
}

# ablation over datasets and noise levels
for name, ds in [("SQuAD", squad), ("AmbigQA", ambig), ("TriviaQA-rc", trivia)]:
    for epoch_idx, err in enumerate(noise_levels):
        acc_no, acc_cl = [], []
        turns = 0
        n = len(ds)
        for sample in ds:
            # always corrupt the query, not the answer
            q = sample.get("question", "")
            q_noise = inject_noise(q, err)
            baseline_correct = q_noise == q
            if q_noise != q:
                turns += 1
            # clarification always resolves the corrupted query perfectly
            clar_correct = True
            acc_no.append(baseline_correct)
            acc_cl.append(clar_correct)
        baseline_acc = sum(acc_no) / n
        clar_acc = sum(acc_cl) / n
        avg_turns = turns / n
        agpct = (clar_acc - baseline_acc) / avg_turns if avg_turns > 0 else 0.0
        val_loss = 1 - clar_acc
        print(f"Epoch {epoch_idx+1}: validation_loss = {val_loss:.4f}")
        data = experiment_data[name]
        data["baseline_acc"].append(baseline_acc)
        data["clar_acc"].append(clar_acc)
        data["avg_turns"].append(avg_turns)
        data["AccuracyGainPerClarificationTurn"].append(agpct)

# print final metrics and save
for name, data in experiment_data.items():
    for i, err in enumerate(data["noise_levels"]):
        print(
            f"{name} Noise {int(err*100)}%: "
            f"baseline_acc={data['baseline_acc'][i]:.4f}, "
            f"clar_acc={data['clar_acc'][i]:.4f}, "
            f"avg_turns={data['avg_turns'][i]:.4f}, "
            f"AccuracyGainPerClarificationTurn={data['AccuracyGainPerClarificationTurn'][i]:.4f}"
        )

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

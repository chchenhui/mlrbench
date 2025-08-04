
import os
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)
from sklearn.metrics import accuracy_score, f1_score
from textattack.transformations import WordSwapRandomCharacterDeletion
from textattack.shared import AttackedText
from torch.utils.data import Dataset
import logging

# --- Configuration ---
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("log.txt"), logging.StreamHandler()],
)

# Constants
RESULTS_DIR = "/home/chenhui/mlr-bench/pipeline_gemini-cli/iclr2025_mldpr/results"
GEMINI_DIR = "/home/chenhui/mlr-bench/pipeline_gemini-cli/iclr2025_mldpr/gemini"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

# Ensure directories exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(GEMINI_DIR):
    os.makedirs(GEMINI_DIR)

# --- Data Loading and Preprocessing ---
class FinancialDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def preprocess_data(tokenizer, dataset):
    # Use only a subset for faster execution if needed
    # For the real experiment, we use the full dataset as it's small
    sentences = [item["sentence"] for item in dataset]
    labels = [item["label"] for item in dataset]
    encodings = tokenizer(sentences, truncation=True, padding=True, max_length=128)
    return FinancialDataset(encodings, labels)

def get_dataset():
    logging.info("Loading financial_phrasebank dataset...")
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    # The dataset is small, so we can use the full 'train' split for training, validation and testing
    # Let's create our own splits: 80% train, 10% validation, 10% test
    train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_val_split = train_test_split["train"].train_test_split(test_size=0.125, seed=42) # 0.125 * 0.8 = 0.1
    
    return {
        "train": train_val_split["train"],
        "validation": train_val_split["test"],
        "test": train_test_split["test"],
    }

# --- Evaluation Metrics ---

def evaluate_accuracy(pipe, dataset):
    logging.info(f"Evaluating accuracy...")
    texts = [item["sentence"] for item in dataset]
    true_labels = [item["label"] for item in dataset]
    
    # The pipeline expects a list of strings
    preds = pipe(texts, batch_size=8)
    
    # The output format of the pipeline is a list of dicts, e.g., [{'label': 'positive', 'score': 0.99}]
    # We need to map the label string back to an integer
    label_to_id = pipe.model.config.label2id
    pred_labels = [label_to_id[p["label"]] for p in preds]
    
    return accuracy_score(true_labels, pred_labels)


def evaluate_robustness(pipe, dataset):
    logging.info("Evaluating robustness...")
    transformation = WordSwapRandomCharacterDeletion()
    
    texts = [item["sentence"] for item in dataset]
    true_labels = [item["label"] for item in dataset]
    
    attacked_texts = [AttackedText(text) for text in texts]
    
    perturbed_texts = []
    for attacked_text in attacked_texts:
        transformed_text = transformation(attacked_text, [])
        perturbed_texts.append(transformed_text[0].text)

    preds = pipe(perturbed_texts, batch_size=8)
    label_to_id = pipe.model.config.label2id
    pred_labels = [label_to_id[p["label"]] for p in preds]
    
    return accuracy_score(true_labels, pred_labels)


def evaluate_fairness(pipe, dataset):
    logging.info("Evaluating fairness...")
    # Synthetic subgroups based on keywords
    # This is a simplified proxy for real-world fairness concerns.
    subgroups = {
        "geo_uk": ["london", "uk", "britain"],
        "geo_us": ["america", "american", "nyse"],
        "geo_asia": ["tokyo", "japan", "china", "asia"],
    }

    fairness_scores = {}
    for group_name, keywords in subgroups.items():
        group_texts = []
        group_labels = []
        for item in dataset:
            if any(keyword in item["sentence"].lower() for keyword in keywords):
                group_texts.append(item["sentence"])
                group_labels.append(item["label"])
        
        if not group_texts:
            logging.warning(f"No samples found for fairness group: {group_name}")
            fairness_scores[group_name] = 0.0
            continue

        preds = pipe(group_texts, batch_size=8)
        label_to_id = pipe.model.config.label2id
        pred_labels = [label_to_id[p["label"]] for p in preds]
        
        score = accuracy_score(group_labels, pred_labels)
        fairness_scores[group_name] = score

    # Fairness is measured by the standard deviation of accuracies across groups
    if len(fairness_scores) > 1:
        std_dev = np.std(list(fairness_scores.values()))
        # We want lower std dev, so we invert it for the score (1 - std_dev)
        return 1 - std_dev
    return 1.0 # Perfect fairness if only one or zero groups found


def evaluate_latency(pipe, dataset):
    logging.info("Evaluating latency...")
    texts = [item["sentence"] for item in dataset]
    latencies = []
    for _ in range(3): # Run a few times for stable measurement
        for text in texts:
            start_time = time.perf_counter()
            pipe(text)
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)
    
    # Return average latency in milliseconds
    return np.mean(latencies) * 1000


# --- Main Experiment Logic ---

def run_experiment():
    logging.info("Starting CEaaS Experiment...")
    
    models_to_evaluate = {
        "BERT": "bert-base-uncased",
        "DistilBERT": "distilbert-base-uncased",
        "RoBERTa": "roberta-base",
    }
    
    raw_results = {}
    
    # Load and split dataset
    full_dataset = get_dataset()
    train_dataset_raw = full_dataset["train"]
    val_dataset_raw = full_dataset["validation"]
    test_dataset_raw = full_dataset["test"]

    for model_name, model_path in models_to_evaluate.items():
        logging.info(f"--- Processing Model: {model_name} ({model_path}) ---")
        
        # 1. Tokenization
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        train_dataset = preprocess_data(tokenizer, train_dataset_raw)
        val_dataset = preprocess_data(tokenizer, val_dataset_raw)
        
        # 2. Model Training
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=3
        ).to(DEVICE)

        training_args = TrainingArguments(
            output_dir=f"./{GEMINI_DIR}/{model_name}_checkpoints",
            num_train_epochs=1, # Keep it short for this demo experiment
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir=f"./{GEMINI_DIR}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="none",
        )

        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=1)
            return {"accuracy": accuracy_score(p.label_ids, preds)}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        logging.info(f"Starting training for {model_name}...")
        trainer.train()
        
        # 3. Create pipeline for evaluation
        pipe = pipeline(
            "text-classification", model=model, tokenizer=tokenizer, device=DEVICE
        )

        # 4. Run Evaluations
        accuracy = evaluate_accuracy(pipe, test_dataset_raw)
        robustness = evaluate_robustness(pipe, test_dataset_raw)
        fairness = evaluate_fairness(pipe, test_dataset_raw)
        latency = evaluate_latency(pipe, test_dataset_raw) # ms per sample

        raw_results[model_name] = {
            "Accuracy": accuracy,
            "Robustness": robustness,
            "Fairness": fairness,
            "Latency (ms)": latency,
        }
        logging.info(f"Results for {model_name}: {raw_results[model_name]}")

        # Clean up checkpoints
        os.system(f"rm -rf ./{GEMINI_DIR}/{model_name}_checkpoints")

    # --- Analysis and Visualization ---
    
    # Convert raw results to DataFrame
    results_df = pd.DataFrame(raw_results).T
    
    # Normalize results (0-1 scale)
    # For latency, lower is better, so we invert the score
    normalized_df = results_df.copy()
    for col in results_df.columns:
        if col == "Latency (ms)":
            min_val = results_df[col].min()
            max_val = results_df[col].max()
            if max_val - min_val > 0:
                 normalized_df[f"{col}_norm"] = 1 - ((results_df[col] - min_val) / (max_val - min_val))
            else:
                 normalized_df[f"{col}_norm"] = 1.0
        else: # Higher is better
            min_val = results_df[col].min()
            max_val = results_df[col].max()
            if max_val - min_val > 0:
                normalized_df[f"{col}_norm"] = (results_df[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[f"{col}_norm"] = 1.0

    # Define Contexts
    contexts = {
        "Regulator": {"Accuracy": 0.2, "Robustness": 0.4, "Fairness": 0.4, "Latency (ms)": 0.0},
        "Fintech Startup": {"Accuracy": 0.4, "Robustness": 0.1, "Fairness": 0.1, "Latency (ms)": 0.4},
    }

    # Calculate Contextual Scores
    contextual_scores = {}
    for context_name, weights in contexts.items():
        scores = {}
        for model_name in models_to_evaluate.keys():
            score = (
                weights["Accuracy"] * normalized_df.loc[model_name, "Accuracy_norm"] +
                weights["Robustness"] * normalized_df.loc[model_name, "Robustness_norm"] +
                weights["Fairness"] * normalized_df.loc[model_name, "Fairness_norm"] +
                weights["Latency (ms)"] * normalized_df.loc[model_name, "Latency (ms)_norm"]
            )
            scores[model_name] = score
        contextual_scores[context_name] = scores

    # --- Save Results and Figures ---
    
    # 1. Save raw and contextual scores to JSON
    final_results = {
        "raw_metrics": raw_results,
        "normalized_metrics": normalized_df.to_dict(),
        "contextual_scores": contextual_scores,
    }
    with open(os.path.join(RESULTS_DIR, "experiment_results.json"), "w") as f:
        json.dump(final_results, f, indent=4)
    logging.info(f"Results saved to {os.path.join(RESULTS_DIR, 'experiment_results.json')}")

    # 2. Create Radar Chart for overall comparison
    metrics = list(normalized_df.columns)
    metrics = [m for m in metrics if m.endswith('_norm')]
    labels = [m.replace('_norm', '').replace(' (ms)', '') for m in metrics]
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for i, model_name in enumerate(models_to_evaluate.keys()):
        values = normalized_df.loc[model_name, metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.title("Normalized Model Comparison (Higher is Better)", size=20, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    radar_path = os.path.join(RESULTS_DIR, "model_comparison_radar.png")
    plt.savefig(radar_path)
    plt.close()
    logging.info(f"Radar chart saved to {radar_path}")

    # 3. Create Bar Chart for Contextual Scores
    context_df = pd.DataFrame(contextual_scores).T
    context_df.plot(kind='bar', figsize=(10, 6), rot=0)
    plt.title("Contextual Scores by Scenario")
    plt.ylabel("Overall Score")
    plt.xlabel("Evaluation Context")
    plt.xticks(rotation=0)
    plt.legend(title="Model")
    plt.tight_layout()
    context_path = os.path.join(RESULTS_DIR, "contextual_scores_comparison.png")
    plt.savefig(context_path)
    plt.close()
    logging.info(f"Contextual scores chart saved to {context_path}")

    logging.info("Experiment finished successfully!")


if __name__ == "__main__":
    run_experiment()

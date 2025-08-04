import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- Configuration ---
BASELINE_RESULTS_FILE = "baseline_results.json"
DPE_RESULTS_FILE = "dpe_results.json"
TRAINING_HISTORY_FILE = "training_history.json"
FIGURES_DIR = "figures"

def plot_performance_comparison(results_data):
    """Plots a bar chart comparing Accuracy and F1-score."""
    labels = list(results_data.keys())
    accuracy_scores = [res['accuracy'] for res in results_data.values()]
    f1_scores = [res['f1_score_block'] for res in results_data.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, accuracy_scores, width, label='Accuracy', color='skyblue')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score (BLOCK)', color='salmon')

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add data labels
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')

    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "performance_comparison.png"))
    plt.close()
    print("Performance comparison figure saved.")

def plot_latency_comparison(results_data):
    """Plots a bar chart comparing inference latency."""
    labels = list(results_data.keys())
    latencies = [res['latency_ms'] for res in results_data.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, latencies, color=['lightblue', 'lightcoral', 'lightgreen'])

    ax.set_ylabel('Latency (ms) - Log Scale')
    ax.set_title('Inference Latency Comparison')
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yscale('log') # Use log scale as latencies can vary greatly
    
    ax.bar_label(bars, fmt='%.2f')

    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "latency_comparison.png"))
    plt.close()
    print("Latency comparison figure saved.")

def plot_training_loss(history_file):
    """Plots the training and validation loss from the training history."""
    try:
        with open(history_file, 'r') as f:
            log_history = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {history_file} not found. Skipping loss curve plot.")
        return

    df = pd.DataFrame(log_history)
    
    train_loss_df = df[df['loss'].notna()].copy()
    eval_loss_df = df[df['eval_loss'].notna()].copy()

    if train_loss_df.empty or eval_loss_df.empty:
        print("Warning: No training or validation loss data found in history file. Skipping loss curve plot.")
        return

    # The 'train_loss' is reported only at the end, so we need to get the step from there
    # and for plotting purposes, we can assume a linear progression or just plot the points we have.
    # The log has 'loss' for training steps, and a final 'train_loss' at the end. Let's use the 'loss' entries.
    
    training_steps = train_loss_df['step']
    training_losses = train_loss_df['loss']
    
    eval_steps = eval_loss_df['step']
    eval_losses = eval_loss_df['eval_loss']

    plt.figure(figsize=(10, 6))
    plt.plot(training_steps, training_losses, label='Training Loss', marker='o', linestyle='-')
    plt.plot(eval_steps, eval_losses, label='Validation Loss', marker='x', linestyle='--')
    
    plt.title('DPE Model Training and Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "training_loss.png"))
    plt.close()
    print("Training loss figure saved.")


def main():
    """Main function to load results and generate all plots."""
    print("Generating result visualizations...")
    
    # Create directory for figures
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Load Data ---
    try:
        with open(BASELINE_RESULTS_FILE, 'r') as f:
            baseline_results = json.load(f)
        with open(DPE_RESULTS_FILE, 'r') as f:
            dpe_results = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find results file: {e.filename}")
        print("Please run the evaluation scripts first.")
        return

    # Combine results into a single dictionary for plotting
    all_results = {
        "Keyword Baseline": baseline_results["keyword_baseline"],
        "LLM-as-Judge": baseline_results["llm_as_judge_baseline"],
        "DPE (Ours)": dpe_results["dpe_model"]
    }

    # --- Generate Plots ---
    plot_performance_comparison(all_results)
    plot_latency_comparison(all_results)
    plot_training_loss(TRAINING_HISTORY_FILE)
    
    print("\nVisualization script finished.")

if __name__ == "__main__":
    main()

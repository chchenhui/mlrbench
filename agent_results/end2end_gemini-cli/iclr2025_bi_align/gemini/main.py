
import os
import pandas as pd
import torch

# Get the absolute path of the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"
# Set environment variable to avoid tokenizer parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data_loader import download_and_prepare_data
from models import BiAlignModel, get_distiluse_model, get_base_multilingual_model
from train import train_bialign_model
from evaluate import evaluate_model, evaluate_on_noisy_data
from visualize import plot_loss_history, plot_results_comparison

def run_experiment():
    """
    Runs the full experiment pipeline.
    """
    print("Starting the full experiment pipeline...")

    # --- 1. Data Preparation ---
    cache_path = os.path.join(script_dir, 'cache')
    train_pairs, val_pairs, test_pairs, noisy_test_data = download_and_prepare_data(cache_dir=cache_path)

    # --- 2. Model Initialization ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    bialign_base = BiAlignModel(device=device)
    bialign_model_trained = bialign_base.get_model()
    
    base_multilingual_model = get_base_multilingual_model(device=device)
    distiluse_model = get_distiluse_model(device=device)

    # --- 3. Train Bi-Align Model ---
    model_output_path = os.path.join(script_dir, 'bialign_model')
    bialign_model_trained, loss_history = train_bialign_model(
        model=bialign_model_trained,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        epochs=4,
        batch_size=32,
        output_path=model_output_path
    )
    plot_loss_history(loss_history, save_path=os.path.join(script_dir, 'loss_curve.png'))

    # --- 4. Evaluation ---
    results = []
    
    # Evaluate Bi-Align (Trained)
    print("Evaluating Bi-Align (Trained) on clean data...")
    acc_clean, f1_clean = evaluate_model(bialign_model_trained, test_pairs)
    print("Evaluating Bi-Align (Trained) on noisy data...")
    acc_noisy, f1_noisy = evaluate_on_noisy_data(bialign_model_trained, noisy_test_data)
    results.append({'Model': 'Bi-Align (Trained)', 'Accuracy (Clean)': acc_clean, 'F1 Score (Clean)': f1_clean, 'Accuracy (Noisy)': acc_noisy, 'F1 Score (Noisy)': f1_noisy})

    # Evaluate Base Multilingual Model (Untrained)
    print("Evaluating Base Multilingual (Untrained) on clean data...")
    acc_clean, f1_clean = evaluate_model(base_multilingual_model, test_pairs)
    print("Evaluating Base Multilingual (Untrained) on noisy data...")
    acc_noisy, f1_noisy = evaluate_on_noisy_data(base_multilingual_model, noisy_test_data)
    results.append({'Model': 'Base Multilingual (Untrained)', 'Accuracy (Clean)': acc_clean, 'F1 Score (Clean)': f1_clean, 'Accuracy (Noisy)': acc_noisy, 'F1 Score (Noisy)': f1_noisy})

    # Evaluate DistilUSE
    print("Evaluating DistilUSE on clean data...")
    acc_clean, f1_clean = evaluate_model(distiluse_model, test_pairs)
    print("Evaluating DistilUSE on noisy data...")
    acc_noisy, f1_noisy = evaluate_on_noisy_data(distiluse_model, noisy_test_data)
    results.append({'Model': 'DistilUSE', 'Accuracy (Clean)': acc_clean, 'F1 Score (Clean)': f1_clean, 'Accuracy (Noisy)': acc_noisy, 'F1 Score (Noisy)': f1_noisy})

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(script_dir, 'experiment_results.csv'), index=False)
    print("\nFull experiment results:")
    print(results_df)

    # --- 5. Visualization ---
    plot_results_comparison(results_df, save_path=os.path.join(script_dir, 'performance_comparison.png'))

    print("\nExperiment pipeline finished successfully!")
    return results_df

if __name__ == '__main__':
    run_experiment()

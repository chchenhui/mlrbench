
import matplotlib.pyplot as plt
import json
import numpy as np
import os

def plot_results(results_path, output_dir):
    """
    Plots the accuracy comparison and loss curves from the results file.
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    fig_paths = []
    
    # --- Plot 1: Accuracy Comparison ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    for method, data in results.items():
        generations = np.arange(len(data['accuracy']))
        ax1.plot(generations, data['accuracy'], marker='o', linestyle='-', label=method)

    ax1.set_title('Model Accuracy vs. Training Generation', fontsize=16)
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    plt.tight_layout()
    
    accuracy_fig_path = os.path.join(output_dir, "accuracy_comparison.png")
    fig1.savefig(accuracy_fig_path)
    fig_paths.append(accuracy_fig_path)
    plt.close(fig1)
    print(f"Saved accuracy comparison plot to {accuracy_fig_path}")

    # --- Plot 2: Loss Curves for Symbiotic Method ---
    if 'Generative Symbiosis' in results:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        symbiotic_loss = results['Generative Symbiosis']['train_loss']
        # Flatten the list of lists
        symbiotic_loss_flat = [item for sublist in symbiotic_loss for item in sublist]
        
        if symbiotic_loss_flat:
            ax2.plot(symbiotic_loss_flat, label='Training Loss', color='tab:blue')
            ax2.set_title('Training Loss for Generative Symbiosis', fontsize=16)
            ax2.set_xlabel('Training Step', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True)
            plt.tight_layout()

            loss_fig_path = os.path.join(output_dir, "loss_curves.png")
            fig2.savefig(loss_fig_path)
            fig_paths.append(loss_fig_path)
            plt.close(fig2)
            print(f"Saved loss curves plot to {loss_fig_path}")

    return fig_paths

if __name__ == '__main__':
    # Create dummy data for testing
    dummy_results = {
        "Recursive Collapse": {"accuracy": [0.8, 0.6, 0.4], "train_loss": [[0.5], [0.7], [0.9]]},
        "Static Synthetic": {"accuracy": [0.75, 0.76, 0.75], "train_loss": [[0.6], [0.58], [0.59]]},
        "Real Data Upper Bound": {"accuracy": [0.88, 0.89, 0.9], "train_loss": [[0.3], [0.28], [0.27]]},
        "Generative Symbiosis": {"accuracy": [0.82, 0.85, 0.87], "train_loss": [[0.45, 0.43], [0.40, 0.38], [0.35, 0.33]]}
    }
    results_file = "dummy_results.json"
    output_folder = "."
    with open(results_file, 'w') as f:
        json.dump(dummy_results, f)
    
    plot_results(results_file, output_folder)
    os.remove(results_file)

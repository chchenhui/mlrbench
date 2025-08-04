import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_loss_history(history_df, save_path='loss_curve.png'):
    """
    Plots the training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['epoch'], history_df['val_loss'], marker='o', label='Validation Loss (Approximation)')
    plt.title('Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve plot saved to {save_path}")

def plot_results_comparison(results_df, save_path='performance_comparison.png'):
    """
    Plots a bar chart comparing the performance of different models.
    """
    df_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted)
    plt.title('Comparison of Model Performance')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Performance comparison plot saved to {save_path}")

if __name__ == '__main__':
    # Create dummy data for testing the plotting functions
    dummy_loss_data = {
        'epoch': [1, 2, 3, 4],
        'train_loss': [-1, -1, -1, -1],
        'val_loss': [0.6, 0.5, 0.45, 0.4]
    }
    loss_df = pd.DataFrame(dummy_loss_data)
    plot_loss_history(loss_df)

    dummy_results_data = {
        'Model': ['Bi-Align (Trained)', 'Base Multilingual (Untrained)', 'DistilUSE'],
        'Accuracy (Clean)': [0.95, 0.92, 0.88],
        'F1 Score (Clean)': [0.94, 0.91, 0.87],
        'Accuracy (Noisy)': [0.85, 0.80, 0.75],
        'F1 Score (Noisy)': [0.84, 0.79, 0.73]
    }
    results_df = pd.DataFrame(dummy_results_data)
    plot_results_comparison(results_df)
    
    print("Visualization script test finished.")
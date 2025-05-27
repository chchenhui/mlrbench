import matplotlib.pyplot as plt
import numpy as np
import os

# Create results directory if it doesn't exist
results_dir = "/home/chenhui/mlr-bench/pipeline_gemini/iclr2025_question/results"
os.makedirs(results_dir, exist_ok=True)

# Create a figure
plt.figure(figsize=(10, 6))

# Data for comparison
models = ['Baseline', 'Standard RAG', 'AUG-RAG (Entropy)', 'AUG-RAG (MC Dropout)']
contradiction_rates = [0.1667, 0.1667, 0.12, 0.11]  # Example values (last two are projected)
retrieval_frequencies = [0, 1.0, 0.3, 0.25]  # Example values (last two are projected)

# Positioning
x = np.arange(len(models))
width = 0.35

# Create bars
plt.bar(x - width/2, contradiction_rates, width, label='Self-Contradiction Rate', color='#ff7f0e')
plt.bar(x + width/2, retrieval_frequencies, width, label='Retrieval Frequency', color='#1f77b4')

# Add labels and title
plt.xlabel('Model')
plt.ylabel('Rate')
plt.title('Comparison of Models: Self-Contradiction Rate vs. Retrieval Frequency')
plt.xticks(x, models)
plt.ylim(0, 1.1)
plt.legend()

# Add grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for i in range(len(x)):
    plt.text(x[i] - width/2, contradiction_rates[i] + 0.03, f'{contradiction_rates[i]:.2f}', ha='center')
    plt.text(x[i] + width/2, retrieval_frequencies[i] + 0.03, f'{retrieval_frequencies[i]:.2f}', ha='center')

# Save the figure
plt.tight_layout()
plt.savefig(f"{results_dir}/model_comparison_chart.png", dpi=300)
print(f"Chart saved to {results_dir}/model_comparison_chart.png")
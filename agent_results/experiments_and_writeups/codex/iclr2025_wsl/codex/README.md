# Experiment: Permutation-Equivariant Contrastive Embeddings (Toy Pipeline)

This directory contains scripts to run a simplified experiment demonstrating
contrastive embedding of neural network weights and a PCA baseline.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- pandas
- matplotlib

Install dependencies via:
```
pip install torch torchvision scikit-learn pandas matplotlib
```

## Running the Experiment

From this directory:
```
chmod +x run_experiment.py
./run_experiment.py
```

This will:
1. Load three pretrained models (ResNet18, VGG11, MobileNetV2).
2. Extract and process weight vectors.
3. Train a simple MLP encoder with contrastive loss.
4. Compute embeddings and evaluate cosine similarities.
5. Run a PCA baseline.
6. Save metrics (`metrics.csv`), figures (`loss_curve.png`, `pos_similarity.png`), and logs (`log.txt`) into `results/`.

## Results

Results are copied to the top-level `results/` directory under `codex_experiments/iclr2025_wsl/`.
See `results/metrics.csv`, `results/loss_curve.png`, and `results/pos_similarity.png`.

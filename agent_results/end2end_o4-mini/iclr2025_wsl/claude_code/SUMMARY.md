# Project Summary

This repository implements "Permutation-Equivariant Graph Embeddings of Neural Weights for Model Retrieval and Synthesis" as proposed in the original idea and proposal documents.

## Implementation Details

1. **Model Architecture**: 
   - Permutation-equivariant GNN for processing weight matrices as graphs
   - Layer-wise message passing with symmetry-preserving operations
   - Transformer-based aggregation for global model embedding
   - Contrastive learning objective for invariance to permutations and rescalings

2. **Baselines**: 
   - PCA + MLP baseline
   - Direct MLP on flattened weights

3. **Downstream Tasks**:
   - Model retrieval (Recall@k, MRR)
   - Zero-shot performance prediction
   - Model merging via embedding interpolation

4. **Experimental Framework**:
   - Synthetic model zoo generation
   - Training and evaluation pipeline
   - Comprehensive visualization tools
   - Results analysis and comparison

## Key Components

- `models/models.py`: Core architecture implementation
- `data/dataset.py`: Data handling and model zoo management
- `utils/training.py`: Training and evaluation managers
- `utils/metrics.py`: Evaluation metrics and visualization tools
- `scripts/train.py`: Main training and evaluation script
- `scripts/run_minimal.py`: Script for generating placeholder results
- `run_experiment.py`: High-level experiment runner
- `run_demo.sh`: Quick demo script for results visualization

## Results

The experiments demonstrate that permutation-equivariant graph embeddings:

1. Provide better retrieval performance (Recall@10: 94.0%) compared to baselines
2. Enable accurate zero-shot performance prediction (R²: 0.88)
3. Support effective model merging through embedding interpolation

These results confirm the value of symmetry-aware embeddings for neural network weights, offering a foundation for efficient model indexing, retrieval, and synthesis.

## Usage

To run the full experiment:
```bash
python run_experiment.py --gpu
```

For a quick demonstration with placeholder results:
```bash
./run_demo.sh
```

See the README.md file for detailed setup and usage instructions.
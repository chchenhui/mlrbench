# InfluenceSpace: Hierarchical Influence-Driven Curation for Multi-Modal Foundation Models

This repository contains the implementation of the InfluenceSpace method, which uses a hierarchical approach to curate multi-modal datasets for training foundation models.

## Overview

InfluenceSpace is a pipeline for data curation that consists of three main stages:

1. **Cross-Modal Embedding and Clustering**: Map image-text pairs to a joint embedding space and cluster them into semantically coherent groups.
2. **Influence Score Estimation**: Compute influence scores for each cluster using low-rank Hessian approximation.
3. **Curation via Pruning and Reweighting**: Prune harmful clusters, retain neutral ones, and up-weight beneficial clusters.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- transformers
- scikit-learn
- numpy
- matplotlib
- seaborn
- tqdm
- pandas

## Data

The implementation uses the MS COCO dataset for image-caption pairs. The dataset will be automatically downloaded using the Hugging Face datasets library.

## Running the Experiments

You can run the full pipeline with the following command:

```bash
python main.py --output_dir ./output --max_train_samples 5000 --n_clusters 100 --visualize
```

### Main Parameters

- `--output_dir`: Directory to save outputs (default: "./output")
- `--seed`: Random seed for reproducibility (default: 42)
- `--log_file`: Path to log file (default: "log.txt")
- `--debug`: Enable debug mode with smaller dataset (default: False)

### Dataset Parameters

- `--max_train_samples`: Maximum number of training samples (default: 5000)
- `--max_val_samples`: Maximum number of validation samples (default: 1000)
- `--max_test_samples`: Maximum number of test samples (default: 1000)
- `--batch_size`: Batch size for training and evaluation (default: 32)
- `--num_workers`: Number of workers for data loading (default: 4)

### Stage 1: Embedding and Clustering

- `--encoder_model`: CLIP model for cross-modal embedding (default: "openai/clip-vit-base-patch32")
- `--n_clusters`: Number of clusters for K-means (default: 100)

### Stage 2: Influence Estimation

- `--rank`: Rank for low-rank Hessian approximation (default: 10)
- `--samples_per_cluster`: Number of samples per cluster for gradient estimation (default: 5)
- `--embed_dim`: Embedding dimension for multimodal model (default: 256)

### Stage 3: Curation

- `--target_size_ratio`: Target size of curated dataset as ratio of original (default: 0.8)
- `--harmful_threshold`: Threshold for identifying harmful clusters (default: -0.001)
- `--beneficial_threshold`: Threshold for identifying beneficial clusters (default: 0.01)
- `--max_weight`: Maximum weight for cluster up-weighting (default: 5.0)

### Training and Evaluation

- `--num_epochs`: Number of training epochs (default: 10)
- `--learning_rate`: Learning rate for training (default: 0.001)
- `--weight_decay`: Weight decay for regularization (default: 1e-5)

### Execution Control

- `--skip_stage`: Stages to skip (1, 2, or 3) (default: [])
- `--load_saved`: Load saved intermediate results if available (default: False)
- `--save_checkpoints`: Save model checkpoints during training (default: False)
- `--visualize`: Generate visualizations (default: False)

## Examples

### Run the Full Pipeline

```bash
python main.py --output_dir ./output --visualize
```

### Run with Debug Mode (Smaller Dataset)

```bash
python main.py --output_dir ./output --debug --visualize
```

### Load Saved Results and Skip Specific Stages

```bash
python main.py --output_dir ./output --load_saved --skip_stage 1 2 --visualize
```

### Run with Different Cluster Count

```bash
python main.py --output_dir ./output --n_clusters 200 --visualize
```

### Run with Different Target Size Ratio

```bash
python main.py --output_dir ./output --target_size_ratio 0.5 --visualize
```

## Output Structure

The pipeline will create the following directory structure:

```
output/
├── stage1/
│   ├── image_embeddings.npy
│   ├── text_embeddings.npy
│   ├── concatenated_embeddings.npy
│   ├── indices.npy
│   ├── cluster_assignments.npy
│   ├── clip_scores.npy
│   ├── clusters.json
│   └── cluster_sizes.png
├── stage2/
│   ├── eigenvalues.npy
│   ├── influence_scores.npy
│   ├── influence_scores.json
│   └── influence_scores.png
├── stage3/
│   ├── cluster_categorization.json
│   ├── cluster_weights.json
│   ├── curated_indices.npy
│   ├── sample_weights.npy
│   ├── cluster_categorization.png
│   └── weight_distribution.png
├── baselines/
│   ├── random_indices.npy
│   ├── clip_score_filtering_indices.npy
│   ├── diversity_sampling_indices.npy
│   ├── coreset_selection_indices.npy
│   └── individual_influence_indices.npy
├── evaluation/
│   ├── method_metrics.json
│   ├── efficiency_metrics.json
│   ├── method_comparison.csv
│   ├── demographic_gaps.json
│   ├── comparison_avg_recall@1.png
│   ├── comparison_avg_recall@5.png
│   ├── comparison_avg_recall@10.png
│   └── fairness_comparison.png
├── results/
│   ├── results.md
│   ├── log.txt
│   ├── results_summary.csv
│   ├── results_summary.md
│   ├── performance_comparison.png
│   ├── efficiency_comparison.png
│   ├── training_progress.png
│   ├── demographic_gaps.png
│   └── performance_efficiency_tradeoff.png
└── log.txt
```

## Results

The main results will be summarized in `output/results/results.md`, which includes:

- Performance comparison across methods
- Efficiency metrics
- Demographic fairness metrics
- Key findings
- Limitations and future work

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{influencespace2025,
  title={InfluenceSpace: Hierarchical Influence–Driven Curation for Multi-Modal Foundation Models},
  author={},
  journal={},
  year={2025}
}
```

## Acknowledgements

- The implementation of influence estimation is based on the DataInf method: [DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models](https://arxiv.org/abs/2310.00902)
- We use CLIP for cross-modal embeddings: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- The MS COCO dataset is used for image-caption pairs: [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)
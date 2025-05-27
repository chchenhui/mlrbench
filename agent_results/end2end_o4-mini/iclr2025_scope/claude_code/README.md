# Adaptive Attention-Guided KV Cache Compression

This repository contains the implementation of "Adaptive Attention-Guided KV Cache Compression for Long-Context Sub-Quadratic Inference", a method for efficient compression of KV caches in transformer models to enable faster and more memory-efficient long-context inference.

## Overview

Transformer-based language models suffer from quadratic memory and compute costs due to the growing key-value (KV) caches during inference on long contexts. This implementation provides a novel on-the-fly KV cache compression module that:

1. Leverages the model's own attention weights to identify and prune low-information tokens
2. Clusters and summarizes retained KV pairs into a low-rank representation via an online k-means procedure
3. Fine-tunes models with a distillation objective to align compressed inference with full-cache baseline

The code includes our proposed method as well as implementations of several baselines for comparison:
- Full KV cache (baseline)
- ZACK: Zero-Overhead LLM Inference Acceleration via Dimensionality Compression
- DynamicKV: Task-Aware Adaptive KV Cache Compression
- RazorAttention: Efficient KV Cache Compression Through Retrieval Heads
- UNComp: Uncertainty-Aware Long-Context Compressor

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
.
├── baselines/               # Baseline implementations
│   ├── dynamic_kv.py        # DynamicKV compression
│   ├── razor_attention.py   # RazorAttention compression
│   ├── uncomp.py            # UNComp compression
│   ├── zack.py              # ZACK compression
│   └── __init__.py
├── data/                    # Data loading and preprocessing
│   ├── dataset_loader.py    # Dataset utilities
│   └── __init__.py
├── logs/                    # Log files
├── models/                  # Model implementations
│   ├── kv_cache_compressor.py  # Core compression module
│   ├── transformer_with_compression.py  # Transformer with KV cache compression
│   └── __init__.py
├── utils/                   # Utility functions
│   ├── metrics.py           # Evaluation metrics
│   ├── visualization.py     # Visualization utilities
│   └── __init__.py
├── results/                 # Results and visualizations
├── evaluate.py              # Evaluation script
├── requirements.txt         # Dependencies
├── run_experiments.py       # Main experiment runner
├── train.py                 # Training script
└── README.md                # This file
```

## Usage

### Running All Experiments

The easiest way to run all experiments is to use the `run_experiments.py` script:

```bash
python run_experiments.py \
  --model_name_or_path gpt2 \
  --output_dir results \
  --dataset_name wikitext \
  --dataset_config wikitext-103-v1 \
  --max_length 2048 \
  --sample_size 10 \
  --run_evaluation \
  --run_ablations \
  --fp16
```

This will:
1. Run evaluation on all compression methods
2. Run ablation studies to analyze parameter sensitivity
3. Generate visualizations and result tables

### Training with Compression

To train a model with KV cache compression and distillation:

```bash
python train.py \
  --model_name_or_path gpt2 \
  --dataset_name wikitext/wikitext-103-v1 \
  --output_dir results/training \
  --max_length 2048 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --warmup_steps 500 \
  --evaluate_during_training \
  --max_cache_size 1024 \
  --num_clusters 256 \
  --pruning_interval 512 \
  --lookback_window 256 \
  --use_compression \
  --fp16
```

### Evaluation Only

To evaluate different compression methods on language modeling:

```bash
python evaluate.py \
  --model_name_or_path gpt2 \
  --dataset_name wikitext/wikitext-103-v1 \
  --output_dir results/evaluation \
  --max_length 2048 \
  --batch_size 1 \
  --methods full ours zack dynamic_kv razor uncomp \
  --sequence_lengths 512 1024 2048 4096 8192 \
  --max_cache_size 1024 \
  --num_clusters 256 \
  --pruning_interval 512 \
  --lookback_window 256 \
  --run_ablations \
  --fp16
```

To include summarization evaluation:

```bash
python evaluate.py \
  --model_name_or_path gpt2 \
  --dataset_name wikitext/wikitext-103-v1 \
  --output_dir results/evaluation \
  --max_length 2048 \
  --batch_size 1 \
  --methods full ours zack dynamic_kv razor uncomp \
  --summarization \
  --summarization_dataset cnn_dailymail \
  --fp16
```

## Compression Parameters

The main parameters for our KV cache compression method are:

- `max_cache_size`: Maximum number of KV pairs to retain after pruning (B in the paper)
- `num_clusters`: Number of cluster centroids for low-rank summarization (K in the paper)
- `pruning_interval`: Interval (in tokens) between pruning operations (P in the paper)
- `lookback_window`: Number of recent positions to consider for importance scoring (Δ in the paper)
- `kmeans_learning_rate`: Learning rate for online k-means updates (η in the paper)
- `temperature`: Temperature for distillation loss (T in the paper)
- `distillation_weight`: Weight of distillation loss (λ in the paper)

## Results

After running the experiments, results will be saved in the specified output directory:

- Visualizations in `results/visualizations/`
- Evaluation metrics in `results/evaluation_results.json`
- Analysis and conclusions in `results/results.md`

## Citation

If you use this code in your research, please cite our paper:

```
@article{...,
  title={Adaptive Attention-Guided KV Cache Compression for Long-Context Sub-Quadratic Inference},
  author={...},
  journal={...},
  year={2025}
}
```

## License

[MIT License](LICENSE)
# Adaptive Attention-Guided KV Cache Compression

This folder contains results from experiments on KV cache compression for efficient long-context inference in transformer models.

## Overview

Transformer-based language models face quadratic memory and compute costs due to growing key-value (KV) caches during inference on long contexts. Our approach implements an on-the-fly KV cache compression module that leverages the model's own attention weights to identify and prune low-importance tokens, followed by clustering and summarizing retained KV pairs into a low-rank representation.

## Experiment Results

The `results.md` file contains detailed analysis and visualizations of our experimental findings, including:

- Latency scaling with sequence length
- Throughput comparison
- Memory usage 
- Performance trade-offs

## Key Findings

1. **Improved Scaling**: Our approach achieves near-linear latency scaling compared to the quadratic scaling of the baseline full KV cache method.
2. **Significant Speedup**: With 75% compression, our method achieves a 5.3x speedup over the baseline for long sequences.
3. **Memory Efficiency**: Our method reduces memory usage by 65.8% compared to the full KV cache baseline.

## Visualizations

- `latency_vs_sequence_length.png`: Shows how per-token latency scales with sequence length
- `throughput_comparison.png`: Compares throughput across different methods
- `memory_comparison.png`: Compares peak memory usage
- `memory_scaling.png`: Shows how memory requirements scale with sequence length
- `tradeoff_analysis.png`: Visualizes the performance trade-offs between throughput, memory, and latency

## Implementation

The implementation of our method, including detailed code for token importance scoring, pruning strategy, online clustering for KV pairs, and distillation loss for fine-tuning, can be found in the `claude_code/` directory.
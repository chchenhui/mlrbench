# Adaptive Token-Relevance Sparse KV-Cache (ATSKV)

This repository contains the implementation of Adaptive Token-Relevance Sparse KV-Cache (ATSKV), a novel approach for efficient long context understanding in Large Language Models. ATSKV dynamically predicts token relevance and selectively retains the most important information in the KV cache, significantly reducing memory requirements while maintaining model performance.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Running Tests](#running-tests)
  - [Running Experiments](#running-experiments)
  - [Using a Custom Model](#using-a-custom-model)
- [Method Details](#method-details)
  - [Token Relevance Prediction](#token-relevance-prediction)
  - [Adaptive Sparsity Management](#adaptive-sparsity-management)
  - [External Memory Integration](#external-memory-integration)
- [Baselines](#baselines)
- [Benchmarks](#benchmarks)
- [Results](#results)
- [License](#license)

## Overview

As Large Language Models (LLMs) process increasingly longer contexts, the memory requirements for storing the key-value (KV) cache grow quadratically with sequence length, creating a significant bottleneck. ATSKV addresses this challenge by:

1. Dynamically predicting the relevance of each token's KV representation
2. Selectively retaining only the most important tokens in the KV cache
3. Adaptively adjusting sparsity patterns during inference
4. Offloading less relevant tokens to external memory when needed

This approach enables processing of significantly longer contexts with reduced memory requirements, while maintaining model performance within 1-2% of the full dense cache baseline.

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

Required packages include:
- PyTorch
- Transformers
- NumPy
- Matplotlib
- seaborn
- tqdm
- datasets

## Project Structure

```
.
├── README.md                   # This file
├── utils.py                    # Utility functions
├── relevance_predictor.py      # Token relevance prediction module
├── sparse_kv_cache.py          # Adaptive sparse KV-cache implementation
├── baselines.py                # Baseline KV-cache implementations
├── evaluation.py               # Evaluation metrics and benchmarks
├── run_experiment.py           # Main experiment script
├── run_test.py                 # Test script
├── data/                       # Directory for datasets (created during execution)
└── outputs/                    # Output directory for results (created during execution)
```

## Usage

### Running Tests

To verify that the implementation is working correctly, run the test script:

```bash
python run_test.py
```

This will run a series of tests on the core components of the system, including:
- Token relevance predictor
- Attention statistics extractor
- Handcrafted feature extractor
- Threshold controller
- KV cache factory
- Sparse KV cache
- Model integration

### Running Experiments

To run the full experiments, use the main experiment script:

```bash
python run_experiment.py [OPTIONS]
```

#### Options

- `--model_name`: Name or path of the pre-trained model to use (default: "meta-llama/Llama-2-7b-hf")
- `--max_seq_len`: Maximum sequence length (default: 4096)
- `--use_fp16`: Use FP16 precision for model weights (flag)
- `--methods`: KV cache methods to evaluate (default: full sliding_window dynamic_kv rocket_kv atskv)
- `--context_sizes`: Context sizes to evaluate (default: 512 1024 2048 4096)
- `--output_dir`: Directory to save outputs (default: ./outputs)
- `--seed`: Random seed (default: 42)
- `--use_api`: Use API-based model (e.g., OpenAI or Claude) (flag)
- `--api_model`: API model to use if `--use_api` is set (default: claude-3-7-sonnet-20250219)

#### Example

```bash
# Run experiments with all methods and context sizes
python run_experiment.py --model_name "meta-llama/Llama-2-7b-hf" --max_seq_len 4096 --use_fp16 --output_dir "./outputs"

# Run experiments with specific methods and context sizes
python run_experiment.py --methods full atskv --context_sizes 512 1024 2048 --output_dir "./outputs/custom_run"

# Run experiments with API-based model
python run_experiment.py --use_api --api_model "claude-3-7-sonnet-20250219" --output_dir "./outputs/api_run"
```

### Using a Custom Model

To use a custom pre-trained model with ATSKV:

1. Ensure the model is compatible with the Hugging Face Transformers library
2. Specify the model using the `--model_name` option with the local path or Hugging Face model ID
3. Adjust the `--max_seq_len` parameter to match the model's capabilities

```bash
python run_experiment.py --model_name "/path/to/custom/model" --max_seq_len 2048
```

## Method Details

### Token Relevance Prediction

The token relevance predictor is a lightweight neural network that runs alongside the main model, taking as input:
- Token hidden state representations
- Attention patterns
- Handcrafted features

It outputs a relevance score for each token at each layer, indicating how important the token's KV representation is for understanding the context.

The formula for computing the relevance score is:

```
r_i^l = σ(w_r^T · [h_i^l; a_i^l; f_i^l] + b_r)
```

where:
- h_i^l is the hidden state representation of token i at layer l
- a_i^l is a feature vector derived from attention patterns
- f_i^l is a set of handcrafted features
- w_r and b_r are learnable parameters
- σ is the sigmoid activation function

### Adaptive Sparsity Management

Based on the predicted relevance scores, ATSKV employs a dynamic thresholding approach to determine which tokens' KV pairs to retain:

```
M_i^l = 1[r_i^l > τ^l(t)]
```

where M_i^l is a binary mask indicating whether to retain the KV pair, and τ^l(t) is a layer-specific threshold that varies with decoding step t.

The threshold is computed adaptively based on memory usage and target sparsity:

```
τ^l(t) = β^l · quantile({r_i^l}, q^l(t))
```

where β^l is a layer-specific scaling factor, and q^l(t) determines the quantile threshold.

### External Memory Integration

For extremely long contexts, ATSKV integrates an external memory system that stores KV pairs for tokens with lower relevance scores. This allows offloading less important information to slower but larger memory while keeping the most relevant information in the GPU-accessible KV cache.

The system implements a two-tier storage:
1. **Active Cache**: Contains KV pairs for tokens with high relevance scores (GPU memory)
2. **Passive Store**: Contains KV pairs for tokens with lower relevance scores (CPU memory)

## Baselines

We compare ATSKV against several baseline methods:

1. **Full KV Cache**: Standard full KV cache without compression (baseline)
2. **Sliding Window**: KV cache with a sliding window approach that keeps only the most recent tokens
3. **DynamicKV**: Layer-wise token retention based on task-specific patterns
4. **RocketKV**: Two-stage KV cache compression with coarse-grain eviction and fine-grain sparsification

## Benchmarks

We evaluate on multiple benchmarks for long-context understanding:

1. **LongBench**: A comprehensive benchmark for evaluating LLMs on long context understanding tasks
2. **ZeroSCROLLS**: A zero-shot benchmark for long text understanding
3. **Synthetic**: A controlled benchmark with key information placed at strategic positions

## Results

After running the experiments, you can find the results in the specified output directory:

- **Figures**: Various plots showing memory usage, inference time, throughput, and accuracy
- **Tables**: Summary tables comparing different methods across benchmarks
- **Report**: A comprehensive report (results.md) analyzing the results

## License

This project is licensed under the MIT License - see the LICENSE file for details.
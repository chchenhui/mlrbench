# Dynamic Sparse Retrieval-Augmented Sub-Quadratic Models

This repository contains the implementation of "Dynamic Sparse Retrieval-Augmented Sub-Quadratic Models for Efficient Long Context Adaptation", a novel approach for efficient processing of long contexts in foundation models.

## Overview

This research addresses the challenge of enabling foundation models to efficiently process long contextual information while maintaining inference efficiency. The approach integrates three key components:

1. **Dynamic Sparse Retriever (DSR)**: A lightweight retriever module that selectively fetches only the most relevant context tokens for a given query, minimizing redundant prefill operations.

2. **Sub-Quadratic Sparse Attention (SQA)**: An attention mechanism that processes only the retrieved tokens, reducing the computational complexity from quadratic to sub-quadratic.

3. **Rotating Compressive KV Cache (RCKV)**: A memory-efficient mechanism that maintains fixed-size latent representations of historical context using low-rank projections, preventing unbounded memory growth during extended sequences.

These components are co-optimized end-to-end with a Hybrid Optimization Framework (HOF) that balances task accuracy and computational efficiency.

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd [repository-directory]

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
claude_code/
├── models/
│   ├── dsr.py                  # Dynamic Sparse Retriever implementation
│   ├── sqa.py                  # Sub-Quadratic Sparse Attention implementation
│   ├── rckv.py                 # Rotating Compressive KV Cache implementation
│   ├── hof.py                  # Hybrid Optimization Framework implementation
│   └── baselines.py            # Baseline models for comparison
├── datasets/
│   └── long_context.py         # Dataset loading and preprocessing utilities
├── utils/
│   ├── evaluation.py           # Evaluation metrics and utilities
│   └── visualization.py        # Visualization utilities
├── main.py                     # Main script for running experiments
├── README.md                   # This file
└── results/                    # Directory for storing experimental results
```

## Usage

### Running the experiments

To run the full experiment with all baselines and ablation studies:

```bash
python main.py --model dsrsq --dataset nq --num_epochs 10 --ablation
```

### Command-line Arguments

- `--model`: Model architecture to use (`dsrsq`, `standard`, `rag`, `attention_rag`, `gca`, `razor`, `pyramid`)
- `--dataset`: Dataset to use (`nq`, `eli5`, `cnn_dm`, `github`, `s2orc`)
- `--num_epochs`: Number of epochs for training
- `--ablation`: Run ablation studies
- `--output_dir`: Directory to save results (default: 'results')
- `--seed`: Random seed for reproducibility
- `--device`: Device to run experiments on ('cuda' or 'cpu')
- `--debug`: Enable debug mode (smaller datasets, fewer iterations)

For more command-line options, see the `parse_args` function in `main.py`.

## Model Components

### Dynamic Sparse Retriever (DSR)

The DSR module selectively identifies and retrieves the most relevant tokens from a context window based on the current query. It uses a bi-encoder architecture with reduced-dimension projections to compute relevance scores efficiently. The token selection budget is dynamically adjusted based on query complexity.

### Sub-Quadratic Sparse Attention (SQA)

The SQA module implements a sparse attention mechanism that operates only on tokens selected by the DSR. It further reduces complexity by using cluster-based attention sparsification, which groups similar key-value pairs and computes attention only with cluster representatives.

### Rotating Compressive KV Cache (RCKV)

The RCKV module maintains a fixed-size representation of historical context using low-rank projections. It employs a rotating buffer mechanism with importance-weighted replacement to ensure that the most valuable information is retained over time.

### Hybrid Optimization Framework (HOF)

The HOF integrates all components and provides an end-to-end training approach with a multi-objective loss function that balances task performance, retrieval quality, compression efficiency, and computational cost.

## Baselines

The implementation includes several baseline models for comparison:

1. **Standard Transformer**: A traditional transformer model with full attention
2. **Traditional RAG**: Retrieval-Augmented Generation with naive document concatenation
3. **AttentionRAG**: Attention-guided context pruning in Retrieval-Augmented Generation
4. **GCA**: Grouped Cross Attention for efficient long-context language modeling
5. **RazorAttention**: Efficient KV cache compression through retrieval heads
6. **PyramidKV**: Dynamic KV cache compression based on pyramidal information funneling

## Datasets

The experiments use the following datasets:

1. **Natural Questions-Long**: For evaluating long-context question answering
2. **ELI5**: For evaluating explanatory question answering
3. **CNN/DailyMail**: For evaluating streaming news analysis
4. **GitHub Code**: For evaluating code understanding
5. **S2ORC**: For evaluating scientific literature processing

## Evaluation Metrics

The models are evaluated using several metrics:

1. **Task Performance Metrics**:
   - ROUGE-L and BLEU for generation quality
   - Exact Match and F1 scores for question answering

2. **Efficiency Metrics**:
   - Throughput (tokens/second)
   - Memory usage (peak and average)
   - Token efficiency (ratio of processed to selected tokens)
   - Latency (time to first token and inter-token delay)

3. **Adaptation Metrics**:
   - Information retention over time
   - Temporal consistency in streaming scenarios
   - Adaptation speed to new contexts

## Results

The experimental results are presented in the `results.md` file in the `results` directory. This file includes:

- Task performance comparisons
- Efficiency metrics
- Adaptation capabilities
- Ablation studies
- Visualizations of key findings

## License

[License information]

## Citation

If you use this code in your research, please cite:

```
[Citation information]
```
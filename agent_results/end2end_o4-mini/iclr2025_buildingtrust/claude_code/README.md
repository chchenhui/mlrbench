# Cluster-Driven Certified Unlearning for LLMs

This repository contains the implementation of the Cluster-Driven Certified Unlearning method for Large Language Models (LLMs) as described in the paper "Cluster-Driven Certified Unlearning for Large Language Models".

## Overview

The Cluster-Driven Certified Unlearning method is a novel approach for efficiently removing specific information from LLMs while maintaining their overall utility. The key components of this approach include:

1. **Representation Clustering**: Partitioning a pretrained LLM's latent representation space into semantically coherent clusters via hierarchical spectral clustering.

2. **Influence Score Approximation**: Identifying which clusters encode the concepts or examples to be removed using efficient influence-score approximations.

3. **Targeted Low-Rank Gradient Surgery**: Applying surgical updates within the affected subspaces to erase memorized data without retraining the full model.

4. **Fisher Information Certification**: Providing statistical guarantees that the unlearning operation does not introduce unexpected shifts, ensuring auditors that no residual information remains.

This method significantly reduces computational cost compared to full fine-tuning, while maintaining high model utility and providing formal certification of unlearning.

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.18+
- NumPy
- scikit-learn
- matplotlib
- seaborn
- tqdm

### Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/cluster-driven-unlearning.git
cd cluster-driven-unlearning
pip install -r requirements.txt
```

## Usage

### Quick Start

Run a simplified experiment with default parameters:

```bash
python run_simplified_experiment.py
```

This will run the experiment with a small GPT-2 model on a subset of the WebText dataset, comparing the Cluster-Driven Certified Unlearning method with baseline methods.

### Full Experiments

For more comprehensive experiments with larger models and datasets:

```bash
python run_experiments.py --model_name gpt2-medium --method all --run_sequential --run_size_impact
```

### Configuration Options

#### Model Arguments:
- `--model_name`: Model to use (`gpt2` or `gpt2-medium`)
- `--load_model_path`: Optional path to a pre-trained model

#### Data Arguments:
- `--dataset`: Dataset to use (`webtext`, `medical`, `legal`, or `code`)
- `--max_length`: Maximum sequence length for tokenization
- `--stride`: Stride for tokenization window

#### Unlearning Arguments:
- `--method`: Unlearning method to use (`cluster_driven`, `relearn`, `unlearn_what_you_want`, `code_unlearn`, `undial`, `o3_framework`, or `all`)
- `--num_clusters`: Number of clusters for Cluster-Driven method
- `--n_deletion_sets`: Number of deletion sets to create
- `--deletion_set_size`: Size of each deletion set

#### Training Arguments:
- `--batch_size`: Batch size for training and evaluation
- `--learning_rate`: Learning rate
- `--num_epochs`: Number of epochs
- `--device`: Device to use (`cuda` or `cpu`)

#### Experiment Arguments:
- `--run_sequential`: Run sequential unlearning experiment
- `--run_size_impact`: Run deletion set size impact experiment
- `--seed`: Random seed
- `--output_dir`: Directory to save results

### Examples

Run only the Cluster-Driven method on GPT-2:

```bash
python run_experiments.py --model_name gpt2 --method cluster_driven
```

Run all methods on GPT-2 medium with a larger deletion set:

```bash
python run_experiments.py --model_name gpt2-medium --method all --deletion_set_size 500
```

Run sequential unlearning experiment with the Cluster-Driven method:

```bash
python run_experiments.py --method cluster_driven --run_sequential
```

## Project Structure

- `models/`: Implementation of the Cluster-Driven Certified Unlearning method
  - `cluster_unlearning.py`: Main implementation
  - `spectral_clustering.py`: Implementation of hierarchical spectral clustering
  - `influence_scores.py`: Implementation of influence score approximation
  - `gradient_surgery.py`: Implementation of targeted low-rank gradient surgery
  - `fisher_certification.py`: Implementation of Fisher information certification

- `baselines/`: Implementation of baseline unlearning methods
  - `relearn.py`: ReLearn method
  - `unlearn_what_you_want.py`: Unlearn What You Want method
  - `code_unlearn.py`: CodeUnlearn method
  - `undial.py`: UNDIAL method
  - `o3_framework.py`: O3 Framework method

- `data/`: Data loading and processing utilities
  - `data_utils.py`: Utilities for loading datasets and creating deletion sets

- `evaluation/`: Evaluation metrics for unlearning
  - `metrics.py`: Implementation of Knowledge Forgetting Rate (KFR), Knowledge Retention Rate (KRR), etc.

- `visualization/`: Utilities for visualizing results
  - `visualize.py`: Functions for creating visualizations of the results

- `utils/`: General utilities
  - `experiment_utils.py`: Utilities for running experiments

- `run_experiments.py`: Main script for running experiments
- `run_simplified_experiment.py`: Simplified version for quick experimentation

## Results

After running an experiment, results will be saved in the specified output directory (default: `./results`). The results include:

- `results.json`: Raw JSON results including all metrics
- `results.md`: Markdown summary of the results with analysis
- Visualizations:
  - Perplexity comparison
  - Knowledge retention vs. forgetting
  - Computational efficiency
  - Sequential unlearning performance (if applicable)
  - Deletion set size impact (if applicable)

## Citation

If you use this code in your research, please cite our paper:

```
@article{author2025cluster,
  title={Cluster-Driven Certified Unlearning for Large Language Models},
  author={Author, A. and Author, B.},
  journal={ArXiv},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
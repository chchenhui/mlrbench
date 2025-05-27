# RAG-Informed Dynamic Data Valuation for Fair Marketplaces

This repository contains the implementation and evaluation of a dynamic data valuation framework for retrieval-augmented generation (RAG) systems, designed to ensure fair compensation in data marketplaces.

## Overview

Current data marketplaces struggle to fairly compensate data contributors for Foundation Models, especially as Retrieval-Augmented Generation (RAG) makes the utility of specific data chunks highly dynamic and context-dependent. This project implements a dynamic data valuation framework where prices for data contributions are continuously updated based on their attributed impact within RAG systems.

The framework consists of:

1. **Attribution Mechanisms**: Lightweight techniques to trace RAG outputs back to specific retrieved data chunks.
2. **Contribution Quantification**: Methods to measure how data chunks contribute to answer quality and task success.
3. **Dynamic Pricing**: A mechanism that updates data valuations based on attribution scores, retrieval frequency, and user feedback.

## Repository Structure

```
claude_code/
├── data/                  # Data storage directory
├── models/               
│   ├── rag_system.py      # RAG system implementation with attribution
│   └── data_valuation.py  # Data valuation framework and baselines
├── utils/
│   ├── data_utils.py      # Data processing utilities
│   └── visualization.py   # Visualization and evaluation utilities
├── results/               # Experimental results and visualizations
├── prepare_data.py        # Data preparation script
├── run_experiments.py     # Main experiment runner
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd claude_code
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Preparing Data

```bash
python prepare_data.py --data_type synthetic --num_samples 100 --output_dir data
```

Options:
- `--data_type`: Type of dataset to prepare (`wiki_qa` or `synthetic`)
- `--num_samples`: Number of samples (QA pairs) to prepare
- `--max_chunk_length`: Maximum length of a data chunk in tokens
- `--seed`: Random seed for reproducibility
- `--output_dir`: Directory to save the prepared datasets

### Running Experiments

```bash
python run_experiments.py --data_dir data --output_dir results --num_iterations 100
```

Options:
- `--data_dir`: Directory containing the prepared datasets
- `--output_dir`: Directory to save the results
- `--num_iterations`: Number of simulation iterations
- `--retriever_type`: Type of retriever to use (`bm25` or `dense`)
- `--attribution_methods`: Attribution methods to use (comma-separated list of `attention`, `perturbation`)
- `--seed`: Random seed for reproducibility

### Viewing Results

The experiment results are saved in the `results/` directory, including:
- Metrics CSV files
- Visualizations of price evolution, attribution, and performance
- Results markdown file with analysis and conclusions

## Experimental Setup

The experiments compare our proposed dynamic valuation method with several baselines:

1. **Static Pricing**: Fixed price based on chunk size
2. **Popularity-based Pricing**: Price based on retrieval frequency
3. **Data Shapley (Benchmark)**: Simplified version of the Data Shapley valuation method

We evaluate these methods on:
- Correlation between price and data quality
- Fairness of reward distribution (Gini coefficient)
- Price stability and dynamics
- Impact on RAG system performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{rag_informed_valuation,
  title={RAG-Informed Dynamic Data Valuation for Fair Marketplaces},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## Acknowledgements

This research was conducted as part of the Data Problems in Foundation Models (DATA-FM) workshop.
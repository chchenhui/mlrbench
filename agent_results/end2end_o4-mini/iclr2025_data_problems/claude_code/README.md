# Gradient-Informed Fingerprinting (GIF)

This repository contains the implementation of Gradient-Informed Fingerprinting (GIF), a method for efficient and scalable training data attribution in foundation models.

## Overview

Foundation models (FMs) such as large language models and multimodal transformers are trained on massive datasets, making it challenging to trace model outputs back to their originating training examples. GIF addresses this challenge with a two-stage approach:

1. **Fingerprint Construction & Indexing:** During training, each data sample is assigned a lightweight fingerprint by combining its static embedding with a gradient-based signature extracted from a small probe network. These fingerprints are indexed in an approximate nearest-neighbor database.

2. **Influence-Based Refinement:** At inference, an output's fingerprint is computed and matched against the ANN index to retrieve top-k candidates, which are then refined using a fast approximation of influence functions.

This approach enables sub-second attribution queries over datasets of millions of samples, with high precision and recall.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended but not required)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/gif-attribution.git
cd gif-attribution
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Configuration

The experiment parameters are controlled via JSON configuration files:

- `config.json`: Full experiment configuration
- `config_small.json`: Configuration for smaller experiments
- `config_simplified.json`: Configuration for simplified experiments with synthetic data

### Running Experiments

The primary entry point is the `run.py` script, which offers different modes:

```bash
# Run simplified experiment with synthetic data (fast)
python run.py --mode simplified

# Run full-scale experiment with real data
python run.py --mode full

# Use a custom configuration file
python run.py --mode full --config custom_config.json

# Specify output directory
python run.py --mode simplified --output-dir custom_results
```

For more options:

```bash
python run.py --help
```

### Running Individual Components

You can also run individual components:

- Simplified experiment:
  ```bash
  python run_simplified_experiment.py --config config_simplified.json
  ```

- Full experiment:
  ```bash
  python run_experiments.py --config config.json
  ```

## Project Structure

```
.
├── data/               # Data storage directory
├── models/             # Trained models and indexes
├── results/            # Experiment results and visualizations
├── logs/               # Log files
├── config.json         # Full experiment configuration
├── config_small.json   # Configuration for smaller experiments
├── config_simplified.json  # Configuration for simplified experiments
├── run.py              # Main entry point script
├── run_experiments.py  # Full experiment implementation
├── run_simplified_experiment.py  # Simplified experiment implementation
└── data/               # Data-related modules
    ├── data_loader.py   # Dataset loading and preprocessing
    └── embedding.py     # Embedding and clustering utilities
└── models/             # Model implementations
    ├── probe.py         # Probe network for fingerprint generation
    ├── indexing.py      # ANN indexing with FAISS
    ├── influence.py     # Influence function approximation
    └── baselines.py     # Baseline methods (TRACE, TRAK)
└── utils/              # Utility modules
    ├── metrics.py       # Evaluation metrics
    └── visualization.py # Visualization utilities
```

## Experiment Results

After running experiments, results are saved to the `results/` directory (or a custom output directory if specified). The results include:

- `results.md`: Summary report of the experiment
- Various plots and visualizations
- Performance metrics (precision, recall, MRR)
- Latency measurements

## Baseline Methods

For comparison, the following baseline methods are implemented:

- **TRACE:** TRansformer-based Attribution using Contrastive Embeddings (Wang et al., 2024)
- **TRAK:** Attributing Model Behavior at Scale (Park et al., 2023)

## Contributing

Contributions are welcome! Please open an issue or pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{gif2024,
  title={Gradient-Informed Fingerprinting for Scalable Foundation Model Attribution},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```
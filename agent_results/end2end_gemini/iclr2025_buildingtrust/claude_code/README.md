# Concept-Graph Explanations for LLM Reasoning Chains

This repository contains the implementation of Concept-Graph Explanations for LLM Reasoning Chains, a novel approach to visualize and understand reasoning processes in large language models (LLMs).

## Overview

Concept-Graph Explanations is a technique that extracts and visualizes an LLM's reasoning process as a structured graph of interconnected, human-understandable concepts. This helps users understand *how* an LLM arrives at conclusions and improves transparency and trust, especially in tasks requiring factual accuracy or logical deduction.

The implementation includes:

1. **LLM State Extraction**: Extracts hidden states and attention weights from LLMs during text generation
2. **Concept Identification**: Maps internal states to human-understandable concepts
3. **Concept Graph Construction**: Builds a directed graph representing reasoning flow
4. **Evaluation**: Assesses the utility and faithfulness of the generated Concept-Graphs
5. **Baselines**: Implements baseline explainability methods for comparison

## Setup

### Requirements

```
Python 3.8+
PyTorch 2.0+
Transformers 4.30+
NetworkX 3.0+
Matplotlib 3.5+
scikit-learn 1.0+
openai 0.27+
pandas 1.5+
numpy 1.20+
seaborn 0.12+
tqdm 4.64+
umap-learn 0.5+
```

### Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Setup

The following datasets are used for evaluation:

1. **GSM8K**: Grade school math problems with step-by-step solutions
2. **HotpotQA**: Multi-hop question answering requiring reasoning over multiple documents
3. **StrategyQA**: Questions requiring implicit reasoning steps

Datasets will be automatically downloaded using the Hugging Face `datasets` library.

## Usage

### Run Experiments

To run experiments, use the `run_experiments.py` script:

```bash
python run_experiments.py --config configs/default_config.json
```

You can specify the following parameters in the config file or as command-line arguments:

- `--model_name`: HuggingFace model name (default: "meta-llama/Llama-3.1-8B-Instruct")
- `--device`: Device to run the model on ("cpu" or "cuda")
- `--datasets`: Comma-separated list of datasets to use (default: "gsm8k")
- `--num_samples`: Number of samples per dataset (default: 10)
- `--output_dir`: Directory to save results (default: "results")

### Analyze Results

To analyze the results and generate visualizations:

```bash
python analyze_results.py --results_dir results
```

This will generate a comprehensive analysis of the experimental results, including:

- Performance metrics comparison
- Concept graph visualizations
- Baseline method comparisons

## Code Structure

```
claude_code/
├── models/
│   ├── llm_state_extractor.py  # LLM state extraction
│   ├── concept_mapper.py       # Concept identification and mapping
│   └── concept_graph.py        # Concept graph construction
├── utils/
│   ├── logging_utils.py        # Logging utilities
│   └── data_utils.py           # Data handling utilities
├── visualization/
│   └── visualization.py        # Visualization functions
├── evaluation/
│   ├── dataset_handler.py      # Dataset handling
│   └── baselines.py            # Baseline methods
├── experiments/
│   └── experiment_runner.py    # Experiment orchestration
├── run_experiments.py          # Main experiment script
├── analyze_results.py          # Results analysis script
└── configs/
    └── default_config.json     # Default configuration
```

## Experiments

### What is being tested?

The experiments evaluate the effectiveness of Concept-Graph explanations compared to traditional explainability methods:

1. **Faithfulness**: How well the explanations reflect the model's decision-making process
2. **Interpretability**: How understandable the explanations are to humans
3. **Completeness**: How well the explanations capture key reasoning steps

### Baselines

Concept-Graphs are compared against the following baseline methods:

1. **Attention Visualization**: Visualizing attention weights
2. **Integrated Gradients**: Token-level attribution method
3. **Chain-of-Thought (CoT)**: Explicit reasoning in text form

### Metrics

The following metrics are used for evaluation:

1. **Graph Structure Metrics**: Number of nodes, edges, density, etc.
2. **Comparison with Ground Truth**: Alignment with human-annotated solutions
3. **User Preference**: Subjective ratings of clarity and helpfulness

## Results

Experiment results are stored in the `results` directory, organized by dataset and sample ID. Each sample includes:

- Generated text
- Concept graph visualization
- Baseline visualizations
- Metrics and analysis

For a comprehensive analysis, see `results/experiment_report.md`.

## License

This project is released under the MIT License.

## Citation

If you use this code or method in your research, please cite:

```
@article{concept-graph-2025,
  title={Concept-Graph Explanations for Unveiling Reasoning Chains in Large Language Models},
  author={Author},
  journal={ICLR Workshop on Building Trust in Language Models and Applications},
  year={2025}
}
```
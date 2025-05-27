# ContextBench: A Holistic, Context-Aware Benchmarking Framework

ContextBench is a novel benchmarking framework that evaluates machine learning models across multiple metrics and in different contexts. It addresses the limitations of traditional benchmarking approaches that focus on singular performance metrics by providing a comprehensive evaluation across performance, fairness, robustness, environmental impact, and interpretability dimensions.

## Features

- **Contextual Metadata Schema (CMS)**: A standardized ontology for dataset metadata
- **Multi-Metric Evaluation Suite (MES)**: Comprehensive evaluation across multiple dimensions
- **Dynamic Task Configuration Engine (DTCE)**: Context-specific test set generation and evaluation

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- TensorFlow (for MNIST dataset)
- Hugging Face Datasets (for SST-2 dataset)
- SHAP (for interpretability evaluation)
- Joblib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Project Structure

```
claude_code/
├── data/               # Dataset loading and preprocessing
├── models/             # Model training and baseline methods
├── metrics/            # Multi-metric evaluation suite
├── utils/              # Utility functions
├── run_experiments.py  # Main experiment runner script
└── README.md           # This file
```

## Supported Datasets

- **Adult Census Income**: A tabular classification dataset for income prediction
- **MNIST**: A vision classification dataset for handwritten digit recognition
- **SST-2**: A text classification dataset for sentiment analysis

## Supported Contexts

- **Healthcare**: Emphasizes accuracy on rare cases and avoids false negatives
- **Finance**: Prioritizes precision and robustness against fraud
- **Vision**: Focuses on performance across different lighting and object conditions
- **NLP**: Balances performance across text lengths and languages

## Running Experiments

To run experiments with default settings (Adult dataset, tabular domain, classification task, healthcare and finance contexts):

```bash
python run_experiments.py
```

### Command-line Arguments

- `--dataset`: Dataset to use (`adult`, `mnist`, `sst2`)
- `--domain`: Domain of the dataset (`tabular`, `vision`, `text`)
- `--task_type`: Type of task (`classification`, `regression`)
- `--contexts`: Contexts to evaluate in (e.g., `healthcare`, `finance`)
- `--output_dir`: Directory to save results (default: `../results`)
- `--use_gpu`: Use GPU for training if available
- `--random_state`: Random seed for reproducibility (default: 42)

Example with custom settings:

```bash
python run_experiments.py --dataset mnist --domain vision --contexts healthcare vision --output_dir ../custom_results --random_state 123
```

## Experiment Process

1. Dataset loading and preprocessing
2. Training baseline models
3. Standard evaluation (performance, fairness, robustness, etc.)
4. Context-specific evaluation
5. Visualization generation
6. Results reporting

## Output

The experiment generates several outputs:

- Trained model files (in `models/` directory)
- Evaluation results (in `results/` directory)
- Visualizations (in `results/visualizations/` directory)
- `results.md`: A comprehensive report with tables and figures
- `log.txt`: Detailed log of the experiment process

## Visualizations

The framework generates various visualizations to help understand model performance:

- Performance comparison across models
- Fairness comparison across sensitive attributes
- Robustness comparison (noise, shift, adversarial)
- Environmental impact (energy, carbon emissions)
- Interpretability metrics
- Radar charts for multi-metric comparison
- Trade-off plots between different metrics
- Context-specific performance profiles

## Extending the Framework

### Adding New Datasets

To add a new dataset, implement a data loader and preprocessor in the `data/data_loader.py` file.

### Adding New Models

Implement new baseline models in the `models/baselines.py` file.

### Adding New Metrics

Add new metric functions to the appropriate files in the `metrics/` directory.

### Adding New Contexts

Define new contexts in the `utils/task_config.py` file.

## Citation

If you use ContextBench in your research, please cite:

```
ContextBench: A Holistic, Context-Aware Benchmarking Framework for Responsible Machine Learning
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
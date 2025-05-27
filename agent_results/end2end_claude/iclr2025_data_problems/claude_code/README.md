# Attribution-Guided Training (AGT)

This repository contains the implementation for Attribution-Guided Training (AGT), a novel framework that embeds attribution signals during foundation model training rather than applying them post-hoc. AGT aims to improve transparency and copyright compliance in foundation models by creating models that inherently track and represent the provenance of information.

## Overview

The AGT framework consists of three key components:

1. **Dual-Objective Optimization**: Balancing conventional training loss with attribution loss
2. **Attribution Network**: A parallel network that maps model activations to source documents
3. **Attribution-aware Generation**: Mechanisms for automatic citation during inference

This implementation includes the AGT model, baseline attribution methods for comparison, evaluation metrics, and visualization tools to assess attribution quality.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.15+
- Datasets
- Matplotlib
- NetworkX
- scikit-learn
- NLTK
- tqdm

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Project Structure

- `models.py`: Implementation of AGT and baseline models
- `data_processing.py`: Data loading and preparation utilities
- `training.py`: Training functionality for AGT
- `evaluation.py`: Evaluation metrics and visualization
- `baselines.py`: Implementation of baseline attribution methods
- `factual_checker.py`: Utilities for checking attribution accuracy
- `self_verification.py`: Methods for models to verify their own attributions
- `trust_path.py`: Visualization tools for attribution pathways
- `run_experiment.py`: Main experiment runner

## Running Experiments

To run the full Attribution-Guided Training experiment, use:

```bash
python run_experiment.py --model_name distilroberta-base --batch_size 16 --num_epochs 10 --use_gpu
```

### Command-line Arguments

- `--model_name`: Base model to use (default: "distilroberta-base")
- `--max_length`: Maximum sequence length (default: 256)
- `--batch_size`: Batch size for training (default: 16)
- `--dataset_size`: Number of examples to use per dataset (default: 5000)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay (default: 0.01)
- `--num_epochs`: Number of training epochs (default: 10)
- `--early_stopping_patience`: Early stopping patience (default: 3)
- `--lambda_attr`: Weight of attribution loss (default: 0.1)
- `--dropout`: Dropout rate (default: 0.1)
- `--seed`: Random seed (default: 42)
- `--output_dir`: Output directory (default: "output")
- `--use_gpu`: Use GPU if available (flag)
- `--run_ablations`: Run ablation studies (flag)

For example, to run a smaller experiment with ablation studies:

```bash
python run_experiment.py --model_name distilroberta-base --batch_size 8 --num_epochs 5 --dataset_size 1000 --run_ablations --use_gpu
```

## Implemented Models

### Attribution-Guided Models

- **AttributionGuidedModel**: Base implementation of AGT
- **AttributionGuidedMLM**: AGT with Masked Language Modeling

### Baseline Models

- **PostHocAttributionModel**: Standard post-hoc attribution
- **DataShapleySimulator**: Simulates Data Shapley attribution
- **MinimalSubsetAttributionModel**: Attribution based on minimal interpretable subsets

## Visualization

The experiment generates various visualizations to help understand attribution quality:

- Attribution scores comparison across models
- Learning curves for training
- Ablation studies for attribution loss weight
- Architecture comparison
- Attribution threshold effects
- Computational efficiency comparison
- Trust path visualizations

## Results

After running experiments, results are saved to the specified output directory:

- Model checkpoints in the `models` subdirectory
- Evaluation results in the `results` subdirectory
- Figures in the `figures` subdirectory
- A `results.md` file with analysis and visualizations

## Example

```python
from models import AttributionGuidedMLM
from data_processing import prepare_datasets
from training import train_model
from evaluation import evaluate_model

# Prepare datasets
datasets = prepare_datasets(
    model_name="distilroberta-base",
    max_length=128,
    batch_size=16,
    dataset_size=1000
)

# Create AGT model
model = AttributionGuidedMLM(
    model_name="distilroberta-base",
    num_sources=datasets["train_dataset"].num_sources,
    attribution_type="multi_layer",
    lambda_attr=0.1
)

# Train the model
train_results = train_model(
    model=model,
    train_loader=datasets["train_loader"],
    val_loader=datasets["val_loader"],
    task_type="mlm_with_attribution",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Evaluate the model
metrics = evaluate_model(
    model=model,
    dataloader=datasets["test_loader"],
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

print(f"Test metrics: {metrics}")
```

## Citation

If you use this code in your research, please cite our paper:

```
@article{agt2025,
  title={Attribution-Guided Training: Enhancing Foundation Model Transparency and Copyright Compliance Through Embedded Attribution Mechanisms},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```
# LLM-Assisted Spuriousity Scout (LASS)

This repository contains the implementation of LASS, a framework for LLM-Driven Discovery and Mitigation of Unknown Spurious Correlations in machine learning models.

## Overview

LASS is a framework that leverages Large Language Models (LLMs) to automatically identify and mitigate unknown spurious correlations in machine learning models. The key components include:

1. **Error-Driven Hypothesis Generation**: Identify clusters of confident model errors and use LLMs to generate hypotheses about potential spurious correlations.
2. **Hypothesis Validation and Refinement**: Validate and refine the generated hypotheses, focusing on those that are most plausible and actionable.
3. **LLM-Guided Robustification**: Apply targeted interventions based on the validated hypotheses to improve model robustness.

The framework is designed to reduce the need for manual annotation of spurious correlations, making robust model development more accessible.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

The experiments use the Waterbirds dataset, which will be automatically downloaded when running the experiments. The dataset is a variant of the CUB dataset with birds superimposed on land and water backgrounds, where landbirds are spuriously correlated with land backgrounds and waterbirds with water backgrounds.

## Running Experiments

### Quick Start

To run the full experiment pipeline with default settings:

```bash
python run.py --gpu
```

This will:
1. Download and prepare the dataset (if needed)
2. Train and evaluate baseline models (ERM, Group-DRO, Balanced Sampling)
3. Run the LASS pipeline
4. Generate visualizations and results summary

### Options

- `--data_dir`: Directory for storing datasets (default: './data')
- `--output_dir`: Directory for saving results (default: './output')
- `--dataset`: Dataset to use (default: 'waterbirds', other options: 'celeba', 'civilcomments')
- `--llm_provider`: LLM provider for hypothesis generation (default: 'anthropic', other option: 'openai')
- `--skip_baselines`: Skip running baseline models
- `--fast_run`: Run with reduced epochs for faster execution
- `--seed`: Random seed (default: 42)
- `--log_file`: Path to log file (default: './log.txt')
- `--gpu`: Use GPU for training (if available)

Example for a fast run with only LASS (no baselines):

```bash
python run.py --gpu --skip_baselines --fast_run
```

### Advanced Configuration

For more fine-grained control over the experiment parameters, you can directly use the `run_experiments.py` script:

```bash
python run_experiments.py --run_lass --run_baselines --cuda --data_dir ./data --output_dir ./output --dataset waterbirds --num_epochs 30 --batch_size 32 --lr 1e-4 --intervention reweighting --llm_provider anthropic
```

Run with `--help` to see all available options:

```bash
python run_experiments.py --help
```

## Experiment Phases

The LASS framework operates in four main phases:

1. **Initial Task Model Training**: Train an initial model using standard Empirical Risk Minimization (ERM).
2. **Error Analysis and Clustering**: Identify clusters of confident model errors and visualize them.
3. **LLM Hypothesis Generation**: Use LLMs to generate hypotheses about potential spurious correlations.
4. **LLM-Guided Robustification**: Apply targeted interventions based on the validated hypotheses.

## Results

After running the experiments, results will be saved in the following locations:

- `output/`: Raw experiment outputs, including trained models, error clusters, and hypotheses
- `results/`: Final results, including visualizations and a detailed results summary (`results.md`)

The results include:
- Comparison of model performance (overall and worst-group accuracy)
- Learning curves for all models
- Visualizations of error clusters
- Group-wise performance analysis
- List of LLM-generated hypotheses about spurious correlations

## LLM API Keys

To use OpenAI or Anthropic models for hypothesis generation, you'll need to set up your API keys as environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY=your_openai_api_key

# For Anthropic
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Citation

If you use LASS in your research, please cite our paper:

```
@article{lass2025,
  title={LLM-Driven Discovery and Mitigation of Unknown Spurious Correlations},
  author={Authors},
  journal={Conference/Journal},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
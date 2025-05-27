# Co-Evolutionary Value Alignment (CEVA) Framework Experiments

This repository contains the implementation and experiments for the Co-Evolutionary Value Alignment (CEVA) framework, a novel approach to bidirectional human-AI alignment that explicitly models and facilitates the reciprocal relationship between evolving human values and developing AI capabilities.

## Overview

The CEVA framework addresses a critical gap in bidirectional alignment: how human values and AI capabilities can co-evolve in a mutually beneficial way. It combines:

1. **Value Evolution Modeling**: Formal models that capture how human values evolve through interaction with AI systems and society
2. **Multi-level Value Representation**: Different adaptation rates for core safety values, cultural values, and personal preferences
3. **Bidirectional Feedback Mechanisms**: Explicit mechanisms for transparent communication about value changes
4. **Evaluation Framework**: Metrics for assessing alignment quality across different timescales and adaptation scenarios

## Repository Structure

```
claude_code/
├── config.py                 # Experiment configuration
├── value_evolution.py        # Value evolution modeling
├── alignment_models.py       # Alignment model implementations
├── evaluation.py             # Evaluation metrics and scenarios
├── visualization.py          # Data visualization utilities
├── run_experiment.py         # Main experiment runner
└── README.md                 # This file
```

## Requirements

This code requires the following Python packages:

- numpy
- pandas
- matplotlib
- seaborn

You can install them with:

```bash
pip install numpy pandas matplotlib seaborn
```

## Running the Experiments

To run the experiments with default settings:

```bash
python run_experiment.py
```

This will:
1. Simulate human value evolution across different scenarios
2. Run the baseline and CEVA alignment models
3. Evaluate performance using various metrics
4. Generate visualizations and result tables
5. Save all outputs to the `results/` directory

### Command Line Options

The experiment runner supports the following command-line options:

- `--seed INTEGER`: Set the random seed for reproducibility (default: 42)
- `--output-dir PATH`: Specify the output directory for results (default: parent directory)

Example:

```bash
python run_experiment.py --seed 123 --output-dir /path/to/output
```

## Experiment Design

The experiments evaluate the CEVA framework against baseline alignment methods across various scenarios:

### Models

1. **Static Alignment**: Traditional static alignment model with no adaptation
2. **Adaptive Alignment**: Simple adaptive alignment model with uniform adaptation
3. **Basic CEVA**: CEVA model with multi-level value adaptation
4. **Full CEVA**: CEVA model with bidirectional feedback mechanisms

### Scenarios

1. **Gradual Drift**: Values change slowly over time
2. **Rapid Shift**: Sudden change in response to critical event
3. **Value Conflict**: Tension between different value levels

### Metrics

1. **Adaptation Accuracy**: How well the AI model's values match human values
2. **Adaptation Response Time**: Time to reduce value misalignment below threshold after shift
3. **Stability**: Resistance to spurious adaptation
4. **User Satisfaction**: Simulated user satisfaction with responses
5. **Agency Preservation**: How well human agency is preserved in the process

## Results

After running the experiments, the results will be saved in the `results/` directory:

- `results.json`: Raw experiment results
- `results.md`: Markdown report with analysis and visualizations
- `figures/`: Visualizations of results
- `log.txt`: Experiment execution log
- Various CSV files with tabulated results

## Customization

You can customize the experiments by modifying the `config.py` file:

- Adjust simulation parameters
- Define different scenarios
- Add or modify alignment models
- Change evaluation metrics
- Configure visualization settings

## Acknowledgments

This research builds upon the bidirectional human-AI alignment framework and draws inspiration from various papers on human-AI coevolution and value alignment.

## License

This code is provided for research purposes only.
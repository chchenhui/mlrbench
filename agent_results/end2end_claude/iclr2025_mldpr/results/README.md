# Contextual Dataset Deprecation Framework

This repository contains the implementation and experimental evaluation of the Contextual Dataset Deprecation Framework, a comprehensive solution for standardizing the process of deprecating problematic machine learning datasets while maintaining research integrity and continuity.

## Overview

The Contextual Dataset Deprecation Framework includes five key components:

1. **Tiered Warning System**: A hierarchical warning system with progressively severe labels (caution, limited use, deprecated) based on documented dataset issues.
2. **Notification System**: An automated notification mechanism to alert previous dataset users when a dataset's status changes.
3. **Context-Preserving Deprecation**: Maintained metadata and documentation with appropriate access restrictions.
4. **Alternative Recommendation System**: Required alternative dataset recommendations when deprecating widely-used benchmarks.
5. **Transparent Versioning System**: A comprehensive versioning system documenting the dataset's lifecycle.

## Repository Structure

```
├── baselines.py            # Implementation of baseline methods for comparison
├── create_visualizations.py # Script to generate visualizations from results
├── dataset_generator.py    # Tools for generating synthetic datasets
├── evaluation.py           # Metrics and analysis tools for evaluation
├── experimental_design.py  # Experimental design and simulation setup
├── framework.py            # Core implementation of the Contextual Dataset Deprecation Framework
├── README.md               # This file
└── run_experiment.py       # Main script to run the complete experiment
```

## Requirements

The code requires the following Python packages:
- numpy
- pandas
- matplotlib
- seaborn

You can install them using:

```bash
pip install numpy pandas matplotlib seaborn
```

## Running the Experiments

To run the complete experiment, which includes evaluating the framework against baseline approaches, use:

```bash
python run_experiment.py
```

Optional arguments:
- `--simulations`: Number of simulated users/groups (default: 50)
- `--output-dir`: Directory to save experiment results (default: auto-generated)

## Creating Visualizations

To create additional visualizations from the experiment results:

```bash
python create_visualizations.py
```

Optional arguments:
- `--results-file`: Path to the experiment results JSON file (default: most recent)
- `--output-dir`: Directory to save visualizations (default: 'results/figures')

## Experimental Design

The experiment compares three strategies:

1. **Control (Traditional)**: Simple removal of datasets without structured deprecation
2. **Basic Framework**: Implementation with only warning labels and basic notifications
3. **Full Framework**: Complete implementation of all components of the Contextual Dataset Deprecation Framework

We evaluate these strategies using:
- User response metrics (time to acknowledge deprecation, alternative adoption rate, etc.)
- System performance metrics (recommendation accuracy, notification success, etc.)
- Research impact metrics (citation patterns, benchmark diversity, etc.)

## Results

After running the experiment, results are saved to the 'results' directory:
- Raw data in JSON format
- Figures in the 'figures' subdirectory
- Summary report in 'results.md'

## Example Usage

Here's a simple example of how to use the framework programmatically:

```python
from framework import ContextualDeprecationFramework, DeprecationStrategy
from experimental_design import WarningLevel

# Initialize the framework with the full strategy
framework = ContextualDeprecationFramework(strategy=DeprecationStrategy.FULL)

# Apply a warning level to a dataset
framework.apply_warning_level(
    dataset_id="biased_dataset",
    warning_level=WarningLevel.CAUTION,
    issue_description="This dataset shows demographic bias that may affect model fairness.",
    evidence_links=["https://example.com/bias_analysis"],
    affected_groups=["Demographic group A"],
    recommended_alternatives=["unbiased_dataset"]
)

# Check access permission
access_granted = framework.check_access_permission(
    user_id="researcher_1",
    dataset_id="biased_dataset",
    purpose="Bias mitigation research"
)

# Get alternative recommendations
alternatives = framework.recommend_alternatives(dataset_id="biased_dataset", top_n=3)
```

## Detailed Documentation

Each module includes detailed docstrings explaining the purpose, parameters, and return values of each function. For more information about a specific component, check the corresponding Python file.

## License

This code is provided for research purposes only.

## Citation

If you use this framework in your research, please cite:

```
@article{contextual_dataset_deprecation,
  title={Contextual Dataset Deprecation: A Systematic Framework for Ethical Machine Learning Repositories},
  author={Author, A.},
  journal={Proceedings of the Future of Machine Learning Data Practices and Repositories Workshop},
  year={2025}
}
```
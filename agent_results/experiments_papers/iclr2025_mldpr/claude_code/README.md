# Benchmark Cards Experiment

This repository contains code for validating the Benchmark Cards methodology for holistic evaluation of machine learning models. The experiment tests the hypothesis that incorporating context-specific evaluation metrics leads to more appropriate model selection for different use cases compared to relying on a single metric like accuracy.

## Overview

The Benchmark Cards methodology aims to standardize contextual reporting of benchmark intentions and promote multimetric evaluation that captures fairness, robustness, efficiency, and other domain-specific concerns. This experiment implements a simulated version of the "Phase 2: Adoption Impact" validation described in the proposal, which measures whether using Benchmark Cards improves evaluation practices.

## Requirements

The experiment requires the following Python packages:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Optionally, for enhanced visualizations:
```
plotly
```

## Experiment Structure

The experiment consists of the following steps:

1. **Dataset Loading**: We load several tabular datasets from OpenML.
2. **Benchmark Card Creation**: We create a Benchmark Card for each dataset, defining intended use cases, evaluation metrics, and use case specific metric weights.
3. **Model Training**: We train multiple classification models, including Logistic Regression, Decision Trees, Random Forests, SVM, and Neural Networks.
4. **Model Evaluation**: We evaluate each model on multiple metrics, including accuracy, balanced accuracy, precision, recall, F1 score, inference time, and model complexity.
5. **User Simulation**: We simulate model selection with and without Benchmark Cards for different use cases.
6. **Results Analysis**: We analyze and visualize the results to determine if Benchmark Cards lead to different (and potentially better) model selections.

## Running the Experiment

1. **Single Dataset Experiment**:
   ```
   python main.py --dataset adult --output-dir results/adult
   ```

2. **Multiple Dataset Experiments**:
   ```
   python run_experiments.py --results-dir results
   ```

3. **Generate Benchmark Cards**:
   ```
   python generate_benchmark_cards.py --datasets adult diabetes credit-g --output-dir benchmark_cards
   ```

4. **Additional Visualizations**:
   ```
   python visualize_results.py --results-dir results --dataset adult
   ```

## Results

The experiment produces the following result files for each dataset:

- `[dataset]_benchmark_card.json`: The Benchmark Card used for evaluation
- `[dataset]_model_results.json`: Raw model evaluation results across all metrics
- `[dataset]_simulation_results.json`: Results of simulated model selection with and without Benchmark Cards
- Various PNG files with visualizations of model performance and selection differences
- `results.md`: A Markdown file summarizing the results and conclusions

For the multi-dataset experiment, an additional comparative analysis is provided:

- `comparative/comparative_analysis.md`: Analysis of results across all datasets
- `comparative/selection_differences.png`: Visualization of how often Benchmark Cards lead to different model selections
- `comparative/metric_differences.png`: Visualization of average metric differences between default and card-selected models
- `comparative/use_case_distribution.png`: Visualization of how different use cases are affected by Benchmark Cards

## Interpreting the Results

The primary hypothesis of the experiment is that using Benchmark Cards will lead to different model selections compared to using only accuracy. This would support the argument that holistic, context-aware evaluation changes which models are considered "best" for specific use cases.

Key metrics to consider:
- **Percentage of Different Selections**: The percentage of use cases where Benchmark Cards lead to a different model selection.
- **Metric Improvements**: Whether the models selected using Benchmark Cards have better performance on the metrics prioritized for each use case.
- **Trade-offs**: How the models selected using Benchmark Cards make different trade-offs between accuracy, fairness, efficiency, and other metrics.

## Extending the Experiment

To extend the experiment, you can:

1. **Add more datasets**: Modify the `DATASETS` list in `run_experiments.py`.
2. **Define new use cases**: Add new use cases and metric weights in `create_benchmark_card` in `main.py`.
3. **Add new models**: Add new model types in `train_models` in `main.py`.
4. **Add new metrics**: Define additional evaluation metrics in `evaluate_models` in `main.py`.
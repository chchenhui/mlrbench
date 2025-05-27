# Benchmark Cards Experiment Results: iris

## Dataset Information

- **Dataset**: iris
- **Features**: 4
- **Target Classes**: ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

## Model Performance

The following table shows the performance of different models on the test set:

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 Score | Roc Auc | Inference Time | Model Complexity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.9967 | 0.0008 | 4 |
| decision_tree | 0.9000 | 0.9000 | 0.9024 | 0.9000 | 0.8997 | 0.9250 | 0.0008 | 4 |
| random_forest | 0.9000 | 0.9000 | 0.9024 | 0.9000 | 0.8997 | 0.9867 | 0.0052 | 4 |
| svm | 0.9667 | 0.9667 | 0.9697 | 0.9667 | 0.9666 | 0.9967 | 0.0009 | 4 |
| mlp | 0.9667 | 0.9667 | 0.9697 | 0.9667 | 0.9666 | 0.9967 | 0.0009 | 4 |

## Model Selection Comparison

This table compares model selection with and without Benchmark Cards for different use cases:

| Use Case | Default Selection (Accuracy Only) | Benchmark Card Selection | Different? |
| --- | --- | --- | --- |
| General Performance | svm | svm | No |
| Fairness Focused | svm | logistic_regression | Yes |
| Resource Constrained | svm | svm | No |
| Interpretability Needed | svm | svm | No |
| Robustness Required | svm | logistic_regression | Yes |

**Summary**: Benchmark Cards resulted in different model selections in 2 out of 5 use cases (40.0%).

## Use Case Specific Metric Weights

The Benchmark Card defined the following weights for each use case:

### General Performance

| Metric | Weight |
| --- | --- |
| Accuracy | 0.30 |
| Balanced Accuracy | 0.20 |
| Precision | 0.20 |
| Recall | 0.20 |
| F1 Score | 0.10 |

### Fairness Focused

| Metric | Weight |
| --- | --- |
| Accuracy | 0.20 |
| Fairness Disparity | 0.50 |
| Balanced Accuracy | 0.20 |
| F1 Score | 0.10 |

### Resource Constrained

| Metric | Weight |
| --- | --- |
| Accuracy | 0.40 |
| Inference Time | 0.40 |
| Model Complexity | 0.20 |

### Interpretability Needed

| Metric | Weight |
| --- | --- |
| Accuracy | 0.30 |
| Model Complexity | 0.50 |
| Precision | 0.10 |
| Recall | 0.10 |

### Robustness Required

| Metric | Weight |
| --- | --- |
| Accuracy | 0.20 |
| Balanced Accuracy | 0.30 |
| Fairness Disparity | 0.30 |
| Precision | 0.10 |
| Recall | 0.10 |

## Conclusions

The experiment showed that using Benchmark Cards sometimes leads to different model selections compared to using accuracy as the only metric. This highlights the potential value of holistic, context-aware evaluation, especially in specific use cases where non-accuracy metrics are important.

### Key Insights

- For the **Fairness Focused** use case, the Benchmark Card selected **logistic_regression** instead of **svm** (highest accuracy).
  - Key metric differences:
    - **Accuracy**: 0.9333 vs 0.9667 (better in default model)
    - **Balanced Accuracy**: 0.9333 vs 0.9667 (better in default model)

- For the **Robustness Required** use case, the Benchmark Card selected **logistic_regression** instead of **svm** (highest accuracy).
  - Key metric differences:
    - **Accuracy**: 0.9333 vs 0.9667 (better in default model)
    - **Balanced Accuracy**: 0.9333 vs 0.9667 (better in default model)

## Limitations and Future Work

This experiment demonstrates the concept of Benchmark Cards, but has several limitations:

1. The experiment used a single dataset and a small set of models.
2. The use cases and metric weights were defined artificially rather than by domain experts.
3. The simulation doesn't fully capture the complexity of real-world model selection decisions.

Future work could address these limitations by:

1. Expanding to multiple datasets across different domains.
2. Conducting surveys with domain experts to define realistic use cases and metric weights.
3. Implementing a more sophisticated composite scoring formula that better handles trade-offs.
4. Developing an interactive tool that allows users to explore model performance with custom weights.

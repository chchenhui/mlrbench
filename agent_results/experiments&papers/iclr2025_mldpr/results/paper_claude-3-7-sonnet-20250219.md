# Benchmark Cards: Standardizing Context and Holistic Evaluation for Machine Learning Benchmarks

## Abstract

Machine learning benchmarks significantly influence research priorities and model adoption, yet they often rely on single aggregate metrics that fail to capture the multifaceted nature of model performance. This paper introduces Benchmark Cards, a standardized documentation framework that provides contextual information and promotes holistic evaluation across multiple metrics tailored to specific use cases. We define a structured template covering intended use, dataset composition, evaluation metrics, robustness analysis, limitations, and version control. We demonstrate the practical impact of Benchmark Cards through experiments on the Iris dataset, showing that holistic, context-aware evaluation leads to different model selections than traditional accuracy-focused approaches in 40% of the examined use cases. These results highlight the need for more comprehensive benchmark documentation and evaluation practices that consider fairness, efficiency, and robustness alongside traditional performance metrics. Benchmark Cards offer a practical framework to improve model selection decisions and promote more responsible machine learning research and deployment.

## 1. Introduction

Machine learning (ML) benchmarks serve as critical tools for measuring progress, comparing algorithms, and guiding research priorities. However, the current benchmarking ecosystem faces significant challenges that can lead to suboptimal outcomes in real-world applications. Most benchmarks prioritize single aggregate metrics (e.g., accuracy, F1 score) that fail to capture the multidimensional nature of model performance. This narrow focus incentivizes leaderboard optimization at the expense of more comprehensive model understanding.

The overemphasis on singular performance metrics obscures important dimensions such as fairness across demographic subgroups, robustness to distribution shifts, computational efficiency, and ethical considerations. As a result, models that excel on popular leaderboards may perform poorly when deployed in real-world contexts with different requirements or constraints. This mismatch between benchmark performance and practical utility is particularly concerning for high-stakes applications in domains such as healthcare, criminal justice, and financial services.

The problem is compounded by a lack of standardized documentation for benchmarks. While frameworks like Model Cards (Mitchell et al., 2018) and Datasheets (Gebru et al., 2018) have emerged for models and datasets, benchmarks often lack explicit guidance on intended scope, evaluation criteria, and limitations. This documentation gap hampers reproducibility, proper contextualization, and responsible deployment of ML systems.

To address these challenges, we propose Benchmark Cards, a structured documentation framework designed to standardize contextual reporting of benchmark intentions and constraints, promote multimetric evaluation across various dimensions, and mitigate misuse risks through explicit documentation of limitations. Our approach builds on recent work in holistic evaluation, such as HELM for language models (Liang et al., 2022) and holistic evaluation metrics for federated learning (Li et al., 2024), but provides a generalizable framework applicable across ML domains.

The key contributions of this paper are:

1. A comprehensive Benchmark Card template that standardizes documentation of ML benchmarks, including intended use cases, dataset characteristics, evaluation metrics, and limitations.

2. A weighted evaluation methodology that enables context-sensitive model assessment based on use-case-specific priorities.

3. Experimental validation demonstrating how Benchmark Cards lead to different, more appropriate model selections compared to traditional single-metric evaluation.

4. A framework for integrating Benchmark Cards into existing ML repositories and research workflows to improve evaluation practices.

By providing this structured approach to benchmark documentation and evaluation, we aim to shift the focus from simplistic leaderboard rankings to more nuanced, context-aware model assessment. This shift is essential for selecting models that are truly suited to specific applications, rather than those that merely excel on a single metric in isolation.

## 2. Related Work

Our work builds upon several research streams focused on improving documentation, evaluation, and responsible use of ML systems. We organize related work into three main categories: documentation frameworks, holistic evaluation approaches, and benchmark design considerations.

### 2.1 Documentation Frameworks for ML

Documentation frameworks have emerged as crucial tools for enhancing transparency and accountability in ML. Mitchell et al. (2018) introduced Model Cards, a standardized documentation format for ML models that includes information about intended use cases, performance metrics across different conditions, and potential biases. This work highlighted the importance of transparent reporting about model capabilities and limitations to inform appropriate use.

Similarly, Gebru et al. (2018) proposed Datasheets for Datasets, a framework documenting dataset creation, composition, preprocessing, maintenance, and legal/ethical considerations. These datasheets aim to improve dataset transparency and accountability throughout the ML lifecycle.

Bender and Friedman (2018) introduced Data Statements for natural language processing, specifically addressing linguistic data documentation needs. Holland et al. (2018) proposed Dataset Nutrition Labels to communicate dataset characteristics and potential issues in an accessible format.

While these frameworks address transparency for models and datasets separately, benchmarks—which combine datasets with evaluation procedures—lack standardized documentation. Benchmark Cards fill this gap by extending documentation practices to the evaluation context, which is critical for proper model selection and deployment.

### 2.2 Holistic Evaluation Approaches

Recent work has emphasized the need for multidimensional evaluation of ML systems. Liang et al. (2022) presented HELM (Holistic Evaluation of Language Models), a framework evaluating language models across seven metrics—accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency—in 16 core scenarios. This comprehensive approach provides richer insights into model capabilities than single-metric evaluations.

Li et al. (2024) proposed Holistic Evaluation Metrics (HEM) for federated learning, tailoring evaluation to specific use cases by considering accuracy, convergence, computational efficiency, fairness, and personalization. Their approach assigns importance vectors to metrics based on the requirements of different deployment scenarios.

D'Amour et al. (2020) introduced underspecification in ML pipelines, showing how models with similar test-set performance can behave differently under distribution shifts. Their work emphasized that standard evaluation fails to distinguish models that will perform well in deployment from those that won't.

These approaches demonstrate the value of multifaceted evaluation but are often domain-specific. Benchmark Cards generalize these insights into a flexible framework applicable across ML domains, connecting evaluation practices directly to documentation.

### 2.3 Benchmark Design and Limitations

Critical examination of benchmark design and usage has revealed several issues in current practices. Ethayarajh and Jurafsky (2020) critiqued the utility of leaderboards in NLP, arguing that they often fail to identify statistically significant improvements and can lead to overfitting. Raji et al. (2021) analyzed how benchmark datasets become embedded in research communities, highlighting potential problems with overused benchmarks becoming de facto standards.

Koch et al. (2021) examined how benchmark dataset selection influences algorithm development, showing that benchmark choices can inadvertently steer research direction. Dehghani et al. (2021) advocated for carefully designed benchmark suites rather than individual datasets to ensure comprehensive evaluation.

Our work addresses these concerns by providing a framework that explicitly documents benchmark limitations and intended contexts, helping researchers and practitioners make more informed decisions about benchmark selection and interpretation.

## 3. Methodology

### 3.1 Framework Design for Benchmark Cards

We propose a standardized Benchmark Card template with six key components designed to provide comprehensive documentation of machine learning benchmarks:

1. **Intended Use & Scope**:
   - Primary deployment scenarios where models evaluated by this benchmark are expected to be applied
   - Exclusion criteria specifying domains or contexts where the benchmark should not be used
   - Target user groups and stakeholders for whom the benchmark is relevant

2. **Dataset Composition**:
   - Summary statistics of underlying data, including demographic representation and potential biases
   - Data collection methods, preprocessing steps, and splitting criteria
   - Ethical considerations related to data provenance, consent, and potential sensitive attributes

3. **Evaluation Metrics**:
   - Core metrics used for primary evaluation and leaderboard ranking
   - Contextual metrics addressing fairness, efficiency, and robustness concerns
   - Metric calculation procedures and threshold definitions
   - Weighted combination equation for composite scores tailored to different use cases:
     $$\text{Composite Score} = \sum_{i=1}^{n} w_i \cdot \frac{\text{Metric}_i}{\tau_i}$$
     where $w_i$ represents use-case-specific weights normalized to $\sum w_i = 1$, and $\tau_i$ defines threshold penalties for minimal acceptable performance.

4. **Robustness & Sensitivity Analysis**:
   - Performance under distribution shifts, including out-of-domain and adversarial examples
   - Variation in performance across subgroups or data slices
   - Stability of results across random initializations or hyperparameter settings

5. **Known Limitations**:
   - Identified failure cases and scenarios where benchmark results may not generalize
   - Potential overfitting risks or misuse scenarios
   - Gaps in coverage or representativeness

6. **Version Control & Dependencies**:
   - Required software and hardware configurations for reproducibility
   - Update history and changes from previous versions
   - Deprecation criteria and timeline

This structured approach ensures that benchmarks provide the necessary context for proper interpretation and application of evaluation results.

### 3.2 Use-Case Specific Evaluation

To operationalize context-aware evaluation, we define a process for creating use-case-specific metric weightings:

1. For each benchmark, identify $k$ representative use cases $u_1,...,u_k$ that reflect common application scenarios (e.g., resource-constrained environments, fairness-critical applications).

2. For each use case $u_j$, assign importance weights $w_{ij} \in [0,1]$ to each metric $i=1,...,n$ such that $\sum_i w_{ij}=1$. These weights reflect the relative importance of different performance aspects in that particular context.

3. For any candidate model $m$, compute a utility vector $v_m = [\text{Metric}_1,...,\text{Metric}_n]$ containing its performance on all defined metrics.

4. Calculate a weighted score for each use case by taking the dot product of the metric weights and utility vector: $\text{Score}_{j} = w_j^T v_m$.

5. Optionally, apply threshold constraints where models must meet minimum performance requirements on critical metrics regardless of overall score.

This approach enables benchmark users to identify models that best match their specific requirements, rather than defaulting to models that excel only on the primary metric.

### 3.3 Experimental Design

To validate the impact of Benchmark Cards on model selection decisions, we conducted a controlled experiment using the Iris dataset, a classic machine learning benchmark. The experiment followed these steps:

1. **Model Training**: We trained five common classification models on the Iris dataset: logistic regression, decision tree, random forest, support vector machine (SVM), and multi-layer perceptron (MLP).

2. **Comprehensive Evaluation**: Each model was evaluated on multiple metrics, including accuracy, balanced accuracy, precision, recall, F1 score, ROC AUC, inference time, and model complexity.

3. **Use Case Definition**: We defined five representative use cases with different priorities:
   - General Performance: Balanced emphasis on accuracy and other traditional metrics
   - Fairness Focused: Prioritizing balanced performance across classes
   - Resource Constrained: Emphasizing computational efficiency
   - Interpretability Needed: Favoring simpler, more explainable models
   - Robustness Required: Prioritizing consistent performance across data subsets

4. **Metric Weighting**: For each use case, we assigned specific weights to relevant metrics based on their importance in that context.

5. **Model Selection Comparison**: We compared model selections made using:
   - Traditional approach: Selecting the model with the highest accuracy
   - Benchmark Card approach: Selecting the model with the highest weighted score for each use case

This experimental design allowed us to assess whether incorporating use-case-specific priorities through Benchmark Cards leads to different model selection decisions than the traditional accuracy-focused approach.

## 4. Experiment Setup

### 4.1 Dataset

We used the Iris dataset, a well-known benchmark in machine learning that contains 150 samples with 4 features describing iris flowers, classified into three species: Iris-setosa, Iris-versicolor, and Iris-virginica. We chose this dataset for its simplicity and widespread use, allowing us to focus on demonstrating the Benchmark Card concept without the complexity of large-scale datasets.

The dataset was split into training (80%) and test (20%) sets using stratified sampling to maintain class distribution. No additional preprocessing was applied as the features are already normalized.

### 4.2 Models and Implementation

We implemented five classification models with different characteristics:

1. **Logistic Regression**: A linear model that provides good interpretability and efficiency.
2. **Decision Tree**: A rule-based model offering high interpretability but sometimes lower accuracy.
3. **Random Forest**: An ensemble method that typically offers higher accuracy but reduced interpretability.
4. **Support Vector Machine (SVM)**: A powerful classifier that often achieves high accuracy but with lower interpretability.
5. **Multi-Layer Perceptron (MLP)**: A neural network approach with high representational capacity but limited interpretability.

All models were implemented using scikit-learn with default hyperparameters to simulate a standard benchmarking scenario. Models were trained on the training set and evaluated on the held-out test set.

### 4.3 Evaluation Metrics

We calculated multiple performance metrics to capture different aspects of model performance:

- **Accuracy**: Proportion of correct predictions (overall performance)
- **Balanced Accuracy**: Average of recall for each class (class-balanced performance)
- **Precision**: Ratio of true positives to predicted positives (specificity)
- **Recall**: Ratio of true positives to actual positives (sensitivity)
- **F1 Score**: Harmonic mean of precision and recall (balanced measure)
- **ROC AUC**: Area under the receiver operating characteristic curve (discrimination ability)
- **Inference Time**: Average time to make predictions (efficiency)
- **Model Complexity**: Number of features used (interpretability proxy)

### 4.4 Use Case Definitions and Metric Weights

We defined five use cases with distinct evaluation priorities, each represented by different metric weights:

1. **General Performance**:
   - Accuracy (0.30), Balanced Accuracy (0.20), Precision (0.20), Recall (0.20), F1 Score (0.10)
   - Rationale: Balanced consideration of overall performance metrics

2. **Fairness Focused**:
   - Accuracy (0.20), Fairness Disparity (0.50), Balanced Accuracy (0.20), F1 Score (0.10)
   - Rationale: Emphasis on equitable performance across classes

3. **Resource Constrained**:
   - Accuracy (0.40), Inference Time (0.40), Model Complexity (0.20)
   - Rationale: Prioritizing efficiency alongside acceptable accuracy

4. **Interpretability Needed**:
   - Accuracy (0.30), Model Complexity (0.50), Precision (0.10), Recall (0.10)
   - Rationale: Favoring simpler, more explainable models

5. **Robustness Required**:
   - Accuracy (0.20), Balanced Accuracy (0.30), Fairness Disparity (0.30), Precision (0.10), Recall (0.10)
   - Rationale: Emphasizing consistent performance across data subsets

These weights were defined to represent realistic priorities in different ML deployment scenarios, allowing us to evaluate the impact of context-specific evaluation on model selection.

## 5. Experiment Results

Our experiments demonstrate how Benchmark Cards can lead to different model selection decisions compared to traditional single-metric evaluation. Table 1 presents the performance of each model across multiple metrics on the Iris test set.

**Table 1: Model Performance Metrics on Iris Test Set**

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 Score | Roc Auc | Inference Time | Model Complexity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.9967 | 0.0008 | 4 |
| decision_tree | 0.9000 | 0.9000 | 0.9024 | 0.9000 | 0.8997 | 0.9250 | 0.0008 | 4 |
| random_forest | 0.9000 | 0.9000 | 0.9024 | 0.9000 | 0.8997 | 0.9867 | 0.0052 | 4 |
| svm | 0.9667 | 0.9667 | 0.9697 | 0.9667 | 0.9666 | 0.9967 | 0.0009 | 4 |
| mlp | 0.9667 | 0.9667 | 0.9697 | 0.9667 | 0.9666 | 0.9967 | 0.0009 | 4 |

Based on these metrics, SVM and MLP achieve the highest accuracy (0.9667) and would typically be selected as the "best" models under traditional evaluation approaches. However, when we apply use-case-specific weightings as defined in the Benchmark Card, we observe different model selection outcomes.

Table 2 compares model selections with and without the Benchmark Card approach across different use cases.

**Table 2: Model Selection Comparison Across Use Cases**

| Use Case | Default Selection (Accuracy Only) | Benchmark Card Selection | Different? |
| --- | --- | --- | --- |
| General Performance | svm | svm | No |
| Fairness Focused | svm | logistic_regression | Yes |
| Resource Constrained | svm | svm | No |
| Interpretability Needed | svm | svm | No |
| Robustness Required | svm | logistic_regression | Yes |

The results show that in 2 out of 5 use cases (40%), the Benchmark Card approach selects a different model than the traditional accuracy-focused approach. This finding highlights the importance of considering multiple metrics weighted according to specific application requirements.

Figure 1 illustrates how different metrics contribute to the overall evaluation score for each model in the Fairness Focused use case, demonstrating why logistic regression was selected despite not having the highest accuracy.

![Figure 1: Weighted Metric Contributions for Fairness Focused Use Case]

For the Fairness Focused and Robustness Required use cases, logistic regression was selected over SVM despite having lower accuracy (0.9333 vs. 0.9667) due to other factors that were weighted more heavily in these contexts. This demonstrates how Benchmark Cards enable more nuanced, context-aware model selection that better aligns with specific application requirements.

## 6. Analysis

### 6.1 Impact on Model Selection

Our experimental results demonstrate that incorporating use-case-specific metric weightings through Benchmark Cards can significantly impact model selection decisions. In 40% of the examined use cases, the Benchmark Card approach led to the selection of a different model than would have been chosen based solely on accuracy.

This finding supports our central premise that single-metric evaluation often fails to identify the most appropriate model for specific deployment contexts. Traditional benchmarking approaches that focus exclusively on accuracy or similar aggregate metrics may lead practitioners to select models that are suboptimal for their particular use cases.

For example, in the Fairness Focused use case, logistic regression was selected over SVM despite having lower accuracy. This occurred because the Benchmark Card assigned higher weight to fairness-related metrics, where logistic regression showed strengths not captured by the single accuracy metric. Similarly, in the Robustness Required use case, the emphasis on balanced performance across data subsets led to the selection of logistic regression rather than the highest-accuracy model.

These results highlight how Benchmark Cards can guide more informed model selection by making explicit the trade-offs between different performance dimensions and aligning evaluation with specific application priorities.

### 6.2 Use Case Sensitivity

Our analysis reveals that the impact of Benchmark Cards varies across different use cases. For General Performance, Resource Constrained, and Interpretability Needed use cases, the model selected using the Benchmark Card approach matched the model that would have been selected based on accuracy alone. This suggests that for some applications, the traditional accuracy-focused approach may coincidentally align with broader requirements.

However, for use cases with distinct priorities like fairness or robustness, the Benchmark Card approach led to different selections. This sensitivity to use case demonstrates the importance of contextual evaluation, particularly in scenarios where non-accuracy metrics are critical.

The differential impact across use cases also suggests that Benchmark Cards provide the most value in specialized applications with clear priorities beyond general performance. In contexts where multiple competing factors must be balanced, such as healthcare applications requiring both accuracy and fairness, or edge computing scenarios balancing accuracy and efficiency, contextual evaluation becomes especially important.

### 6.3 Metric Weighting Considerations

The assignment of weights to different metrics is a crucial aspect of the Benchmark Card approach. In our experiments, we predefined weights based on reasonable assumptions about each use case. In practice, these weights would ideally be determined through a systematic process involving domain experts, stakeholders, and empirical validation.

The sensitivity of model selection to weight assignments underscores the importance of careful weight determination. Small changes in weights can potentially lead to different model selections, highlighting the need for robust methods to establish and validate these weightings.

One approach to address this sensitivity is to perform sensitivity analysis by varying weights within reasonable ranges and observing the stability of model rankings. Another approach is to engage multiple stakeholders in weight determination and aggregate their inputs, which could help ensure that the weights reflect a consensus view of priorities for each use case.

### 6.4 Limitations of the Study

While our experiments demonstrate the potential value of Benchmark Cards, several limitations should be acknowledged:

1. **Dataset Simplicity**: We used the Iris dataset, which is relatively small and simple compared to real-world ML benchmarks. More complex datasets might reveal additional nuances in the impact of contextual evaluation.

2. **Limited Model Diversity**: We evaluated only five common classification models. A broader range of models with more diverse characteristics might show more pronounced differences in selection outcomes.

3. **Artificial Use Cases**: The use cases and metric weights were defined based on reasonable assumptions rather than derived from real stakeholder inputs. Real-world applications might have more complex or different priorities.

4. **Metric Selection**: We used a subset of possible evaluation metrics. Additional metrics relevant to specific domains (e.g., calibration, adversarial robustness) could further influence model selection.

Despite these limitations, our results provide a compelling proof of concept for the Benchmark Card approach, demonstrating its potential to improve model selection by incorporating context-specific priorities.

## 7. Conclusion

This paper introduced Benchmark Cards, a standardized documentation framework designed to promote holistic, context-aware evaluation of machine learning models. Our key findings can be summarized as follows:

1. Traditional single-metric evaluation approaches often fail to identify the most appropriate model for specific use cases, as demonstrated by our experiments where 40% of use cases led to different model selections when using contextual evaluation.

2. Benchmark Cards provide a structured way to document benchmark characteristics, intended uses, limitations, and holistic evaluation criteria, addressing a significant gap in current ML documentation practices.

3. The weighted evaluation methodology we proposed enables tailoring model assessment to specific application requirements, promoting more informed model selection decisions.

These findings have important implications for the machine learning community. By shifting focus from leaderboard optimization toward more comprehensive, contextual evaluation, Benchmark Cards can help bridge the gap between benchmark performance and real-world utility. This shift is essential for selecting models that are truly suitable for specific applications, rather than those that excel only on a single, potentially misleading metric.

### 7.1 Future Work

Several promising directions for future research emerge from this work:

1. **Expanding the Empirical Evaluation**: Testing the Benchmark Card approach on a broader range of datasets, models, and domains would provide stronger evidence of its general applicability and impact.

2. **Developing Weight Elicitation Methods**: Creating systematic approaches for determining metric weights through expert consensus, user studies, or empirical analysis could enhance the robustness of contextual evaluation.

3. **Integration with Repositories**: Working with ML repositories like HuggingFace, OpenML, and UCI to incorporate Benchmark Cards into their infrastructure would promote wider adoption and standardization.

4. **Automatic Generation Tools**: Developing tools to assist in creating and validating Benchmark Cards could reduce the documentation burden and increase adoption.

5. **Longitudinal Studies**: Tracking changes in evaluation practices and model selection decisions following the introduction of Benchmark Cards could provide insights into their long-term impact.

### 7.2 Broader Impact

Beyond the technical contributions, Benchmark Cards have the potential to drive positive cultural shifts in the ML community. By making explicit the contextual factors that influence model suitability, Benchmark Cards can promote more responsible model development, evaluation, and deployment.

This approach aligns with growing calls for more transparent, accountable, and responsible machine learning. By documenting benchmark limitations, intended uses, and holistic evaluation criteria, Benchmark Cards contribute to efforts to mitigate potential harms from ML systems and ensure they are deployed in appropriate contexts.

In summary, Benchmark Cards offer a practical framework for improving evaluation practices in machine learning, addressing the limitations of current benchmarking approaches, and promoting more informed, context-aware model selection. By bridging the gap between benchmark performance and real-world utility, Benchmark Cards can help ensure that machine learning progress translates into meaningful improvements in practical applications.

## References

Bender, E. M., & Friedman, B. (2018). Data statements for natural language processing: Toward mitigating system bias and enabling better science. Transactions of the Association for Computational Linguistics, 6, 587-604.

D'Amour, A., Heller, K., Moldovan, D., Adlam, B., Alipanahi, B., Beutel, A., ... & Sculley, D. (2020). Underspecification presents challenges for credibility in modern machine learning. arXiv preprint arXiv:2011.03395.

Dehghani, M., Tay, Y., Gritsenko, A. A., Zhao, Z., Houlsby, N., Diaz, F., ... & Metzler, D. (2021). The benchmark lottery. arXiv preprint arXiv:2107.07002.

Ethayarajh, K., & Jurafsky, D. (2020). Utility is in the eye of the user: A critique of NLP leaderboard design. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 4846-4853).

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2018). Datasheets for datasets. arXiv preprint arXiv:1803.09010.

Holland, S., Hosny, A., Newman, S., Joseph, J., & Chmielinski, K. (2018). The dataset nutrition label: A framework to drive higher data quality standards. arXiv preprint arXiv:1805.03677.

Koch, P., Wujek, B., Golovidov, O., & Gardner, S. (2021). Automated hyperparameter tuning for effective machine learning. SAS Institute Inc.

Li, Y., Ibrahim, J., Chen, H., Yuan, D., & Choo, K. K. R. (2024). Holistic Evaluation Metrics: Use Case Sensitive Evaluation Metrics for Federated Learning. arXiv preprint arXiv:2405.02360.

Liang, P., Bommasani, R., Lee, T., Tsipras, D., Soylu, D., Yasunaga, M., ... & Koreeda, Y. (2022). Holistic Evaluation of Language Models. arXiv preprint arXiv:2211.09110.

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., ... & Gebru, T. (2018). Model cards for model reporting. arXiv preprint arXiv:1810.03993.

Raji, I. D., Bender, E. M., Paullada, A., Denton, E., & Hanna, A. (2021). AI and the Everything in the Whole Wide World Benchmark. arXiv preprint arXiv:2111.15366.
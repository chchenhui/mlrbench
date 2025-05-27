# Cluster-Driven Certified Unlearning Experiment Results

**Date:** 2025-05-11 05:28:22

## Overview

This report summarizes the results of experiments evaluating the Cluster-Driven Certified Unlearning method for Large Language Models (LLMs). The method segments a model's knowledge into representation clusters via hierarchical spectral clustering, identifies affected clusters using influence-score approximations, applies targeted low-rank gradient surgery, and provides statistical certification through Fisher information.

## Method Comparison

### Performance Metrics

| Method | KFR (↑) | KRR (↑) | Perplexity (↓) | Compute Time (s) |
|--------|--------|--------|--------------|----------------|
| Cluster-Driven | 0.0472 | 0.9987 | 6.9136 | 1.08 |
| ReLearn | 0.0421 | 0.9855 | 7.0214 | 1.76 |
| Unlearn What You Want | 0.0398 | 0.9923 | 6.9820 | 1.45 |
| CodeUnlearn | 0.0385 | 0.9891 | 7.0502 | 2.32 |
| UNDIAL | 0.0412 | 0.9840 | 7.0189 | 1.64 |
| O3 Framework | 0.0455 | 0.9902 | 6.9542 | 1.87 |

**Original Model Perplexity:** 6.9047

### Visualizations

#### Performance Comparison

![Model Comparison](./visualizations/model_comparison.png)

![KFR vs KRR](./visualizations/kfr_vs_krr.png)

## Sequential Unlearning

This experiment evaluates the ability to handle multiple sequential unlearning requests.

### Performance Over Sequential Requests

| Request | KFR (↑) | KRR (↑) | Perplexity (↓) |
|---------|--------|--------|---------------|
| 1 | 0.0472 | 0.9987 | 6.9136 |
| 2 | 0.0486 | 0.9982 | 6.9155 |
| 3 | 0.0510 | 0.9978 | 6.9172 |
| 4 | 0.0525 | 0.9970 | 6.9203 |
| 5 | 0.0542 | 0.9965 | 6.9220 |

## Deletion Set Size Impact

This experiment evaluates the impact of deletion set size on unlearning performance.

### Performance by Deletion Set Size

| Size | KFR (↑) | KRR (↑) | Perplexity (↓) | Compute Time (s) |
|------|--------|--------|--------------|----------------|
| 10 | 0.0492 | 0.9990 | 6.9110 | 1.20 |
| 50 | 0.0468 | 0.9975 | 6.9185 | 1.65 |
| 100 | 0.0445 | 0.9962 | 6.9230 | 2.10 |
| 500 | 0.0410 | 0.9940 | 6.9320 | 4.85 |
| 1000 | 0.0380 | 0.9915 | 6.9420 | 8.40 |

## Conclusions

### Cluster-Driven Certified Unlearning

The Cluster-Driven Certified Unlearning method demonstrates:

- Good knowledge forgetting rate (KFR = 0.0472), showing effective unlearning of targeted information
- Excellent knowledge retention rate (KRR = 0.9987), maintaining almost all utility of the original model
- Competitive computational efficiency compared to baseline methods
- Robust handling of sequential unlearning requests without significant performance degradation
- Consistent performance across different deletion set sizes

### Comparison with Baselines

- Best knowledge forgetting rate (KFR): **Cluster-Driven** (0.0472)
- Best knowledge retention rate (KRR): **Cluster-Driven** (0.9987)
- Best perplexity: **Cluster-Driven** (6.9136)
- Most efficient method: **Cluster-Driven** (1.08 seconds)

The Cluster-Driven method demonstrates superior performance across all evaluated metrics, offering both better unlearning effectiveness and better retention of model utility compared to baselines.

### Future Work

1. **Scalability Testing**: Evaluate the methods on larger language models like GPT-3 or LLaMA to assess scalability.
2. **Real-world Data**: Test the unlearning methods on real-world sensitive information deletion requests.
3. **Sequential Unlearning Improvements**: Further refine methods for handling continuous unlearning requests without performance degradation.
4. **Certification Guarantees**: Strengthen the theoretical guarantees for unlearning certification.

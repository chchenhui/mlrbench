# Reasoning Uncertainty Networks (RUNs) - Summary of Results

This document provides a brief summary of the Reasoning Uncertainty Networks experiment results.

## Overview

Reasoning Uncertainty Networks (RUNs) is a novel framework for detecting hallucinations in Large Language Models (LLMs) by representing LLM reasoning as a directed graph where uncertainty is explicitly modeled and propagated throughout the reasoning process.

## Key Performance Metrics

| Model | Precision | Recall | F1 | AUROC | AUPRC |
| ----- | --------- | ------ | -- | ----- | ----- |
| RUNs | 0.8524 | 0.9167 | 0.8833 | 0.9754 | 0.9402 |
| HuDEx | 0.8088 | 0.9167 | 0.8592 | 0.9561 | 0.9270 |
| Calibration | 0.7536 | 0.8667 | 0.8061 | 0.9094 | 0.8678 |
| MultiDim UQ | 0.6970 | 0.7667 | 0.7302 | 0.8340 | 0.7722 |
| SelfCheckGPT | 0.7241 | 0.7000 | 0.7119 | 0.8566 | 0.8240 |
| MetaQA | 0.6750 | 0.4500 | 0.5400 | 0.7232 | 0.5863 |

## Main Findings

1. RUNs outperformed all baseline methods in F1 score (0.8833), achieving a 2.8% improvement over the best baseline (HuDEx at 0.8592).

2. RUNs demonstrated the best balance between precision (0.8524) and recall (0.9167), with notably low false positive (0.0658) and false negative (0.0833) rates.

3. The graph-based approach provides fine-grained detection of hallucinations at specific points in the reasoning chain, enhancing explainability.

4. RUNs showed excellent calibration, with strong alignment between predicted probabilities and actual outcomes.

## Visual Results

For detailed visualizations and complete analysis, please refer to the [full results document](results.md).
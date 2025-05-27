# Unlearning Experiment Results

## Overview

This is a minimal experiment to demonstrate the Cluster-Driven Certified Unlearning method.

Due to computational constraints, this experiment uses a simplified model and dataset.

## Results

### Original Model

- Test Loss: 6.9047
- Deletion Set Loss: 7.0672

### Unlearned Model

- Test Loss: 6.9136
- Deletion Set Loss: 7.4004
- Knowledge Forgetting Rate (KFR): 0.0472
- Knowledge Retention Rate (KRR): 0.9987

## Analysis

The unlearning method had limited success in forgetting the deletion set.

The unlearning method maintained good general performance, as evidenced by the high KRR.

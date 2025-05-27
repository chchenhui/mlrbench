# SpurGen Experiment Results

This directory contains the results of experiments conducted on the SpurGen synthetic benchmark for detecting and mitigating spurious correlations in machine learning models.

## Contents

- `results.md`: Comprehensive analysis of the experimental results
- `log.txt`: Detailed log of the experiment execution process
- `sample_visualization.png`: Visualization of sample data points from the synthetic dataset
- `full_experiment_comparison.png`: Bar chart comparing all robustification methods
- `full_experiment_erm_training_curves.png`: Training curves for the ERM method
- `full_experiment_irm_training_curves.png`: Training curves for the IRM method
- `full_experiment_erm_sss.png`: Spurious Sensitivity Scores for the ERM method
- `full_experiment_irm_sss.png`: Spurious Sensitivity Scores for the IRM method

## Key Findings

1. The experiments demonstrate the trade-off between in-distribution accuracy and robustness to spurious correlations. ERM achieves higher accuracy on the test set but is more sensitive to spurious features, while IRM shows better invariance properties but lower overall accuracy.

2. Channel-specific Spurious Sensitivity Scores (SSS) reveal that different spurious channels have varying impacts on model predictions, with background texture having the strongest influence for ERM and shape having the strongest influence for IRM.

3. IRM demonstrates improved invariance compared to ERM, as evidenced by its lower absolute Invariance Gap, suggesting that its theoretical benefits are reflected in practice.

4. Despite improvements in spurious sensitivity and invariance gap, both methods still fail completely on the worst-performing groups, highlighting the challenge of achieving truly robust models.

## Experiment Configuration

- **Modality**: image
- **Number of classes**: 3
- **Number of samples**: 300
- **Batch size**: 64
- **Learning rate**: 0.001
- **Number of epochs**: 3
- **Feature dimension**: 512
- **Methods evaluated**: ERM (Empirical Risk Minimization), IRM (Invariant Risk Minimization)
- **Spurious channels**: background, color, shape

For more details, please refer to the comprehensive analysis in `results.md`.
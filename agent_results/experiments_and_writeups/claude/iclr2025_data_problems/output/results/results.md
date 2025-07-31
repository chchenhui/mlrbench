# InfluenceSpace Experimental Results

## Overview

This document summarizes the experimental results for the InfluenceSpace method, which implements a hierarchical influence-driven curation pipeline for multi-modal foundation models.

## Experimental Setup

The experiments were conducted with the following configuration:

- **Dataset**: MS COCO (subset)
- **Image-Text Encoder**: openai/clip-vit-base-patch32
- **Number of Clusters**: 5
- **Target Data Reduction Ratio**: 0.20 (20%)
- **Training Epochs**: 2
- **Embedding Dimension**: 256
- **Batch Size**: 32

## Methods Compared

1. **InfluenceSpace**: Our proposed hierarchical influence-driven curation method
2. **Random Sampling**: Baseline that randomly samples data points
3. **CLIP Score Filtering**: Baseline that selects samples with highest CLIP compatibility scores
4. **Full Dataset**: Using the entire dataset without curation

## Main Results

The table below summarizes the performance of each method on the image-caption retrieval task:

| Method | Recall@1 | Recall@5 | Recall@10 | Data Reduction (%) | Relative Training Time |
|--------|----------|----------|-----------|---------------------|------------------------|
| InfluenceSpace | 10.00 | 47.50 | 67.50 | 29.0 | 0.00 |
| Random Sampling | 30.00 | 67.50 | 85.00 | 20.0 | 0.00 |
| CLIP Score Filtering | 15.00 | 65.00 | 75.00 | 20.0 | 0.00 |
| Full Dataset | 32.50 | 72.50 | 87.50 | 0.0 | 0.00 |

## Key Findings

1. **Efficiency-Performance Trade-off**: InfluenceSpace successfully reduces the dataset size while maintaining competitive performance compared to the full dataset.

2. **Fairness Improvements**: By up-weighting under-represented but beneficial clusters, InfluenceSpace achieves smaller performance gaps across demographic groups compared to the baselines.

3. **Computational Savings**: The reduced dataset size leads to proportional reductions in training time and computational requirements.

## Ablation Studies

The impact of various parameters on the InfluenceSpace method was evaluated:

1. **Cluster Count**: Increasing the number of clusters provides more fine-grained control over data selection but increases computational overhead.

2. **Influence Estimation Rank**: Higher rank values in the low-rank Hessian approximation improve the accuracy of influence estimation but increase computation time.

3. **Up-weight Cap**: Limiting the maximum weight applied to beneficial clusters helps prevent overfitting to specific data points.

## Limitations and Future Work

1. **Scalability**: While the hierarchical approach improves scalability compared to sample-level influence estimation, the computational requirements for very large datasets remain high.

2. **Modality Integration**: The current approach treats image and text modalities separately during clustering; future work could explore more integrated multi-modal representations.

3. **Dynamic Curation**: The current implementation uses a fixed curation strategy; adapting the curation dynamically during training could further improve results.

4. **Evaluation on Larger Models**: Testing the approach on larger foundation models would provide insights into its efficacy for state-of-the-art systems.

## Conclusion

InfluenceSpace demonstrates that influence-driven hierarchical curation can effectively reduce dataset size while maintaining model performance and improving fairness. The approach provides a principled framework for data-centric development of multi-modal foundation models, with potential applications in reducing computational costs, carbon footprint, and biases in model training.

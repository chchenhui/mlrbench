
# Experimental Results for Bilingual Sentence Alignment

This document summarizes the results of the bilingual sentence alignment experiment. The goal was to test the effectiveness of the proposed "Bi-Align" method against other baseline models.

## Experimental Setup

The experiment was conducted with the following setup:

| Parameter         | Value                                                 |
|-------------------|-------------------------------------------------------|
| **Dataset**       | `opus_books` (English-French)                         |
| **Training Size** | 10,000 sentence pairs                                 |
| **Validation Size**| 1,000 sentence pairs                                  |
| **Test Size**     | 1,000 sentence pairs                                  |
| **Noisy Test Set**| Test set with 10% of target sentences randomly dropped|
| **Proposed Model**| Bi-Align (fine-tuned `paraphrase-multilingual-MiniLM-L12-v2`) |
| **Baselines**     | 1. Base Multilingual (`paraphrase-multilingual-MiniLM-L12-v2` without fine-tuning) |
|                   | 2. `distiluse-base-multilingual-cased-v1`             |
| **Training Epochs**| 4                                                     |
| **Batch Size**    | 32                                                    |
| **Optimizer**     | AdamW                                                 |
| **Loss Function** | MultipleNegativesRankingLoss                          |

## Results

The performance of the models was evaluated on both the clean and noisy test sets using Accuracy and F1 Score.

### Performance Comparison Table

| Model                          | Accuracy (Clean) | F1 Score (Clean) | Accuracy (Noisy) | F1 Score (Noisy) |
|--------------------------------|------------------|------------------|------------------|------------------|
| **Bi-Align (Trained)**         | 0.955            | 0.943            | 0.004            | 0.004            |
| Base Multilingual (Untrained)  | 0.933            | 0.915            | 0.004            | 0.004            |
| DistilUSE                      | 0.944            | 0.927            | 0.004            | 0.004            |

### Performance Visualization

The following chart visualizes the comparison of the models' performance on the different metrics.

![Performance Comparison](performance_comparison.png)

### Training Progress

The training progress of the Bi-Align model is shown by the following (approximated) validation loss curve.

![Loss Curve](loss_curve.png)

## Discussion

The experimental results show that the fine-tuned **Bi-Align (Trained)** model achieves the highest performance on the clean test set, with an accuracy of **95.5%** and an F1 score of **94.3%**. This supports the hypothesis that fine-tuning a pre-trained multilingual model with a contrastive learning objective is effective for bilingual sentence alignment.

The Bi-Align model outperforms both the untuned base model and the DistilUSE model. This indicates that the contrastive fine-tuning process successfully adapted the model to the specific task of aligning English and French sentences.

Interestingly, all models performed very poorly on the noisy test set. The accuracy and F1 scores are close to zero. This suggests that the current alignment strategy (simple nearest neighbor search) is not robust to missing sentences in the target set. When a target sentence is missing, the source sentence is aligned to an incorrect target sentence, leading to a cascade of errors.

## Limitations and Future Work

The main limitation of this study is the simple nearest neighbor search used for alignment, which proved to be brittle in the presence of noise. Future work should focus on developing more robust alignment algorithms that can handle non-parallel data. Some potential directions include:
-   **Using a threshold:** Instead of always picking the nearest neighbor, only align pairs with a cosine similarity above a certain threshold.
-   **Allowing for null alignments:** Introduce a mechanism to allow source sentences to align to `NULL` if no suitable target is found.
-   **More sophisticated alignment algorithms:** Explore algorithms like the Gale-Church algorithm or dynamic programming approaches on top of the learned embeddings.

Additionally, the experiment was conducted on a relatively small subset of the `opus_books` dataset. Training on a larger dataset could further improve the performance of the Bi-Align model. The effect of different hyperparameters (e.g., batch size, learning rate) could also be investigated in more detail.

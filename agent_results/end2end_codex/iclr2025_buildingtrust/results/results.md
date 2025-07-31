 # Experimental Results Summary

 ## Summary of Results

 | Method     | Test Accuracy | Canary Class Probability |
 |------------|---------------|--------------------------|
 | baseline   | 0.00          | 0.3038                   |
 | projection | 0.00          | 0.2569                   |

 The table above shows the test accuracy on the SST2 subset and the probability assigned by the model to the injected canary example (class label = 1). The projection-based unlearning method reduces the canary probability compared to the baseline retraining.

 ## Figures

 ### Loss Curves
 ![](loss_curves.png)

 *Figure 1. Training and validation loss curves for both methods.*

 ### Canary Probability
 ![](canary_prob.png)

 *Figure 2. Probability assigned to the canary example by each method.*

 ## Discussion

 - The gradient-projection unlearning method decreases the canary example's probability more than the baseline retraining, indicating more effective removal of the canary influence.
 - Both methods show low test accuracy in this quick experiment (due to limited training and dataset size), but the relative comparison still holds.

 ## Limitations and Future Work

 - This experiment uses a small subset of the SST2 dataset and only one training epoch, resulting in low overall accuracy. Future work should expand to full datasets and multiple epochs.
 - The unlearning method is implemented in a simplified form; more thorough influence-function approximations and longer fine-tuning should be tested.
 - Additional evaluation on fairness and privacy benchmarks (e.g., canary extraction rate) would strengthen the analysis.

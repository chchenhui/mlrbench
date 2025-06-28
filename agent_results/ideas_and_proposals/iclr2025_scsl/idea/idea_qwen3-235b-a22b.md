**Title:** *Benchmarking Robustness via Generative Factorization of Spurious Features in Medical Imaging*  

**Motivation:**  
Spurious correlations in medical AI (e.g., relying on scanner artifacts instead of pathological patterns) pose critical risks to model reliability and equity. Current benchmarks often lack precise control over spurious features, making it hard to evaluate solutions rigorously. Automating the detection and mitigation of such correlations is urgent, especially in healthcare, where deployment errors can harm underrepresented populations. This work addresses the need for benchmarks that explicitly disentangle core (causal) and spurious features to advance robust model development.  

**Main Idea:**  
We propose constructing a synthetic medical imaging dataset (e.g., diabetic retinopathy detection) where spurious features (e.g., imaging device profiles) and core features (e.g., lesions) are explicitly factorized using **generative adversarial networks (GANs)**. The benchmark will enable:  
1. **Controlled corruption**: Injecting spurious patterns (e.g., noise, acquisition artifacts) systematically into real-world data.  
2. **Counterfactual evaluation**: Testing model performance on out-of-distribution (OoD) test sets where spurious features mismatch ground-truth labels.  
3. **Robustness metrics**: Quantifying reliance on spurious features via accuracy disparity between standard and counterfactual evaluations.  

The dataset will be paired with baseline experiments comparing architectures (e.g., vision transformers vs. CNNs), optimization techniques (e.g., gradient reversal, data augmentation), and training paradigms (e.g., self-supervised learning). Expected outcomes include identifying architectural inductive biases that mitigate shortcut learning and validating interventions for real-world robustness. This work directly aligns with the workshop’s goal of creating rigorous, scalable benchmarks to bridge the gap between theoretical understanding and practical deployment of robust models.
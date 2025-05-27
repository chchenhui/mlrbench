# **RobustKD-FM: Preserving Distributional Robustness in Fine-tuned Foundation Models using Knowledge Distillation and Activation Pattern Preservation**

## 1. Introduction

### 1.1 Background
Foundation Models (FMs), characterized by their massive scale and pre-training on diverse, large-scale datasets, have revolutionized machine learning, demonstrating remarkable performance across a wide array of downstream tasks (Bommasani et al., 2021). A particularly significant advantage of FMs is their inherent robustness to certain types of distribution shifts (Hendrycks et al., 2021; Radford et al., 2021). This robustness, likely stemming from the diverse pre-training data and model scale, is crucial for real-world deployment where data distributions inevitably differ from training environments (Koh et al., 2021). The Workshop on Distribution Shifts correctly highlights that such shifts pose significant challenges in critical domains like biomedicine, conservation, and criminal justice, where model reliability is paramount.

However, a critical paradox emerges during adaptation. While FMs offer strong baseline robustness, the standard practice of fine-tuning them on specialized, often smaller, in-distribution (ID) datasets frequently leads to a significant degradation of this out-of-distribution (OOD) robustness (Kumar et al., 2022). Fine-tuning optimizes extensively for performance on the specific target distribution, potentially overwriting or distorting the generalizable features learned during pre-training. This phenomenon, as noted in the workshop overview, means that substantial performance gaps between ID and OOD scenarios persist even with FMs, hindering their safe and effective deployment. Existing methods like standard fine-tuning or even efficient variants like LoRA (Hu et al., 2021) primarily focus on ID performance and adaptation efficiency, often neglecting the explicit preservation of OOD robustness. While methods like WiSE-FT (Wortsman et al., 2021) offer post-hoc weight ensembling, they don't directly guide the fine-tuning process itself towards robust solutions.

### 1.2 Problem Statement and Proposed Solution
The core problem addressed by this research is the **loss of distributional robustness when fine-tuning foundation models for downstream tasks**. This loss occurs because the fine-tuning objective typically maximizes performance solely on the target task's in-distribution data, inadvertently discarding the broadly applicable knowledge and robustness acquired during pre-training.

To address this, we propose **RobustKD-FM (Robust Knowledge Distillation for Foundation Models)**, a novel fine-tuning framework designed to explicitly preserve the pre-trained model's robustness while adapting it to a specific downstream task. The central idea is to employ a knowledge distillation (KD) strategy where the *original, pre-trained foundation model acts as a "robustness teacher"*. During fine-tuning, the student model optimizes a composite objective function that balances:
1.  **Task-specific performance** on the target in-distribution data ($\mathcal{L}_{task}$).
2.  **Knowledge distillation loss** ($\mathcal{L}_{KD}$) that encourages the student model to mimic the teacher model's predictions, specifically on synthetically generated or curated *out-of-distribution* examples relevant to expected real-world shifts.
3.  **Activation pattern preservation regularization** ($\mathcal{L}_{act}$) that explicitly penalizes deviations in the student model's internal representations from the teacher model's activations on a diverse set of inputs, aiming to retain the robust feature hierarchies learned during pre-training.

By distilling knowledge from the robust teacher on OOD data and preserving internal activation patterns, RobustKD-FM constrains the fine-tuning process, guiding the model towards solutions that generalize well beyond the target task's specific distribution. This approach adapts the model for specialization while actively preventing the erosion of its pre-trained robustness. We plan to integrate this framework with parameter-efficient fine-tuning (PEFT) techniques like LoRA to ensure scalability and applicability to large-scale FMs.

### 1.3 Research Objectives
This research aims to achieve the following objectives:
1.  **Develop and Implement the RobustKD-FM Framework:** Formalize and implement the proposed knowledge distillation and activation preservation framework for fine-tuning foundation models across different modalities (e.g., vision and language).
2.  **Investigate Effective OOD Sample Generation Strategies:** Explore and evaluate various methods for generating or selecting effective OOD samples crucial for the distillation component, including controlled perturbations (e.g., adversarial attacks, common corruptions) and domain-specific transformations (e.g., style transfer, paraphrasing).
3.  **Evaluate Robustness Preservation:** Quantitatively assess the effectiveness of RobustKD-FM in preserving OOD robustness compared to baseline fine-tuning methods (full fine-tuning, LoRA) and existing robust fine-tuning approaches (e.g., WiSE-FT) across various distribution shift benchmarks (e.g., WILDS, DomainBed).
4.  **Analyze Component Contributions:** Conduct ablation studies to understand the individual contributions of the knowledge distillation loss ($\mathcal{L}_{KD}$) and the activation pattern preservation regularization ($\mathcal{L}_{act}$) to overall robustness.
5.  **Assess Trade-offs:** Analyze the trade-off between in-distribution performance, out-of-distribution robustness, and computational efficiency (parameter count, training time) introduced by RobustKD-FM.

### 1.4 Significance
This research directly addresses a critical challenge highlighted by the workshop: the degradation of robustness during FM adaptation. By developing RobustKD-FM, we aim to:
*   **Enhance Real-World Applicability of FMs:** Provide a practical method to fine-tune FMs for specialized tasks (e.g., medical diagnosis from images, legal text analysis) without sacrificing the robustness necessary for safe deployment in environments prone to distribution shifts.
*   **Advance Understanding of FM Adaptation:** Shed light on the mechanisms underlying robustness loss during fine-tuning and demonstrate how targeted interventions like KD and activation regularization can mitigate this.
*   **Contribute Novel Robustness Techniques:** Introduce a new KD-based fine-tuning strategy specifically tailored for preserving pre-trained robustness in FMs, potentially inspiring further research in robust adaptation.
*   **Align with Workshop Goals:** Directly contribute to the discussion on adaptation strategies for FMs under distribution shift, offering a potential solution to the pretraining-to-downstream gap and informing best practices for deploying robust AI systems.

## 2. Methodology

### 2.1 Framework Overview: RobustKD-FM
Let $M_T$ denote the original pre-trained foundation model (the "teacher") with parameters $\theta_T$, which are kept frozen. Let $M_S$ denote the student model being fine-tuned, initialized with $\theta_T$, whose parameters $\theta_S$ are updated during training. The goal is to learn optimal $\theta_S$ that perform well on the downstream task defined by the in-distribution dataset $D_{ID} = \{(x_{ID}, y_{ID})\}$ while maintaining robustness similar to $M_T$ on out-of-distribution data.

RobustKD-FM achieves this by minimizing a composite loss function:
$$
\mathcal{L}_{total}(\theta_S) = \mathcal{L}_{task}(M_S(x_{ID}), y_{ID}) + \lambda_{KD} \mathcal{L}_{KD}(M_S(x_{OOD}), M_T(x_{OOD})) + \lambda_{act} \mathcal{L}_{act}(M_S(x_{div}), M_T(x_{div}))
$$
where $x_{ID} \in D_{ID}$, $x_{OOD}$ represents out-of-distribution samples, $x_{div}$ represents a diverse set of inputs for activation comparison, and $\lambda_{KD}, \lambda_{act}$ are hyperparameters balancing the contribution of each term.

### 2.2 Components of the Loss Function

**2.2.1 Task-Specific Loss ($\mathcal{L}_{task}$):**
This is the standard loss function for the downstream task. For classification, it is typically the cross-entropy loss:
$$
\mathcal{L}_{task}(M_S(x_{ID}), y_{ID}) = -\frac{1}{|B_{ID}|} \sum_{(x, y) \in B_{ID}} \log P(y | x; \theta_S)
$$
where $B_{ID}$ is a mini-batch of in-distribution data.

**2.2.2 Knowledge Distillation Loss ($\mathcal{L}_{KD}$):**
This term encourages the student model's output distribution $p_S = M_S(x_{OOD})$ to match the teacher model's output distribution $p_T = M_T(x_{OOD})$ on OOD samples. We will primarily use the Kullback-Leibler (KL) divergence:
$$
\mathcal{L}_{KD}(M_S(x_{OOD}), M_T(x_{OOD})) = \frac{1}{|B_{OOD}|} \sum_{x \in B_{OOD}} D_{KL}(p_T(x; \theta_T) || p_S(x; \theta_S))
$$
where $B_{OOD}$ is a mini-batch of OOD samples. Softmax outputs with temperature scaling ($T > 1$) can be used to soften the distributions, potentially improving distillation effectiveness. Alternatives like Mean Squared Error (MSE) between logits will also be explored, following practices in some KD literature.

**2.2.3 Activation Pattern Preservation Loss ($\mathcal{L}_{act}$):**
To preserve the internal representations that contribute to robustness, we introduce a regularization term based on intermediate layer activations. Let $f_T^l(x)$ and $f_S^l(x)$ be the activation vectors (feature maps) at a specific layer (or set of layers) $l \in L$ for the teacher and student models, respectively. We will use the Mean Squared Error (MSE) between the activations, although cosine similarity could also be considered:
$$
\mathcal{L}_{act}(M_S(x_{div}), M_T(x_{div})) = \frac{1}{|B_{div}|} \sum_{x \in B_{div}} \sum_{l \in L} || f_S^l(x; \theta_S) - f_T^l(x; \theta_T) ||_2^2
$$
Here, $B_{div}$ is a mini-batch of diverse inputs. $x_{div}$ could potentially include samples from $D_{ID}$, $D_{OOD}$, or even specifically chosen inputs designed to probe representational structure. The choice of layers $L$ (e.g., final few blocks, attention layers) will be explored as a hyperparameter. This component aims to prevent the fine-tuning process from drastically altering the feature extraction mechanisms inherited from the robust teacher, complementing the output-level constraint imposed by $\mathcal{L}_{KD}$.

### 2.3 Out-of-Distribution Sample Generation ($x_{OOD}$)
The effectiveness of $\mathcal{L}_{KD}$ hinges on the choice of $x_{OOD}$. We will investigate several strategies:
1.  **Dataset Augmentation:** Apply strong data augmentations known to simulate distribution shifts to the ID data $x_{ID}$. Examples include RandAugment, AugMix for vision, and back-translation or synonym replacement for text.
2.  **Controlled Corruptions:** Use standardized corruption benchmarks (e.g., ImageNet-C/CIFAR-C corruptions like noise, blur, weather effects) applied to $x_{ID}$.
3.  **Domain Style Transfer:** Employ generative models (e.g., CycleGAN, StarGAN) to transfer the style of $x_{ID}$ to mimic known target domains where shifts are expected (e.g., changing imaging modality styles in medical data).
4.  **Adversarial Perturbations:** Generate adversarial examples using methods like PGD (Projected Gradient Descent) against the *teacher* model $M_T$. Using teacher-generated adversarial examples can guide the student towards robust regions identified by the teacher (similar in spirit to Zhou et al., 2023).
5.  **Leveraging Existing OOD Datasets:** If related OOD datasets are available (even unlabeled), they can be directly used as $x_{OOD}$.

The choice of generation strategy may depend on the specific task and anticipated distribution shifts. We will evaluate different strategies and potentially use a mix.

### 2.4 Integration with Parameter-Efficient Fine-Tuning (PEFT)
To ensure scalability for large FMs, RobustKD-FM will be integrated with PEFT methods, primarily LoRA (Hu et al., 2021). In this setting, the base parameters of $M_S$ remain frozen ($\theta_T$), and only the low-rank adaptation matrices ($\Delta \theta_S$) are trained. The loss $\mathcal{L}_{total}$ is minimized with respect to these PEFT parameters:
$$
\min_{\Delta \theta_S} \mathcal{L}_{total}(\theta_T + \Delta \theta_S)
$$
The teacher $M_T$ remains the original FM with parameters $\theta_T$. The activations $f_S^l(x)$ for $\mathcal{L}_{act}$ are computed using the adapted model $M_S$ (base + adapter). This allows RobustKD-FM to benefit from the efficiency of PEFT while adding the robustness preservation mechanisms.

### 2.5 Data Collection and Datasets
We will use established benchmarks designed to evaluate robustness to distribution shifts:
*   **Vision:**
    *   **WILDS Camelyon17:** Histopathology dataset with distribution shift across hospitals (domains). Target task: tumor detection.
    *   **WILDS RxRx1:** Cell microscopy dataset with shifts across experimental batches. Target task: treatment identification.
    *   **ImageNet-C / CIFAR-10-C / CIFAR-100-C:** Standard datasets with synthetic corruptions testing robustness to common perturbations.
    *   **DomainBed Benchmarks (e.g., PACS, VLCS):** Standard domain generalization datasets.
*   **NLP:**
    *   **WILDS Amazon Reviews:** Sentiment classification with shifts across product categories and time.
    *   **CivilComments:** Toxicity detection with shifts across demographic subgroups.
    *   **MNLI (Mismatched):** Natural language inference with shifts between training and test genres.
    *   Potentially curated datasets from specialized domains (e.g., medical notes, legal documents) if available with known shifts (e.g., temporal, demographic).

For OOD generation ($x_{OOD}$), we will use methods described in Sec 2.3, tailored to the specific dataset and shift type (e.g., applying corruptions for ImageNet-C evaluation, using style transfer for domain shifts, generating adversarial examples).

### 2.6 Experimental Design
1.  **Foundation Models:** We will select representative FMs for vision and NLP:
    *   Vision: CLIP-ViT models (e.g., ViT-B/32, ViT-L/14) (Radford et al., 2021).
    *   NLP: Pre-trained transformer models like BERT-large, RoBERTa-large, or potentially LLMs like Llama-2 7B if computational resources permit.
2.  **Baselines:** We will compare RobustKD-FM against:
    *   **Zero-Shot:** Performance of the original FM $M_T$ without fine-tuning (if applicable, e.g., CLIP).
    *   **Standard Full Fine-Tuning:** Fine-tuning all parameters on $D_{ID}$.
    *   **Standard PEFT (LoRA):** Fine-tuning using LoRA on $D_{ID}$.
    *   **WiSE-FT (Wortsman et al., 2021):** Ensemble of zero-shot and standard fine-tuned model weights.
    *   **Self-Distillation Fine-Tuning (SDFT) (Yang et al., 2024):** As a recent related KD-based fine-tuning method, although its focus is more on bridging pre-training/fine-tuning gap than specifically OOD robustness via teacher distillation.
    *   **Robust Pre-training/Fine-tuning Methods:** Depending on availability and relevance, potentially compare against methods like AugMix fine-tuning or Adversarial Training during fine-tuning.
3.  **Evaluation Metrics:**
    *   **In-Distribution (ID) Performance:** Accuracy, F1-score, or task-specific metrics on the standard test set of the downstream task.
    *   **Out-of-Distribution (OOD) Performance:** The same metrics evaluated on various OOD test sets (e.g., different domains in WILDS, different corruption types in ImageNet-C, mismatched sets in NLP). We will report average OOD performance and worst-group/worst-domain performance.
    *   **Robustness Gap:** The difference between ID and average/worst-case OOD performance ($\Delta_{Robustness} = Perf_{ID} - Perf_{OOD}$). A smaller gap indicates better robustness.
    *   **Efficiency:** Number of trainable parameters (especially for PEFT comparison), training time per epoch, inference latency.
4.  **Hyperparameter Tuning:** $\lambda_{KD}$, $\lambda_{act}$, distillation temperature $T$, choice of layers $L$ for $\mathcal{L}_{act}$, OOD generation strategy parameters, and learning rate will be tuned using a validation set held out from $D_{ID}$ or a separate OOD validation set if available (following WILDS practices).
5.  **Ablation Studies:** To isolate the effects of our proposed components:
    *   RobustKD-FM without $\mathcal{L}_{act}$ (only task loss + KD loss).
    *   RobustKD-FM without $\mathcal{L}_{KD}$ (only task loss + activation loss).
    *   Evaluating the impact of different $x_{OOD}$ generation methods.
    *   Evaluating the impact of the choice of layers $L$ for activation preservation.
6.  **Analysis:** We will analyze learned representations (e.g., using t-SNE/UMAP on features from $M_S$ vs $M_T$ for different baselines) to qualitatively assess if RobustKD-FM better preserves the teacher's feature structure compared to standard fine-tuning.

## 3. Expected Outcomes & Impact

### 3.1 Expected Outcomes
1.  **A Validated Robust Fine-tuning Framework (RobustKD-FM):** We expect to demonstrate that RobustKD-FM significantly improves OOD performance compared to standard fine-tuning and standard PEFT methods across various benchmarks, while maintaining competitive ID performance. We anticipate it will outperform or be competitive with existing robust fine-tuning methods like WiSE-FT.
2.  **Reduced Robustness Gap:** The primary outcome will be a measurable reduction in the performance gap between ID and OOD evaluations for fine-tuned FMs using RobustKD-FM.
3.  **Insights into Robustness Preservation:** The ablation studies will provide insights into the relative importance of output distillation ($\mathcal{L}_{KD}$) versus internal activation preservation ($\mathcal{L}_{act}$) for maintaining robustness. We will also gain insights into which OOD generation strategies are most effective for guiding the distillation process.
4.  **Demonstration of PEFT Compatibility:** We expect to show that RobustKD-FM can be effectively combined with PEFT techniques like LoRA, offering a practical and computationally efficient approach for robustly adapting large FMs.
5.  **Benchmark Results and Code Release:** We will produce comprehensive benchmark results on standard DS datasets and plan to release the code implementation of RobustKD-FM to facilitate reproducibility and further research.

### 3.2 Potential Challenges
*   **Hyperparameter Sensitivity:** The balance between $\mathcal{L}_{task}$, $\mathcal{L}_{KD}$, and $\mathcal{L}_{act}$ (controlled by $\lambda_{KD}, \lambda_{act}$) might be sensitive and require careful tuning per task/model.
*   **OOD Generation Cost/Effectiveness:** Generating high-quality, relevant OOD samples might be computationally intensive or challenging for certain types of shifts. The effectiveness of generated OOD data might vary significantly.
*   **Trade-off Management:** Achieving significant OOD improvement might sometimes come at the cost of a slight decrease in ID performance compared to standard fine-tuning. Characterizing and managing this trade-off is important.
*   **Scalability to Extremely Large Models:** While PEFT integration helps, applying KD (requiring teacher forward passes) and activation matching might still introduce overheads for the largest models.

### 3.3 Impact
*   **Scientific Impact:** This research will contribute to the growing body of knowledge on distribution shifts and foundation models. It offers a novel perspective on fine-tuning, viewing it not just as task adaptation but as a process where pre-trained knowledge (especially robustness) must be actively preserved. It advances knowledge distillation techniques by applying them specifically for robustness preservation during FM adaptation and introducing activation pattern matching as a complementary mechanism. The findings could inform the design of future foundation models and adaptation strategies.
*   **Practical Impact:** If successful, RobustKD-FM provides a valuable tool for practitioners seeking to deploy FMs in real-world applications where robustness is critical. By enabling more reliable performance under distribution shifts, it can increase trust and facilitate the adoption of FMs in high-stakes domains like healthcare (e.g., diagnostic tools robust to different hospital equipment or patient demographics), autonomous driving (robust perception systems), finance (models robust to market regime shifts), and fairness-critical applications (robustness to demographic shifts). The integration with PEFT makes the approach potentially accessible even with limited computational resources.
*   **Contribution to the Workshop:** This proposal directly addresses central themes of the workshop, particularly the challenges of adaptation under distribution shifts and the pretraining-to-downstream gap. The results and insights generated will provide concrete contributions to the discussion on methods, evaluations, and the fundamental understanding of robustness in foundation models, fostering further research in this crucial area.

## 4. References (Implicit based on Literature Review)

1.  Zhou, A., Wang, J., Wang, Y.-X., & Wang, H. (2023). Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models. *arXiv preprint arXiv:2311.01441*.
2.  Kim, S., Ham, G., Cho, Y., & Kim, D. (2023). Robustness-Reinforced Knowledge Distillation with Correlation Distance and Network Pruning. *arXiv preprint arXiv:2311.13934*.
3.  Yang, Z., Pang, T., Feng, H., Wang, H., Chen, W., Zhu, M., & Liu, Q. (2024). Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning. *arXiv preprint arXiv:2402.13669*.
4.  Kumar, A., Raghunathan, A., Jones, R., Ma, T., & Liang, P. (2022). Fine-Tuning Can Distort Pretrained Features and Underperform Out-of-Distribution. *arXiv preprint arXiv:2201.10066*.
5.  Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. In *International Conference on Machine Learning* (pp. 8748-8763). PMLR.
6.  Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.
7.  Wortsman, M., Ilharco, G., Kim, J. W., Li, M., Kornblith, S., & Roelofs, R. (2021). Robust Fine-Tuning of Zero-Shot Models. *arXiv preprint arXiv:2109.01903*.
8.  Cuenca, P., & Paul, S. (2023). Parameter-Efficient Fine-Tuning Using PEFT. *arXiv preprint arXiv:2303.10130*.
9.  Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., ... & Liang, P. (2021). On the Opportunities and Risks of Foundation Models. *arXiv preprint arXiv:2108.07258*.
10. Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J., & Song, D. (2021). Natural Adversarial Examples. *arXiv preprint arXiv:1907.07174*. (Relevant foundational work on distribution shift robustness).
11. Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., Balsubramani, A., ... & Liang, P. (2021). WILDS: A Benchmark of in-the-Wild Distribution Shifts. In *International Conference on Machine Learning* (pp. 5637-5664). PMLR. (Provides key benchmarks).
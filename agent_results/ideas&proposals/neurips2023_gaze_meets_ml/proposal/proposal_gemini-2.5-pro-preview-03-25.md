Okay, here is a detailed research proposal based on the provided task description, research idea, and literature review.

---

**1. Title:** **GazeCon: Self-Supervised Contrastive Learning Guided by Expert Gaze for Feature Prioritization in Medical Image Analysis**

**2. Introduction**

**2.1 Background**
Medical image analysis is a cornerstone of modern diagnostics and treatment planning. Machine learning (ML), particularly deep learning, has shown remarkable promise in automating tasks like disease detection, classification, and segmentation across various modalities (e.g., X-ray, CT, MRI, histopathology). However, developing high-performing supervised ML models typically requires vast amounts of meticulously annotated data, often necessitating expert clinicians (e.g., radiologists, pathologists) to delineate regions of interest or provide diagnostic labels. This annotation process is time-consuming, expensive, and can suffer from inter-observer variability, creating a significant bottleneck for deploying AI in clinical settings.

To overcome the reliance on labeled data, self-supervised learning (SSL) has emerged as a powerful paradigm. SSL methods learn rich feature representations from unlabeled data by defining pretext tasks, such as predicting image rotations, solving jigsaw puzzles, or, more recently, leveraging contrastive learning objectives (e.g., SimCLR, MoCo, BYOL). Contrastive learning, in particular, trains models to pull representations of augmented views of the same instance closer while pushing representations of different instances further apart. While effective in learning general visual features, standard SSL approaches often lack domain-specific priors. In medical imaging, this means they may not inherently focus on the subtle, clinically relevant patterns that experts use for diagnosis, potentially limiting their downstream task performance, especially in identifying localized anomalies.

Eye-tracking technology offers a unique window into human cognition and visual attention. In radiology, a radiologist's gaze pattern while examining an image is not random; it reflects a dynamic process of hypothesis testing, information seeking, and feature identification critical to diagnosis. Fixations and saccades often correlate strongly with clinically significant regions. This rich, implicit information embedded in gaze data represents a largely untapped resource for guiding AI models. As highlighted by the "Gaze Meets ML" workshop, leveraging cost-efficient human supervision signals like eye-gaze can bridge the gap between human cognition and AI, equipping agents with the ability to mimic or predict human behavior and improve interpretability.

Recent works ([1, 2, 3] in the provided literature review) have started exploring the integration of gaze data into medical image AI. McGIP [1] uses gaze similarity to define positive pairs for contrastive learning. FocusContrast [2] uses gaze to guide data augmentation, preserving salient regions. GazeGNN [3] integrates raw gaze sequences directly into a graph neural network for classification. These studies demonstrate the potential of gaze but also highlight challenges, including data scarcity, gaze variability, and integration complexity [Literature Review: Key Challenges 1, 2, 3].

**2.2 Research Gap and Proposed Idea**
While existing methods use gaze for defining instance pairs [1] or guiding augmentation [2], there remains an opportunity to leverage gaze more directly to influence the *feature learning process itself* at a regional level within a self-supervised framework. Standard SSL treats all image regions implicitly equally during representation learning, whereas expert gaze explicitly prioritizes certain areas.

This proposal introduces **GazeCon (Gaze-guided Contrastive Self-Supervision)**, a novel self-supervised learning framework designed to explicitly prioritize clinically relevant features in medical images by leveraging radiologists' eye-gaze patterns. The core idea is to incorporate a gaze-guided contrastive objective that operates at the level of image regions or patches, complementing standard instance-level contrastive learning. Specifically, GazeCon will encourage the model encoder to learn representations where image regions frequently fixated upon by experts (high-gaze regions) have similar embeddings, while distinguishing them from regions that receive little or no expert attention (low-gaze regions). This approach uses readily available gaze data as weak supervision, aligning the model's learned feature space with expert diagnostic attention without requiring explicit annotations like bounding boxes or segmentation masks.

**2.3 Research Objectives**
The primary objectives of this research are:

1.  **Develop the GazeCon Framework:** Design and implement a self-supervised learning framework that integrates a novel gaze-guided regional contrastive loss with a standard instance-level contrastive loss (e.g., InfoNCE).
2.  **Incorporate Gaze Data:** Process raw eye-tracking data from radiologists viewing medical images (e.g., chest X-rays) into gaze density maps or fixation sequences suitable for guiding the regional contrastive loss.
3.  **Train and Evaluate GazeCon:** Pre-train deep learning models (e.g., ResNets, Vision Transformers) using GazeCon on large-scale unlabeled medical image datasets with corresponding gaze data. Evaluate the quality of the learned representations on downstream tasks such as anomaly detection, disease classification, and potentially segmentation, comparing against standard SSL methods and supervised baselines.
4.  **Assess Interpretability:** Analyze the attention mechanisms or feature importance maps derived from models trained with GazeCon. Quantitatively and qualitatively evaluate the alignment of model attention with expert gaze patterns and ground-truth lesion locations.
5.  **Investigate Robustness:** Study the impact of gaze data variability (e.g., different radiologists, varying data quality) and quantity on GazeCon's performance and explore strategies for mitigation.

**2.4 Significance**
This research holds significant potential contributions:

*   **Improved SSL for Medical Imaging:** By incorporating expert priors via gaze, GazeCon aims to learn more clinically relevant features than standard SSL, potentially leading to better performance on downstream diagnostic tasks, especially in low-data or subtle anomaly settings.
*   **Reduced Annotation Burden:** Leveraging gaze data as free weak supervision can significantly reduce the need for expensive expert annotations, accelerating the development and deployment of medical AI models.
*   **Enhanced Interpretability and Trust:** Aligning model attention with expert gaze patterns can make AI decisions more transparent and interpretable to clinicians, fostering greater trust and facilitating human-AI collaboration. The model effectively learns *where* to look based on expert behavior.
*   **Advancement in Gaze-Guided ML:** This work contributes to the growing field of gaze-assisted machine learning by proposing a novel mechanism for integrating gaze into the core representation learning process via regional contrastive objectives.
*   **Neuro-AI Synergy:** Provides a framework for studying the relationship between human visual attention strategies (captured by gaze) and the feature representations learned by deep neural networks, bridging neuroscience and AI.

**3. Methodology**

**3.1 Overall Framework: GazeCon**
The proposed GazeCon framework integrates two complementary contrastive learning objectives: a standard instance-level contrastive loss ($\mathcal{L}_{inst}$) and a novel gaze-guided regional contrastive loss ($\mathcal{L}_{gaze}$). The overall training objective is a weighted sum of these two losses:
$$\mathcal{L}_{total} = \mathcal{L}_{inst} + \lambda \mathcal{L}_{gaze}$$
where $\lambda$ is a hyperparameter balancing the contribution of the two losses.

The framework consists of:
*   A deep neural network backbone (encoder) $f(\cdot)$ (e.g., ResNet-50, ViT-B) that maps an input image $x$ to a representation vector.
*   A projection head $h(\cdot)$ (typically a small MLP) that maps the encoder's output to a lower-dimensional space where the contrastive loss is computed, following practices in SimCLR/MoCo.
*   A mechanism to process gaze data into a spatial guidance signal (e.g., a gaze density map).
*   The implementation of the two loss functions.

**3.2 Data Collection and Preprocessing**
*   **Datasets:** We plan to utilize publicly available medical image datasets that include corresponding eye-tracking data from radiologists. Key candidates include:
    *   **REFLACX (Reports and Eye-Tracking Labels for Chest X-ray Interpretation):** Contains chest X-rays (CXRs) from the MIMIC-CXR dataset with gaze recordings (fixations, timestamps) from multiple radiologists during diagnostic reporting. Also includes bounding boxes for abnormalities.
    *   **Eye Gaze Data on Chest X-rays (Associated with [1, 2]):** Datasets used in prior work, potentially accessible through collaboration or public release.
    *   If necessary and feasible, we may explore collecting additional gaze data under IRB approval or using synthetic gaze generation models trained on existing data as an augmentation strategy.
*   **Image Preprocessing:** Standard medical image preprocessing steps will be applied, including intensity normalization, resizing to a consistent input dimension (e.g., 224x224 or 512x512), and potentially data augmentation (random crop, flips, color jitter – carefully chosen as per [4]'s findings).
*   **Gaze Data Processing:** Raw gaze data (sequences of fixation points $(x, y, duration)$) associated with each image will be processed to generate spatial **Gaze Density Maps (GDMs)**. This typically involves:
    1.  Aggregating fixation points across time (and potentially across multiple radiologists viewing the same image).
    2.  Applying a Gaussian kernel centered at each fixation point, weighted by fixation duration, to create a smooth density map $G$. The kernel bandwidth $\sigma$ will be a hyperparameter.
    3.  Normalizing the GDM $G$ so its values represent relative attention density, e.g., ranging from 0 to 1.

**3.3 Gaze-Guided Contrastive Learning Mechanism**

*   **Instance-Level Contrastive Loss ($\mathcal{L}_{inst}$):**
    We will adopt a standard instance-level contrastive loss like InfoNCE, as used in SimCLR. For an input image $x_i$, two different augmented views $v_{i,1}$ and $v_{i,2}$ are generated. The encoder $f(\cdot)$ and projector $h(\cdot)$ compute embeddings $z_{i,1} = h(f(v_{i,1}))$ and $z_{i,2} = h(f(v_{i,2}))$. The loss for a positive pair $(z_{i,1}, z_{i,2})$ within a minibatch of size $N$ is:
    $$ \mathcal{L}_{inst}^{(i,1,2)} = -\log \frac{\exp(\text{sim}(z_{i,1}, z_{i,2}) / \tau)}{\sum_{j=1, j \neq i}^{N} \exp(\text{sim}(z_{i,1}, z_{j,2}) / \tau) + \exp(\text{sim}(z_{i,1}, z_{i,2}) / \tau)} $$
    where $\text{sim}(u, v) = u^T v / (||u|| ||v||)$ is the cosine similarity and $\tau$ is the temperature hyperparameter. The total instance loss is averaged over all positive pairs in the batch.

*   **Gaze-Guided Regional Contrastive Loss ($\mathcal{L}_{gaze}$):**
    This novel component operates on regions *within* an image, guided by the corresponding GDM $G$.
    1.  **Region Sampling:** For an input image $x$ (or one of its augmented views $v$), sample multiple overlapping or non-overlapping patches/regions $\{p_k\}$. The features for these patches can be obtained from an intermediate layer of the encoder $f(\cdot)$ or by passing cropped patches through the encoder. Let $e_k = f_{region}(p_k)$ be the embedding for patch $p_k$.
    2.  **Gaze Score Association:** For each patch $p_k$, calculate an average gaze score $g_k$ by averaging the corresponding GDM values $G(y, x)$ over the spatial extent of the patch.
    3.  **Positive/Negative Set Definition:** Define a threshold $\theta_g$ on the gaze scores. Patches with $g_k > \theta_g$ are considered "high-gaze" (attended) regions, and patches with $g_k < \theta_l$ (where $\theta_l$ might be equal to $\theta_g$ or lower) are "low-gaze" (unattended) regions.
    4.  **Regional Contrastive Objective:** For a given high-gaze anchor patch $p_a$ (with embedding $e_a$), other high-gaze patches $\{p_p\}$ from the *same* image form the positive set. Low-gaze patches $\{p_n\}$ from the same image, as well as potentially all patches from *other* images in the batch, form the negative set. A contrastive loss (e.g., InfoNCE formulation adapted for regions) is applied:
        $$ \mathcal{L}_{gaze}^{(a)} = -\sum_{p_p} \log \frac{\exp(\text{sim}(e_a, e_p) / \tau_r)}{\sum_{p_p} \exp(\text{sim}(e_a, e_p) / \tau_r) + \sum_{p_n} \exp(\text{sim}(e_a, e_n) / \tau_r)} $$
        where $\tau_r$ is a temperature parameter for the regional loss. The total $\mathcal{L}_{gaze}$ is averaged over all high-gaze anchor patches in the batch.
    *Alternative Formulation:* A simpler approach could involve maximizing the similarity between embeddings of all high-gaze patches within an image and minimizing their similarity to low-gaze patch embeddings, potentially using triplet loss or a simpler pairwise contrast.

**3.4 Algorithmic Steps (Training GazeCon)**

1.  Initialize encoder $f(\cdot)$ and projector $h(\cdot)$.
2.  Load a minibatch of $N$ images $\{x_i\}$ and their corresponding Gaze Density Maps $\{G_i\}$.
3.  For each image $x_i$:
    a.  Generate two augmented views $v_{i,1}, v_{i,2}$.
    b.  Compute instance embeddings $z_{i,1} = h(f(v_{i,1}))$ and $z_{i,2} = h(f(v_{i,2}))$.
    c.  Select one view (e.g., $v_{i,1}$) or the original image $x_i$.
    d.  Extract regional features/embeddings $\{e_{i,k}\}$ for patches $\{p_{i,k}\}$.
    e.  Determine gaze scores $\{g_{i,k}\}$ for each patch using $G_i$.
    f.  Identify high-gaze patches $\{p_{i,p}\}$ and low-gaze patches $\{p_{i,n}\}$.
4.  Compute $\mathcal{L}_{inst}$ using pairs $(z_{i,1}, z_{i,2})$ across the batch.
5.  Compute $\mathcal{L}_{gaze}$ using the regional embeddings $\{e_{i,k}\}$ and their gaze scores $\{g_{i,k}\}$ within each image (and potentially across images for negatives).
6.  Compute the total loss $\mathcal{L}_{total} = \mathcal{L}_{inst} + \lambda \mathcal{L}_{gaze}$.
7.  Perform backpropagation and update the parameters of $f(\cdot)$ and $h(\cdot)$.
8.  Repeat steps 2-7 for a desired number of epochs.

**3.5 Experimental Design and Validation**

*   **Datasets:** Primarily REFLACX (using MIMIC-CXR images).
*   **Baselines:**
    *   Supervised baseline: Train the same encoder architecture from scratch with available labels (e.g., disease classification labels from MIMIC-CXR).
    *   Standard SSL: SimCLR, MoCo v2/v3, BYOL trained on the same unlabeled images *without* gaze data.
    *   Existing Gaze-Guided SSL: Implementations of relevant parts of McGIP [1] and FocusContrast [2] for comparison, if feasible based on published details.
*   **Downstream Tasks & Evaluation:**
    *   **Linear Probing:** Freeze the pre-trained encoder $f(\cdot)$ and train a linear classifier on top for tasks like multi-label pathology classification (using MIMIC-CXR labels). Evaluate using micro/macro AUC, F1-score.
    *   **Fine-tuning:** Fine-tune the entire pre-trained model on the same classification tasks. Evaluate using the same metrics.
    *   **Anomaly Detection:** Evaluate the ability to distinguish abnormal from normal images (e.g., using KNN distance in the embedding space or training a simple classifier). Evaluate using AUC.
    *   **(Optional) Segmentation Transfer:** Use the pre-trained encoder as a backbone for a segmentation model (e.g., U-Net) and evaluate performance on tasks where segmentation masks are available (potentially using the bounding boxes in REFLACX as weak localization targets). Evaluate using Dice score, IoU.
*   **Interpretability Evaluation:**
    *   **Attention Map Generation:** Use techniques like Grad-CAM or attention rollout (for ViTs) to visualize model attention on test images.
    *   **Qualitative Assessment:** Visually compare generated attention maps with expert GDMs and ground-truth lesion locations (if available).
    *   **Quantitative Assessment:**
        *   **Pointersect:** Calculate the Intersection over Union (IoU) or Normalized Scanpath Saliency (NSS) between thresholded model attention maps and expert GDMs.
        *   **Correlation:** Compute Spearman or Pearson correlation between continuous model attention map values and GDM values.
        *   Compare these metrics for GazeCon vs. baseline SSL models.
*   **Ablation Studies:**
    *   **Impact of Gaze Loss:** Compare GazeCon ($\mathcal{L}_{inst} + \lambda \mathcal{L}_{gaze}$) vs. only instance loss ($\mathcal{L}_{inst}$, i.e., $\lambda=0$).
    *   **Impact of $\lambda$:** Vary the weighting factor $\lambda$ to understand the balance between instance and regional gaze guidance.
    *   **Regional Loss Formulation:** Compare different variants of $\mathcal{L}_{gaze}$ (e.g., InfoNCE vs. Triplet).
    *   **Gaze Data Quality:** If data permits, train GazeCon using gaze from single vs. multiple radiologists, or subsets of gaze data, to assess robustness to variability [Challenge 2] and data limitations [Challenge 1].
    *   **Encoder Architecture:** Test with both CNN (ResNet) and Transformer (ViT) backbones.

**3.6 Addressing Challenges**
*   **Data Availability [Challenge 1]:** Leverage existing public datasets like REFLACX. Acknowledge limitations and suggest future work involving broader datasets.
*   **Gaze Variability [Challenge 2]:** Aggregate gaze from multiple radiologists when available to create averaged GDMs. Investigate the impact of single vs. multiple observer gaze in ablations. The regional contrastive objective might inherently focus on commonly attended areas, potentially mitigating some variability.
*   **Integration Complexity [Challenge 3]:** Build upon well-established SSL frameworks (e.g., PyTorch implementations of SimCLR/MoCo) and carefully implement the additional regional loss component. Modular design will be key.
*   **Data Privacy [Challenge 4]:** Use anonymized public datasets (REFLACX is derived from MIMIC, which has undergone de-identification). Ensure compliance with data usage agreements. Avoid any attempt to re-identify radiologists from gaze patterns.
*   **Scalability [Challenge 5]:** Design the framework using standard deep learning libraries (PyTorch/TensorFlow) capable of handling large datasets. Employ efficient patch extraction and loss computation. Test scalability during implementation.

**4. Expected Outcomes & Impact**

**4.1 Expected Outcomes**
*   **Improved Representation Quality:** We expect GazeCon pre-trained models to outperform standard SSL baselines (SimCLR, MoCo) on downstream medical image analysis tasks (classification, potentially detection/segmentation), particularly when evaluated via linear probing and fine-tuning. The performance might approach, though likely not exceed, fully supervised levels.
*   **Enhanced Interpretability:** We anticipate that attention maps generated from GazeCon models will show significantly better alignment with radiologists' gaze patterns (higher Pointersect/Correlation scores) and ground-truth abnormalities compared to baseline SSL models. Visualizations should confirm that the model learns to focus on clinically relevant regions identified by experts.
*   **Demonstration of Gaze Benefit:** Ablation studies are expected to clearly demonstrate the positive contribution of the $\mathcal{L}_{gaze}$ component to both downstream performance and interpretability.
*   **Framework Release:** A well-documented implementation of the GazeCon framework will be made publicly available (subject to data usage permissions) to facilitate further research.
*   **Insights into Gaze-Feature Correlation:** The research will provide empirical evidence on how incorporating gaze patterns influences learned feature representations in deep networks for medical images.

**4.2 Potential Impact**
*   **Clinical AI Development:** GazeCon offers a pathway to develop more accurate and reliable medical AI systems with reduced reliance on costly manual annotations. By learning features prioritized by experts, these models may detect subtle signs of disease more effectively.
*   **Human-AI Collaboration:** Enhanced interpretability and alignment with clinical reasoning processes can foster greater trust in AI tools among clinicians, paving the way for more seamless human-AI collaboration in diagnostic workflows. For instance, an AI using GazeCon could highlight areas consistent with where experts typically look for certain pathologies.
*   **Advancing Self-Supervised Learning:** This work pushes the boundaries of SSL by demonstrating how implicit human cognitive signals (gaze) can be effectively integrated to instill domain-specific priors, offering a blueprint for similar approaches in other domains where expert attention data is available.
*   **Ethical Considerations:** While leveraging gaze, we remain mindful of privacy [Challenge 4]. By focusing on anonymized, aggregated gaze for improving model performance and interpretability rather than individual monitoring, we aim for responsible innovation. The alignment with expert behavior could also potentially mitigate certain biases if the expert pool is diverse, although this needs careful verification.
*   **Contribution to Gaze Meets ML Community:** This research directly addresses core themes of the workshop, including ML supervision with eye-gaze, attention mechanisms, understanding human intention (implicitly via gaze), using gaze for Computer Vision and Explainable AI, and applications in radiology.

**5. Conclusion**

The proposed GazeCon framework represents a novel approach to self-supervised learning in medical imaging, leveraging expert eye-gaze patterns to guide the model towards learning clinically relevant features. By combining standard instance-level contrastive learning with a unique gaze-guided regional contrastive objective, we aim to improve downstream task performance, enhance model interpretability, and reduce the dependence on expensive manual annotations. Through rigorous experimental validation and ablation studies, we expect to demonstrate the efficacy of GazeCon and contribute valuable insights to the intersection of machine learning, computer vision, cognitive science, and medical informatics. This research has the potential to significantly impact the development of trustworthy and effective AI tools for clinical practice.

---
*References are based on the provided literature review numbers.*
[1] Zhao, Z., Wang, S., Wang, Q., & Shen, D. (2023). Mining Gaze for Contrastive Learning toward Computer-Assisted Diagnosis. arXiv preprint arXiv:2312.06069.
[2] Wang, S., Zhuang, Z., Ouyang, X., Zhang, L., Li, Z., Ma, C., Liu, T., Shen, D., & Wang, Q. (2023). Learning Better Contrastive View from Radiologist's Gaze. arXiv preprint arXiv:2305.08826.
[3] Wang, B., Pan, H., Aboah, A., Zhang, Z., Keles, E., Torigian, D., Turkbey, B., Krupinski, E., Udupa, J., & Bagci, U. (2023). GazeGNN: A Gaze-Guided Graph Neural Network for Chest X-ray Classification. arXiv preprint arXiv:2305.18221.
[4] Cheng, Z., Li, B., Altmann, A., Keane, P. A., & Zhou, Y. (2025). Enhancing Contrastive Learning for Retinal Imaging via Adjusted Augmentation Scales. arXiv preprint arXiv:2501.02451. [Note: Year adjusted as per typical preprint practice/likely typo in source]
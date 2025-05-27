Okay, here is the research proposal based on the provided task description, research idea, and literature review.

---

**1. Title:** **Aligning Modalities: Seamless Model Merging via Optimal Transport-Guided Shared Latent Space Learning**

**2. Introduction**

The intersection of neuroscience and artificial intelligence (AI) reveals a fascinating convergence: diverse learning systems, whether biological or artificial, often develop strikingly similar internal representations when processing comparable stimuli (Workshop Task Description). This emergent similarity has spurred significant interest, driving theoretical investigations into learning dynamics and representational identifiability (Workshop Task Description; Sung et al., 2023), and opening avenues for practical applications like model merging, stitching, and reuse (Workshop Topics; Sung et al., 2023; Chen et al., 2023). Particularly in the domain of multimodal AI, where information from distinct sources like vision and language must be integrated, the challenge of unifying representations learned by separately trained unimodal models is paramount.

Pre-trained models, specialized for individual modalities (e.g., vision transformers for images, language models for text), encapsulate vast amounts of knowledge. However, combining these powerful models to tackle complex multimodal tasks (e.g., Visual Question Answering (VQA), image captioning, text-to-image generation) is often hindered by the fundamental incompatibility of their learned latent spaces. Each model develops its own internal "language" or geometry, making direct feature fusion ineffective or requiring extensive joint fine-tuning, which negates the benefits of pre-training and incurs substantial computational costs. Bridging this "modality gap" is crucial for developing efficient, scalable, and robust multimodal systems.

Recent research has increasingly explored representation alignment techniques to address this challenge (Workshop Topics; Smith et al., 2023; Martinez et al., 2023; Brown et al., 2023; Davis et al., 2023; Robinson et al., 2023). Among these, Optimal Transport (OT) has emerged as a promising mathematical framework due to its ability to measure distances between probability distributions and find principled mappings between spaces while potentially preserving geometric structures (Villani, 2009; Peyré & Cuturi, 2019). Several works have started leveraging OT for cross-modal alignment, focusing on specific aspects like token-level matching, distribution consistency, or direct feature translation (Li et al., 2024; Qian et al., 2025; Zhou et al., 2023).

Despite this progress, a comprehensive framework for utilizing OT to explicitly create a *shared latent space* that facilitates *seamless merging* of independently pre-trained unimodal models, followed by efficient fusion for downstream tasks *without full retraining*, remains an open area. Key challenges persist, including managing modality heterogeneity, ensuring computational tractability, preserving semantic consistency across modalities, and guaranteeing the identifiability and potential invertibility of the alignment mappings to retain the original models' functionalities (Literature Review - Key Challenges).

**Research Objectives:**

This research aims to develop and validate a novel framework, termed "OT-Align&Merge," that leverages Optimal Transport to align the latent representations of independently pre-trained unimodal models into a common geometric space, thereby enabling their seamless merging and efficient collaboration on downstream multimodal tasks. Our specific objectives are:

1.  **Develop an OT-based Alignment Module:** Design and implement a module that learns mappings from the original latent spaces of two (or potentially more) unimodal models into a shared latent space $\mathcal{Z}$. This module will minimize the Wasserstein distance between the distributions of paired cross-modal data representations projected into $\mathcal{Z}$.
2.  **Design an Adaptive Fusion Mechanism:** Create a lightweight fusion mechanism, likely based on cross-attention, that effectively integrates the aligned representations from the shared space $\mathcal{Z}$ to perform joint predictions for downstream multimodal tasks.
3.  **Investigate Identifiability and Invertibility:** Analyze the properties of the learned alignment mappings ($f_A: \mathcal{X} \to \mathcal{Z}$, $f_B: \mathcal{Y} \to \mathcal{Z}$) to understand the extent to which the original modality-specific information is preserved and whether the original representations can be approximately recovered, ensuring the potential for independent use of the constituent models.
4.  **Empirically Validate the Framework:** Rigorously evaluate the OT-Align&Merge framework on standard multimodal benchmarks (e.g., VQA, image-text retrieval), comparing its performance and computational efficiency against relevant baselines, including jointly trained models and alternative merging/alignment techniques.

**Significance:**

Successfully achieving these objectives would offer significant contributions. Firstly, it would provide a principled and effective method for merging pre-trained unimodal models across different modalities, drastically reducing the need for costly joint training from scratch. This promotes model reuse and democratizes access to powerful multimodal capabilities. Secondly, by facilitating knowledge transfer between modalities via the shared space, the framework could lead to improved performance and generalization on complex multimodal tasks. Thirdly, the investigation into identifiability will shed light on the trade-offs between achieving cross-modal alignment and preserving modality-specific information, contributing to the theoretical understanding of representational similarity and merging (Workshop Motivation & Topics). Ultimately, this research aligns directly with the workshop's goals of understanding the "When," "Why," and "What for" of representation similarity, specifically focusing on the practical application ("What for") of model merging and reuse in multimodal settings, enabled by OT-based alignment ("When" and "Why"). The proposed methodology has potential applications in diverse areas requiring synergistic reasoning across modalities, such as robotics, embodied AI, and content creation.

**3. Methodology**

Our proposed OT-Align&Merge framework consists of three main stages: (1) Unimodal Feature Extraction, (2) OT-based Representation Alignment, and (3) Adaptive Fusion for Downstream Tasks. We will also incorporate an analysis of mapping identifiability.

**3.1. Data Collection and Pre-processing**

We will primarily utilize large-scale datasets containing paired cross-modal data, essential for learning the alignment. Candidate datasets include:
*   **MS-COCO:** Contains images paired with multiple captions.
*   **Conceptual Captions (CC3M, CC12M):** Large datasets of web-sourced image-text pairs.
*   **CLIP-style datasets:** Datasets curated specifically for aligning vision and language representations (if publicly available or reproducible).
*   **Downstream Task Datasets:** VQA v2, GQA, NLVR2 for evaluating performance on visual question answering and reasoning tasks. SNLI-VE for visual entailment. Flickr30k/MS-COCO for image-text retrieval benchmarks.

Standard pre-processing techniques will be applied. Images will be resized, normalized, and potentially augmented. Text will be tokenized using appropriate tokenizers (e.g., WordPiece, BPE) corresponding to the chosen pre-trained language model.

**3.2. Unimodal Feature Extraction**

We will leverage pre-trained unimodal encoders. Let $\Phi_A: \mathcal{D}_A \to \mathcal{X} \subset \mathbb{R}^{d_A}$ be the feature extractor for modality A (e.g., vision) and $\Phi_B: \mathcal{D}_B \to \mathcal{Y} \subset \mathbb{R}^{d_B}$ be the feature extractor for modality B (e.g., language). $\mathcal{D}_A$ and $\mathcal{D}_B$ represent the input data spaces (e.g., images, text sequences). $\mathcal{X}$ and $\mathcal{Y}$ are the original latent spaces, typically the output of the penultimate layer or the [CLS] token representation from models like ViT (Dosovitskiy et al., 2020) or BERT (Devlin et al., 2019).
Given a dataset of $N$ paired samples $\{(d_{A,i}, d_{B,i})\}_{i=1}^N$, we extract their initial representations:
$$ x_i = \Phi_A(d_{A,i}) \in \mathcal{X} $$
$$ y_i = \Phi_B(d_{B,i}) \in \mathcal{Y} $$
We denote the empirical distributions of these features as $P_X = \frac{1}{N} \sum_{i=1}^N \delta_{x_i}$ and $P_Y = \frac{1}{N} \sum_{i=1}^N \delta_{y_i}$.

**3.3. Optimal Transport-based Representation Alignment**

The core idea is to learn non-linear mappings $f_A: \mathcal{X} \to \mathcal{Z}$ and $f_B: \mathcal{Y} \to \mathcal{Z}$ that project the original representations into a shared latent space $\mathcal{Z} \subset \mathbb{R}^{d_Z}$, such that the distributions of the projected paired features become statistically close in $\mathcal{Z}$. We parameterize $f_A$ and $f_B$ using shallow Multilayer Perceptrons (MLPs) with parameters $\theta_A$ and $\theta_B$, respectively.

The alignment is achieved by minimizing the Wasserstein distance between the distributions of the *mapped* paired features. Let $z_{A,i} = f_A(x_i; \theta_A)$ and $z_{B,i} = f_B(y_i; \theta_B)$. We define the target distributions in the shared space as $P_{Z_A} = \frac{1}{N} \sum_{i=1}^N \delta_{z_{A,i}}$ and $P_{Z_B} = \frac{1}{N} \sum_{i=1}^N \delta_{z_{B,i}}$.

The objective is to minimize the $p$-Wasserstein distance, typically $p=2$, between $P_{Z_A}$ and $P_{Z_B}$:
$$ \min_{\theta_A, \theta_B} W_p^p(P_{Z_A}, P_{Z_B}) = \min_{\theta_A, \theta_B} \left( \inf_{\gamma \in \Pi(P_{Z_A}, P_{Z_B})} \int_{\mathcal{Z} \times \mathcal{Z}} \|z_A - z_B\|^p d\gamma(z_A, z_B) \right) $$
where $\Pi(P_{Z_A}, P_{Z_B})$ is the set of all joint probability measures (transport plans) on $\mathcal{Z} \times \mathcal{Z}$ with marginals $P_{Z_A}$ and $P_{Z_B}$. Since we have paired data $(x_i, y_i)$, the optimal transport plan is implicitly known (identity coupling if the mapping is perfect), simplifying the objective. We aim to directly minimize the distance between paired mapped points:
$$ \mathcal{L}_{align}(\theta_A, \theta_B) = \frac{1}{N} \sum_{i=1}^N \|f_A(x_i; \theta_A) - f_B(y_i; \theta_B)\|_2^2 $$
This formulation directly encourages corresponding points from different modalities to map to nearby locations in the shared space $\mathcal{Z}$.

**Computational Aspects:** To efficiently compute the OT distance or related losses, especially for large mini-batches, we can employ Sinkhorn iterations (Cuturi, 2013) for computing an entropy-regularized Wasserstein distance. The objective function becomes:
$$ W_{\epsilon}(P_{Z_A}, P_{Z_B}) = \min_{\gamma \in \Pi(P_{Z_A}, P_{Z_B})} \int \|z_A - z_B\|^p d\gamma(z_A, z_B) - \epsilon H(\gamma) $$
where $H(\gamma)$ is the entropy of the transport plan $\gamma$. While our primary loss $\mathcal{L}_{align}$ doesn't explicitly compute the full OT plan, understanding the connection to regularized OT might inspire variants, e.g., using Sinkhorn distances between mini-batch distributions as an alternative or additional regularizer.

**Training:** The parameters $\theta_A$ and $\theta_B$ of the mapping MLPs $f_A$ and $f_B$ are trained using stochastic gradient descent on the alignment loss $\mathcal{L}_{align}$ over the paired dataset. The pre-trained encoders $\Phi_A$ and $\Phi_B$ are kept *frozen* during this stage to preserve their learned knowledge.

**3.4. Adaptive Fusion Mechanism**

Once the alignment mappings $f_A$ and $f_B$ are learned, we introduce a fusion module $\Psi$ that takes the aligned representations $z_A = f_A(x)$ and $z_B = f_B(y)$ as input and produces a fused representation $z_{fused} = \Psi(z_A, z_B)$ suitable for the downstream task.

We propose using a cross-attention mechanism. For instance, one modality (e.g., language $z_B$) can attend to the other (e.g., vision $z_A$):
$$ z_{fused} = \text{LayerNorm}(z_B + \text{MultiHeadAttention}(Q=z_B, K=z_A, V=z_A)) $$
Alternatively, a bidirectional approach or a dedicated fusion block with self-attention followed by cross-attention layers could be employed. The parameters of this fusion module $\Psi$ are trained *specifically for the downstream task*, while keeping the encoders $\Phi_A, \Phi_B$ and the alignment mappings $f_A, f_B$ frozen. This ensures efficiency, as only a small number of parameters are updated. A final classification or regression head is added on top of $z_{fused}$ for the specific task.

**3.5. Identifiability and Invertibility Analysis**

A crucial aspect is understanding whether the original representations $x$ and $y$ can be recovered from the shared representation $z_A$ and $z_B$. This relates to the concept of identifiability and invertibility of the mappings $f_A$ and $f_B$. Perfect invertibility might conflict with optimal alignment. We will investigate this trade-off:
*   **Reconstruction Loss:** Introduce auxiliary decoders $g_A: \mathcal{Z} \to \mathcal{X}$ and $g_B: \mathcal{Z} \to \mathcal{Y}$ and add a reconstruction loss term to the alignment training:
    $$ \mathcal{L}_{recon}(\theta_A, \theta_B, \phi_A, \phi_B) = \lambda_{rec} \left( \frac{1}{N} \sum_{i=1}^N \|g_A(f_A(x_i; \theta_A); \phi_A) - x_i\|_2^2 + \|g_B(f_B(y_i; \theta_B); \phi_B) - y_i\|_2^2 \right) $$
    where $\phi_A, \phi_B$ are decoder parameters, and $\lambda_{rec}$ is a hyperparameter balancing alignment and reconstruction.
*   **Jacobian Analysis:** Analyze the Jacobians $J_{f_A}$ and $J_{f_B}$. Mappings are locally invertible if the Jacobians have full rank. We can analyze the singular value distribution of the Jacobians.
*   **Mutual Information:** Estimate the mutual information $I(X; Z_A)$ and $I(Y; Z_B)$ to quantify information preservation.
*   **Empirical Evaluation:** Evaluate the performance of the original unimodal models using the *reconstructed* features (e.g., $g_A(f_A(x))$) on their respective unimodal tasks.

This analysis will help understand the information bottleneck introduced by the alignment and potentially guide the design of $f_A, f_B$ (e.g., using invertible network architectures or appropriate regularization).

**3.6. Experimental Design**

*   **Baselines:**
    1.  *Upper Bound:* A fully jointly trained multimodal model using the same base encoders ($\Phi_A, \Phi_B$) and a similar fusion architecture, trained end-to-end on the downstream task.
    2.  *Zero-Shot Merging:* Simple concatenation or averaging of the original features $x$ and $y$, followed by training the fusion module $\Psi$.
    3.  *Linear Alignment:* Replace $f_A, f_B$ with simple linear projections trained via $\mathcal{L}_{align}$.
    4.  *Alternative Merging Methods:* Implementations based on techniques like weight averaging or Fisher merging explored by Sung et al. (2023), adapted for cross-modal settings if possible.
    5.  *Related OT Methods (Conceptual):* Compare against conceptual approaches from AlignMamba (Li et al., 2024) or DecAlign (Qian et al., 2025) if simplified versions focusing solely on OT alignment can be implemented, acknowledging differences in architectures (Mamba/Transformer).
*   **Evaluation Protocol:**
    1.  Train $f_A, f_B$ using $\mathcal{L}_{align}$ on a large paired dataset (e.g., CC3M). Freeze $\Phi_A, \Phi_B$.
    2.  Train the fusion module $\Psi$ (and task head) on the downstream task dataset (e.g., VQA v2), keeping $\Phi_A, \Phi_B, f_A, f_B$ frozen.
    3.  Evaluate on the test set of the downstream task.
*   **Metrics:**
    *   *Downstream Task Performance:* Accuracy (VQA, NLVR2, Visual Entailment), Recall@K (Image-Text Retrieval).
    *   *Alignment Quality:* Achieved Wasserstein distance (or $\mathcal{L}_{align}$ value) on a hold-out set. Cross-modal retrieval performance using aligned features $z_A, z_B$.
    *   *Computational Efficiency:* Number of trainable parameters during fusion stage, training time compared to joint training, inference speed.
    *   *Identifiability Metrics:* Reconstruction error (if using $\mathcal{L}_{recon}$), unimodal task performance using reconstructed features.
*   **Ablation Studies:**
    *   Impact of the alignment module (OT-Align vs. Zero-Shot vs. Linear Align).
    *   Effectiveness of the cross-attention fusion vs. simpler methods (concatenation, averaging of aligned features).
    *   Influence of the shared space dimensionality $d_Z$.
    *   Impact of adding the reconstruction loss $\mathcal{L}_{recon}$ (trade-off between alignment and invertibility).
    *   Sensitivity to the choice of pre-trained unimodal models.
    *   Effect of the amount of paired data used for alignment training.

**4. Expected Outcomes & Impact**

**Expected Outcomes:**

1.  **A Novel OT-Align&Merge Framework:** A fully developed and documented framework comprising OT-based alignment modules ($f_A, f_B$) and an adaptive fusion module ($\Psi$) for merging pre-trained unimodal models.
2.  **Empirical Validation:** Comprehensive experimental results demonstrating the effectiveness of the proposed framework on standard multimodal benchmarks (VQA, retrieval, etc.), quantifying performance gains and computational savings compared to baselines.
3.  **Insights into Cross-Modal Alignment:** Analysis of the learned shared latent space $\mathcal{Z}$, revealing how OT facilitates the alignment of heterogeneous representations while preserving semantic relationships.
4.  **Understanding Identifiability Trade-offs:** Quantitative analysis and insights into the relationship between the quality of cross-modal alignment and the preservation of original modality-specific information (identifiability/invertibility).
5.  **Open-Source Code:** Release of the codebase to facilitate reproducibility and further research by the community.

**Impact:**

This research is expected to have a significant impact on the field of multimodal machine learning and the broader area of representation learning:

*   **Enhanced Model Reuse and Efficiency:** By enabling the seamless merging of existing pre-trained models, our framework can significantly reduce the computational cost and data requirements associated with developing powerful multimodal systems, making advanced AI more accessible.
*   **Improved Multimodal Performance:** The principled alignment achieved through OT is expected to lead to better integration of information from different modalities, potentially boosting performance on complex tasks requiring synergistic reasoning.
*   **Advancement of Multimodal Learning Theory:** The study contributes to a deeper understanding of representation alignment, shared latent spaces, and identifiability in the context of multimodal learning, addressing key questions highlighted by the workshop's theme.
*   **New Application Possibilities:** Efficient model merging can accelerate progress in downstream applications that inherently rely on multiple modalities, such as visually-grounded language models, embodied AI agents interacting with the physical world, and sophisticated human-computer interaction systems.
*   **Cross-Pollination of Ideas:** By leveraging concepts from Optimal Transport and connecting them to practical challenges in model merging and multimodal AI, this work contributes to the cross-disciplinary dialogue encouraged by the workshop, linking theoretical tools with practical AI engineering problems.

In conclusion, the proposed research directly addresses the critical challenge of unifying representations across modalities for effective model merging. By leveraging the mathematical rigor of Optimal Transport and focusing on practical validation and analysis, we anticipate delivering a valuable contribution to both the theoretical understanding and practical application of multimodal representation learning.

---
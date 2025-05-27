Okay, here is the research proposal based on the provided task description, idea, and literature review.

---

## **1. Title:** Cross-Domain Representational Alignment via Domain-Invariant Feature Learning

## **2. Introduction**

### 2.1 Background

Understanding the nature of intelligence, whether biological or artificial, hinges on deciphering the representations systems use to process information, reason about the world, and guide actions. A central challenge in cognitive science, neuroscience, and machine learning is comparing these representations across different systems (Sucholutsky et al., 2023). Are the computations underlying object recognition in the primate visual cortex similar to those in a Convolutional Neural Network (CNN)? Do Large Language Models (LLMs) process language in ways analogous to the human brain? Answering such questions requires robust methods for quantifying **representational alignment** – the degree to which the geometric or statistical structure of representational spaces is similar across systems.

Current approaches to measuring representational alignment, such as Representational Similarity Analysis (RSA) (Kriegeskorte et al., 2008) or Centered Kernel Alignment (CKA) (Kornblith et al., 2019), have yielded valuable insights, particularly within specific domains (e.g., comparing different layers within a deep network or across different brain regions). However, their effectiveness often diminishes when comparing representations across fundamentally different domains, such as between neural recordings (fMRI, ECoG) and artificial neural network (ANN) activations. These cross-domain comparisons are hampered by significant differences in data modalities (e.g., blood-oxygen-level-dependent signals vs. unit activations), dimensionality, signal-to-noise ratios, underlying data distributions, and the lack of explicit one-to-one correspondence between components (e.g., neurons vs. artificial units). This limits our ability to draw meaningful conclusions about shared computational principles or functional equivalence.

The Workshop on Representational Alignment (Re-Align) highlights the pressing need for methods that can bridge this gap. Key questions revolve around developing more generalizable measures of alignment, understanding when and why systems learn aligned representations, and exploring how alignment can be systematically manipulated. Addressing these questions requires moving beyond metrics that assume homogeneity in the representational formats.

### 2.2 Research Objectives

This research proposes a novel framework to address the limitations of current alignment techniques by focusing on learning **domain-invariant feature spaces**. The core idea is that if representations from disparate systems ($S$ and $T$) can be projected into a shared latent space $\mathcal{Z}$ such that domain-specific characteristics are factored out while functionally relevant structures are preserved, then alignment measured within $\mathcal{Z}$ will provide a more robust and meaningful comparison.

The primary objectives of this research are:

1.  **To develop a computational framework based on domain adaptation techniques (specifically leveraging adversarial and contrastive learning) to learn transformations that map representations from heterogeneous sources (e.g., biological, artificial) into a common, domain-invariant latent space.** This framework aims to explicitly handle differences in modality, scale, and distribution.
2.  **To define and validate a novel representational alignment metric computed within this learned invariant space.** This metric should reflect functional similarity rather than superficial statistical correlations sensitive to domain-specific properties.
3.  **To systematically evaluate the proposed framework and alignment metric across diverse cross-domain pairings,** including visual representations (primate visual cortex vs. CNNs/Transformers) and language representations (human neural recordings during language tasks vs. LLM activations).
4.  **To investigate the relationship between the proposed alignment metric and behavioral congruence.** Specifically, we will test the hypothesis that higher alignment scores in the invariant space predict greater similarity in task performance or error patterns between the systems being compared.
5.  **To explore the potential of this framework for systematically influencing alignment,** for instance, by using the learned invariant space as a target for training AI models or interpreting neural data.

### 2.3 Significance

This research directly addresses central questions posed by the Re-Align workshop. By developing a domain-agnostic alignment framework, we aim to:

*   **Advance Measurement Approaches:** Provide a more robust and generalizable method for quantifying alignment across diverse systems and data types, overcoming limitations of current metrics.
*   **Uncover Shared Computational Strategies:** Facilitate more meaningful comparisons between biological and artificial intelligence, potentially revealing universal principles of information processing that transcend specific implementations. Higher alignment in the invariant space could more reliably indicate shared computational strategies.
*   **Enable Systematic Intervention:** The framework could offer mechanisms to *increase* alignment (e.g., by regularizing ANN training towards a biologically-informed invariant space) or *decrease* it, allowing for controlled experiments on the functional consequences of representational similarity.
*   **Broader Implications:** Understanding and manipulating cross-domain alignment has potential implications for designing AI systems that better mimic or interface with biological systems (e.g., improved brain-computer interfaces), for value alignment by identifying shared representational substrates for goals or concepts, and for developing more interpretable AI models by comparing their internal workings to human cognition. Progress in this area contributes to a unified science of intelligence bridging machine learning, neuroscience, and cognitive science.

## 3. Methodology

### 3.1 Conceptual Framework

Let $\mathcal{X}_S$ and $\mathcal{X}_T$ be the original representational spaces of a source system $S$ (e.g., fMRI voxel activations) and a target system $T$ (e.g., ANN layer activations) in response to a common set of stimuli or task conditions. Our goal is to learn mapping functions $f_S: \mathcal{X}_S \to \mathcal{Z}$ and $f_T: \mathcal{X}_T \to \mathcal{Z}$ that project these representations into a shared latent space $\mathcal{Z}$ of dimension $d$. The key property of $\mathcal{Z}$ is **domain invariance**: the distribution of projected representations $P_{\mathcal{Z}}(z | \text{domain}=S)$ should be indistinguishable from $P_{\mathcal{Z}}(z | \text{domain}=T)$, while preserving the information relevant for the underlying task or stimuli discrimination. Alignment is then measured between the projected representations $z_S = f_S(x_S)$ and $z_T = f_T(x_T)$ within $\mathcal{Z}$.

### 3.2 Data Collection and Selection

We will leverage existing and potentially collect new datasets that provide paired representations from biological and artificial systems responding to the same stimuli or performing analogous tasks. Examples include:

*   **Vision:**
    *   **Biological:** Primate electrophysiology data (e.g., V1, V4, IT) or human fMRI/MEG data responding to natural images (e.g., datasets from Kriegeskorte lab, THINGS database (Hebart et al., 2019), Algonauts Project ( compréhension et al., 2021)).
    *   **Artificial:** Layer activations from pre-trained CNNs (e.g., ResNet, VGG) and Vision Transformers (ViT) processing the same images. We will focus on layers hypothesized to correspond functionally to different stages of visual processing.
*   **Language:**
    *   **Biological:** Human fMRI or ECoG data recorded while participants read sentences, process narratives, or perform specific linguistic tasks (e.g., datasets from Fedorenko lab, Pereira et al. (2018)).
    *   **Artificial:** Layer activations from pre-trained LLMs (e.g., BERT, GPT variants, T5) processing the same text stimuli or performing analogous NLP tasks (e.g., sentiment analysis, question answering).

Representations will be extracted appropriately for each modality (e.g., time-averaged fMRI response patterns for specific stimuli/conditions, activation vectors from specific ANN layers). We will ensure a sufficient number of shared stimuli/conditions ($N$) to estimate the representational geometry reliably. Let $X_S \in \mathbb{R}^{N \times D_S}$ and $X_T \in \mathbb{R}^{N \times D_T}$ denote the representation matrices for $N$ stimuli in the source and target domains, with dimensions $D_S$ and $D_T$, respectively.

### 3.3 Algorithmic Approach: Hybrid Contrastive-Adversarial Domain Adaptation

We propose a hybrid approach inspired by recent advances in unsupervised domain adaptation, particularly combining adversarial and contrastive learning principles (Yadav et al., 2023; Wang et al., 2021; Thota & Leontidis, 2021). This combination allows leveraging the strengths of both: adversarial learning for global domain alignment and contrastive learning for preserving fine-grained, class-conditional structure.

Let $E_S$ and $E_T$ be feature extractors (e.g., neural networks) that implement the mappings $f_S$ and $f_T$, mapping from $\mathcal{X}_S$ and $\mathcal{X}_T$ to $\mathcal{Z}$. These extractors might share some parameters initially or be trained jointly.

**1. Adversarial Domain Alignment:**
A domain discriminator $D$ is trained to distinguish between projected representations $z_S = E_S(x_S)$ and $z_T = E_T(x_T)$. The feature extractors $E_S$ and $E_T$ are trained concurrently to *fool* the discriminator, thereby encouraging the distributions $P(z_S)$ and $P(z_T)$ in the latent space $\mathcal{Z}$ to become similar. This can be formulated as a minimax game:
$$
\min_{E_S, E_T} \max_{D} \mathcal{L}_{adv}(E_S, E_T, D) = \mathbb{E}_{x_S \sim X_S} [\log D(E_S(x_S))] + \mathbb{E}_{x_T \sim X_T} [\log(1 - D(E_T(x_T)))]
$$
This addresses the global domain shift but may not ensure alignment of semantically similar points across domains, potentially leading to misalignment near class boundaries (addressing Challenge 2 partially).

**2. Cross-Domain Contrastive Alignment:**
To enforce finer-grained alignment based on semantic similarity (assuming stimuli belong to implicit or explicit classes/conditions), we employ a cross-domain contrastive loss. For an anchor representation $z_i$ from one domain, we want to pull it closer to positive representations $z_j^+$ from the *other* domain that correspond to the same stimulus or class, while pushing it away from negative representations $z_k^-$ from both domains corresponding to different stimuli/classes.

Let $z_i^S = E_S(x_i^S)$ and $z_j^T = E_T(x_j^T)$ be projected representations for stimuli $i$ and $j$ from source and target domains, respectively. Sim($u, v$) denotes a similarity function (e.g., cosine similarity). The contrastive loss for an anchor $z_i^S$ can be defined similar to InfoNCE:
$$
\mathcal{L}_{con}(z_i^S) = - \log \frac{\sum_{j: \text{same class}} \exp(\text{Sim}(z_i^S, z_j^T) / \tau)}{\sum_{j: \text{same class}} \exp(\text{Sim}(z_i^S, z_j^T) / \tau) + \sum_{k: \text{diff class}} (\exp(\text{Sim}(z_i^S, z_k^T) / \tau) + \exp(\text{Sim}(z_i^S, z_k^S) / \tau)) }
$$
where $\tau$ is a temperature hyperparameter. A symmetric loss $\mathcal{L}_{con}(z_j^T)$ is defined for anchors from the target domain. The total contrastive loss is $\mathcal{L}_{con} = \mathbb{E}[\mathcal{L}_{con}(z^S)] + \mathbb{E}[\mathcal{L}_{con}(z^T)]$.

*   **Handling Lack of Labels (Challenge 3):** In purely unsupervised scenarios (no class labels), "positives" can be defined as representations corresponding to the *same stimulus* across domains ($i=j$). If comparing representations averaged over stimulus classes, pseudo-labeling via clustering in the latent space (as in Wang et al., 2021) can be employed to identify likely positive and negative pairs dynamically.
*   **Handling False Negatives (Challenge 4):** We will explore techniques like removing potentially similar negatives based on feature similarity (Thota & Leontidis, 2021) or using more sophisticated sampling strategies to mitigate the impact of false negatives.
*   **Handling Modality Differences (Challenge 1):** The use of separate encoders $E_S$ and $E_T$ (potentially with different architectures tailored to input modalities, e.g., 1D CNN for fMRI time series, MLP for static activations) coupled with the domain adaptation objective specifically addresses the modality issue.

**3. Combined Objective:**
The overall training objective combines the adversarial and contrastive losses:
$$
\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_{adv} \mathcal{L}_{adv} + \lambda_{con} \mathcal{L}_{con}
$$
where $\mathcal{L}_{task}$ is an optional task-specific loss (if available, e.g., classification loss on source domain data or a reconstruction loss to ensure information preservation), and $\lambda_{adv}, \lambda_{con}$ are hyperparameters balancing the contributions of the alignment terms. We may adopt a two-stage training approach as in CDA (Yadav et al., 2023), first focusing on adversarial alignment then refining with contrastive learning.

**4. Alignment Metric in $\mathcal{Z}$:**
Once the mappings $f_S=E_S$ and $f_T=E_T$ are learned, we obtain the projected representation matrices $Z_S \in \mathbb{R}^{N \times d}$ and $Z_T \in \mathbb{R}^{N \times d}$. We can then compute alignment using standard metrics within this shared space $\mathcal{Z}$. We propose using **CKA (Centered Kernel Alignment)** due to its invariance to isotropic scaling and orthogonality, making it suitable for comparing potentially rotated/scaled representations within $\mathcal{Z}$.
$$
\text{CKA}(Z_S, Z_T) = \frac{\text{HSIC}(K_S, K_T)}{\sqrt{\text{HSIC}(K_S, K_S) \text{HSIC}(K_T, K_T)}}
$$
where $K_S = Z_S Z_S^T$ and $K_T = Z_T Z_T^T$ are centered Gram matrices (using a linear kernel here, but other kernels are possible), and HSIC is the Hilbert-Schmidt Independence Criterion. Alternatively, Procrustes analysis or RSA computed on $Z_S$ and $Z_T$ can also be used and compared.

### 3.4 Experimental Design and Validation

**1. Datasets and Tasks:** We will use the datasets mentioned in 3.2 (e.g., THINGS, Algonauts, Pereira et al. data) paired with standard pre-trained ANN models (ResNets, ViTs, BERT, GPT variants).

**2. Baselines:**
    *   **Direct Alignment:** Calculate CKA, RSA, and linear Procrustes distance directly between the original representations $X_S$ and $X_T$ (after appropriate pre-processing like dimensionality reduction via PCA if needed).
    *   **Alternative DA Methods:** Implement and compare with alignment scores obtained using only adversarial DA ($\lambda_{con}=0$) and only contrastive DA ($\lambda_{adv}=0$). We may also compare with canonical correlation analysis (CCA) based methods if applicable.

**3. Evaluation Metrics:**
    *   **Alignment Score:** The proposed CKA score (or alternative) calculated in the learned latent space $\mathcal{Z}$. Higher scores indicate better alignment.
    *   **Behavioral/Functional Congruence:** This is a critical validation step. We will measure the correlation between the Alignment Score and metrics of behavioral similarity.
        *   *For vision:* Correlate alignment score with the similarity in classification accuracy patterns across image categories, or similarity in confusion matrices between the biological system (e.g., human psychophysics data if available) and the ANN.
        *   *For language:* Correlate alignment score with similarity in performance on downstream tasks (e.g., next word prediction accuracy, question answering scores), or similarity in error patterns (e.g., types of grammatical errors, semantic confusions). We can measure behavioral similarity using metrics like correlation of accuracy vectors across stimuli/tasks, or distance between error distributions.
    *   **Domain Discriminator Accuracy:** During training, monitor the accuracy of the domain discriminator $D$. Lower accuracy (approaching chance level) indicates successful domain confusion / invariant feature learning.
    *   **Qualitative Analysis:** Visualize the latent space $\mathcal{Z}$ using techniques like t-SNE or UMAP, coloring points by domain and stimulus class/condition, to visually inspect the degree of domain mixing and semantic clustering.

**4. Ablation Studies:**
    *   Vary the dimension $d$ of the latent space $\mathcal{Z}$.
    *   Analyze the effect of hyperparameters $\lambda_{adv}$ and $\lambda_{con}$.
    *   Compare different contrastive sampling strategies or pseudo-labeling techniques.
    *   Evaluate different choices for the feature extractors $E_S$ and $E_T$.

**5. Scalability and Generalization (Challenge 5):** We will evaluate the computational cost of the proposed method with increasing data size ($N$) and representation dimensions ($D_S, D_T$). We will also test generalization by training the framework on one set of stimuli/tasks and evaluating alignment on a held-out set.

### 3.5 Addressing Implementation Challenges

*   **Hyperparameter Tuning:** Extensive tuning of $\lambda_{adv}, \lambda_{con}, \tau$, learning rates, and network architectures will be required, likely using a validation set of stimuli.
*   **Optimization Stability:** Training adversarial networks can be unstable. Techniques like gradient penalty (WGAN-GP) or spectral normalization might be needed for the discriminator.
*   **Meaningful Alignment:** Ensuring that the learned alignment is functionally meaningful, not just statistically convenient, is paramount. The correlation with behavioral congruence is the primary check for this.

## 4. Expected Outcomes & Impact

**Expected Outcomes:**

1.  **A Novel Framework (Code & Publication):** A publicly available software implementation of the proposed hybrid contrastive-adversarial domain adaptation framework for learning invariant feature spaces from heterogeneous representational data.
2.  **A Validated Cross-Domain Alignment Metric:** A robust metric (e.g., CKA applied in $\mathcal{Z}$) demonstrated to quantify representational alignment across diverse pairs of biological and artificial systems, overcoming modality and scale differences.
3.  **Empirical Alignment Results:** Quantitative alignment scores for specific pairings (e.g., primate IT vs. CNN layers for object recognition, human language cortex vs. LLM layers for sentence processing). These results will provide concrete data on the degree of representational similarity under various conditions.
4.  **Evidence for Alignment-Behavior Link:** Empirical evidence demonstrating the extent to which the proposed alignment metric predicts behavioral congruence (task performance similarity, shared error patterns) between the systems being compared. This would validate the functional relevance of the learned invariant space.
5.  **Insights into Conserved Representations:** Analysis of the structure of the learned invariant space $\mathcal{Z}$ may reveal which representational features are preserved across domains, shedding light on potentially universal computational strategies employed by different intelligences.
6.  **Demonstration of Alignment Manipulation:** Preliminary results showing how the framework could be used to guide ANN training towards higher (or lower) alignment with biological data by using the learned mapping $f_S$ or the invariant space $\mathcal{Z}$ as a target or regularizer.

**Impact:**

This research is expected to make significant contributions aligned with the goals of the Re-Align workshop:

*   **Improved Measurement:** Provides a direct answer to the need for "more robust and generalizable measures of alignment that work across different domains and types of representations."
*   **Understanding Shared Computation:** Offers a more principled way to address "to what extent does representational alignment indicate shared computational strategies," by factoring out domain-specific noise and focusing on functional similarity in the invariant space.
*   **Enabling Intervention:** Directly addresses "how can we systematically increase (or decrease) representational alignment," by providing a framework where the mapping to the invariant space can be used as a target for intervention (e.g., guiding model training).
*   **Investigating Implications:** By linking the proposed alignment metric to behavioral congruence, the research contributes to understanding "the implications (positive and negative) of increasing or decreasing representational alignment... on behavioral alignment."
*   **Interdisciplinary Bridge:** This work fosters collaboration between ML, neuroscience, and cognitive science by providing common ground and tools for comparing representations across their respective systems of study.
*   **Potential Applications:** Beyond fundamental science, the ability to align representations across domains could benefit brain-computer interface design (mapping neural signals to intelligible representations), AI safety (understanding and aligning model representations with human values/concepts), and transfer learning (leveraging insights from biological systems to improve AI).

In summary, this research proposes a theoretically grounded and empirically driven approach to tackle the critical challenge of cross-domain representational alignment. By learning domain-invariant feature spaces, we aim to provide deeper insights into the shared principles of computation across biological and artificial intelligence, paving the way for a more unified understanding of intelligent systems.

---
**References:**

*   compréhension, R. A. C., et al. (2021). The Algonauts Project 2021 Challenge: Object Decoding from Human Brain Activity. *arXiv preprint arXiv:2109.09704*.
*   Hebart, M. N., et al. (2019). THINGS: A database of 1,854 object concepts and more than 26,000 naturalistic object images. *PLOS ONE*, 14(10), e0223792.
*   Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. *Proceedings of the 36th International Conference on Machine Learning (ICML)*.
*   Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational similarity analysis - connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2, 4.
*   Liu, W., Ferstl, D., Schulter, S., Zebedin, L., Fua, P., & Leistner, C. (2021). Domain Adaptation for Semantic Segmentation via Patch-Wise Contrastive Learning. *arXiv preprint arXiv:2104.11056*.
*   Pereira, F., Lou, B., Pritchett, B., Ritter, S., Gershman, S. J., Kanwisher, N., ... & Fedorenko, E. (2018). Toward a universal decoder of linguistic meaning from brain activation patterns. *Nature Communications*, 9(1), 963.
*   Sucholutsky, I., et al. (2023). Getting aligned on representational alignment. *arXiv preprint arXiv:2310.13017*.
*   Thota, M., & Leontidis, G. (2021). Contrastive Domain Adaptation. *arXiv preprint arXiv:2103.15566*.
*   Wang, R., Wu, Z., Weng, Z., Chen, J., Qi, G. J., & Jiang, Y. G. (2021). Cross-domain Contrastive Learning for Unsupervised Domain Adaptation. *arXiv preprint arXiv:2106.05528*.
*   Yadav, N., Alam, M., Farahat, A., Ghosh, D., Gupta, C., & Ganguly, A. R. (2023). CDA: Contrastive-adversarial Domain Adaptation. *arXiv preprint arXiv:2301.03826*.
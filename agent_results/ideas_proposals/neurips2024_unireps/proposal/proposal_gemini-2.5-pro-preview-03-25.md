Okay, here is a detailed research proposal based on the provided task description, research idea, and literature review.

---

**1. Title:** **Task-Conditioned Functional Alignment for Efficient Cross-Architecture Model Merging**

**2. Introduction**

**2.1 Background**
The proliferation of large-scale pre-trained neural models across various domains (e.g., computer vision, natural language processing) has revolutionized machine learning capabilities. However, training these models from scratch is computationally exorbitant and environmentally demanding. This motivates the exploration of techniques to reuse and combine existing pre-trained models effectively. Model merging, the process of combining parameters or representations from two or more pre-trained models into a single, functional model, emerges as a promising strategy to leverage collective knowledge, reduce training costs, and potentially enhance performance or robustness (Matena & Raffel, 2022; Wortsman et al., 2022).

Despite its potential, current model merging techniques often face significant limitations. Simple methods like parameter averaging (averaging weights element-wise) typically require models to have identical architectures and often fail when models are trained on even slightly different tasks or data distributions (Ainsworth et al., 2022). More sophisticated methods attempt to find permutations or transformations in parameter space, but these can be computationally complex and still struggle with architectural heterogeneity.

Concurrently, a fundamental question is arising in both machine learning and neuroscience, as highlighted by the "Workshop on Unifying Representations in Neural Models": *When, how, and why do different neural systems (biological or artificial) learn similar internal representations when exposed to similar stimuli or tasked with similar goals?* Recent findings suggest that functional similarities often emerge despite underlying structural differences (Insulla et al., 2025; Ziyin et al., 2024). Understanding this phenomenon is not only crucial for theoretical insights into learning dynamics and representation formation but also holds the key to practical advancements like robust model merging, stitching, and cross-modal learning. If different models converge towards functionally similar representations for specific tasks or concepts, exploiting this convergence could enable more effective merging strategies that operate at the level of functional activation spaces rather than brittle parameter spaces.

**2.2 Research Problem and Proposed Solution**
The core research problem we address is the difficulty of merging pre-trained neural models that possess *different architectures* or are trained on *related but distinct task distributions*. Existing methods predominantly focus on parameter-space alignment, which is inherently sensitive to architectural mismatches and often fails to capture functional equivalence.

We propose a novel approach termed **Task-Conditioned Functional Alignment (TCFA)**. Instead of aligning parameters directly, TCFA focuses on aligning the *activation spaces* of different models. Crucially, this alignment is *conditioned on specific, shared task properties*. The central hypothesis is that even if models differ architecturally or were trained slightly differently, the activation manifolds corresponding to the same underlying task condition (e.g., recognizing the same object class, processing text with similar sentiment) within specific layers might exhibit alignable structures. TCFA aims to identify these shared functional structures and learn minimal, lightweight transformations (acting as "stitching" layers) to map between them. By focusing on functional equivalence relevant to downstream tasks, we anticipate that TCFA can facilitate effective merging across heterogeneous models with significantly fewer trainable parameters compared to full fine-tuning or more complex parameter-space alignment techniques.

**2.3 Research Objectives**
This research aims to achieve the following objectives:

1.  **Develop the TCFA Algorithm:** Formulate and implement the TCFA methodology, including defining task conditions, strategies for probing activations, and learning alignment transformations using techniques like Optimal Transport (OT) or Canonical Correlation Analysis (CCA) variants.
2.  **Investigate Conditions for Functional Alignment:** Empirically study the factors influencing the success of TCFA, such as the degree of architectural difference, the type and granularity of task conditioning, the choice of layers for alignment, and the similarity of the source models' training data/objectives.
3.  **Evaluate TCFA for Model Merging:** Assess the effectiveness of TCFA in merging diverse pre-trained models (e.g., different CNN architectures like ResNet and EfficientNet, or CNNs and Vision Transformers) on downstream tasks. Compare performance against baseline merging techniques and standard fine-tuning.
4.  **Analyze Efficiency and Representation Quality:** Quantify the computational efficiency of TCFA (e.g., number of trainable parameters in the stitch layer, merging time) and analyze the quality of representations in the merged model using metrics like Centered Kernel Alignment (CKA).

**2.4 Significance**
This research holds significant potential contributions:

*   **Practical:** TCFA could provide a computationally efficient method for reusing and combining diverse pre-trained models, drastically reducing the need for full retraining or fine-tuning, saving significant computational resources and energy. This enables practitioners to leverage a wider range of existing models.
*   **Theoretical:** By investigating *when* functional alignment occurs under specific task conditions despite architectural differences, this work contributes to the fundamental understanding of representation learning in neural networks, directly addressing the core questions of the "Unifying Representations in Neural Models" workshop. It explores the emergence of potentially canonical computations (Ziyin et al., 2024) tied to task semantics rather than specific network implementations.
*   **Methodological:** Developing robust functional alignment techniques advances the toolkit for model analysis, comparison, and integration, potentially benefiting areas beyond merging, such as transfer learning, multi-task learning, and interpretability. It connects to the learning-theoretic perspective on stitching and alignment discussed by Insulla et al. (2025).

**3. Methodology**

**3.1 Overall Research Design**
Our research will follow a structured empirical approach. We will select pairs of pre-trained models with varying degrees of architectural and task-distribution differences. We will then apply the TCFA algorithm to learn alignment transformations between their activation spaces, conditioned on specific task properties. These transformations will be used to merge the models. The performance of the merged models will be rigorously evaluated on downstream tasks and compared against relevant baselines. Ablation studies will be conducted to understand the impact of different methodological choices.

**3.2 Data Collection and Preparation**
We will primarily use publicly available datasets and pre-trained models to ensure reproducibility.

*   **Datasets:**
    *   **Source Task Data (for probing):** We will use datasets like ImageNet-1k (Deng et al., 2009) for vision models and potentially subsets of GLUE (Wang et al., 2018) or similar benchmarks for NLP models (if extending). To create task conditions, we will utilize subsets of the data based on specific properties:
        *   *Class-Conditioning:* Select input samples belonging to specific classes (e.g., activations for 'dog' images vs. 'cat' images).
        *   *Transformation-Conditioning:* Apply specific data augmentations or corruptions (e.g., rotations, noise levels from ImageNet-C (Hendrycks & Dietterich, 2019), styles from ImageNet-Sketch (Wang et al., 2019)) to probe robustness and invariance.
    *   **Downstream Task Data (for evaluation):** We will evaluate merged models on standard downstream tasks, such as image classification accuracy on ImageNet validation set, or transfer learning performance on datasets like CIFAR-100 (Krizhevsky, 2009), Food-101 (Bossard et al., 2014), or relevant NLP classification tasks.

**3.3 Model Selection**
We will select pairs of pre-trained models ($M_1, M_2$) exhibiting diversity:

*   **Architecture:** Include models with significant architectural differences, e.g.,
    *   ResNet family (e.g., ResNet-18 vs. ResNet-50) (He et al., 2016)
    *   CNN vs. Transformer (e.g., ResNet-50 vs. ViT-Base) (Dosovitskiy et al., 2020)
    *   Different CNN families (e.g., ResNet vs. EfficientNet) (Tan & Le, 2019)
*   **Training:** Models pre-trained on ImageNet-1k, potentially with slight variations in training protocols or initializations if available through standard model hubs (e.g., PyTorch Hub, Hugging Face). We may also consider models trained on related but distinct large datasets if applicable.

**3.4 Task-Conditioned Functional Alignment (TCFA) Algorithm**

1.  **Layer Selection:** Identify pairs of layers ($l_1$ in $M_1$, $l_2$ in $M_2$) where functional alignment is expected or desired. These could be intermediate feature extraction layers or final embedding layers before the classification head. Let the dimensionality of activations at these layers be $d_1$ and $d_2$, respectively.

2.  **Task Conditioning and Probing:** Define a set of relevant task conditions $C = \{c_1, c_2, ..., c_k\}$. For each condition $c \in C$, select a representative set of input samples $X_c$ (e.g., images from a specific class, images with a specific augmentation).

3.  **Activation Extraction:** For each condition $c$, pass the inputs $X_c$ through both models $M_1$ and $M_2$ up to the selected layers $l_1$ and $l_2$. Extract the corresponding activation vectors, forming sets of activation matrices $A_1^c \in \mathbb{R}^{n_c \times d_1}$ and $A_2^c \in \mathbb{R}^{n_c \times d_2}$, where $n_c = |X_c|$.

4.  **Learning the Alignment Transformation:** The core of TCFA is to find a transformation $T: \mathbb{R}^{d_1} \rightarrow \mathbb{R}^{d_2}$ that aligns activations $A_1^c$ with $A_2^c$ *simultaneously* or *on average* across relevant conditions $c \in C$. We will explore two main families of methods:

    *   **a) Optimal Transport (OT) based Alignment:**
        *   **Goal:** Find a mapping $T$ such that the transformed activations $T(A_1^c)$ are "close" to $A_2^c$ in distribution, for relevant $c$.
        *   **Approach:** We can frame this as finding a map $T$ that minimizes the Wasserstein distance between the empirical distributions defined by $T(A_1^c)$ and $A_2^c$. For simplicity and computational tractability, we might approximate this by learning a linear map $T \in \mathbb{R}^{d_2 \times d_1}$ (or an affine map) that minimizes a squared Euclidean distance, potentially weighted by OT coupling or based on aligning centroids:
            $$ \min_{T} \sum_{c \in C} w_c \sum_{i=1}^{n_c} \| T(a_{1,i}^c) - a_{2,i}^c \|^2_2 + \lambda \Omega(T) $$
            where $a_{1,i}^c$ is the $i$-th row of $A_1^c$, $a_{2,i}^c$ is the corresponding activation in $A_2^c$, $w_c$ are weights for each condition (e.g., uniform or based on condition importance), and $\Omega(T)$ is a regularization term (e.g., Frobenius norm $\|T\|_F^2$ or a low-rank constraint). More sophisticated OT approaches involving Gromov-Wasserstein distance (comparing intra-distribution distances) could also be explored if simple alignment fails.

    *   **b) Canonical Correlation Analysis (CCA) / Subspace Alignment based:**
        *   **Goal:** Find linear projections that maximize the correlation between the activation spaces, conditioned on tasks.
        *   **Approach:** For each condition $c$, compute covariance matrices $\Sigma_{11}^c$, $\Sigma_{22}^c$, and cross-covariance $\Sigma_{12}^c$. Find projection matrices $W_1 \in \mathbb{R}^{k \times d_1}$ and $W_2 \in \mathbb{R}^{k \times d_2}$ (where $k \le \min(d_1, d_2)$ is the dimensionality of the shared subspace) that maximize the average canonical correlations across conditions:
            $$ \max_{W_1, W_2} \sum_{c \in C} w_c \cdot \text{trace}( \text{corr}( W_1 A_1^c, W_2 A_2^c ) ) $$
            subject to constraints (e.g., $W_1 \Sigma_{11}^c W_1^T = I$, $W_2 \Sigma_{22}^c W_2^T = I$). The alignment transformation $T$ can then be derived from $W_1$ and $W_2$, e.g., by finding $T$ that minimizes $\| T A_1^c W_1^T - A_2^c W_2^T \|^2_F$ across conditions, potentially mapping to the canonical space and back, or more simply $T \approx W_2^\dagger W_1$ if using a simplified Procrustes-like alignment aiming to match $W_1 A_1^c \approx W_2 A_2^c$.

5.  **Parameterization of T:** The transformation $T$ will be parameterized as a lightweight neural network layer (the "stitching layer"), e.g., a single linear layer, possibly with low-rank factorization, or a small MLP.

**3.5 Model Merging Procedure**
Given two models $M_1$ (ending at layer $l_1$) and $M_2$ (starting from layer $l_2+1$), and the learned transformation $T$ mapping activations from $l_1$ to the space of $l_2$, the merged model $M_{merged}$ is constructed by composing $M_1$ (up to $l_1$), the stitch layer $T$, and $M_2$ (from $l_2+1$ onwards).

*   **Training/Fine-tuning:** We will investigate different strategies:
    *   *Zero-shot merge:* Use $T$ directly without further training.
    *   *Stitch tuning:* Freeze $M_1$ and $M_2$, and only fine-tune the parameters of the stitch layer $T$ on a small amount of downstream task data.
    *   *Partial fine-tuning:* Fine-tune $T$ and a small number of subsequent layers in $M_2$.

**3.6 Experimental Design and Evaluation**

*   **Baselines:**
    *   *Naive Parameter Averaging:* (Where applicable, i.e., identical architectures) Average weights of $M_1$ and $M_2$.
    *   *Linear Stitching:* A baseline alignment method using a simple linear regression or probe trained on paired activations without task conditioning.
    *   *Full Fine-tuning:* Fine-tune $M_1$ (or $M_2$) entirely on the downstream task data.
    *   *Individual Model Performance:* Performance of $M_1$ and $M_2$ (potentially fine-tuned) on the downstream task.
    *   *Existing Merging Methods:* Compare against relevant methods like Git Re-Basin (Ainsworth et al., 2022) or Activation Matching (Li et al., 2023, if applicable to cross-architecture).

*   **Evaluation Tasks:** Primarily image classification on held-out data (e.g., ImageNet validation) and transfer learning performance on standard benchmarks (e.g., CIFAR-100, Food-101).

*   **Evaluation Metrics:**
    *   *Performance:* Accuracy, F1-score, etc., on downstream tasks.
    *   *Efficiency:* Number of trainable parameters added/tuned during merging, computational cost (FLOPs) of merging and inference.
    *   *Representation Similarity:* Use metrics like Centered Kernel Alignment (CKA) (Kornblith et al., 2019) or Representational Similarity Analysis (RSA) (Kriegeskorte et al., 2008) to analyze the similarity of representations *before* and *after* alignment via $T$, and within the merged model.
    *   *Robustness:* Evaluate performance on out-of-distribution or corrupted data (e.g., ImageNet-C).

*   **Ablation Studies:**
    *   *Impact of Layer Choice:* Compare alignment quality and downstream performance when aligning different layers (early, middle, late).
    *   *Impact of Alignment Method:* Compare OT-based vs. CCA-based alignment.
    *   *Impact of Task Conditioning:* Compare TCFA with alignment learned without task conditioning, and vary the type/number of conditions ($k$).
    *   *Impact of Architectural Difference:* Systematically vary the architectural similarity between $M_1$ and $M_2$.
    *   *Impact of Stitch Tuning Strategy:* Compare zero-shot, stitch tuning, and partial fine-tuning.

**4. Expected Outcomes & Impact**

**4.1 Expected Outcomes**

1.  **A Novel TCFA Algorithm:** A well-defined and implemented algorithm for Task-Conditioned Functional Alignment, capable of finding transformations between activation spaces of heterogeneous models based on task semantics.
2.  **Empirical Validation:** Demonstration of TCFA's effectiveness in merging diverse pre-trained models (e.g., ResNet-ViT merge), achieving competitive or superior performance compared to baselines, particularly in cross-architecture scenarios.
3.  **Efficiency Gains:** Quantitative evidence showing that TCFA requires significantly fewer trainable parameters and potentially less tuning data/time compared to full fine-tuning while achieving comparable or better performance than naive merging techniques.
4.  **Insights into Functional Alignment:** Identification of factors (layer depth, architectural similarity, task condition granularity, alignment method) that govern the success of functional alignment. This analysis will shed light on *when* and *how* different networks develop functionally equivalent representations for specific task aspects.
5.  **Representation Analysis:** Characterization of the learned representations within the merged model using CKA/RSA, providing insights into how TCFA preserves or integrates information from the source models.
6.  **Benchmarks and Codebase:** (Potentially) Publicly released code implementing TCFA and benchmark results on standard model pairs and tasks, facilitating further research.

**4.2 Impact**

*   **Practical Impact:** TCFA could significantly lower the barrier to reusing and combining powerful pre-trained models, democratizing access to state-of-the-art AI capabilities by reducing computational costs. It enables flexible model combination previously difficult due to architectural constraints, potentially leading to novel applications by merging specialized models.
*   **Scientific Impact:** This research directly contributes to the central theme of the "Unifying Representations in Neural Models" workshop by providing empirical evidence and a methodological framework for understanding functional representation similarity conditioned on tasks. It offers insights into the principles governing representation learning, potentially revealing invariances that emerge across different learning systems (Lehalleur et al., 2025). While focused on AI models, the concept of task-conditioned functional alignment may offer conceptual parallels to how biological neural circuits might coordinate or transfer information despite structural variation.
*   **Future Research Directions:** This work can spur further research into more sophisticated alignment techniques, dynamic or adaptive alignment methods, applications to multi-modal models, and theoretical analyses linking functional alignment to generalization and robustness. The insights gained could inform the design of more efficient and inherently alignable future architectures.

By focusing on task-conditioned functional equivalence, TCFA offers a promising path towards truly leveraging the collective knowledge embedded in the vast ecosystem of pre-trained models, pushing the boundaries of both practical AI engineering and our fundamental understanding of learned representations.

---
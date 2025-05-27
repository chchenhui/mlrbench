**1. Title:**

Causal Diffusion Models: Learning Disentangled Latent Causal Factors for Controllable Generation

**2. Introduction**

*   **Background:** Deep generative models, particularly diffusion models (Ho et al., 2020; Song & Ermon, 2020), have achieved state-of-the-art performance in synthesizing high-fidelity data across various domains, including images, audio, and text. These models excel at learning complex data distributions by progressively adding noise to data samples and then learning to reverse this process. However, their remarkable generative capabilities often stem from capturing intricate statistical correlations present in the training data, without necessarily understanding the underlying causal mechanisms that generated the data. This reliance on correlation rather than causation can lead to several critical limitations:
    *   **Spurious Correlations:** Models may learn associations that are statistically prevalent but causally meaningless, leading to outputs that fail or behave unexpectedly under distributional shifts or interventions.
    *   **Lack of Controllability:** While conditional diffusion models allow generation based on certain labels or inputs, precise control over specific, semantically meaningful factors (especially those corresponding to underlying causal variables) remains challenging. Users cannot easily manipulate one aspect of the generated data (e.g., disease severity in a medical scan) while keeping others (e.g., patient anatomy) constant.
    *   **Trustworthiness and Interpretability:** In high-stakes applications like healthcare, finance, or autonomous systems, the inability to explain *why* a model generates a specific output or to guarantee that it relies on causally sound reasoning severely undermines trust and hinders deployment. Algorithmic bias, often rooted in spurious correlations involving sensitive attributes, is another significant concern.

    Concurrently, the field of Causal Representation Learning (CRL) has emerged (Schölkopf et al., 2021) aiming to discover latent variables that not only represent the data effectively but also correspond to underlying causal factors. The goal is to learn disentangled representations where changes in individual latent variables correspond to independent causal mechanisms in the real world. Integrating CRL principles with powerful generative frameworks like diffusion models holds immense potential for overcoming the aforementioned limitations. Existing works like DeCaFlow (Almodóvar et al., 2025) and CausalBGM (Liu & Wong, 2025) explore causal generative modeling, often focusing on VAEs or Bayesian approaches and tackling confounding, while C2VAE (Zhao et al., 2024) explicitly aims for property control in VAEs using causal structures. However, a dedicated framework that embeds causal graphical structures directly within the latent dynamics of high-fidelity diffusion models is still largely unexplored territory. Such integration could unlock unprecedented levels of control and interpretability in generative AI. The survey by Komanduri et al. (2023) highlights the growing interest and need for bridging deep generative models with structural causal models (SCMs).

*   **Research Objectives:** This research aims to develop and evaluate **Causal Diffusion Models (CDMs)**, a novel class of generative models that integrates causal discovery and reasoning within the diffusion process framework. Our primary objectives are:
    1.  **Develop the CDM Framework:** To formally define and implement a diffusion model architecture where the latent space is explicitly structured according to a learned causal graph.
    2.  **Integrate Latent Causal Discovery:** To incorporate or develop methods for discovering causal relationships (represented as a Directed Acyclic Graph - DAG) among latent variables derived during the diffusion/denoising process, potentially leveraging observational data, interventional data (if available), or domain knowledge constraints.
    3.  **Enable Causal Conditioning:** To modify the denoising process of the diffusion model such that it respects the learned causal graph structure, enabling fine-grained control over the generation process by intervening on specific latent causal factors.
    4.  **Achieve Controllable Counterfactual Generation:** To empower CDMs to generate counterfactual samples (e.g., "What would this image look like if causal factor Z were different?") in a way that is faithful to the intervention while minimally affecting causally independent factors.
    5.  **Evaluate Robustness and Interpretability:** To empirically demonstrate that CDMs exhibit improved robustness to spurious correlations and offer enhanced interpretability compared to standard diffusion models and other relevant baselines, particularly in contexts requiring trustworthy generation.

*   **Significance:** The successful development of CDMs would represent a significant advancement in both generative modeling and causal representation learning.
    *   **Enhanced Trustworthiness:** By grounding generation in causal structures, CDMs can mitigate reliance on spurious correlations, leading to more reliable and robust models, especially crucial for applications in healthcare (e.g., generating synthetic medical images reflecting causal disease progression), finance, and fairness-aware AI.
    *   **Precise Control and Editability:** CDMs would offer a powerful interface for users to control the generative process based on meaningful causal factors, enabling targeted data augmentation, scenario simulation, and intuitive content creation tools. For instance, generating variations of a chest X-ray by manipulating only the "pneumonia severity" factor while preserving anatomical features.
    *   **Scientific Discovery:** CDMs could serve as tools for causal hypothesis testing. By learning causal graphs from data and generating counterfactuals, they might help scientists explore causal relationships in complex systems (e.g., biology, economics).
    *   **Bridging Generative Models and Causality:** This work contributes to the fundamental understanding of how to integrate causal principles into state-of-the-art deep learning architectures, potentially paving the way for Causal Foundation Models with improved reasoning capabilities.
    *   **Addressing Key Challenges:** This research directly tackles several key challenges identified in the literature (Komanduri et al., 2023), such as identifying latent causal variables, ensuring interpretability, and achieving robustness, within the powerful framework of diffusion models.

**3. Methodology**

Our proposed methodology involves developing the Causal Diffusion Model framework, designing the causal discovery module, integrating causal conditioning into the diffusion process, and establishing a rigorous experimental protocol for validation.

*   **Conceptual Framework:** A standard Denoising Diffusion Probabilistic Model (DDPM) defines a forward noising process $q(x_t | x_{t-1})$ that gradually adds Gaussian noise to data $x_0$ over $T$ steps, and learns a reverse denoising process $p_\theta(x_{t-1} | x_t)$ parameterized by a neural network $\epsilon_\theta$ (typically a U-Net) to reconstruct the data from noise. CDM extends this by assuming that the data $x$ is generated from underlying latent causal variables $z = (z_1, ..., z_d)$ structured according to a causal graph $\mathcal{G}$. We aim to learn this structure $\mathcal{G}$ and leverage it within the diffusion process. The core idea is to associate latent representations derived during the diffusion process with these causal variables $z$ and ensure the denoising steps respect the causal dependencies defined by $\mathcal{G}$.

*   **Data Collection and Preparation:** We will utilize a combination of datasets:
    1.  **Synthetic Datasets:** Datasets generated from known SCMs (e.g., simple graphs generating image patterns) where the ground truth causal graph $\mathcal{G}$ and latent variables $z$ are known. This allows for quantitative evaluation of causal discovery accuracy (e.g., Structural Hamming Distance - SHD) and disentanglement.
    2.  **Benchmark Datasets with Attribute Labels:** Datasets like CelebA-HQ, SynFace or others where attributes (e.g., hair color, glasses, expression) can serve as proxies for potential latent causal factors. While the true causal graph is unknown, we can evaluate controllable generation based on these attributes.
    3.  **Domain-Specific Datasets:** Datasets from target application areas, such as medical imaging (e.g., ChestX-ray14, TCGA pathology images) or physics simulations, where domain knowledge can provide weak supervision or constraints on the causal graph. We may explore generating synthetic interventional data if feasible (e.g., simulating the effect of a treatment on medical images based on domain expertise).

*   **Algorithmic Steps: Causal Diffusion Model (CDM)**

    1.  **Diffusion Backbone:** We start with a standard diffusion model backbone (e.g., DDPM or DDIM). The forward process remains unchanged:
        $$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$$
        The standard objective is to train a network $\epsilon_\theta(x_t, t)$ to predict the noise $\epsilon$ added at step $t$, minimizing the loss:
        $$L_{diffusion} = \mathbb{E}_{t, x_0, \epsilon} [ || \epsilon - \epsilon_\theta(x_t, t) ||^2 ]$$

    2.  **Latent Representation Extraction:** We need to define the latent variables $z$ upon which the causal graph $\mathcal{G}$ will operate. These can be derived from intermediate representations of the denoising network $\epsilon_\theta$. For instance, we can extract a feature map from a bottleneck layer of the U-Net at time $t$, $h_t = \text{Encoder}(x_t, t)$, and apply a mapping (e.g., pooling or MLP) to obtain a $d$-dimensional latent vector $z_t = g_\psi(h_t)$. The parameters $\psi$ could be learned jointly. Alternatively, a separate encoder could map $x_t$ (or $x_0$) to $z$. For simplicity, we initially consider $z$ that is relatively stable across time $t$, potentially derived from an initial encoding of $x_0$ or averaged features across time.

    3.  **Latent Causal Discovery Module:** This module aims to learn the causal DAG $\mathcal{G}$ over the latent variables $z = (z_1, ..., z_d)$. We will explore several approaches:
        *   **Score-based Methods:** Adapting methods like NOTEARS (Zheng et al., 2018) to the latent space. This involves parameterizing the graph structure (e.g., via adjacency matrix $A$) and optimizing a score function that encourages good fit to the data (e.g., minimizing prediction error under the implied factorization) regularized by an acyclicity constraint. A potential loss term could be:
            $$L_{causal\_score} = \mathcal{R}(z, A) + \lambda_{acyclic} h(A)$$
            where $\mathcal{R}$ is a data fit score (e.g., negative log-likelihood assuming linear SEMs or more complex non-linear models) and $h(A)$ enforces acyclicity (e.g., $h(A) = \text{tr}(e^{A \odot A}) - d$).
        *   **Constraint-based Methods:** Using principles from algorithms like PC or FCI, identifying conditional independencies among latent variables $z_i$ and building the graph accordingly. This might be harder to integrate directly into end-to-end training but could be used in an alternating optimization scheme.
        *   **Variational Inference Approaches:** Framing causal discovery as variational inference over graph structures, potentially incorporating prior knowledge.
        *   **Leveraging Interventions:** If interventional data (pairs of $(x, x')$ resulting from an intervention on some $z_k$) is available (even synthetically generated), it can strongly guide graph learning by penalizing structures inconsistent with observed intervention effects.
        The causal discovery can happen jointly during diffusion training or in a pre-training/alternating phase. We will add a causal loss term $L_{causal}$ to the overall objective.

    4.  **Causal Conditioning in the Reverse Process:** This is the crucial step where $\mathcal{G}$ influences generation. The denoising network $\epsilon_\theta$ needs to be conditioned not just on $x_t$ and $t$, but also implicitly or explicitly on the causal structure $\mathcal{G}$. We propose several mechanisms:
        *   **Causal Masking/Attention:** Modify the attention layers (e.g., self-attention in the U-Net) such that the computation for a latent variable $z_i$'s corresponding features primarily attends to features associated with its causal parents $Pa(z_i)$ in $\mathcal{G}$.
        *   **Graph Neural Network (GNN) Integration:** Use a GNN operating on the latent variables $z_t$ according to $\mathcal{G}$ to produce context vectors that modulate the denoising process. The denoising step becomes:
            $$\epsilon_\theta(x_t, t, \text{GNN}(z_t, \mathcal{G}))$$
        *   **Causally-Aware Loss:** Introduce a term that encourages the denoising step to respect causal factorization. For example, if we predict the denoised $x_0$ (as in some diffusion variants), denoted $\hat{x}_0$, we could encourage the latent representation $z(\hat{x}_0)$ derived from it to satisfy the conditional independencies implied by $\mathcal{G}$.
        *   **Direct Intervention Mechanism:** Design the architecture such that we can perform interventions $do(z_i = \alpha)$ during sampling by fixing the value corresponding to $z_i$ and letting the model propagate its effects according to $\mathcal{G}$ through the causally conditioned denoising steps.

    5.  **Joint Optimization:** The overall objective function combines the standard diffusion loss, the causal discovery loss, and potentially a term encouraging adherence to the causal structure during denoising ($L_{control}$):
        $$L_{total} = L_{diffusion} + \lambda_{causal} L_{causal} + \lambda_{control} L_{control}$$
        The hyperparameters $\lambda_{causal}$ and $\lambda_{control}$ balance the objectives of reconstruction fidelity, causal structure accuracy, and causal control during generation. Training proceeds end-to-end or using alternating optimization schedules.

*   **Experimental Design and Validation:**
    *   **Baselines:**
        *   Standard Diffusion Models (e.g., DDPM, DDIM).
        *   Conditional Diffusion Models (trained on attributes corresponding to target controllable factors).
        *   State-of-the-art CRL models based on VAEs/GANs (e.g., C2VAE, iVAE, potentially adapting DeCaFlow concepts if feasible).
        *   Ablation studies of CDM (e.g., without $L_{causal}$, without causal conditioning).
    *   **Evaluation Tasks & Metrics:**
        *   **Generative Quality:** Fréchet Inception Distance (FID), Inception Score (IS), Precision, Recall to assess the fidelity and diversity of generated samples.
        *   **Causal Discovery Accuracy (Synthetic Data):** Structural Hamming Distance (SHD), Structural Intervention Distance (SID) between the learned graph $\mathcal{G}$ and the ground truth graph.
        *   **Disentanglement Metrics:** Mutual Information Gap (MIG), SAP score, DCI score applied to the learned latent variables $z$, measuring statistical independence. *Crucially*, we need metrics for *causal* disentanglement: evaluate if intervening on $z_i$ primarily changes the feature corresponding to $z_i$ while leaving features corresponding to non-descendants of $z_i$ unchanged.
        *   **Controllable Generation:**
            *   *Task:* Intervene on a specific latent variable $z_i$ (identified or mapped to a known attribute) and generate samples.
            *   *Metrics:* Visual inspection by humans. Quantitative evaluation using classifiers trained to recognize the target attribute (e.g., accuracy of generated images having the intended attribute). Measure unintended changes in other attributes (e.g., evaluate attribute preservation for causally unrelated factors).
        *   **Counterfactual Generation:**
            *   *Task:* Given a sample $x$, generate a counterfactual $x'$ corresponding to an intervention $do(z_k = \alpha)$.
            *   *Metrics:* Visual faithfulness to the intervention. Minimal change principle: evaluate distance $d(x, x')$ focusing only on regions/features *not* causally downstream of $z_k$. Task-specific metrics (e.g., in medical imaging, does changing a "disease" factor generate a plausible counterfactual image judged by expert criteria?).
        *   **Robustness to Spurious Correlations:** Train models on datasets with known spurious correlations (e.g., attribute A always co-occurs with attribute B, but B is not caused by A). Test if CDM can generate samples where A is present but B is absent (by intervening on the latent corresponding to A, assuming the learned $\mathcal{G}$ correctly identifies them as non-causally linked or linked via a confounder). Compare failure rates against standard diffusion models.
        *   **Interpretability:** Analyze the learned graph $\mathcal{G}$ for semantic coherence (do learned links align with intuition or domain knowledge?). Visualize the effect of traversing latent variables corresponding to causal factors.

**4. Expected Outcomes & Impact**

*   **Expected Outcomes:**
    1.  **A Novel Causal Diffusion Model Framework:** A well-defined architecture and training procedure for CDMs, implemented and publicly released as code.
    2.  **Demonstrated Controllability:** Empirical results showcasing CDM's superior ability to control specific attributes/factors during generation compared to baseline diffusion models and other generative approaches, particularly for factors corresponding to identified latent causes.
    3.  **Faithful Counterfactual Generation:** Validation of CDM's capability to generate plausible counterfactuals via interventions on the latent causal space, satisfying minimal change constraints for unrelated factors.
    4.  **Improved Robustness:** Evidence that CDMs are less susceptible to spurious correlations present in the training data compared to standard diffusion models.
    5.  **Validated Causal Discovery:** On synthetic data, quantitative results confirming the model's ability to recover the ground truth latent causal graph. On real data, qualitative analysis of the learned graph's plausibility.
    6.  **Insights and Limitations:** A clear understanding of the strengths and weaknesses of the proposed approach, including sensitivity to hyperparameters, assumptions about the causal structure (e.g., acyclicity, sufficiency), scalability, and requirements for data (observational vs. interventional).

*   **Impact:**
    *   **Advancing Generative AI:** Push the boundary of generative modeling beyond correlation capture towards causal understanding, leading to more reliable, controllable, and trustworthy AI systems.
    *   **Enabling High-Stakes Applications:** Facilitate the use of generative models in sensitive domains like healthcare (e.g., generating realistic synthetic data that respects causal disease mechanisms for training diagnostic models, simulating treatment effects), finance (causal modeling of market factors), and fairness (mitigating bias by controlling generation based on causal effects of sensitive attributes).
    *   **New Tools for Science and Creativity:** Provide researchers with tools to explore causal hypotheses in complex datasets through simulation and counterfactual generation. Offer artists and designers more intuitive and powerful tools for content creation by manipulating underlying causal factors.
    *   **Foundation for Causal AI:** Contribute to the development of Causal Foundation Models by demonstrating how causal reasoning can be integrated into large-scale generative architectures.
    *   **Cross-Disciplinary Collaboration:** This research lies at the intersection of machine learning, causality, and specific application domains, fostering collaboration and aligning perfectly with the goals of the Causal Representation Learning Workshop.

This research holds the promise of significantly enhancing the capabilities and trustworthiness of generative AI, moving from models that merely mimic data distributions to models that understand and leverage the underlying causal processes.
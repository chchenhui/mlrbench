Okay, here is a research proposal based on the provided task description, idea, and literature review.

---

## **1. Title:** **Causal Discovery in Biological Systems through Active Learning-Guided Multi-Omics Perturbation Analysis**

---

## **2. Introduction**

**2.1 Background**

The translation of biological insights into effective therapeutics remains a significant challenge, largely due to our incomplete understanding of the complex causal mechanisms underlying disease pathophysiology (Tejada-Lapuerta et al., 2023). Traditional drug discovery often relies on identifying correlations between biological entities (genes, proteins) and disease states, derived primarily from observational data. However, correlation does not imply causation, and the presence of confounding variables frequently leads to spurious associations, contributing to the high failure rates observed in clinical trials. Inferring causal relationships is paramount for identifying effective drug targets – molecules whose perturbation yields a predictable and desirable downstream effect on disease phenotype.

Recent technological advances have generated an unprecedented wealth of high-dimensional biological data across multiple modalities, including genomics (bulk and single-cell RNA sequencing), proteomics, metabolomics, and spatial omics. Concurrently, powerful perturbation technologies like CRISPR-Cas9 gene editing and RNA interference (RNAi) allow researchers to systematically probe biological systems by introducing targeted interventions (Lopez et al., 2022). Integrating these rich, multimodal datasets with interventional data offers a powerful avenue for moving beyond correlation towards causal inference.

However, several key challenges hinder progress. Firstly, the sheer dimensionality and inherent noise in biological data make causal structure learning computationally demanding and statistically complex (Key Challenge 1). Secondly, effectively integrating information from disparate omics modalities to build a cohesive causal picture requires sophisticated modeling approaches (Key Challenge 2; Sun et al., 2024; White et al., 2023). Thirdly, perturbation experiments, while informative, are often resource-intensive (time, cost). Naively performing all possible perturbations is infeasible, necessitating intelligent experimental design strategies to maximize information gain (Key Challenge 4; Williams et al., 2023). Finally, for causal models to be actionable, they must be interpretable by biologists, and the uncertainty associated with inferred causal links must be rigorously quantified to guide decision-making (Key Challenge 3 & 5; Tejada-Lapuerta et al., 2023; Johnson et al., 2024).

Existing approaches often tackle these challenges in isolation. Some focus on causal representation learning from multimodal data (Sun et al., 2024; Brown et al., 2024), others on using interventions to refine causal graphs (Wu et al., 2024), and others on optimizing experimental design (Doe et al., 2024; Williams et al., 2023). Yet, a unified framework that dynamically integrates multi-omics data, leverages perturbation experiments through active learning, and provides interpretable, uncertainty-aware causal models is currently lacking.

**2.2 Research Objectives**

This research proposes to develop a novel computational framework that synergistically combines causal graphical models, multi-omics data integration, and active learning to efficiently uncover causal relationships in complex biological systems. The primary objectives are:

1.  **Develop an Integrated Causal Representation Learning Model:** Create a model, likely based on structured variational autoencoders (SVAEs) or similar deep generative frameworks, capable of learning interpretable, low-dimensional latent representations that capture shared and modality-specific causal factors from diverse omics data (e.g., single-cell RNA-seq, proteomics, potentially spatial transcriptomics) incorporating both observational and interventional datasets.
2.  **Implement Robust Causal Structure Learning Incorporating Interventions:** Utilize the learned latent representations and original data to infer causal graphical models (e.g., Directed Acyclic Graphs - DAGs) that explicitly leverage interventional data (e.g., from CRISPR or RNAi screens) to orient edges and distinguish direct causal effects from correlations, potentially using counterfactual reasoning approaches (Lee et al., 2023; Wu et al., 2024).
3.  **Design an Active Learning Strategy for Optimal Perturbation Selection:** Develop and implement an active learning algorithm that iteratively suggests the most informative perturbation experiments to perform next. This strategy will aim to maximally reduce uncertainty in the estimated causal graph structure or specific causal pathways of interest, thereby optimizing resource allocation.
4.  **Quantify Uncertainty in Causal Inferences:** Integrate methods for quantifying uncertainty (e.g., Bayesian approaches, bootstrapping) over the inferred causal graph structure (presence/absence/direction of edges) and the magnitude of causal effects (Johnson et al., 2024).
5.  **Validate the Framework:** Rigorously evaluate the proposed framework's performance on synthetic datasets with known ground truth causal structures and on real-world multi-omics perturbation datasets (e.g., LINCS L1000, Perturb-seq data) by comparing against baseline methods and known biological pathways.

**2.3 Significance**

This research sits at the intersection of machine learning, causal inference, and genomics, addressing critical bottlenecks in fundamental biology and drug discovery. By successfully achieving the outlined objectives, this work will provide:

*   **Accelerated Target Identification:** A more efficient and reliable method for identifying and prioritizing potential drug targets based on strong causal evidence, reducing the costly trial-and-error inherent in current approaches.
*   **Deeper Mechanistic Understanding:** Interpretable causal models that offer insights into the complex interplay between genes, proteins, and cellular phenotypes, potentially revealing novel biological mechanisms underlying health and disease.
*   **Optimized Experimental Design:** A data-driven strategy for designing perturbation experiments, maximizing the scientific value obtained from limited experimental resources.
*   **Enhanced Reproducibility:** By focusing on causal relationships validated through interventions and quantifying uncertainty, the framework aims to improve the reproducibility of biological findings.
*   **Methodological Advancements:** Contributions to machine learning methodology in areas such as causal representation learning from multimodal data, active learning for causal discovery, and uncertainty quantification in high-dimensional biological systems.

Ultimately, this research aims to bridge the gap between high-throughput data generation and actionable biological knowledge, paving the way for more rational, efficient, and successful drug development pipelines.

---

## **3. Methodology**

Our proposed methodology iteratively refines a causal graphical model by integrating multi-omics data and actively selecting informative perturbation experiments. The framework consists of four core components operating within an iterative loop: (1) Multi-Omics Causal Representation Learning, (2) Causal Graph Discovery and Refinement, (3) Uncertainty Quantification, and (4) Active Learning for Experiment Selection.

**3.1 Data Collection and Preprocessing**

We will utilize both synthetic and real-world datasets.

*   **Synthetic Data:** We will generate synthetic datasets simulating multi-omics measurements (e.g., gene expression, protein abundance) governed by known ground truth causal DAGs. Data generation will incorporate various complexities, including linear and non-linear relationships, different noise models (Gaussian, Poisson), varying levels of latent confounding, and simulated perturbation effects following specific intervention mechanisms (e.g., perfect vs. leaky knockdowns). This allows for controlled evaluation of model performance.
*   **Real-World Data:** We will leverage publicly available datasets such as:
    *   **LINCS L1000:** Contains gene expression profiles of human cell lines before and after perturbation by thousands of small molecules and genetic reagents (shRNAs). Provides observational and interventional data.
    *   **DepMap/Achilles:** Large-scale CRISPR and RNAi screens measuring gene essentiality/dependency across hundreds of cancer cell lines. Provides rich interventional data.
    *   **Single-Cell Perturbation Screens (e.g., Perturb-seq, CROP-seq):** Datasets combining single-cell RNA sequencing with CRISPR-based perturbations, offering high-resolution insights into heterogeneous cellular responses.
    *   **Multi-Omics Studies:** Datasets profiling matched samples across multiple modalities (e.g., TCGA with transcriptomics and proteomics, single-cell multi-omics). We may need to combine observational multi-omics data with separate perturbation datasets initially.

**Preprocessing:** Standard preprocessing steps will be applied, including quality control, normalization (e.g., library size normalization for RNA-seq, log-transformation), batch effect correction where necessary, feature selection (e.g., selecting highly variable genes), and handling missing values (e.g., imputation using matrix factorization or deep learning methods). Data from different modalities will be aligned based on samples/cells.

**3.2 Algorithmic Steps**

The core of the methodology is an iterative loop:

**Initial Step:** Start with available observational and initial interventional multi-omics data $D_0 = \{ (X_{obs}^{(i)}, M^{(i)}), (X_{int}^{(j)}, M^{(j)}, I^{(j)}) \}$, where $X$ represents the multi-omics measurements for sample $i$ or $j$, $M$ denotes the modality type (e.g., RNA, protein), and $I^{(j)}$ indicates the intervention applied to sample $j$.

**(Iteration $k=1, 2, ...$)**

**Step 3.2.1: Multi-Omics Causal Representation Learning**

*   **Model:** We propose using a Structured Variational Autoencoder (SVAE) framework, potentially extending existing models (e.g., inspired by Lopez et al. 2022, Brown et al. 2024, Sun et al. 2024) to handle multimodal inputs and explicitly model interventions. The VAE consists of an encoder $q_\phi(Z|X, M, I)$ and a decoder $p_\theta(X|Z, M, I)$. We aim to learn a latent representation $Z$ that is disentangled and reflects the underlying causal structure.
*   **Multimodal Integration:** The encoder and decoder will be designed to handle multiple modalities, potentially using modality-specific layers feeding into shared latent variables, or techniques like product-of-experts or mixture-of-experts.
*   **Intervention Modeling:** Interventions $I$ will be explicitly incorporated. Drawing inspiration from mechanism shift models (Lopez et al., 2022), interventions are assumed to affect specific, sparse components of the latent causal mechanism. The VAE objective (Evidence Lower Bound - ELBO) will be adapted:
    $$ \mathcal{L}(\theta, \phi; X, M, I) = \mathbb{E}_{q_\phi(Z|X, M, I)}[\log p_\theta(X|Z, M, I)] - D_{KL}(q_\phi(Z|X, M, I) || p(Z|I)) $$
    where $p(Z|I)$ is the intervention-dependent prior on the latent space. For observational data, $I=null$.
*   **Output:** A low-dimensional latent representation $Z_k$ for each sample, learned from data $D_k$. Potential structure (e.g., sparsity, graph priors) can be imposed on $Z$ to enhance interpretability.

**Step 3.2.2: Causal Graph Discovery and Refinement**

*   **Input:** Latent representations $Z_k$ and potentially selected original features $X_k$, along with intervention information $I$.
*   **Method:** We will employ causal structure learning algorithms suitable for observational and interventional data. Options include:
    *   **Score-based methods:** Extending approaches like NOTEARS or GES to incorporate interventional data by modifying the scoring function (e.g., penalizing edges inconsistent with interventions). The score $S(G, D_k)$ measures the goodness-of-fit of a graph $G$ to the data $D_k$.
    *   **Constraint-based methods:** Using conditional independence tests (e.g., Kernel Conditional Independence Test - KCI) adapted for interventional data (e.g., ICP, GIES). Interventions break specific edges, altering conditional independencies.
    *   **Hybrid methods:** Combining the strengths of both approaches.
*   **Handling Latent Confounders:** The VAE aims to capture major latent factors in $Z$. We may also explore algorithms robust to unobserved confounders (e.g., FCI-based methods) if necessary.
*   **Counterfactual Queries:** The learned causal graph $G_k=(V, E_k)$ can be used to estimate interventional distributions $P(Y|do(X=x))$ and answer counterfactual queries (Lee et al., 2023), predicting the effects of potential interventions.
*   **Output:** An estimated causal graph $G_k$ (often represented as a DAG or CPDAG) over latent variables and/or observed variables.

**Step 3.2.3: Uncertainty Quantification**

*   **Input:** Estimated causal graph $G_k$ and data $D_k$.
*   **Method:** We will quantify uncertainty primarily over the graph structure.
    *   **Bayesian Approaches:** Employ Bayesian structure learning methods (e.g., using MCMC or variational inference over graph space) to obtain posterior probabilities $P(e \in E_k | D_k)$ for each potential edge $e$. This provides a distribution over possible graphs rather than a single point estimate (Johnson et al., 2024).
    *   **Bootstrapping:** Resample the data $D_k$ multiple times, learn a causal graph for each bootstrap sample, and aggregate the results to estimate edge confidence/stability.
*   **Output:** A probabilistic causal graph or a set of high-probability graphs, along with confidence scores for individual edges or causal paths.

**Step 3.2.4: Active Learning for Experiment Selection**

*   **Input:** The current estimate of the causal graph $G_k$ and its associated uncertainty (e.g., posterior distribution over graphs $P(G|D_k)$).
*   **Goal:** Select the next perturbation experiment $I_{next}$ from a set of candidate interventions $\mathcal{I}_{cand}$ that is expected to yield the most information for refining the causal graph.
*   **Acquisition Function:** We will explore several acquisition functions $A(I)$ aiming to maximize information gain or uncertainty reduction (Doe et al., 2024; Williams et al., 2023):
    *   **Maximum Entropy Reduction:** Select the intervention $I$ that maximizes the expected reduction in the entropy of the posterior distribution over graphs:
        $$ I_{next} = \arg \max_{I \in \mathcal{I}_{cand}} [H(G|D_k) - \mathbb{E}_{Y_{I} \sim P(Y|do(I), G_k)}[H(G|D_k, Y_I)]] $$
        where $Y_I$ is the anticipated outcome of intervention $I$.
    *   **Uncertainty Sampling:** Focus on resolving uncertainty about specific edges or paths. For example, target interventions expected to resolve the orientation or presence of edges with posterior probability close to 0.5.
    *   **Causal Effect Variance Reduction:** Select interventions that maximally reduce the variance of a specific target causal effect estimate $\mathbb{E}[Y|do(X=x)]$.
*   **Selection:** Choose $I_{next} = \arg \max_{I \in \mathcal{I}_{cand}} A(I)$.
*   **Output:** The selected intervention $I_{next}$ to be performed (or simulated).

**(End of Iteration $k$)**

*   Perform (or simulate) the selected experiment $I_{next}$, obtain new data $(X_{int}^{(new)}, M^{(new)}, I_{next})$.
*   Update the dataset: $D_{k+1} = D_k \cup \{ (X_{int}^{(new)}, M^{(new)}, I_{next}) \}$.
*   Proceed to the next iteration ($k \leftarrow k+1$).

**3.3 Experimental Design and Validation**

*   **Baseline Methods:** We will compare our framework against relevant baselines:
    *   Standard causal discovery methods on observational data only (e.g., PC algorithm, GES).
    *   Causal discovery methods using observational + randomly selected interventional data.
    *   Multi-omics integration methods without explicit causal modeling (e.g., standard VAEs, Canonical Correlation Analysis).
    *   Existing causal discovery methods leveraging interventions but without active learning or multi-omics integration (e.g., GIES, Wu et al. 2024 method applied unimodally).

*   **Evaluation on Synthetic Data:**
    *   **Metrics:**
        *   **Structure Learning Accuracy:** Structural Hamming Distance (SHD), Precision, Recall, F1-score comparing the learned graph ($G_k$) to the ground truth graph ($G_{true}$).
        *   **Causal Effect Estimation Accuracy:** Mean Squared Error (MSE) between estimated interventional distributions/effects and true effects.
        *   **Active Learning Efficiency:** Plot structure learning accuracy (e.g., SHD) as a function of the number of interventions performed. Compare the curve for our active learning strategy against random selection.
    *   **Analysis:** Assess performance under varying conditions (dimensionality, sample size, noise levels, graph density, intervention types).

*   **Evaluation on Real-World Data:**
    *   **Metrics:**
        *   **Consistency with Known Biology:** Assess the overlap between top-confidence inferred causal edges/paths and known signaling pathways or protein-protein interactions from databases (e.g., KEGG, Reactome, StringDB). Calculate precision/recall against these curated pathways.
        *   **Prediction of Held-out Interventions:** Train the model on a subset of interventions and evaluate its ability to predict the molecular profiles (e.g., gene expression changes) resulting from held-out interventions. Use metrics like R-squared or MSE.
        *   **Reproducibility Across Datasets:** Apply the framework to different datasets representing similar biological contexts and assess the consistency of the inferred causal networks.
        *   **Identification of Known Causal Regulators:** Evaluate if known key regulators (e.g., master transcription factors, critical signaling nodes) are identified with high confidence in the inferred graph.
        *   **Qualitative Assessment:** Collaborate with domain experts to evaluate the biological plausibility and interpretability of the discovered causal relationships.
    *   **Analysis:** Demonstrate the framework's ability to integrate multi-omics data and leverage interventions effectively. Highlight novel, high-confidence causal links identified. Benchmark against baseline methods on these real-world tasks. Evaluate the reduction in experimental effort suggested by the active learning component compared to naive strategies.

*   **Computational Aspects:** Evaluate the scalability and computational efficiency of the proposed framework.

---

## **4. Expected Outcomes & Impact**

This research is expected to yield several significant outcomes, contributing both methodologically and biologically:

**4.1 Expected Outcomes**

1.  **A Novel Computational Framework:** A publicly available software implementation of the integrated framework combining multi-omics SVAE, intervention-aware causal discovery, uncertainty quantification, and active learning for experimental design.
2.  **Validated Causal Networks:** Application of the framework to benchmark synthetic and real-world datasets (e.g., LINCS, DepMap, single-cell perturbation data) will produce high-confidence, interpretable causal graphical models representing regulatory relationships between genes, proteins, and potentially cellular phenotypes. These models will be annotated with uncertainty estimates for each inferred relationship.
3.  **Demonstration of Active Learning Efficiency:** Quantitative evidence showing that the active learning strategy significantly reduces the number of perturbation experiments required to achieve a desired level of accuracy in causal graph reconstruction, compared to random or exhaustive experimentation.
4.  **Prioritized Candidate Targets:** For specific biological systems studied (e.g., a particular cancer type or signaling pathway), the framework will output a ranked list of potential intervention targets (genes/proteins) based on the inferred strength and confidence of their causal influence on downstream variables or phenotypes of interest.
5.  **Methodological Insights:** New insights into the challenges and best practices for causal representation learning from implicitly causal multimodal data, causal discovery leveraging targeted interventions, and effective strategies for active learning in the context of causal structure identification in high dimensions.

**4.2 Impact**

The proposed research holds the potential for significant impact across multiple domains:

*   **Accelerating Drug Discovery:** By providing a more principled and efficient way to identify and validate causal drivers of disease, this work can significantly accelerate the early stages of drug discovery. Identifying targets with strong causal links to disease phenotypes increases the likelihood of successful translation to clinical trials, potentially reducing the high attrition rates currently observed.
*   **Improving Mechanistic Understanding:** The interpretable causal models generated will provide biologists with valuable tools to dissect complex biological systems. This can lead to a deeper understanding of disease mechanisms, gene function, and cellular responses to perturbations, moving beyond correlative associations.
*   **Resource Optimization in Biological Research:** The active learning component directly addresses the high cost of perturbation experiments. By guiding researchers to perform the most informative experiments first, the framework can optimize the use of time, reagents, and funding, maximizing knowledge gain from limited resources.
*   **Advancing Machine Learning for Genomics:** This project will push the boundaries of machine learning applied to genomics, particularly in the areas of causal inference, multimodal learning, active learning, and uncertainty quantification. The developed methods may be adaptable to other scientific domains where causal understanding from observational and interventional data is crucial.
*   **Fostering Interdisciplinary Collaboration:** The project inherently requires collaboration between machine learning researchers and biologists, fostering the cross-pollination of ideas and expertise necessary to tackle complex biomedical challenges, aligning perfectly with the goals of the "Machine Learning for Genomics Explorations" workshop.

In conclusion, this research proposes a rigorous, integrated, and efficient approach to causal discovery in complex biological systems. By combining cutting-edge machine learning techniques with the power of multi-omics and perturbation data, we aim to deliver actionable causal insights that can significantly advance both fundamental biological understanding and translational therapeutic development.

---
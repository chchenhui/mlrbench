Okay, here is a detailed research proposal based on the provided task description, research idea, and literature review.

---

**1. Title:** **DiffuNA: A 3D Equivariant Diffusion Model for Structure-Conditioned Generative Design of Functional RNA Therapeutics**

---

**2. Introduction**

**Background:**
Nucleic acids, particularly RNA, are emerging as a powerful class of therapeutics with diverse modalities, including mRNA vaccines, antisense oligonucleotides, small interfering RNAs (siRNAs), aptamers, and ribozymes. These molecules can modulate gene expression, catalyze reactions, or bind specific targets with high affinity and specificity. However, the rational design of functional RNA molecules remains a significant bottleneck. Current approaches often rely on high-throughput screening of vast random libraries (e.g., SELEX for aptamers) or iterative, intuition-driven modifications of known motifs. These methods are labor-intensive, costly, time-consuming, and often yield suboptimal candidates or fail to explore the full potential of the RNA chemical space.

The central challenge lies in navigating the complex relationship between RNA sequence, secondary structure (base pairing), tertiary structure (3D fold), and ultimately, function (e.g., binding affinity, catalytic activity). Predicting RNA structure from sequence is difficult, and the inverse problem – designing sequences that fold into a desired functional structure – is even more challenging. Existing computational tools often focus on specific aspects, such as secondary structure prediction (e.g., UFold [6], SPOT-RNA [7], EternaFold [10]), tertiary structure prediction (e.g., trRosettaRNA [5], RNAComposer [8], FARFAR2 [9]), or inverse folding (structure-to-sequence, e.g., RiboDiffusion [1]). While valuable, these tools do not provide an end-to-end framework for generating novel RNA sequences *and* structures tailored to specific therapeutic functions, such as binding a predefined protein pocket.

Recent advances in artificial intelligence, particularly deep generative models, have shown remarkable success in designing small molecules and proteins (e.g., DiffSBDD [3], Luo et al. [4], Tang et al. survey [2]). Diffusion models, in particular, have demonstrated state-of-the-art performance in generating complex, high-dimensional data like images, audio, and molecular structures. These models learn to reverse a diffusion process that gradually adds noise to data, allowing them to generate realistic samples starting from noise. Applying such powerful generative frameworks to RNA design holds immense promise for accelerating therapeutic discovery.

**Problem Statement:**
There is a critical need for a principled, efficient, and automated computational framework capable of generating novel RNA molecules with desired structural and functional properties *de novo*. Specifically, a method is required that can jointly design RNA sequence, secondary, and tertiary structure, conditioned on a target specification (e.g., a protein binding pocket or a structural scaffold), and optimized for therapeutic relevance (e.g., stability and binding affinity). This framework should overcome the limitations of traditional methods and existing computational tools by providing a unified generative approach.

**Proposed Solution: DiffuNA**
We propose **DiffuNA**, a novel generative framework based on 3D graph-based diffusion models, designed specifically for the *de novo* generation of functional RNA therapeutics. DiffuNA will represent RNA molecules as 3D graphs where nodes are nucleotides and edges represent covalent bonds and base-pairing interactions. Crucially, it will leverage SE(3)-equivariant graph neural networks to properly handle the geometric nature of RNA structures. The core diffusion model will be trained on publicly available databases of RNA structures (e.g., PDB, RNA STRAND) and associated experimental data (e.g., SHAPE reactivity for secondary structure constraints) to jointly learn the distributions of sequence, secondary structure, and tertiary structure coordinates.

During training, DiffuNA learns to reverse a corruption process that adds noise to the 3D coordinates, masks sequence elements, and perturbs secondary structure information. At inference time, DiffuNA can generate novel RNA candidates conditioned on specific structural constraints, such as a target protein's binding pocket geometry or a desired structural scaffold. Starting from random noise, the model iteratively refines the RNA graph through the learned reverse diffusion process to produce candidate sequence-structure pairs.

Furthermore, to explicitly optimize for therapeutic function, DiffuNA will incorporate a reinforcement learning (RL) based refinement loop. This loop will leverage pre-trained predictors for RNA folding stability (e.g., based on energy models or deep learning predictors like EternaFold/UFold) and binding affinity (e.g., using docking surrogates or ML-based predictors) to guide the generation process towards candidates with high predicted stability and target affinity.

**Research Objectives:**

1.  **Develop the DiffuNA Core Model:** Implement and train a 3D graph-based SE(3)-equivariant diffusion model capable of jointly generating RNA sequence, secondary structure, and tertiary structure coordinates.
2.  **Implement Conditional Generation:** Enable the DiffuNA model to generate RNA candidates conditioned on user-defined structural constraints, such as target protein binding pockets or predefined structural scaffolds.
3.  **Integrate RL-based Refinement:** Develop and integrate a reinforcement learning loop that utilizes pre-trained folding stability and binding affinity predictors to fine-tune the generated RNA candidates for desired functional properties.
4.  **Validate and Benchmark:** Evaluate DiffuNA's performance on standard RNA design benchmarks (e.g., designing thrombin-binding aptamers, hammerhead ribozymes) and compare its capabilities against existing state-of-the-art computational methods and relevant baselines (e.g., RiboDiffusion for inverse folding aspects, potentially adapting protein/molecule design methods).
5.  **Demonstrate Novel Design Capability:** Apply DiffuNA to design novel RNA binders for a therapeutically relevant protein target not used during training.

**Significance:**
This research addresses a critical gap in RNA therapeutic design by proposing a powerful, AI-driven generative framework. DiffuNA has the potential to:

*   **Accelerate Discovery:** Significantly reduce the time and cost associated with identifying novel RNA therapeutic leads compared to traditional screening methods.
*   **Expand Therapeutic Potential:** Enable the design of RNA molecules with tailored functionalities and potentially novel mechanisms of action, targeting proteins or cellular processes previously considered undruggable by RNA.
*   **Advance AI for Biology:** Contribute novel deep learning methodologies (specifically equivariant diffusion models and RL integration) for complex biomolecular design problems, pushing the frontiers of AI applications in nucleic acid research, aligning perfectly with the goals of the AI4NA workshop.
*   **Provide New Tools:** Deliver a computational tool that can aid researchers in hypothesis generation and rational design of functional RNAs for diverse applications.

By tackling the inherent complexity of RNA sequence-structure-function relationships through a unified generative model, DiffuNA represents a significant step towards automated, efficient, and innovative RNA therapeutic design.

---

**3. Methodology**

This section details the research design, including data collection and preprocessing, the architectural components of DiffuNA, the RL refinement loop, and the experimental validation plan.

**3.1 Data Collection and Preprocessing:**

*   **Data Sources:** We will primarily utilize publicly available datasets:
    *   **Protein Data Bank (PDB):** Extract experimentally determined 3D structures of RNA molecules (both free and protein/ligand-bound).
    *   **RNA STRAND / Rfam:** Obtain curated RNA sequences, secondary structures, and family classifications.
    *   **SHAPE-MaP Databases (e.g., RMDB):** Collect SHAPE (Selective 2'-Hydroxyl Acylation analyzed by Primer Extension) reactivity data, which provides experimental constraints on RNA secondary structure in solution.
*   **Preprocessing:**
    1.  **Filtering:** Select high-resolution RNA structures (e.g., < 3.5 Å) from the PDB. Remove highly redundant structures using sequence and structural clustering (e.g., using RNA Equivalents from RNArchitecture or custom clustering). Filter out structures with too few or too many nucleotides based on model capacity.
    2.  **Standardization:** Ensure consistent nucleotide naming and representation. Process modified nucleotides – either by mapping them to canonical bases or treating them as distinct node types if sufficient data exists.
    3.  **Structure/Sequence Integration:** For each PDB structure, extract 3D coordinates of backbone atoms (e.g., C4', P, O5', C5', C3') and key base atoms, the corresponding sequence, and derive the secondary structure using tools like DSSR or RNApdbee. Where available, integrate SHAPE data as additional structural constraints or features.
    4.  **Graph Representation:** Convert each RNA molecule into a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$.
        *   **Nodes ($\mathcal{V}$):** Each node $v_i$ represents a nucleotide. Node features $\mathbf{h}_i$ will include:
            *   Nucleotide type (A, U, G, C, potentially modified bases) - one-hot encoded.
            *   Local structural information (e.g., derived torsion angles, sugar pucker).
            *   Optionally, SHAPE reactivity scores.
        *   **Edges ($\mathcal{E}$):** Edges represent relationships between nucleotides:
            *   *Covalent Bonds:* Edges connect $v_i$ and $v_{i+1}$ along the backbone.
            *   *Base Pairing:* Edges connect nucleotides $v_i$ and $v_j$ involved in canonical (Watson-Crick) or non-canonical base pairs, annotated with pair type (e.g., WC, Hoogsteen, Wobble, Stacked).
            * Edge features $\mathbf{e}_{ij}$ could include distance or pairwise interaction types.
        *   **Coordinates ($\mathbf{X}$):** A matrix $\mathbf{X} \in \mathbb{R}^{N \times k \times 3}$ stores the 3D coordinates of $k$ representative atoms (e.g., C4', P, base center-of-mass) for each of the $N$ nucleotides.

**3.2 DiffuNA Core Model Architecture:**

DiffuNA will be based on a score-based generative model operating on the 3D graph representation of RNA. It learns to reverse a diffusion process that gradually perturbs the RNA structure and sequence/secondary structure features.

*   **Forward Diffusion Process:**
    *   **Structure:** A variance-preserving stochastic differential equation (VP-SDE) or a discrete-time Markov chain adds Gaussian noise to the 3D coordinates $\mathbf{X}$ over time $t \in [0, T]$.
        $$ d\mathbf{X}_t = \sqrt{\frac{d[\sigma^2(t)]}{dt}} d\mathbf{w}_t $$
        where $\sigma^2(t)$ is the noise schedule and $\mathbf{w}_t$ is a standard Wiener process.
    *   **Sequence & Secondary Structure:** A discrete diffusion process (analogous to methods for categorical data) will corrupt the node features (base type) and edge features (pair type) over time, potentially transitioning them towards a uniform or uninformative distribution. For instance, base identities can be randomly masked or mutated with increasing probability as $t$ increases.
*   **Reverse Denoising Process (Score Network):**
    *   **Architecture:** We will employ an SE(3)-equivariant graph neural network (GNN) architecture, such as EGNN (Equivariant Graph Neural Network) or GVP-GNN (Geometric Vector Perceptron GNN), adapted for RNA. This ensures that the model's predictions are invariant/equivariant to rotations and translations of the input RNA structure. The network $\epsilon_\theta(\mathcal{G}_t, t)$ takes the noisy graph $\mathcal{G}_t$ (including noisy coordinates $\mathbf{X}_t$, corrupted sequence/SS features $\mathbf{h}_t, \mathbf{e}_t$) and the noise level/time $t$ as input.
    *   **Output:** The network predicts the noise added to the structure ($\nabla_{\mathbf{X}_t} \log p_t(\mathbf{X}_t | \mathbf{h}_t, \mathbf{e}_t)$ or the noise itself) and the original sequence/secondary structure features.
    *   **Training Objective:** The model $\epsilon_\theta$ is trained to minimize the denoising score matching loss. This involves minimizing the difference between the predicted noise and the actual noise added during the forward process for structure, and using cross-entropy loss for the categorical sequence and secondary structure features. The combined loss function will be a weighted sum:
        $$ \mathcal{L}(\theta) = \mathbb{E}_{t, p(\mathcal{G}_0), \epsilon} \left[ \lambda_{3D} || \epsilon_\theta(\mathcal{G}_t, t)_{3D} - \epsilon_{3D} ||^2 + \lambda_{seq} \mathcal{L}_{CE}(\epsilon_\theta(\mathcal{G}_t, t)_{seq}, \mathbf{h}_0) + \lambda_{ss} \mathcal{L}_{CE}(\epsilon_\theta(\mathcal{G}_t, t)_{ss}, \mathbf{e}_0) \right] $$
        where $\epsilon_{3D}$ is the sampled Gaussian noise for coordinates, $\mathbf{h}_0$ and $\mathbf{e}_0$ are the original sequence and secondary structure features, $\mathcal{L}_{CE}$ is the cross-entropy loss, and $\lambda$ terms are weighting factors.

**3.3 Conditional Generation:**

To generate RNA specific to a target (e.g., protein pocket), we will implement conditioning mechanisms:

*   **Input Conditioning:** Provide the target structure (e.g., pocket atoms represented as a point cloud or graph) as an additional input to the denoising network $\epsilon_\theta$. The network learns to generate RNA structures compatible with this context.
*   **Guidance during Sampling:** Use classifier guidance or related techniques during the reverse diffusion sampling. A separately trained classifier $p(\text{pocket} | \mathcal{G}_t)$ could guide the sampling towards RNA structures likely to fit the pocket. Alternatively, modify the score: $\nabla \log p(\mathcal{G}_t | \text{pocket}) \approx \nabla \log p(\mathcal{G}_t) + \nabla \log p(\text{pocket} | \mathcal{G}_t)$.
*   **Scaffold-Based Generation:** If a structural scaffold is provided (e.g., a specific helix arrangement), fix the coordinates and features of the scaffold nucleotides during the denoising process, generating the variable regions.

**3.4 Reinforcement Learning (RL) Refinement Loop:**

To optimize generated candidates for function beyond what is implicitly learned from the training data, we integrate an RL loop.

*   **Setup:**
    *   **Agent:** The DiffuNA sampler acts as the agent's policy generator.
    *   **State:** A partially or fully generated RNA candidate $(\mathcal{G}_t)$.
    *   **Action:** A denoising step in the reverse diffusion process.
    *   **Environment:** Simulates the evaluation of a generated RNA candidate.
*   **Reward Function:** The reward R is assigned upon generating a complete candidate $\mathcal{G}_0$. It combines predicted stability and binding affinity:
    $$ R(\mathcal{G}_0) = w_{stab} \cdot f_{stab}(\mathcal{G}_0) + w_{aff} \cdot f_{aff}(\mathcal{G}_0 | \text{target}) + R_{constraints} $$
    *   $f_{stab}(\mathcal{G}_0)$: Predicted folding stability. Calculated using established tools (e.g., RNAfold from ViennaRNA package for free energy estimate) or a pre-trained deep learning model (e.g., based on UFold/EternaFold principles) evaluating the likelihood of the sequence folding into the generated structure.
    *   $f_{aff}(\mathcal{G}_0 | \text{target})$: Predicted binding affinity to the target pocket. Estimated using fast docking tools (e.g., AutoDock Vina, rDock, HDOCK) or a trained ML-based docking surrogate model for speed. The score will be normalized.
    *   $R_{constraints}$: Penalties for violating constraints (e.g., steric clashes, unrealistic bond lengths/angles).
    *   $w_{stab}, w_{aff}$: Weights to balance stability and affinity.
*   **RL Algorithm:** We will employ a policy gradient algorithm suitable for this high-dimensional action space, potentially Proximal Policy Optimization (PPO) or a similar method, to fine-tune the diffusion model's parameters or guide the sampling process to maximize the expected reward. The reward signal guides the reverse diffusion path towards promising regions of the RNA design space.

**3.5 Experimental Design and Validation:**

*   **Datasets for Validation:**
    *   **Benchmark Regeneration:**
        1.  *Thrombin Binding Aptamer (TBA):* Use the known TBA structure (PDB ID: 1HUT) and its target thrombin. Task: Conditioned on the thrombin binding site, can DiffuNA generate the known TBA sequence and structure, or novel sequences folding into similar G-quadruplex structures with predicted high affinity?
        2.  *Hammerhead Ribozyme:* Use a well-characterized hammerhead ribozyme structure (e.g., PDB ID: 2GOZ). Task: Conditioned on the catalytic core scaffold, can DiffuNA generate functional sequence variants? Stability prediction will be key here.
*   **De Novo Design Task:**
    *   Select a therapeutically relevant protein target *not* present in the training set (e.g., a viral protein or an oncology target). Define its binding pocket. Task: Use DiffuNA conditioned on this pocket to generate novel RNA aptamer candidates.
*   **Baselines for Comparison:**
    *   **RiboDiffusion [1]:** Adapt for conditional generation if possible, or compare on inverse folding sub-tasks.
    *   **Sequence-based Generative Models:** Train a sequence-only generative model (e.g., LSTM, Transformer) and predict structure/function afterwards using separate tools (e.g., RNAfold + docking).
    *   **Structure-based Inverse Folding:** Use RiboDiffusion or similar methods to generate sequences for *designed* or *target* structures.
    *   **Fragment Assembly Methods (e.g., Rosetta FARFAR2 [9]):** Compare structural accuracy and novelty where applicable, though these are typically not generative in the same sense.
*   **Evaluation Metrics:**
    *   **Structural Similarity:** Root Mean Square Deviation (RMSD) between generated structures and reference structures (if applicable). Secondary structure similarity (e.g., F1-score on base pairs).
    *   **Sequence Recovery/Identity:** For regeneration tasks, percentage sequence identity to the native sequence.
    *   **Predicted Folding Stability:** Minimum Free Energy (MFE) calculated by RNAfold, or scores from ML predictors. Distribution of stability scores for generated candidates.
    *   **Predicted Binding Affinity:** Docking scores (e.g., kcal/mol) or predicted $K_d$ values from surrogate models. Distribution of affinity scores.
    *   **Novelty & Diversity:** Measured by sequence and structure similarity (e.g., Tanimoto similarity based on fingerprints) among generated candidates and compared to the training set.
    *   **Validity:** Percentage of generated RNAs that are chemically valid (correct bond lengths, angles) and satisfy basic structural constraints.
    *   **Success Rate:** Percentage of generated candidates passing predefined thresholds for stability and affinity.
    *   **Computational Efficiency:** Time required for training and generation.

---

**4. Expected Outcomes & Impact**

**Expected Outcomes:**

1.  **A Functioning DiffuNA Model:** A robust, validated implementation of the 3D equivariant graph diffusion model (DiffuNA) capable of jointly generating RNA sequence, secondary structure, and tertiary structure. The model, code, and training protocols will be made publicly available.
2.  **Benchmark Performance:** Quantitative results demonstrating DiffuNA's ability to regenerate known functional RNA structures (TBA, Hammerhead Ribozyme) and perform competitively against or outperform existing baseline methods on metrics including structural accuracy, stability, and predicted affinity.
3.  **Novel RNA Designs:** A set of *de novo* designed RNA sequences and structures predicted to bind a specific therapeutic target with high affinity and stability, showcasing the model's potential for practical drug discovery applications.
4.  **Insights into RNA Design Principles:** Analysis of the learned representations and generated structures may reveal novel sequence-structure motifs or principles governing RNA folding and interaction, contributing to our fundamental understanding.
5.  **Publications and Presentations:** Dissemination of findings through publications in leading AI/ML and computational biology journals/conferences, and presentations at relevant workshops like AI4NA.

**Impact:**

*   **Transforming RNA Therapeutic Design:** DiffuNA aims to shift the paradigm from slow, experimental screening towards rapid, automated, *in silico* design of RNA therapeutics. This could dramatically accelerate the preclinical development pipeline for RNA drugs.
*   **Addressing Key Challenges:** By jointly modeling sequence and structure, DiffuNA tackles the complex sequence-structure relationship (#3 from lit review challenges). The RL loop directly addresses the need for functional optimization. While data scarcity (#1) remains a challenge, leveraging existing PDB and SHAPE data, potentially augmented with transfer learning, offers a path forward. The use of diffusion models addresses the generation of complex structures (#2), and SE(3)-equivariant networks ensure proper geometric handling. Careful model design will be needed for computational efficiency (#4), and validation on unseen targets will assess generalization (#5).
*   **Broader Applications in Synthetic Biology:** The ability to generate novel functional RNAs could extend beyond therapeutics to applications in synthetic biology, diagnostics (biosensors), and nanotechnology (RNA origami).
*   **Contribution to AI for Science:** This work will advance the application of cutting-edge generative AI techniques (diffusion models, equivariant GNNs, RL) to fundamental problems in structural biology and drug discovery, demonstrating the power of AI in accelerating scientific progress.
*   **Community Resource:** A successful DiffuNA framework, released as open-source software, would provide a valuable tool for the wider research community engaged in RNA biology and drug design.

In conclusion, the DiffuNA project promises to deliver a powerful new approach for generative RNA design, with significant potential impact on therapeutic discovery and our ability to engineer functional nucleic acid molecules. Its development aligns strongly with the focus of the AI4NA workshop, bridging the gap between advanced AI methodologies and critical challenges in nucleic acid research.

---
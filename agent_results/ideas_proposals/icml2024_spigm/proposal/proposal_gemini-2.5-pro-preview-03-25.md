# **Research Proposal**

## 1. Title: Physics-Informed Graph Normalizing Flows for Efficient and Valid Molecular Conformation Generation

## 2. Introduction

### 2.1 Background

The three-dimensional structure (conformation) of a molecule dictates its physical, chemical, and biological properties. Generating computationally feasible and physically realistic molecular conformations is a cornerstone of modern chemistry, materials science, and structural biology, particularly in drug discovery where molecular shape determines interactions with biological targets (Ragoza et al., 2017; Skalic et al., 2019). Given a molecular graph (atoms and bonds), the task is to generate a diverse set of low-energy 3D arrangements of its atoms. This problem is challenging due to the high dimensionality of the conformational space, the complex interplay of local (bond lengths, angles) and global (non-bonded interactions, torsional strains) energetic factors, and the requirement for generated structures to adhere strictly to the laws of physics and chemistry.

Traditional computational methods for conformation generation, such as molecular dynamics (MD) simulations or distance geometry methods, can be computationally expensive, often requiring significant simulation time to adequately sample the conformational landscape (Leach, 2001). In recent years, deep generative models, including Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), diffusion models, and normalizing flows, have shown promise in learning complex data distributions and generating novel samples (Kingma & Welling, 2013; Goodfellow et al., 2014; Ho et al., 2020; Dinh et al., 2016). These models have been adapted for molecular generation tasks, operating either on graph representations (Jin et al., 2018; Liu et al., 2021; Kuznetsov & Polykovskiy, 2021) or directly in 3D coordinate space (Xu et al., 2022; Shah & Koltun, 2024).

However, many existing deep learning approaches struggle with several key challenges highlighted in the literature. Firstly, ensuring chemical *validity* and physical *realism* is difficult; models trained solely on data may generate structures with incorrect bond lengths, steric clashes, or unrealistically high potential energies. Secondly, incorporating fundamental physical principles and domain knowledge, such as energy functions and geometric constraints (e.g., roto-translational invariance), into the deep learning architecture remains an active area of research (Schütt et al., 2017; Xu et al., 2022). Thirdly, achieving both *diversity* in generated conformations and efficient *sampling* of low-energy states is crucial for practical applications but often involves trade-offs (Liu et al., 2021). While recent methods like GeoDiff (Xu et al., 2022) and ConfFlow (Shah & Koltun, 2024) have made significant progress, GeoDiff relies on a potentially slow iterative diffusion process, and ConfFlow lacks explicit physical constraints during training, potentially limiting the physical realism of generated samples, especially for out-of-distribution predictions.

This research proposes a novel approach, **Physics-Informed Graph Normalizing Flows (PI-GNF)**, to address these challenges. Normalizing flows are attractive due to their exact likelihood calculation, invertible nature (allowing direct mapping between data and a simple latent space), and potential for efficient sampling in a single forward pass. We leverage Graph Neural Networks (GNNs) within the flow architecture to handle the inherent graph structure of molecules. Crucially, we embed physical priors directly into the training process by incorporating a differentiable physics-based energy penalty alongside the standard likelihood objective. This hybrid approach aims to guide the generative model towards learning not only the statistical patterns in conformational data but also the underlying energy landscapes dictated by fundamental physics. The model is designed to be inherently roto-translationally invariant, ensuring geometric consistency. This work aligns directly with the themes of the Workshop on Structured Probabilistic Inference & Generative Modeling, focusing on inference and generative methods for structured graph data, encoding domain knowledge (physics) into probabilistic models, and applications in natural science (chemistry).

### 2.2 Research Objectives

The primary goal of this research is to develop and evaluate the PI-GNF framework for generating chemically valid, physically realistic, and diverse molecular conformations efficiently. The specific objectives are:

1.  **Develop a Graph Normalizing Flow Architecture for Molecular Conformations:** Design an invertible GNF architecture based on GNNs that operates directly on molecular graphs and their 3D coordinates, ensuring efficient computation of the likelihood and its Jacobian determinant.
2.  **Integrate Physics-Based Priors:** Incorporate a differentiable potential energy term, derived from a computationally efficient force field approximation, into the GNF training objective. This term will penalize generated conformations with high energies or physically implausible geometries (e.g., incorrect bond lengths/angles).
3.  **Ensure Roto-Translational Invariance:** Design the GNF transformations or input representations such that the learned distribution and sampling process are inherently invariant to rigid rotations and translations of the molecule.
4.  **Optimize and Train the PI-GNF Model:** Develop a stable training procedure that balances the negative log-likelihood (NLL) objective (data fit) and the physics-based energy penalty (physical realism) using appropriate regularization and optimization techniques.
5.  **Comprehensive Evaluation and Comparison:** Rigorously evaluate the performance of PI-GNF on established benchmark datasets (e.g., QM9, GEOM-Drugs) against state-of-the-art conformation generation methods. Evaluation will focus on chemical validity, physical realism (energy distribution), conformational diversity, sampling efficiency, and accuracy (similarity to ground truth conformations).

### 2.3 Significance

This research holds significant potential for advancing both machine learning methodology and scientific discovery:

1.  **Improved Molecular Design Tools:** By generating higher quality (more valid, diverse, low-energy) molecular conformations more efficiently, PI-GNF can accelerate workflows in drug discovery and materials science, potentially leading to faster identification of promising candidate molecules.
2.  **Bridging Machine Learning and Physical Sciences:** This work provides a concrete framework for integrating fundamental physical laws (domain knowledge) into sophisticated deep generative models. This physics-informed approach can lead to more robust, interpretable, and generalizable models for scientific data, addressing a key challenge highlighted by the workshop.
3.  **Advancing Structured Probabilistic Modeling:** The proposed PI-GNF model contributes to the field of generative modeling for structured data (graphs), specifically addressing the challenges of incorporating constraints and invariances within the normalizing flow framework.
4.  **Efficient Sampling for Complex Distributions:** If successful, PI-GNF will offer a method for fast, single-pass sampling of low-energy states from a complex, physically-constrained distribution, contrasting with potentially slower iterative methods like diffusion models or MCMC-based approaches (e.g., GraphEBM; Liu et al., 2021).
5.  **Contribution to Open Science:** We plan to make the model implementation and trained weights publicly available, fostering reproducibility and further research in this area.

## 3. Methodology

### 3.1 Data Representation

A molecule will be represented as a graph $G = (V, E)$, where $V$ is the set of nodes (atoms) and $E$ is the set of edges (bonds). Each node $v \in V$ is associated with atomic features $h_v$ (e.g., atom type, charge) and its 3D coordinates $x_v \in \mathbb{R}^3$. The collection of all atomic coordinates forms the conformation matrix $X \in \mathbb{R}^{|V| \times 3}$. Edge features $e_{uv}$ can represent bond types (single, double, etc.). The input to our model during training will be pairs $(G, X)$ from known molecular conformation datasets.

### 3.2 Model Architecture: Graph Normalizing Flow

We propose a normalizing flow model $f: \mathcal{X} \rightarrow \mathcal{Z}$ that maps the space of molecular conformations $\mathcal{X}$ (conditioned on the graph structure $G$) to a simple base distribution $p_Z(z)$ in the latent space $\mathcal{Z}$, typically a standard multivariate Gaussian ($\mathcal{N}(0, I)$). The flow $f$ is constructed as a composition of $K$ invertible transformations (layers): $f = f_K \circ \dots \circ f_1$. Each transformation $f_k$ must be differentiable and have a computationally tractable Jacobian determinant. The probability density of a conformation $X$ is given by the change of variables formula:
$$ p_X(X | G) = p_Z(f(X)) \left| \det \left( \frac{\partial f(X)}{\partial X} \right) \right| $$
The log-likelihood is then:
$$ \log p_X(X | G) = \log p_Z(f(X)) + \sum_{k=1}^K \log \left| \det \left( \frac{\partial z_k}{\partial z_{k-1}} \right) \right| $$
where $z_0 = X$ and $z_k = f_k(z_{k-1})$.

Each layer $f_k$ will incorporate Graph Neural Networks (GNNs) to process the molecular graph structure $G$ and the current representation $z_{k-1}$ (which includes coordinate information). We will adapt existing GNF architectures suitable for graph data, such as graph coupling layers (similar to RealNVP adapted for graphs) or graph autoregressive flows. For instance, a graph coupling layer might split the nodes (or their coordinate dimensions) into two sets $A$ and $B$, updating one set based on the other using GNN-computed parameters:
$$ z_{k, A} = z_{k-1, A} $$
$$ z_{k, B} = s(z_{k-1, A}; G) \odot z_{k-1, B} + t(z_{k-1, A}; G) $$
where $s(\cdot)$ and $t(\cdot)$ are scaling and translation functions parameterized by GNNs operating on $z_{k-1, A}$ and the graph $G$. The Jacobian determinant for such layers is easily computable (sum of log-scales).

### 3.3 Roto-Translational Invariance

To ensure the learned distribution $p_X(X|G)$ is invariant to rotations and translations, we will enforce this property within the model architecture. One effective approach is to operate on internal coordinates (bond lengths, angles, dihedrals) or relative coordinates within the GNN layers. Alternatively, we can employ equivariant GNN layers (e.g., EGNN; Satorras et al., 2021) within the flow transformations ($f_k$). These layers are specifically designed to preserve SE(3)-equivariance (translation, rotation, and reflection equivariance). We will primarily investigate using relative inter-atomic distances and angles as inputs to standard GNNs within the flow layers, as this avoids the complexity of equivariant layers while still achieving invariance for the overall likelihood computation. The final output coordinates will be reconstructed from these internal or relative representations, potentially centered at the origin.

### 3.4 Physics-Informed Training Objective

The core innovation is the integration of physical knowledge via an energy penalty. We define a differentiable potential energy function $E_{physics}(X | G)$ that approximates the molecule's internal energy based on its conformation $X$ and graph structure $G$. This function will be based on terms from classical force fields, focusing on computationally inexpensive components:
$$ E_{physics}(X | G) = E_{bond}(X, G) + E_{angle}(X, G) + E_{nonbonded}(X, G) $$
where:
*   $E_{bond} = \sum_{(i,j) \in E} k_{b,ij} (d(x_i, x_j) - l_{0,ij})^2$ penalizes deviations of bond lengths $d(x_i, x_j)$ from their equilibrium values $l_{0,ij}$.
*   $E_{angle} = \sum_{(i,j,k) \text{ s.t. } (i,j),(j,k) \in E} k_{a,ijk} (\theta_{ijk} - \theta_{0,ijk})^2$ penalizes deviations of bond angles $\theta_{ijk}$ from their equilibrium values $\theta_{0,ijk}$.
*   $E_{nonbonded} = \sum_{i<j, (i,j) \notin E} V_{LJ}(d(x_i, x_j))$ could include a simple repulsive term (e.g., Lennard-Jones repulsion) to prevent atomic clashes for non-bonded atoms $i, j$. Van der Waals terms will be included if computationally feasible.
The parameters ($k_b, l_0, k_a, \theta_0$) will be based on standard force field values (e.g., derived from UFF or MMFF) depending on atom and bond types specified in $G$.

The final training objective combines the standard maximum likelihood estimation (MLE) for the normalizing flow with the minimization of this physics-based energy term. We aim to maximize the following objective function over the training dataset $\mathcal{D} = \{(G_i, X_i)\}$:
$$ \mathcal{L}(\theta) = \frac{1}{|\mathcal{D}|} \sum_{(G, X) \in \mathcal{D}} \left[ \log p_X(X | G; \theta) - \lambda E_{physics}(X | G) \right] $$
where $\theta$ represents the parameters of the GNF ($f_\theta$), and $\lambda \ge 0$ is a hyperparameter controlling the strength of the physical regularization. A larger $\lambda$ encourages the model to generate lower-energy conformations, potentially at the cost of slightly lower likelihood for the training data. The log-likelihood term $\log p_X(X | G; \theta)$ is computed as described in Section 3.2.

### 3.5 Training and Optimization

The model parameters $\theta$ will be optimized using stochastic gradient descent (SGD) or variants like Adam (Kingma & Ba, 2014) on mini-batches of molecular conformations. The energy term $E_{physics}$ must be differentiable with respect to atom coordinates $X$ to allow gradient-based optimization, which is true for standard force field terms. We will carefully tune the learning rate, batch size, and the regularization hyperparameter $\lambda$. A curriculum learning strategy might be employed, potentially starting with $\lambda=0$ and gradually increasing it during training to first learn the overall data distribution and then refine it towards lower-energy regions.

### 3.6 Sampling

Generating new conformations for a given molecular graph $G$ is efficient. We first sample a latent vector $z$ from the base distribution $p_Z(z)$ (e.g., $z \sim \mathcal{N}(0, I)$). Then, we compute the conformation $X$ by passing $z$ through the inverse flow $X = f^{-1}(z) = f_1^{-1} \circ \dots \circ f_K^{-1}(z)$. Since each layer $f_k$ is designed to be invertible, this requires only a single forward pass through the inverse network, making sampling significantly faster than iterative methods like MCMC or diffusion models.

### 3.7 Implementation Details

We plan to implement the PI-GNF model using Python with standard deep learning libraries such as PyTorch (Paszke et al., 2019) and GNN libraries like PyTorch Geometric (Fey & Lenssen, 2019) or DGL (Wang et al., 2019). For calculating the physics-based energy term and its gradients, we can either implement the simple force field terms directly using PyTorch's automatic differentiation or potentially leverage existing libraries like OpenMM (Eastman et al., 2017) if interoperability allows efficient batched computation within the training loop. RDKit (rdkit.org) will be used for cheminformatics tasks like molecule validation and input processing.

### 3.8 Experimental Design

#### 3.8.1 Datasets

We will evaluate our PI-GNF model on widely used molecular conformation datasets:
1.  **QM9:** Contains ~134k small organic molecules (up to 9 heavy atoms) with DFT-calculated geometries and properties (Ramakrishnan et al., 2014). Useful for initial validation and analysis of physical properties.
2.  **GEOM-Drugs:** A larger dataset containing ~300k drug-like molecules with >50 million conformations generated via classical force fields and DFT calculations (Axen et al., 2017; Ganea et al., 2021). This dataset provides greater chemical diversity and conformational complexity, suitable for evaluating scalability and performance on relevant structures.

#### 3.8.2 Baselines

We will compare PI-GNF against several state-of-the-art and representative baseline methods:
1.  **GeoDiff (Xu et al., 2022):** A strong diffusion-based model known for good performance, especially on larger molecules.
2.  **ConfFlow (Shah & Koltun, 2024):** A flow-based model using transformers, representing the high end of performance without explicit physics.
3.  **Standard GNF (PI-GNF with $\lambda=0$):** An ablation study to quantify the impact of the physics-based energy term.
4.  **Classical Methods (if feasible):** Potentially compare sampling efficiency and quality against standard methods like RDKit's ETKDG (Riniker & Landrum, 2015) or a simple MMFF94 optimization, although deep generative models focus on learning distributions rather than single conformer optimization.

#### 3.8.3 Evaluation Metrics

Performance will be assessed using a comprehensive set of metrics:

1.  **Chemical Validity:**
    *   **% Valid Molecules:** Percentage of generated conformations that pass basic chemical checks using RDKit (e.g., reasonable bond lengths, no atomic clashes).
    *   **Bond Length Distribution:** Comparison (e.g., using Wasserstein distance or KL divergence) between the distribution of bond lengths in generated conformations and the ground truth data or equilibrium values.
    *   **Bond Angle Distribution:** Similar comparison for bond angles.

2.  **Physical Realism:**
    *   **Energy Distribution:** Distribution of potential energies of the generated conformations, calculated using a consistent (potentially more accurate, e.g., MMFF94) external force field not used during training. Compare the average energy and spread against baselines and ground truth data. We expect PI-GNF to produce conformations with lower energies.

3.  **Conformational Diversity:**
    *   **Average Pairwise RMSD:** Calculate the root-mean-square deviation (RMSD) between pairs of unique conformations generated for the same molecule. Higher average RMSD indicates greater diversity.
    *   **Coverage of Known Conformers (CoCo Score):** For molecules with multiple known low-energy conformers in the dataset, measure the fraction of known conformers that are closely matched (low RMSD) by at least one generated conformer.

4.  **Accuracy/Fidelity:**
    *   **Affinity (Minimum RMSD):** For each ground truth conformation in the test set, generate multiple conformations with the model and report the minimum RMSD achieved between generated samples and the ground truth. Lower average minimum RMSD indicates better reconstruction fidelity.
    *   **Test Log-Likelihood:** Evaluate the average log-likelihood of the model on unseen test conformations (only applicable to flow-based models).

5.  **Efficiency:**
    *   **Sampling Time:** Measure the average wall-clock time required to generate one conformation for a molecule of a given size.

#### 3.8.4 Evaluation Protocol

For each molecule in the test split of the datasets, we will generate a fixed number of conformations (e.g., N=100 or N=1000) using PI-GNF and each baseline model. The metrics described above will then be computed based on these generated sets and compared against the ground truth conformations available in the test set. Statistical significance tests will be used where appropriate to compare performance differences. We will also perform ablation studies by varying the weight $\lambda$ of the physics penalty to understand the trade-off between data likelihood and physical realism.

## 4. Expected Outcomes & Impact

We anticipate the following outcomes from this research:

1.  **Demonstration of PI-GNF Effectiveness:** We expect PI-GNF to outperform baseline models (especially standard GNF and potentially non-physics-informed flows like ConfFlow) in generating chemically valid and physically realistic (lower energy) conformations, as measured by validity checks, energy distributions, and bond/angle statistics.
2.  **Competitive Diversity and Accuracy:** We aim for PI-GNF to achieve conformational diversity and accuracy (fidelity to known low-energy states) comparable or superior to state-of-the-art methods like GeoDiff, while offering significantly faster sampling.
3.  **Quantification of Physics-Informed Benefit:** Ablation studies ($\lambda=0$ vs $\lambda>0$) will clearly quantify the contribution of the integrated physics-based energy term to improving the quality of generated conformations.
4.  **Efficient Sampling:** PI-GNF is expected to demonstrate significantly faster sampling speeds (time per conformation) compared to iterative methods like diffusion models (GeoDiff) or MCMC-based approaches, owing to the single-pass nature of normalizing flows.
5.  **Insights into Physics-ML Integration:** The research will provide valuable insights into how to effectively combine domain-specific physical knowledge (force fields) with flexible deep generative models (GNFs) for scientific data modeling, particularly for structured data like molecular graphs.

**Impact:** Success in this project will deliver a powerful new tool for computational chemistry and drug discovery, enabling faster exploration of relevant conformational space with higher physical fidelity. It will represent a significant step forward in building more reliable and interpretable AI systems for science by demonstrating a robust method for encoding physical principles into deep learning models. This directly addresses the core themes of the workshop by advancing structured probabilistic inference, showcasing effective encoding of domain knowledge, and providing practical applications in a key scientific domain. The findings and open-source implementation will benefit the broader machine learning and scientific communities interested in generative modeling, graph representation learning, and AI for science. Potential future work includes extending the framework to handle larger molecules, incorporate more sophisticated physics (e.g., QM calculations approximations, solvent effects), and apply it to downstream tasks like property prediction or molecular docking.

**References:** (Note: A full proposal would include a complete bibliography. Key references mentioned in the text are listed below for context.)

*   Axen, S. D., et al. (2017). A simple representation of three-dimensional molecular structure. *Journal of Medicinal Chemistry*.
*   Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using Real NVP. *arXiv preprint arXiv:1605.08803*.
*   Eastman, P., et al. (2017). OpenMM 7: Rapid development of high performance algorithms for molecular dynamics. *PLoS Computational Biology*.
*   Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *arXiv preprint arXiv:1903.02428*.
*   Ganea, O.-E., et al. (2021). GEOM: Geometric Deep Learning for Molecular Conformation Generation. *ICLR Workshop on Geometrical and Topological Representation Learning*.
*   Goodfellow, I., et al. (2014). Generative adversarial nets. *Advances in Neural Information Processing Systems*.
*   Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*.
*   Jin, W., Barzilay, R., & Jaakkola, T. (2018). Junction tree variational autoencoder for molecular graph generation. *International Conference on Machine Learning*.
*   Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
*   Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*.
*   Kuznetsov, M., & Polykovskiy, D. (2021). MolGrow: A Graph Normalizing Flow for Hierarchical Molecular Generation. *arXiv preprint arXiv:2106.05856*.
*   Leach, A. R. (2001). *Molecular modelling: principles and applications*. Pearson Education.
*   Liu, M., et al. (2021). GraphEBM: Molecular Graph Generation with Energy-Based Models. *arXiv preprint arXiv:2102.00546*.
*   Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*.
*   Ragoza, M., et al. (2017). Protein–ligand scoring with convolutional neural networks. *Journal of Chemical Information and Modeling*.
*   Ramakrishnan, R., et al. (2014). Quantum chemistry structures and properties of 134 kilo molecules. *Scientific Data*.
*   Riniker, S., & Landrum, G. A. (2015). Better informed distance geometry: using what we know to improve conformation generation. *Journal of Chemical Information and Modeling*.
*   Satorras, V. G., et al. (2021). E(n) equivariant graph neural networks. *International Conference on Machine Learning*.
*   Schütt, K. T., et al. (2017). SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. *Advances in Neural Information Processing Systems*.
*   Shah, S. A., & Koltun, V. (2024). Conformation Generation using Transformer Flows. *arXiv preprint arXiv:2411.10817* (Note: Fictional arXiv ID used from prompt).
*   Skalic, M., et al. (2019). Shape-based generative modeling for de novo drug design. *Journal of Chemical Information and Modeling*.
*   Wang, M., et al. (2019). Deep graph library: A graph-centric, highly-performant package for graph neural networks. *arXiv preprint arXiv:1909.01315*.
*   Xu, M., et al. (2022). GeoDiff: a Geometric Diffusion Model for Molecular Conformation Generation. *International Conference on Learning Representations (ICLR)*.
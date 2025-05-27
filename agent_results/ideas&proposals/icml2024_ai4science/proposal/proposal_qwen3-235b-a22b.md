# Symmetry-Driven Foundation Model Scaling for Molecular Dynamics

## Introduction

Over the past decade, machine learning has revolutionized scientific discovery, particularly in domains such as materials science and molecular biology, where it enables accurate prediction of quantum mechanical properties, acceleration of molecular dynamics (MD) simulations, and exploration of vast chemical spaces with unprecedented precision. Foundation models, pretrained on massive datasets, have demonstrated remarkable success in computer vision, natural language processing, and graph-based systems, offering the potential for high generalization across downstream tasks. In MD applications, recent progress has been made in incorporating physical symmetries—such as translational, rotational, and permutation invariance—into neural network architectures, leading to substantial improvements in data efficiency and predictive accuracy. However, the challenge of scaling such models for complex, high-dimensional molecular systems remains largely unmet. Naively increasing model size or dataset diversity often results in rapidly diminishing returns, where computational costs escalate without commensurate gains in accuracy, and where interpretability diminishes as models grow more opaque.

Our central objective is to develop a scalable, symmetry-aware foundation model for MD tasks that balances accuracy, interpretability, and compute expenditure through a principled, three-stage pipeline. First, we will construct and pretrain an E(3)-equivariant Transformer architecture—a hybrid of modern attention mechanisms and symmetry-aware graph network primitives—on large-scale atomistic datasets. Second, we will derive and implement physics-informed scaling laws to dynamically adjust model capacity and dataset granularity based on empirical validation metrics and compute budgets, extending the work of arXiv:2302.23456 in a systematic fashion. Third, we will integrate uncertainty quantification (UQ)-driven active sampling to iteratively refine model predictions and identify gaps in chemical motif representation, building upon arXiv:2303.34567.

This proposal directly addresses the limitations identified in recent literature (e.g., the high computational costs of E(3)-equivariant models in arXiv:2204.05249 and the interpretability bottlenecks in arXiv:2305.56789), while introducing a synergistic framework that systematically improves the trade-offs along the methodology–interpretability–discovery frontier. Success in this project could enable orders-of-magnitude acceleration in MD-driven applications such as drug design and materials discovery, thereby lowering the barrier to discovery across chemistry and physics. Additionally, by demonstrating a general protocol for adaptive model scaling guided by physical priors and UQ, we aim to contribute a replicable blueprint for AI-for-science efforts beyond MD.

---

## Methodology

We design a three-stage pipeline for symmetry-driven scaling of foundation models in MD, combining equivariant architecture design, physics-guided scaling laws, and active data refinement.

### Stage 1: Equivariant Transformer Foundation Model for Atomistic Graphs

We propose **SE(3)-Eformer**, a Transformer-style foundation model combining global attention mechanisms with strict E(3) symmetry preservation. Leveraging insights from **Equiformer** arXiv:2206.11990, our architecture replaces standard self-attention with tensor-product attention that respects the spatial geometry of atomic systems. Each atom $i$ is represented with a scalar feature $ s_i \in \mathbb{R} $ and a vector feature $ v_i \in \mathbb{R}^3 $, capturing electronic and structural properties respectively. The attention weight between atoms $i$ and $j$ is derived from a geometric tensor product:

$$
A_{ij} = \text{Softmax} \left( \frac{(W_s s_i + \sigma(W_v [v_i \otimes v_j]))}{\sqrt{d}} \right)
$$

where $ \otimes $ denotes the tensor product, $ W_s $ and $ W_v $ are learnable weights, $ \sigma $ is a differentiable activation function, and $ d $ is the feature dimension. This design ensures that relative atomic displacements and orientations are preserved in feature space, adhering to rotational and translational invariances.

For pretraining, we curate a dataset of over **10^8 molecular conformers** generated via extensive MD simulations of the QM9 arXiv:1406.2209, MD17 arXiv:1806.07253, and OC22 dataset arXiv:2212.09721. Pretraining loss minimizes mean absolute error (MAE) on atomic forces and potential energies:

$$
\mathcal{L}_{\text{pretrain}} = \frac{1}{N} \sum_{i=1}^{N} \left( |\hat{E}_i - E_i| + \lambda |\hat{F}_i - F_i| \right)
$$

where $ \hat{E} $ and $ \hat{F} $ are model outputs, $ \lambda $ is a hyperparameter balancing energy and force errors, and $ N $ is the number of atoms. The model will be optimized via AdamW arXiv:1711.05101, with learning rate warmup and decay over 1 million steps.

### Stage 2: Physics-Informed Adaptive Scaling

We develop **Dynamic Symmetry-Aware Scaling Laws (DiaSL)** to determine when and how to allocate resources for model/data expansion during training. Motivated by arXiv:2302.23456, our laws are derived by modeling validation loss $ V(C, E, M) $ as a function of compute (FLOPs) $ C $, dataset entropy $ E $, and model parameters $ M $. A Pareto optimal regime is found by identifying the point where the marginal gains in test accuracy $ \Delta A $ fall below a threshold $ \epsilon $:

$$
\Delta A(C, E, M) = \frac{A_{\text{new}} - A_{\text{old}}}{C_\text{new} - C_\text{old}} < \epsilon \Rightarrow \text{trigger expansion}
$$

Here, expansion could mean increasing model width (via new attention heads) if underfitting dominates, or diversifying the dataset (via generating new trajectories from high-uncertainty conformers) if overfitting occurs. We will calibrate these laws on validation sets from QM9 and OC22, fitting them to empirical curves of accuracy vs. compute. The scaling protocol will be implemented through a reinforcement learning controller that dynamically selects the optimal next scaling step.

### Stage 3: Uncertainty-Driven Active Sampling for Refinement

To address data inefficiencies and improve model generalization, we introduce an **active conformation refinement (ACRe)** algorithm. At each iteration, SE(3)-EFormer outputs per-atom uncertainty estimates (from Monte Carlo dropout arXiv:1506.02142) and conformational uncertainties from ensemble variance. Underrepresented motifs are identified through:

1. **Geometric Outlier Detection**: Atoms where local symmetry constraints are violated beyond a learned threshold.
2. **Entropy-based Motif Prioritization**: High entropy motifs in attention maps (from Equation 1) indicate poor feature alignment.

These motifs will be used to generate targeted high-fidelity simulations using **Metadynamics with Neural Priors**—a method where CVs are selected via attention weights and used to generate bias potentials using a neural network. The resultant data will be fine-tuned into SE(3)-EFormer, repeating the pipeline until convergence. We will evaluate this via:

- **Conformational Coverage**: Fraction of known chemical motifs correctly represented.
- **Energy Drift**: Stability of model predictions across iteratively refined simulations.
- **Hypothesis-Driven Validation**: Ability to recapitulate novel metastable conformers previously found via rare-event sampling.

### Experimental Design

We benchmark our model on tasks critical for drug and materials discovery:

1. **Free Energy Estimation**: Perform thermodynamic integration using SE(3)-EFormer potentials.
2. **Long Timescale Sampling**: Measure folding rates of peptides like Chignolin (10 residues), comparing to ground truth simulations from Anton 2 arXiv:1811.08926.
3. **High-Throughput Screening**: Use MD trajectories to predict aggregation rates of small molecules.
4. **Interpretability**: Visualizing attention maps for known symmetry-violating conformations (e.g., cis-trans isomerization).

Our comparison baselines include **NequIP** arXiv:2101.03164, **Allegro** arXiv:2204.05249, and **Equiformer** arXiv:2206.11990. Evaluation metrics:

- **Accuracy**: MAE and RMSE on energy (eV), force (eV/Å), and conformational predictions.
- **Compute Efficiency**: Inference steps per second and energy-error vs. FLOPs scaling.
- **Data Efficiency**: Validation error as a function of dataset size.
- **Interpretability**: Correlation between attention maps and known hydrogen-bonding sites; symmetry violation quantified via rotation-invariance tests.

Hyperparameter sweeps (learning rate, λ in Equation 2, dropout rate) will be conducted across 2,000 A100 GPU hours. We use NVIDIA DGX clusters for training and LAMMPS arXiv:1506.02217 for validation simulations.

---

## Expected Outcomes & Impact

Our work will produce **SE(3)-EFormer**, a novel scalable architecture integrating E(3) symmetry into attention-based MD models. By stage 3, we anticipate:

- A **2× improvement** in accuracy-per-FLOP over current foundation models.
- **Enhanced generalization** on out-of-distribution molecules (e.g., those with non-standard bond types).
- Model predictions that **recover rare events** (e.g., protein folding transitions) within 10× fewer steps than traditional MD.

This will transform two areas:

1. **Method Development**: We establish a new paradigm for adaptive scaling, where physical priors dictate architectural growth and training efficiency, rather than heuristic rules.
2. **Discovery in Drug & Materials Science**: More efficient MD simulations mean broader sampling of conformational space and faster evaluation of potential candidates—a critical need in pharmaceutical discovery and battery material design.

Our contribution also advances the foundational understanding of scaling in science. By formalizing symmetry-enforced scaling laws (DiaSL) and uncertainty-guided refinement (ACRe), we provide a replicable strategy for AI-for-science projects seeking interpretability and cost optimization. We aim to share models, code, and datasets at the Open Catalyst Challenge platform to foster open science innovation.

Beyond MD, the principles of symmetry-aware scaling and active refinement could benefit fields like quantum chemistry, where enforcing N-representability or spin symmetries remains challenging, and continuum mechanics, where symmetries constrain fluid dynamics and elasticity. Through this research, we expect AI-for-science to advance toward principled, physics-guided growth, unlocking new avenues for discovery under fixed resource constraints.
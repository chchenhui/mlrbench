1. Title  
Equivariant World Models: Leveraging SE(2)/SE(3) Symmetry for Sample-Efficient Robotic Manipulation and Navigation  

2. Introduction  
Background  
Recent advances in both neuroscience and artificial intelligence point to a convergent principle: biological and artificial neural systems that respect underlying geometric structure exhibit superior efficiency, robustness, and generalization. In neuroscience, circuits representing head direction, grid cells, and motor cortex activity mirror the topological and group structure of sensory and motor variables (Kim et al. 2017; Wolff et al. 2015; Gallego et al. 2017). In parallel, the field of Geometric Deep Learning shows that injecting group and manifold priors into network architectures—through equivariant layers, steerable filters, and related constructions—leads to substantial gains in computational efficiency and out-of-distribution generalization (Bronstein et al. 2021).

In embodied AI, world models serve as predictive simulators of environment dynamics and rewards, enabling model-based reinforcement learning (RL) agents to plan and optimize policies. However, most existing world-model architectures do not exploit the fact that many robotic tasks exhibit strong symmetry: planar navigation is equivariant under SE(2) (rotations + translations in the plane), and manipulation in three-dimensional space is equivariant under SE(3). Ignoring these symmetries forces the model to relearn identical dynamics from differently oriented data, driving up sample complexity and hindering robustness.

Research Objectives  
This proposal aims to develop a framework of Equivariant World Models (EWMs) that explicitly enforce SE(2) or SE(3) equivariance in both state-transition and reward predictors. We will:  
• Derive group-equivariant network layers suitable for high-dimensional sensory inputs (e.g. images, point clouds) and non-Euclidean state variables (e.g. orientations).  
• Integrate these layers into a state-space world-model architecture and train it end-to-end with symmetry-aware data augmentation.  
• Embed the learned model into a planning or model-based RL loop to generate control policies.  
• Validate sample efficiency, generalization to novel rotations/translations/scales, and sim-to-real transfer on robotic navigation and manipulation benchmarks.

Significance  
By embedding known geometric priors directly into corporate world-model architectures, we expect to reduce the number of environment interactions required for reliable planning by up to an order of magnitude, while improving out-of-distribution generalization to unseen poses. Such sample-efficient, robust learning is critical for real-world robotics in unstructured environments (homes, warehouses) where data collection is costly. Moreover, this work forges a clear bridge between geometric deep learning and embodied AI, aligning with emerging neuroscience insights about geometry in neural representations.

3. Methodology  
3.1. Overall Research Design  
We propose a three-stage pipeline: (1) design and implementation of SE(2)/SE(3)-equivariant network modules, (2) integration into a latent-state world-model architecture trained with symmetry-augmented data, and (3) embedding into a model-based RL planner for downstream control tasks.  

3.2. Group-Equivariant Building Blocks  
Let \(G\) denote a Lie group (either SE(2) or SE(3)), and let \(\rho: G \to GL(V)\) be a representation on the feature space \(V\). A layer \(L\) is \(G\)-equivariant if for all \(g\in G\),
\[
L(\rho(g) x) \;=\; \rho'(g)\,L(x)\,,
\]
where \(\rho'\) is the representation used in the next layer.  

3.2.1. Equivariant Convolutions (e-convs)  
For image-based inputs \(x: \mathbb{R}^2\to \mathbb{R}^C\), we implement a discrete SE(2)-convolution:  
\[
(Lf)(y) \;=\; \sum_{g\in G_d} K(g^{-1} y)\,f(g)\,,
\]
where \(G_d\) is a finite sampling of rotations and translations, and \(K\) are learnable steerable kernels parameterized in a basis of spherical harmonics (for SE(3)) or Fourier modes (for SE(2)).  

3.2.2. Equivariant Point-Cloud Processing  
For 3D point clouds we build on Tensor Field Networks (Thomas et al. 2018): each feature \(f_i\) at point \(p_i\) is expanded in a basis of irreducible representations. Convolution weights are radial functions \(W(r)\) times spherical harmonics \(Y_\ell(\hat r)\), enforcing full SE(3) equivariance.  

3.3. World Model Architecture  
We adopt a latent-state sequence model akin to PlaNet or Dreamer (Hafner et al. 2019), augmented with equivariant encoders, decoders, and transition networks.  

3.3.1. Latent Transition Model  
Let \(s_t\in \mathbb{R}^d\) be the latent state at time \(t\). We enforce
\[
f_\theta\bigl(\rho(g) s_t,\; a_t \bigr) \;=\; \rho(g)\,f_\theta(s_t,a_t)\quad\forall g\in G,
\]
where \(a_t\) is the action. We implement \(f_\theta\) as a stack of equivariant MLP blocks, each respecting a chosen representation \(\rho\).  

3.3.2. Observation Encoder/Decoder  
• Encoder \(E_\phi\): maps raw observations \(o_t\) (images or point clouds) to \(s_t\). Built from stacked e-convs or equivariant point-cloud layers, followed by channel-wise pooling into a group representation.  
• Decoder \(D_\psi\): reconstructs \(o_t\) from \(s_t\) using tied equivariant filters.  

3.3.3. Reward Predictor  
A small equivariant network \(R_\eta(s_t)\) outputs scalar rewards. Equivariance implies \(R_\eta(\rho(g)s_t)=R_\eta(s_t)\), i.e. invariance under group action.  

3.3.4. Training Objective  
We minimize a composite loss
\[
\mathcal{L} = \underbrace{\sum_t \|D_\psi(s_t) - o_t\|^2}_{\text{reconstruction}}
\;+\;\underbrace{\sum_t \|f_\theta(s_t,a_t)-s_{t+1}\|^2}_{\text{forward loss}}
\;+\;\underbrace{\sum_t (R_\eta(s_t)-r_t)^2}_{\text{reward loss}}
\;+\;\beta\,\mathcal{L}_{\rm reg}.
\]
Regularization \(\mathcal{L}_{\rm reg}\) enforces norm penalties and eigenvalue constraints on linear operators to maintain stability.  

3.4. Symmetry-Aware Data Augmentation  
Given an observed tuple \((o_t,a_t,o_{t+1})\), sample \(g\sim\text{Uniform}(G_d)\) and generate
\[
(o'_t,a'_t,o'_{t+1}) \;=\; \bigl(g\cdot o_t,\; g\cdot a_t,\; g\cdot o_{t+1}\bigr).
\]
Because the model is equivariant by design, these augmented samples expand the training set without violating symmetry constraints.  

3.5. Model-Based RL Integration  
We use Model-Predictive Control (MPC) in latent space: at each time step, we roll out the learned \(f_\theta\) for \(H\) steps under candidate action sequences \(\{a_{t:t+H-1}\}\), evaluate cumulative predicted reward, and select the best first action. Pseudocode:  

  Initialize world model \((\phi,\theta,\psi,\eta)\)  
  Initialize environment  
  for iteration = 1…N do  
    collect real transitions \((o_t,a_t,o_{t+1},r_t)\) under current MPC policy  
    update world model by minimizing \(\mathcal{L}\) over all data  
    for each new real step:  
      encode \(s_t=E_\phi(o_t)\)  
      sample candidate action sequences using CEM or random shooting  
      for each sequence, roll out \(\hat s_{t+1:t+H}\) with \(f_\theta\), sum \(\sum R_\eta(\hat s)\)  
      execute best action \(a_t\) in real env  
  end  

3.6. Experimental Design and Evaluation  
Tasks:  
• Planar Navigation: 2D maze with goal locations; SE(2) symmetry.  
• Object Manipulation: pick-and-place with different object orientations; SE(3) symmetry.  
• Sim-to-Real Transfer: train in MuJoCo and deploy on a real UR5 arm for rotated pick tasks.  

Baselines:  
• Standard world model without equivariance (CNN + MLP).  
• Data-augmented world model (no equivariant layers).  
• Equivariant policy network but non-equivariant world model.  

Metrics:  
• Sample Efficiency: number of environment steps to reach target success rate (e.g. 90%).  
• Generalization Gap: performance drop when tested on novel rotations/translations/scales.  
• Sim-to-Real Transfer Gap: difference in success rate between simulation and real robot.  

Statistical Validation:  
• Repeat each experiment with 5 random seeds.  
• Report means ± standard error.  
• Perform paired t-tests between EWM and each baseline.  

Implementation Details:  
• Group discretization: 16 rotations in SE(2), 24 rotations in SO(3) combined with 8 radial translations → 192 group elements.  
• Network sizes: 4 equivariant conv layers with 64 channels each, followed by 3 equivariant MLP blocks of width 256.  
• Optimizer: Adam with learning rate \(3\times10^{-4}\), batch size 128, weight decay \(10^{-5}\).  
• Training horizon \(H=15\), MPC planning horizon 10, CEM samples 500.  

4. Expected Outcomes & Impact  
Expected Outcomes  
• A modular library of SE(2)/SE(3)-equivariant layers for images and point clouds, released as open-source.  
• Demonstration that Equivariant World Models achieve 3×–10× faster convergence in sample-efficiency compared to non-equivariant baselines on navigation and manipulation tasks.  
• Empirical generalization: less than 5% performance degradation when faced with unseen rotations or translations, versus >30% for baselines.  
• Successful sim-to-real transfer: achieving ≥80% task success on a real robotic arm with zero fine-tuning.  

Broader Impact  
This work directly addresses the critical bottleneck of data inefficiency in robotic learning, enabling practical deployment in environments where real-world data collection is expensive and time-consuming. By embedding known symmetries into world models, we foster a new class of robust, generalizable, and biologically inspired learning systems. This project bridges foundational neuroscience insights on geometric representations with cutting-edge geometric deep learning, contributing to a unified theory of symmetry in both brains and machines. It aligns with the NeurReps workshop’s theme of symmetry-aware neural representations and paves the way for future work on topology, dynamics, and interpretability in embodied AI.
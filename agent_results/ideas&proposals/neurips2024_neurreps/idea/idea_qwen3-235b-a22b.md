**Title:** *Dynamic Equivariant Neural Networks: Bridging Geometric Deep Learning and Biological Spatial Representations*  

**Motivation:**  
Biological neural circuits, such as grid cells and head-direction neurons, exploit geometric symmetries to represent spatial environments efficiently. Similarly, geometric deep learning uses equivariant architectures to preserve structural priors in data. However, most AI models neglect the dynamic interplay between geometric representations and temporal evolution observed in biological systems. This gap limits their ability to generalize in complex, real-world environments. Bridging this divide could yield robust models for robotics and deepen our understanding of neural computation.  

**Main Idea:**  
We propose *Dynamic Equivariant Neural Networks* (DENNs), which integrate Lie group theory and dynamical systems into geometric deep learning. Inspired by the brain’s use of continuous attractor networks for spatial coding, DENNs will:  
1. Encode symmetries via Lie group representations (e.g., SO(3) for rotations), enabling explicit equivariance to transformations.  
2. Model temporal dynamics using Hamiltonian mechanics on group manifolds, mimicking how neural circuits update representations during motion.  
3. Combine geometric priors with self-supervised learning to adaptively discover task-relevant symmetries in data.  

We will validate DENNs on robotic navigation tasks requiring generalization across novel environments, comparing performance to standard equivariant models and biological data. Expected outcomes include improved sample efficiency and robustness to domain shifts, alongside interpretable latent representations mirroring grid cell activity. This work could unify insights from neuroscience and geometric deep learning, advancing both AI and our understanding of neural coding mechanisms.
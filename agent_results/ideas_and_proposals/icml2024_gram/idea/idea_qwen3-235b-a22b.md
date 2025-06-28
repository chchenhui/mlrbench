**Title:** Geometry-Preserving Diffusion Models for Generating Manifold-Constrained Data  

**Motivation:**  
Generating geometrically accurate 3D structures (e.g., atomic configurations or biomedical surfaces) remains a critical challenge in computational sciences. Traditional diffusion models struggle with non-Euclidean data manifolds (e.g., spherical/hyperbolic spaces), often distorting local geometry or violating physical constraints. This limits their utility in domains like drug discovery or robotics, where preserving structural symmetries and manifold integrity is essential for scientific validity and functional utility.  

**Main Idea:**  
We propose **Geometric Diffusion Flows (GDF)**, a framework that integrates Riemannian geometry and equivariant learning into diffusion-based generative models. The key innovation involves:  
1. *Manifold-aware noise scheduling*: Noise perturbations and denoising steps are formulated in tangent spaces of the data manifold, preserving intrinsic curvature during forward and reverse processes.  
2. *Equivariant neural SDEs*: Symmetry-preserving architectures (e.g., SO(3)-equivariant transformers) parameterize stochastic dynamics on manifolds, ensuring transformations like rotations are exact.  
3. *Energy-based geometric priors*: A physics-informed loss enforces physical constraints (e.g., van der Waals distances in molecules) via differentiable Riemannian metrics.  

This approach will generate high-fidelity geometric objects (e.g., proteins, patients’ anatomical structures) while respecting topological and symmetry constraints. We anticipate significant advancements in data efficiency for scientific applications and foundational insights into combining diffusion processes with geometric structure preservation.
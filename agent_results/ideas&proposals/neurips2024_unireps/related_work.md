1. **Title**: Towards a Learning Theory of Representation Alignment (arXiv:2502.14047)
   - **Authors**: Francesco Insulla, Shuo Huang, Lorenzo Rosasco
   - **Summary**: This paper explores the alignment of representations in AI models as their scale and performance increase. It reviews various notions of alignment and focuses on "stitching," which examines the interplay between different representations in the context of a task. The authors relate stitching properties to kernel alignment, offering a learning-theoretic perspective on representation alignment.
   - **Year**: 2025

2. **Title**: Formation of Representations in Neural Networks (arXiv:2410.03006)
   - **Authors**: Liu Ziyin, Isaac Chuang, Tomer Galanti, Tomaso Poggio
   - **Summary**: The authors propose the Canonical Representation Hypothesis (CRH), suggesting that during training, latent representations, weights, and neuron gradients become mutually aligned. This alignment leads to compact representations invariant to task-irrelevant transformations. The paper also introduces the Polynomial Alignment Hypothesis (PAH), linking the balance between gradient noise and regularization to the emergence of canonical representations.
   - **Year**: 2024

3. **Title**: SARA: Structural and Adversarial Representation Alignment for Training-efficient Diffusion Models (arXiv:2503.08253)
   - **Authors**: Hesen Chen, Junyan Wang, Zhiyu Tan, Hao Li
   - **Summary**: SARA introduces a hierarchical alignment framework for diffusion models, enforcing multi-level representation constraints: patch-wise alignment for local semantic details, autocorrelation matrix alignment for structural consistency, and adversarial distribution alignment for global representation coherence. This approach accelerates convergence and improves generation quality.
   - **Year**: 2025

4. **Title**: You Are What You Eat -- AI Alignment Requires Understanding How Data Shapes Structure and Generalisation (arXiv:2502.05475)
   - **Authors**: Simon Pepin Lehalleur, Jesse Hoogland, Matthew Farrugia-Roberts, Susan Wei, Alexander Gietelink Oldenziel, George Wang, Liam Carroll, Daniel Murfet
   - **Summary**: This position paper argues that understanding the relationship between data distribution structure and trained models' internal structure is central to AI alignment. The authors discuss how two neural networks with equivalent training performance can generalize differently due to differences in internal computation, emphasizing the need for a robust mathematical science of AI alignment.
   - **Year**: 2025

**Key Challenges:**

1. **Architectural Disparities**: Merging models with different architectures poses significant challenges due to variations in layer configurations, activation functions, and parameter distributions.

2. **Task Distribution Variability**: Models trained on slightly varied task distributions may develop divergent representations, complicating the alignment process.

3. **Functional Alignment Complexity**: Achieving functional alignment conditioned on specific downstream task properties requires sophisticated techniques to map activation spaces effectively.

4. **Computational Efficiency**: Developing lightweight "stitching" layers that enable efficient merging without extensive fine-tuning or parameter averaging is a complex task.

5. **Generalization Assurance**: Ensuring that merged models generalize well across diverse tasks and inputs remains a significant challenge in the field. 
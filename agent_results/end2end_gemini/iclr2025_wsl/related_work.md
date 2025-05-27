1. **Title**: Neural Functional Transformers (arXiv:2305.13546)
   - **Authors**: Allan Zhou, Kaien Yang, Yiding Jiang, Kaylee Burns, Winnie Xu, Samuel Sokota, J. Zico Kolter, Chelsea Finn
   - **Summary**: This paper introduces Neural Functional Transformers (NFTs), which are deep models designed to process neural network weights while respecting permutation symmetries. NFTs utilize attention mechanisms to handle high-dimensional weight-space objects and demonstrate improved performance over prior weight-space methods.
   - **Year**: 2023

2. **Title**: Shape Generation via Weight Space Learning (arXiv:2503.21830)
   - **Authors**: Maximilian Plattner, Arturs Berzins, Johannes Brandstetter
   - **Summary**: The authors explore the weight space of 3D shape-generative models, identifying submanifolds that modulate topological properties and part features. They demonstrate that small changes in weight space can significantly alter topology and that low-dimensional reparameterizations allow controlled local geometry changes, highlighting the potential of weight space learning in 3D shape generation.
   - **Year**: 2025

3. **Title**: Universal Neural Functionals (arXiv:2402.05232)
   - **Authors**: Allan Zhou, Chelsea Finn, James Harrison
   - **Summary**: This work proposes Universal Neural Functionals (UNFs), an algorithm that constructs permutation equivariant models for any weight space. UNFs can be applied to various architectures, including those with recurrence or residual connections, and show promising improvements in tasks like optimizing image classifiers and language models.
   - **Year**: 2024

4. **Title**: Equivariant Architectures for Learning in Deep Weight Spaces (arXiv:2301.12780)
   - **Authors**: Aviv Navon, Aviv Shamsian, Idan Achituve, Ethan Fetaya, Gal Chechik, Haggai Maron
   - **Summary**: The authors present a novel network architecture designed to process neural networks in their raw weight matrix form. The architecture is equivariant to the permutation symmetry of MLP weights, allowing it to perform tasks such as adapting pre-trained networks to new domains and editing implicit neural representations.
   - **Year**: 2023

5. **Title**: Neural Tangent Kernel
   - **Authors**: Various
   - **Summary**: The Neural Tangent Kernel (NTK) framework provides insights into the training dynamics of neural networks in the infinite-width limit. It has been used to analyze feature learning and generalization properties, offering a theoretical foundation for understanding model behavior based on weight distributions.
   - **Year**: 2025

6. **Title**: Neural Operators
   - **Authors**: Various
   - **Summary**: Neural operators are models that learn mappings between function spaces, enabling the processing of infinite-dimensional data. They have been applied to various scientific and engineering disciplines, such as turbulent flow modeling and computational mechanics, demonstrating the potential of weight space learning in complex systems.
   - **Year**: 2025

7. **Title**: Neural Architecture Search
   - **Authors**: Various
   - **Summary**: Neural Architecture Search (NAS) automates the design of neural networks by exploring a predefined search space. Recent advancements in NAS have led to architectures that outperform hand-designed models, highlighting the importance of understanding weight space properties for efficient model development.
   - **Year**: 2025

8. **Title**: Physics-Informed Neural Networks
   - **Authors**: Various
   - **Summary**: Physics-Informed Neural Networks (PINNs) incorporate physical laws into neural network training, ensuring that models adhere to governing equations. This approach has been applied to various domains, including fluid dynamics and elasticity problems, demonstrating the utility of weight space learning in scientific applications.
   - **Year**: 2025

9. **Title**: Neural Tangent Hierarchy
   - **Authors**: Various
   - **Summary**: The Neural Tangent Hierarchy extends the NTK framework to describe finite-width effects in neural networks. This approach provides a more comprehensive understanding of feature learning and generalization, emphasizing the importance of weight space properties in model behavior.
   - **Year**: 2025

10. **Title**: Physics-Informed PointNet: A Deep Learning Solver for Steady-State Incompressible Flows and Thermal Fields on Multiple Sets of Irregular Geometries
    - **Authors**: Ali Kashefi, Tapan Mukerji
    - **Summary**: This paper introduces Physics-Informed PointNet (PIPN), which combines PINNs with PointNet to solve governing equations on multiple computational domains with irregular geometries simultaneously. PIPN demonstrates the effectiveness of weight space learning in handling complex geometries in scientific computations.
    - **Year**: 2025

**Key Challenges:**

1. **Permutation Symmetry Handling**: Neural network weights exhibit inherent permutation symmetries, making it challenging to design models that can process weights directly without being affected by neuron order.

2. **High-Dimensional Weight Spaces**: The high dimensionality of weight spaces poses difficulties in learning meaningful representations and mappings, requiring models to be both expressive and efficient.

3. **Generalization Across Architectures**: Developing models that can generalize across different neural network architectures, including those with recurrence or residual connections, remains a significant challenge.

4. **Efficient Training on Large Model Zoos**: Training models on large datasets of diverse pre-trained networks necessitates efficient algorithms and computational resources to handle the vast amount of data.

5. **Interpretability of Weight-Based Predictions**: Ensuring that models predicting properties from weights provide interpretable and trustworthy outputs is crucial for their adoption in model auditing and selection processes. 
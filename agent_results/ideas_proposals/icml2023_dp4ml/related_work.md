1. **Title**: Set-Valued Sensitivity Analysis of Deep Neural Networks (arXiv:2412.11057)
   - **Authors**: Xin Wang, Feilong Wang, Xuegang Ban
   - **Summary**: This paper introduces a sensitivity analysis framework based on set-valued mapping to understand how deep neural network solutions respond to perturbations in training data. It focuses on the expansion and contraction of the solution set, providing insights into the robustness and reliability of DNNs during training.
   - **Year**: 2024

2. **Title**: Metric Tools for Sensitivity Analysis with Applications to Neural Networks (arXiv:2305.02368)
   - **Authors**: Jaime Pizarroso, David Alfaya, José Portela, Antonio Muñoz
   - **Summary**: The authors propose a theoretical framework using metric techniques for sensitivity analysis in machine learning models. They introduce a family of quantitative metrics called α-curves, offering deeper insights into input variable importance compared to existing explainable AI methods.
   - **Year**: 2023

3. **Title**: Topological Interpretability for Deep-Learning (arXiv:2305.08642)
   - **Authors**: Adam Spannaus, Heidi A.Hanson, Lynne Penberthy, Georgia Tourassi
   - **Summary**: This work presents a method employing topological and geometric data analysis to infer prominent features in deep learning models. By creating a graph of a model's feature space, the approach identifies features relevant to the model's decisions, enhancing interpretability.
   - **Year**: 2023

4. **Title**: Application of Sensitivity Analysis Methods for Studying Neural Network Models (arXiv:2504.15100)
   - **Authors**: Jiaxuan Miao, Sergey Matveev
   - **Summary**: The study demonstrates the capabilities of various sensitivity analysis methods, including Sobol global sensitivity analysis and activation maximization, in analyzing neural networks. It applies these methods to feedforward and convolutional neural networks, providing insights into model behavior and input parameter importance.
   - **Year**: 2025

5. **Title**: Utility-Probability Duality of Neural Networks (arXiv:2305.14859)
   - **Authors**: Huang Bojun, Fei Yuan
   - **Summary**: This paper proposes interpreting neural networks as ordinal utility functions rather than probability models. The authors demonstrate that training neural networks can be viewed as a utility learning process, offering an alternative perspective on model behavior and decision-making.
   - **Year**: 2023

6. **Title**: Interpreting and Generalizing Deep Learning in Physics-Based Problems with Functional Linear Models (arXiv:2307.04569)
   - **Authors**: Amirhossein Arzani, Lingxiao Yuan, Pania Newell, Bei Wang
   - **Summary**: The authors propose generalized functional linear models as interpretable surrogates for trained deep learning models in physics-based problems. This approach aims to enhance interpretability and generalization capabilities beyond training data.
   - **Year**: 2023

7. **Title**: Sensitivity Analysis Using Physics-Informed Neural Networks
   - **Authors**: Not specified
   - **Summary**: This paper introduces a method for local sensitivity analysis using Physics-Informed Neural Networks (PINNs). By adding a regularization term to the loss function, the approach enables the computation of solution sensitivities with respect to parameters of interest, demonstrated through various examples.
   - **Year**: 2024

8. **Title**: Goal-Oriented Sensitivity Analysis of Hyperparameters in Deep Learning (arXiv:2207.06216)
   - **Authors**: Paul Novello, Gaël Poëtte, David Lugato, Pietro Marco Congedo
   - **Summary**: The authors study the use of goal-oriented sensitivity analysis, based on the Hilbert-Schmidt Independence Criterion (HSIC), for hyperparameter analysis and optimization in deep learning. This approach quantifies hyperparameters' relative impact on a neural network's final error, aiding in better understanding and optimization.
   - **Year**: 2022

9. **Title**: Machine Learning and Optimization-Based Approaches to Duality in Statistical Physics (arXiv:2411.04838)
   - **Authors**: Andrea E. V. Ferrari, Prateek Gupta, Nabil Iqbal
   - **Summary**: This work explores the use of machine learning and optimization to discover dualities in lattice statistical mechanics models. By parameterizing maps with neural networks and introducing a loss function, the authors formulate duality discovery as an optimization problem, successfully rediscovering known dualities.
   - **Year**: 2024

10. **Title**: Lagrangian Dual Framework for Conservative Neural Network Solutions of Kinetic Equations (arXiv:2106.12147)
    - **Authors**: Hyung Ju Hwang, Hwijae Son
    - **Summary**: The authors propose a conservative formulation for solving kinetic equations via neural networks, formulating the learning problem as a constrained optimization with physical conservation laws as constraints. By relaxing these constraints using Lagrangian duality, the approach achieves accurate approximations of solutions.
    - **Year**: 2021

**Key Challenges:**

1. **Computational Complexity**: Implementing Lagrange duality in deep networks can be computationally intensive, potentially hindering real-time applications.

2. **Non-Convexity of Neural Networks**: Deep neural networks are inherently non-convex, making the application of convex duality principles challenging and requiring careful formulation.

3. **Scalability**: Ensuring that sensitivity analysis methods scale effectively with large and complex models remains a significant challenge.

4. **Interpretability vs. Performance Trade-off**: Balancing the need for interpretability with maintaining high predictive performance can be difficult, as enhancing one may compromise the other.

5. **Robustness to Adversarial Perturbations**: Developing methods that provide reliable sensitivity analysis in the presence of adversarial attacks or distributional shifts is crucial for real-world applicability. 
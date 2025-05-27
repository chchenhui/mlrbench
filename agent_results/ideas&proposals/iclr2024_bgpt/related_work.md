Here is a literature review on the proposed idea of the Dynamic Curvature-Aware Optimizer (DCAO), focusing on related works published between 2023 and 2025.

**1. Related Papers**

1. **Title**: Adaptive Gradient Methods at the Edge of Stability (arXiv:2207.14484)
   - **Authors**: Jeremy M. Cohen, Behrooz Ghorbani, Shankar Krishnan, Naman Agarwal, Sourabh Medapati, Michal Badura, Daniel Suo, David Cardoze, Zachary Nado, George E. Dahl, Justin Gilmer
   - **Summary**: This paper investigates the behavior of adaptive gradient methods like Adam in the full-batch and large-batch settings. It introduces the concept of the "Adaptive Edge of Stability" (AEoS), where the maximum eigenvalue of the preconditioned Hessian equilibrates at a stability threshold. The study highlights how adaptive methods can advance into high-curvature regions by adjusting the preconditioner, differing from non-adaptive methods at the Edge of Stability.
   - **Year**: 2022

2. **Title**: Investigating the Edge of Stability Phenomenon in Reinforcement Learning (arXiv:2307.04210)
   - **Authors**: Rares Iordan, Marc Peter Deisenroth, Mihaela Rosca
   - **Summary**: This work explores the Edge of Stability (EoS) phenomenon within the context of reinforcement learning, particularly in off-policy Q-learning algorithms. The authors observe that, despite differences from supervised learning, the EoS phenomenon can manifest in deep reinforcement learning, with variations depending on the loss function used.
   - **Year**: 2023

3. **Title**: A Hessian-Informed Hyperparameter Optimization for Differential Learning Rate (arXiv:2501.06954)
   - **Authors**: Shiyun Xu, Zhiqi Bu, Yiliang Zhang, Ian Barnett
   - **Summary**: The authors propose Hi-DLR, an efficient approach that utilizes Hessian information to optimize learning rates dynamically during training. By capturing the loss curvature adaptively, Hi-DLR aims to improve convergence and can also identify parameters to freeze, leading to parameter-efficient fine-tuning.
   - **Year**: 2025

4. **Title**: ADLER -- An Efficient Hessian-Based Strategy for Adaptive Learning Rate (arXiv:2305.16396)
   - **Authors**: Dario Balboni, Davide Bacciu
   - **Summary**: This paper introduces ADLER, a strategy that derives a positive semi-definite approximation of the Hessian to inform adaptive learning rates. The method aims to minimize the local quadratic approximation efficiently, achieving performance comparable to grid search on learning rates with reduced computational cost.
   - **Year**: 2023

5. **Title**: Trajectory Alignment: Understanding the Edge of Stability Phenomenon via Bifurcation Theory (arXiv:2307.04204)
   - **Authors**: Minhak Song, Chulhee Yun
   - **Summary**: The authors analyze the Edge of Stability phenomenon through the lens of bifurcation theory. They demonstrate that different gradient descent trajectories align on a specific bifurcation diagram, independent of initialization, providing a theoretical foundation for understanding EoS dynamics.
   - **Year**: 2023

6. **Title**: A PDE-Based Explanation of Extreme Numerical Sensitivities and Edge of Stability in Training Neural Networks (arXiv:2206.02001)
   - **Authors**: Yuxin Sun, Dong Lao, Ganesh Sundaramoorthi, Anthony Yezzi
   - **Summary**: This work presents a theoretical framework using numerical analysis of partial differential equations to explain restrained numerical instabilities observed during neural network training. The authors link these instabilities to the Edge of Stability phenomenon, providing insights into the role of regularization and network complexity.
   - **Year**: 2022

7. **Title**: Understanding Gradient Descent on the Edge of Stability in Deep Learning (arXiv:2205.09745)
   - **Authors**: Sanjeev Arora, Zhiyuan Li, Abhishek Panigrahi
   - **Summary**: This paper mathematically analyzes a new mechanism of implicit regularization in the Edge of Stability phase. The authors demonstrate that gradient descent updates evolve along a deterministic flow on the manifold of minimum loss, providing a theoretical explanation for the EoS phenomenon.
   - **Year**: 2022

8. **Title**: A Survey of Deep Learning Optimizers -- First and Second Order Methods (arXiv:2211.15596)
   - **Authors**: Rohan Kashyap
   - **Summary**: This survey provides a comprehensive review of 14 standard optimization methods used in deep learning, including both first and second-order methods. It offers a theoretical assessment of the challenges in numerical optimization, such as saddle points, local minima, and ill-conditioning of the Hessian.
   - **Year**: 2022

9. **Title**: Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability (arXiv:2209.15594)
   - **Authors**: Alex Damian, Eshaan Nichani, Jason D. Lee
   - **Summary**: The authors introduce the concept of self-stabilization, where gradient descent at the Edge of Stability implicitly follows projected gradient descent under a stability constraint. They provide theoretical predictions for loss, sharpness, and deviation from the projected trajectory, corroborated by empirical studies.
   - **Year**: 2022

10. **Title**: Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability (arXiv:2103.00065)
    - **Authors**: Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, Ameet Talwalkar
    - **Summary**: This seminal paper empirically demonstrates that full-batch gradient descent on neural network training objectives often operates in a regime termed the Edge of Stability. The study observes that the maximum eigenvalue of the training loss Hessian hovers just above a critical value, leading to non-monotonic yet consistent long-term loss decrease.
    - **Year**: 2021

**2. Key Challenges**

1. **Computational Overhead of Hessian Approximations**: Accurately estimating the Hessian or its top eigenvalues can be computationally intensive, potentially offsetting the benefits of curvature-aware optimization.

2. **Stability in High-Curvature Regions**: Navigating high-curvature regions of the loss landscape without causing instability remains a significant challenge, especially when adjusting learning rates dynamically.

3. **Generalization Across Architectures**: Ensuring that curvature-aware optimizers generalize well across various neural network architectures and tasks is non-trivial, requiring extensive empirical validation.

4. **Balancing Adaptivity and Convergence**: Designing optimizers that adapt to curvature information while maintaining robust convergence properties poses a delicate balance.

5. **Theoretical Guarantees Under Non-Smoothness**: Providing rigorous theoretical guarantees for convergence and generalization in non-smooth loss landscapes is complex and often lacks comprehensive solutions. 
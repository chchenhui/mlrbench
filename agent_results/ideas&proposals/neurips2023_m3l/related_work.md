1. **Title**: Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability (arXiv:2103.00065)
   - **Authors**: Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, Ameet Talwalkar
   - **Summary**: This paper empirically demonstrates that full-batch gradient descent on neural network training objectives often operates in a regime termed the Edge of Stability (EoS). In this regime, the maximum eigenvalue of the training loss Hessian hovers just above the value $2 / \text{(step size)}$, and the training loss exhibits non-monotonic behavior over short timescales while consistently decreasing over long timescales.
   - **Year**: 2021

2. **Title**: Understanding Gradient Descent on Edge of Stability in Deep Learning (arXiv:2205.09745)
   - **Authors**: Sanjeev Arora, Zhiyuan Li, Abhishek Panigrahi
   - **Summary**: This work provides a mathematical analysis of the implicit regularization occurring in the EoS phase of gradient descent. The authors demonstrate that gradient descent updates, influenced by the non-smooth loss landscape, evolve along a deterministic flow on the manifold of minimum loss, leading to implicit bias without relying on infinitesimal updates or gradient noise.
   - **Year**: 2022

3. **Title**: Continuous-time stochastic gradient descent for optimizing over the stationary distribution of stochastic differential equations (arXiv:2202.06637)
   - **Authors**: Ziheng Wang, Justin Sirignano
   - **Summary**: This paper introduces a continuous-time stochastic gradient descent method designed for optimizing over the stationary distribution of stochastic differential equation models. The algorithm continuously updates the model's parameters using an estimate for the gradient of the stationary distribution, with convergence proven for linear SDE models and numerical results presented for nonlinear examples.
   - **Year**: 2022

4. **Title**: Convergence of continuous-time stochastic gradient descent with applications to linear deep neural networks (arXiv:2409.07401)
   - **Authors**: Gabor Lugosi, Eulalia Nualart
   - **Summary**: This study examines a continuous-time approximation of the stochastic gradient descent process for minimizing expected loss in learning problems. The authors establish general sufficient conditions for convergence and apply their results to the training of overparametrized linear neural networks.
   - **Year**: 2024

**Key Challenges:**

1. **Understanding the Edge of Stability (EoS) Dynamics**: The EoS phenomenon, where training operates near an unstable regime, is not yet fully understood. Developing a comprehensive theoretical framework to explain and predict EoS behavior remains a significant challenge.

2. **Designing Adaptive Optimization Algorithms**: Creating optimization algorithms that can dynamically adjust learning rates and noise schedules to effectively operate at the EoS without causing divergence is complex and requires further research.

3. **Efficient Curvature Estimation**: Incorporating curvature estimates from low-cost Hessian approximations to modulate updates poses computational challenges, especially for large-scale models.

4. **Balancing Stability and Acceleration**: Ensuring stable convergence while achieving accelerated training in non-convex loss landscapes is a delicate balance that needs to be addressed.

5. **Bridging Theory and Practice**: Translating theoretical insights into practical guidelines that can reduce energy and time costs in training large-scale models is an ongoing challenge in the field. 
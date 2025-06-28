# Dynamic Curvature-Aware Optimizer (DCAO): Bridging Theory and Practice in Deep Learning

## Introduction

The success of deep learning practices has driven the rapid development of learning theory. However, recent studies have pointed out that contrasting scenarios and conclusions exist between many existing theories and their corresponding real-world applications, leading to a significant gap. This workshop aims to bridge this gap by (i) troubleshooting unnoticed gaps between learning theory and practice and (ii) narrowing the existing ones by developing new analyses. We hope that this workshop will not only raise awareness of the challenges in bridging the gap between theory and practice in deep learning but also inspire new solutions and insights that contribute to the advancement of deep learning.

The Dynamic Curvature-Aware Optimizer (DCAO) is proposed to address the theoretical insights into loss-landscape non-smoothness and the Edge-of-Stability phenomenon. By operationalizing curvature-based analyses, DCAO aims to yield more stable convergence and better generalization in practice. This proposal outlines the research plan, methodology, and expected outcomes of the DCAO, focusing on its theoretical foundations, practical implementation, and empirical validation.

## Methodology

### Research Design

The research design for DCAO involves three main phases: theoretical analysis, algorithm development, and empirical validation.

#### Theoretical Analysis

The theoretical analysis focuses on understanding the behavior of neural network training under non-smooth loss landscapes and the Edge-of-Stability (EoS) phenomenon. The study leverages the results from previous works, such as Adaptive Gradient Methods at the Edge of Stability (Cohen et al., 2022) and Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability (Cohen et al., 2021), to derive convergence bounds under non-smooth assumptions.

#### Algorithm Development

The algorithm development phase involves designing the DCAO, which periodically probes local curvature spectra using low-rank Hessian approximations. The algorithm integrates stochastic Lanczos iterations to compute top-k eigenpairs, deriving curvature metrics such as spectral radius and spectral gap. These metrics inform dynamic adjustments of learning rate, momentum, and weight decay. The method is designed to integrate seamlessly into existing training pipelines with minimal overhead.

#### Empirical Validation

The empirical validation phase involves testing DCAO on vision and language models to evaluate its effectiveness in improving training stability, convergence, and generalization. The experiments will compare DCAO with state-of-the-art optimizers, such as Adam and Hi-DLR, to demonstrate its superior performance in the Edge-of-Stability regime.

### Data Collection

The data collection phase involves selecting a diverse set of neural network architectures and datasets for empirical validation. The datasets include CIFAR-10, ImageNet, and GLUE benchmark datasets for vision models, and the Penn Treebank and SQuAD datasets for language models.

### Algorithmic Steps

The DCAO algorithm can be summarized as follows:

1. **Initialization**: Initialize the neural network parameters and set the initial learning rate, momentum, and weight decay.
2. **Periodic Curvature Probing**: At set intervals (e.g., every 1000 iterations), compute the low-rank Hessian approximation using stochastic Lanczos iterations.
3. **Curvature Metrics**: Compute the top-k eigenpairs to derive curvature metrics, such as spectral radius and spectral gap.
4. **Dynamic Adjustments**: Adjust the learning rate, momentum, and weight decay based on the computed curvature metrics:
   - High spectral radius triggers conservative updates.
   - Increased spectral gap allows for acceleration.
5. **Gradient Descent Update**: Perform gradient descent update using the adjusted hyperparameters.
6. **Repeat**: Repeat steps 2-5 until convergence or a maximum number of iterations is reached.

### Mathematical Formulation

The mathematical formulation of DCAO involves the computation of the low-rank Hessian approximation and the adjustment of hyperparameters based on curvature metrics. Let $\mathbf{w}$ denote the neural network parameters, $\mathbf{g}$ denote the gradient, and $\mathbf{H}$ denote the Hessian matrix. The low-rank Hessian approximation can be computed using stochastic Lanczos iterations as follows:

$$ \mathbf{H}_{k} \approx \mathbf{V}_k \mathbf{\Lambda}_k \mathbf{V}_k^T $$

where $\mathbf{V}_k$ is an orthogonal matrix containing the top-k eigenvectors, and $\mathbf{\Lambda}_k$ is a diagonal matrix containing the corresponding eigenvalues. The spectral radius and spectral gap can be computed as follows:

$$ \rho(\mathbf{H}) = \max_{i} |\lambda_i| $$
$$ \gamma(\mathbf{H}) = \max_{i \neq j} |\lambda_i - \lambda_j| $$

where $\lambda_i$ denotes the eigenvalues of the Hessian matrix. The learning rate, momentum, and weight decay can be adjusted based on the computed curvature metrics as follows:

$$ \eta(t) = \eta_0 \exp(-\alpha \rho(\mathbf{H}_k)) $$
$$ \mu(t) = \mu_0 \exp(-\beta \gamma(\mathbf{H}_k)) $$
$$ \lambda(t) = \lambda_0 \exp(-\gamma \rho(\mathbf{H}_k)) $$

where $\eta_0$, $\mu_0$, and $\lambda_0$ denote the initial values of the learning rate, momentum, and weight decay, respectively, and $\alpha$, $\beta$, and $\gamma$ denote the scaling factors for the curvature metrics.

### Experimental Design

The experimental design involves training vision and language models using DCAO and comparing its performance with state-of-the-art optimizers, such as Adam and Hi-DLR. The evaluation metrics include training loss, validation loss, convergence speed, and generalization performance. The experiments will be conducted using a diverse set of neural network architectures and datasets, ensuring the robustness and generalization of DCAO.

## Expected Outcomes & Impact

The expected outcomes of the DCAO research include improved training stability, faster convergence in the Edge-of-Stability regime, and enhanced generalization. The impact of the research is expected to narrow the theory-practice divide by operationalizing curvature-based analyses and providing a practical solution for optimizing neural network training. The findings of the research will contribute to the advancement of deep learning by bridging the gap between learning theory and practice, inspiring new solutions and insights.

The proposed research is expected to have a significant impact on the field of deep learning by:

1. **Improving Training Stability**: DCAO's dynamic adjustments of learning rate, momentum, and weight decay based on curvature metrics will lead to more stable convergence in high-curvature regions of the loss landscape.
2. **Accelerating Convergence**: By adjusting hyperparameters based on curvature metrics, DCAO will enable faster convergence in the Edge-of-Stability regime.
3. **Enhancing Generalization**: The improved training stability and convergence will result in better generalization performance on unseen data.
4. **Bridging Theory and Practice**: DCAO will operationalize theoretical insights into loss-landscape non-smoothness and the Edge-of-Stability phenomenon, narrowing the gap between learning theory and practice.
5. **Inspiring New Solutions**: The research will inspire new solutions and insights in deep learning optimization, contributing to the advancement of the field.

In conclusion, the Dynamic Curvature-Aware Optimizer (DCAO) aims to bridge the gap between theory and practice in deep learning by operationalizing curvature-based analyses. The proposed research plan, methodology, and expected outcomes demonstrate the potential of DCAO to improve training stability, convergence, and generalization in neural network training.
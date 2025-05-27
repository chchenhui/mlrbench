## 1. Title: Dynamic Curvature-Aware Optimizer (DCAO): Bridging Theory and Practice via Adaptive Hessian Eigenspectrum Estimation

## 2. Introduction

### 2.1 Background
Deep learning has achieved remarkable success across diverse domains, largely driven by empirical advancements in architectures, training techniques, and large-scale datasets. This empirical success has spurred significant interest in developing a deeper theoretical understanding of the underlying principles governing deep learning model training and generalization. Key areas of theoretical investigation include optimization dynamics, the role of implicit bias in optimizers, the effects of overparameterization, and the geometry of the loss landscape.

However, a noticeable gap often exists between theoretical findings and practical deep learning applications. Recent theoretical work has shed light on complex phenomena such as the non-smooth nature of neural network loss landscapes and the "Edge of Stability" (EoS) phenomenon (Cohen et al., 2021), where optimizers operate near the boundary of stability, exhibiting non-monotonic loss decreases while still achieving good generalization. These insights suggest that the local curvature of the loss landscape plays a crucial role in determining optimal training dynamics. Standard optimizers like SGD, Adam (Kingma & Ba, 2014), and AdamW (Loshchilov & Hutter, 2019), while widely successful, often rely primarily on first-order gradient information or diagonal approximations of the Hessian (as in Adam/AdamW), potentially underutilizing critical curvature information revealed by the Hessian eigenspectrum. For instance, Adam's adaptive learning rates can sometimes lead to instability in regions of sharp curvature or fail to fully exploit flatter directions (Cohen et al., 2022). Similarly, the EoS phenomenon, where the maximum Hessian eigenvalue $\lambda_{max}$ interacts critically with the learning rate $\eta$ (specifically, $\eta \lambda_{max} \approx 2$), is typically observed with constant learning rates and full-batch gradient descent (Cohen et al., 2021; Arora et al., 2022; Damian et al., 2022), and its implications for adaptive, stochastic optimization remain an active area of research.

The disconnect lies partly in the difficulty of efficiently incorporating second-order (curvature) information into optimizers suitable for large-scale deep learning. Full Hessian computation is prohibitively expensive ($O(d^2)$ memory and computation for $d$ parameters), and even Hessian-vector products can add significant overhead if computed frequently. Existing attempts at Hessian-informed optimization often focus on specific aspects like adaptive learning rates (Balboni & Bacciu, 2023; Xu et al., 2025) but may not dynamically adapt other crucial hyperparameters like momentum or weight decay based on the evolving curvature landscape, nor do they always directly address phenomena like EoS within a unified framework.

### 2.2 Research Objectives
This research proposes the **Dynamic Curvature-Aware Optimizer (DCAO)**, a novel optimization algorithm designed to bridge the gap between theoretical insights on loss landscape geometry and practical optimization strategies. DCAO aims to leverage efficiently computed, low-rank approximations of the local Hessian eigenspectrum to dynamically adapt key hyperparameters during training. The primary objectives of this research are:

1.  **Develop the DCAO Algorithm:** Design and implement an optimizer that periodically estimates the top-$k$ eigenpairs (eigenvalues and eigenvectors) of the local Hessian using efficient stochastic low-rank approximation techniques (e.g., stochastic Lanczos iteration). These spectral properties will be used to dynamically adjust the learning rate ($\eta$), momentum coefficient ($\beta$), and weight decay ($\lambda_{wd}$).
2.  **Theoretically Analyze DCAO:** Derive theoretical guarantees for the convergence of DCAO under realistic assumptions relevant to deep learning, potentially including non-smoothness or conditions like the Polyak-Łojasiewicz (PL) inequality. Analyze DCAO's behavior, particularly concerning stability, especially in the context of the Edge of Stability phenomenon.
3.  **Empirically Validate DCAO:** Conduct extensive experiments on benchmark datasets (e.g., CIFAR-10/100, ImageNet) and diverse architectures (e.g., ResNets, Vision Transformers, potentially LSTMs/Transformers for sequence tasks) to evaluate DCAO's performance against standard optimizers (SGD, Adam, AdamW) and potentially other adaptive methods. Assess improvements in training stability, convergence speed, final model performance (accuracy, perplexity), and generalization.
4.  **Bridge Theory and Practice:** Demonstrate how DCAO operationalizes theoretical concepts (loss curvature, EoS, spectral properties) into a practical optimization algorithm, providing insights into how curvature information can be effectively utilized to improve deep learning training.

### 2.3 Significance
This research holds significant potential for advancing deep learning optimization:

*   **Improved Optimization Performance:** By dynamically adapting hyperparameters based on local curvature, DCAO aims to achieve more stable training dynamics (especially mitigating instabilities near the EoS), potentially faster convergence, and improved final model performance compared to standard first-order methods.
*   **Enhanced Generalization:** Theoretical links exist between loss landscape flatness (related to small Hessian eigenvalues) and generalization. By incorporating spectral information, DCAO might implicitly favor flatter minima, potentially leading to better generalization. The dynamic adjustment of weight decay based on curvature could also contribute to improved regularization.
*   **Bridging the Theory-Practice Gap:** This work directly addresses the workshop's theme by translating theoretical insights about the loss landscape (non-smoothness, curvature, EoS) into a concrete, computationally feasible optimization algorithm. The empirical validation and theoretical analysis will provide valuable feedback, potentially refining both theory and practice.
*   **New Insights into Optimization Dynamics:** Analyzing the behavior of DCAO and the evolution of curvature metrics during training can provide deeper insights into the interplay between optimization algorithms, loss landscape geometry, and model generalization, particularly concerning phenomena like EoS in stochastic and adaptive settings.
*   **Practical Tool for Researchers and Practitioners:** If successful, DCAO could offer a valuable new tool for training deep learning models more effectively and robustly, potentially reducing the need for extensive manual hyperparameter tuning.

## 3. Methodology

### 3.1 Research Design Overview
The research will follow a structured approach combining algorithmic design, theoretical analysis, and empirical validation.
1.  **Algorithmic Development:** Formulate the DCAO algorithm, detailing the curvature estimation and hyperparameter adaptation mechanisms. Implement DCAO within a standard deep learning framework (e.g., PyTorch).
2.  **Theoretical Analysis:** Analyze the convergence properties and stability characteristics of DCAO, leveraging existing theoretical frameworks for stochastic optimization and potentially incorporating insights from EoS literature.
3.  **Empirical Evaluation:** Design and execute a comprehensive set of experiments to compare DCAO against baseline optimizers across various tasks, models, and datasets. Analyze the results using rigorous statistical methods.

### 3.2 The DCAO Algorithm

**3.2.1 Core Idea:** DCAO modifies a base optimizer (e.g., AdamW) by periodically probing the local Hessian eigenspectrum and using this information to adjust $\eta$, $\beta$, and $\lambda_{wd}$. The probing occurs at predefined intervals (e.g., every $T$ training steps or epochs).

**3.2.2 Curvature Estimation via Stochastic Lanczos:**
At each probing step $t$, DCAO estimates the top-$k$ eigenvalues and corresponding eigenvectors of the Hessian matrix $H = \nabla^2 L(w_t)$ using the stochastic Lanczos algorithm. This method is efficient as it only requires Hessian-vector products ($Hv$), which can be computed without explicitly forming the Hessian. A Hessian-vector product can be approximated using finite differences:
$$Hv \approx \frac{\nabla L(w_t + \epsilon v) - \nabla L(w_t - \epsilon v)}{2\epsilon}$$
where $v$ is a random vector (or a carefully chosen one, e.g., based on recent gradients) and $\epsilon$ is a small scalar. The Lanczos algorithm iteratively builds a small tridiagonal matrix $T_k$ whose eigenvalues approximate the extremal eigenvalues of $H$.

*   **Algorithm Sketch (Stochastic Lanczos for top-k eigenpairs):**
    1.  Initialize a random vector $v_1$ with $\|v_1\|_2 = 1$. Set $\beta_0 = 0$, $v_0 = 0$.
    2.  For $j = 1, \dots, k$:
        a. Compute $u_j = H v_j$ (using Hessian-vector product).
        b. Compute $\alpha_j = v_j^T u_j$.
        c. Compute $z_j = u_j - \alpha_j v_j - \beta_{j-1} v_{j-1}$.
        d. Compute $\beta_j = \|z_j\|_2$.
        e. If $\beta_j < \text{tolerance}$, stop (invariant subspace found).
        f. Set $v_{j+1} = z_j / \beta_j$.
    3.  Form the $k \times k$ tridiagonal matrix $T_k$ with $\alpha_j$ on the diagonal and $\beta_j$ on the off-diagonals.
    4.  Compute the eigenpairs $(\theta_i, s_i)$ of $T_k$. The eigenvalues $\theta_i$ approximate the eigenvalues $\lambda_i$ of $H$. The approximate eigenvectors of $H$ are $y_i = V_k s_i$, where $V_k = [v_1, \dots, v_k]$.
*   From the estimated eigenvalues $\{\lambda_1, \dots, \lambda_k\}$ (assuming sorted $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_k$), we extract key curvature metrics:
    *   **Maximum Eigenvalue (Spectral Radius):** $\lambda_{max} \approx \lambda_1$. This indicates the sharpness along the steepest direction.
    *   **Spectral Gap (optional):** $gap = \lambda_1 - \lambda_2$. A larger gap might indicate a more pronounced principal direction of curvature.
    *   **Effective Rank / Eigenvalue Distribution:** Information from $\{\lambda_i\}_{i=1}^k$.

**3.2.3 Dynamic Hyperparameter Adjustment:**
Based on the estimated curvature metrics at probing step $t$, DCAO adjusts the hyperparameters for the subsequent $T$ steps. Let $\eta_t, \beta_t, \lambda_{wd,t}$ be the current parameters. The adjusted parameters $\eta_{t+1}, \beta_{t+1}, \lambda_{wd,t+1}$ are computed using adaptive rules.

*   **Learning Rate ($\eta$):**
    *   **Rule:** If $\lambda_{max}$ is high (e.g., $\eta_t \lambda_{max} \ge C_{stab}$, where $C_{stab} \approx 2$ is the stability threshold inspired by EoS), decrease $\eta$. If $\lambda_{max}$ is relatively low, potentially allow for an increase in $\eta$.
    *   **Formula Example:** $\eta_{t+1} = \eta_t \cdot \text{clip}(\frac{C_{target}}{\eta_t \lambda_{max}}, \gamma_{down}, \gamma_{up})$, where $C_{target}$ is a target stability value (e.g., 1.0-1.8), and $\gamma_{down}, \gamma_{up}$ are clamping factors (e.g., [0.8, 1.1]).
*   **Momentum ($\beta$):**
    *   **Rule:** High curvature ($\lambda_{max}$) might suggest reducing momentum to prevent overshooting. Smoother regions (low $\lambda_{max}$) might benefit from higher momentum. The spectral gap could also play a role; a large gap might suggest aligning with the dominant eigenvector, potentially warranting adjusted momentum.
    *   **Formula Example:** $\beta_{t+1} = \beta_t \cdot (1 - \alpha_\beta (\eta_t \lambda_{max} - C_{target}))$, clamped within a valid range [$\beta_{min}, \beta_{max}$]. $\alpha_\beta$ controls sensitivity.
*   **Weight Decay ($\lambda_{wd}$):**
    *   **Rule:** In sharp regions (high $\lambda_{max}$), increasing weight decay might help regularize and stabilize training. In flatter regions, weight decay could potentially be reduced.
    *   **Formula Example:** $\lambda_{wd,t+1} = \lambda_{wd,t} \cdot (1 + \alpha_{wd} (\eta_t \lambda_{max} - C_{target}))$, clamped within a valid range. $\alpha_{wd}$ controls sensitivity.

**3.2.4 Integration into Training Loop:**
DCAO wraps a standard optimizer (e.g., AdamW). The training loop proceeds as usual, but every $T$ steps, the DCAO module performs curvature estimation and updates the hyperparameters of the underlying optimizer.

*   **Over_head Management:** The computational overhead comes primarily from the $k$ Hessian-vector products computed every $T$ steps. Choosing $k$ small (e.g., 5-10) and $T$ reasonably large (e.g., one epoch, or hundreds/thousands of steps) aims to keep the overhead minimal (e.g., < 5-10% increase in training time). The efficiency of the Hessian-vector product implementation is crucial.

### 3.3 Theoretical Analysis

*   **Convergence:** We will analyze the convergence of DCAO in the stochastic, non-convex setting common to deep learning. We will state explicit assumptions, likely including Lipschitz continuous gradients (possibly locally), bounded gradient variance, and potentially the Polyak-Łojasiewicz (PL) condition or assumptions about non-smoothness. The analysis will need to account for the periodic, adaptive changes in $\eta, \beta, \lambda_{wd}$ based on potentially noisy estimates of $\lambda_{max}$. We aim to derive convergence rates (e.g., to a stationary point or region) and understand how they depend on the curvature adaptation strategy.
*   **Stability Analysis (EoS):** We will specifically analyze DCAO's behavior in regimes where EoS phenomena are expected. Can DCAO detect the proximity to the stability boundary ($\eta \lambda_{max} \approx 2$) and adjust parameters to maintain stability or navigate this regime effectively? How do the adaptive momentum and weight decay interact with the learning rate adjustment near the EoS? We will connect the adaptation rules to theoretical stability conditions.
*   **Non-Smoothness:** The analysis will consider the impact of non-smooth activations (like ReLU) or loss functions, acknowledging that the Hessian may not exist everywhere. Techniques from non-smooth optimization analysis might be required, possibly analyzing convergence in terms of generalized gradients or proximal operators, though we may initially focus on locally smooth regions or use smoothing approximations.

### 3.4 Experimental Design

*   **Datasets:**
    *   **Vision:** CIFAR-10, CIFAR-100 (standard benchmarks), ImageNet (large-scale challenge). Possibly simpler datasets like MNIST for initial debugging and analysis.
    *   **Language:** Penn Treebank (PTB), WikiText-2 (standard language modeling benchmarks).
*   **Models:**
    *   **Vision:** ResNet-18/34/50, Vision Transformer (ViT-Base).
    *   **Language:** LSTMs, small-to-medium sized Transformer models (e.g., GPT-2 small).
*   **Baselines:**
    *   SGD with momentum.
    *   Adam.
    *   AdamW.
    *   *Potential:* K-FAC or other approximate second-order methods (if computationally comparable or for reference), or simplified versions of Hessian-informed LR methods like ADLER (Balboni & Bacciu, 2023).
*   **DCAO Configuration:**
    *   Base Optimizer: AdamW.
    *   Probing Frequency ($T$): Ablate values (e.g., every 100 steps, 500 steps, 1 epoch).
    *   Number of Lanczos Iterations ($k$): Ablate values (e.g., 5, 10, 20).
    *   Hyperparameters for Adaptation Rules: $C_{target}, \gamma_{down}, \gamma_{up}, \alpha_\beta, \alpha_{wd}$. Tune using a validation set or based on theoretical guidelines.
*   **Evaluation Metrics:**
    *   **Performance:** Final test accuracy (vision) / perplexity (language). Training/validation loss curves over epochs and wall-clock time.
    *   **Convergence Speed:** Epochs or time required to reach a target validation performance level.
    *   **Stability:** Monitor loss evolution (oscillations), gradient norms, parameter norms, magnitude of updates. Track the estimated $\lambda_{max}$ and the value $\eta_t \lambda_{max}$ over time to observe behavior relative to the EoS threshold.
    *   **Generalization:** Compare final test performance. Potentially measure sharpness ($\lambda_{max}$) of the final solution or flatness metrics.
    *   **Computational Overhead:** Measure average time per epoch/iteration for DCAO vs. baselines. Report the percentage increase.
*   **Ablation Studies:** Systematically evaluate the contribution of each adaptive component (dynamic $\eta$, dynamic $\beta$, dynamic $\lambda_{wd}$) and the sensitivity to DCAO hyperparameters ($T, k$).
*   **Robustness:** Test sensitivity to initial learning rates and other hyperparameters compared to baselines.

## 4. Expected Outcomes & Impact

### 4.1 Expected Outcomes
1.  **A Novel Optimizer (DCAO):** A fully implemented and documented DCAO algorithm, available as open-source code, integrated with standard deep learning frameworks.
2.  **Theoretical Guarantees:** Formal convergence analysis of DCAO under relevant assumptions, potentially establishing convergence rates and providing theoretical insights into its stability properties, especially concerning the EoS regime.
3.  **Comprehensive Empirical Evaluation:** Rigorous experimental results demonstrating the performance of DCAO across various tasks, models, and datasets compared to state-of-the-art optimizers. This includes quantitative measures of performance, convergence speed, stability, generalization, and computational overhead.
4.  **Analysis of Curvature Dynamics:** Insights into how key spectral properties of the Hessian (e.g., $\lambda_{max}$) evolve during training with DCAO and how the adaptive mechanism responds to these changes.
5.  **Publications and Dissemination:** High-quality publications in top-tier machine learning conferences (e.g., NeurIPS, ICML, ICLR) or journals, presenting the algorithm, theory, and empirical findings. Presentation at relevant workshops, such as the one described in the task description.

### 4.2 Impact
This research is expected to have a significant impact on the field of deep learning optimization and the broader goal of bridging theory and practice:

*   **Practical Advancement:** DCAO has the potential to become a widely used optimizer, offering improved stability, faster convergence, and potentially better generalization with manageable overhead. This could ease the burden of hyperparameter tuning for practitioners.
*   **Bridging Theory and Practice:** By directly incorporating theoretical insights about loss landscape curvature and phenomena like EoS into a practical algorithm, this work serves as a concrete example of how theory can inform and improve deep learning practice. The empirical results will, in turn, provide feedback to theoretical studies, potentially highlighting regimes where current theory holds or needs refinement (e.g., EoS in adaptive, stochastic settings).
*   **Stimulating Further Research:** The methodology and findings could inspire further research into curvature-aware optimization, efficient Hessian spectral estimation, and the dynamic adaptation of hyperparameters based on landscape geometry. The theoretical analysis might open new avenues for understanding optimization in non-smooth and adaptive settings.
*   **Deeper Understanding of Deep Learning:** The analysis of DCAO's behavior and the associated curvature dynamics will contribute to a more nuanced understanding of why deep learning models train effectively and how optimization choices influence the final solution and its generalization properties.

In summary, the proposed research on the Dynamic Curvature-Aware Optimizer (DCAO) directly addresses the critical need to connect deep learning theory with practice. By developing, analyzing, and validating an optimizer that dynamically adapts to the loss landscape curvature, we aim to deliver both a practical tool and deeper theoretical insights, contributing significantly to the advancement of reliable and efficient deep learning.
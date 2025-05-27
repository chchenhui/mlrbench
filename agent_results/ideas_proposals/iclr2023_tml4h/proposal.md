### Introduction  

Multi-modal medical data fusion has gained immense importance in healthcare due to its potential to enhance diagnostic accuracy by leveraging complementary information from different data sources. Medical imaging modalities, such as computed tomography (CT), magnetic resonance imaging (MRI), and ultrasound, provide detailed anatomical and functional insights, while structured clinical data from electronic healthcare records (EHRs) offer vital patient-specific information such as comorbidities, medication history, and lab results. When combined, these modalities enable a holistic understanding of patient conditions, leading to more reliable and data-driven clinical decision-making. However, ensuring the trustworthiness of multi-modal fusion models remains a significant challenge, as real-world deployment introduces complexities such as missing modalities, domain shifts, and inherent noise in medical data.  

Despite the widespread adoption of machine learning in healthcare, the practical integration of multi-modal fusion techniques remains limited due to issues of reliability and generalization. Traditional fusion approaches often assume that all modalities are equally reliable and available, which leads to suboptimal performance when certain modalities are degraded, missing, or subject to domain shifts. Trustworthy fusion models must therefore not only improve predictive accuracy but also be robust to such uncertainties. Recent advances in medical ML have emphasized the importance of uncertainty estimation, interpretability, and domain adaptation—key pillars of trustworthy AI in healthcare. The Trustworthy Machine Learning for Healthcare workshop highlights these concerns, particularly in the context of multi-modal fusion, where models must account for heterogeneous data sources, missing information, and modality-specific degradation.  

The core motivation for this research lies in addressing critical gaps in current multi-modal fusion methods. Specifically, existing approaches either fail to dynamically assess the reliability of individual modalities during inference or do not adapt to varying levels of uncertainty within each modality. This limitation results in unreliable predictions, especially when input data suffer from quality degradation or incomplete modality coverage. By addressing this issue, the proposed method aims to improve robustness while maintaining interpretability by explicitly quantifying modality-specific uncertainty and dynamically adjusting attention weights accordingly. This aligns with the workshop’s goal of advancing trustworthy machine learning by incorporating uncertainty-aware mechanisms into multi-modal fusion models. Additionally, the proposed approach introduces self-supervised techniques to enhance reliability estimation in real-world deployment scenarios where labeled data may be scarce or noisy, further strengthening the model's applicability in clinical settings.  

Building upon these insights, this research will explore a novel framework that dynamically estimates modality reliability using Bayesian neural networks and attention mechanisms. The primary objective is to develop a multi-modal fusion model that robustly handles heterogeneous, noisy, and partially missing medical data while maintaining predictive reliability. Through comprehensive validation on real-world benchmarks, this work will contribute a practical solution for trustworthy multi-modal fusion, addressing critical challenges in deployable medical AI.

### Research Objectives and Literature Review  

The primary objective of this research is to develop a trustworthy multi-modal fusion framework that dynamically estimates modality reliability during inference to enhance robustness, uncertainty quantification, and interpretability. Specifically, the model aims to:  

1. **Dynamically assess modality reliability based on input conditions:** By integrating modality-specific uncertainty estimation into the fusion process, the model will adjust attention weights in real-time based on the perceived reliability of each modality. This capability will help mitigate the impact of noisy, degraded, or missing input data, ensuring more reliable predictions even in adverse conditions.  

2. **Enable interpretable attention mechanisms to explain modality contributions:** Attention-based fusion strategies have shown promise in prioritizing relevant modalities for medical decision-making. However, most existing approaches lack explicit mechanisms for quantifying how modality reliability influences attention weights. This research will incorporate an uncertainty-driven attention module that highlights which modalities contributed most to a given prediction, increasing transparency and trustworthiness.  

3. **Demonstrate robustness to modality degradation and missing data without predefined assumptions:** Traditional multi-modal fusion models often rely on static assumptions about modality reliability, which limits their real-world applicability. By leveraging a self-supervised auxiliary task that exposes the model to synthetic modality corruption during training, the proposed framework will learn to recognize and adapt to different levels of input degradation, improving generalization in real-world clinical settings.  

These objectives align directly with the Trustworthy Machine Learning in Healthcare workshop’s focus on explainability, generalization, and uncertainty estimation. Recent advancements in multi-modal medical fusion, such as the Modal-Domain Attention (MDA) model [1], DRIFA-Net [2], and HEALNet [3], have made progress in handling modality heterogeneity, missing data, and intrinsic noise. However, these methods often assume static reliability across modalities, limiting their ability to adapt to dynamic real-world conditions. DrFuse [4] improves upon this by disentangling modality-specific and shared features, but it does not explicitly model reliability as a function of input degradation.  

The key distinction of the proposed work lies in its integration of Bayesian neural networks with attention-based fusion to explicitly model modality reliability. Unlike existing models, the proposed approach will incorporate uncertainty-aware mechanisms that dynamically adjust attention weights based on perceived modality quality. This advancement will address a critical gap in current multi-modal fusion research, providing a more robust and trustworthy framework for healthcare applications.

### Methodology  

To achieve the research objectives, this study proposes a multi-modal fusion framework that incorporates Bayesian neural networks and dynamic attention mechanisms to estimate modality reliability. This approach will not only enhance model robustness to noisy and missing data but also provide interpretable modality-specific attention weights that reflect reliability levels during inference. The methodology includes three core components: (1) a Bayesian fusion architecture with uncertainty estimation, (2) an attention-based fusion module to dynamically weight modality contributions, and (3) a self-supervised auxiliary task for modality corruption detection during training.  

#### 1. Bayesian Fusion with Uncertainty Estimation  

The framework employs Bayesian neural networks to quantify modality-specific uncertainty, enabling dynamic reliability estimation. Following the principles of Bayesian deep learning, each modality branch $ m \in \{1, ..., M\} $ will be modeled as a latent function $ f_m(x) $, where $ x $ represents the input data and $ \hat{y}_m, \sigma_y $ denote the predicted output and uncertainty estimates, respectively. Specifically, the uncertainty estimation follows a Bayesian variational approximation:  

$$
p(\theta | D) \approx q_\phi(\theta) = \prod_{m=1}^M q_\phi(\theta_m)
$$

Here, $ q_\phi(\theta_m) $ represents a variational posterior distribution over model parameters for modality $ m $, approximating the true posterior $ p(\theta | D) $. The predictive distribution for each modality can then be approximated using Monte Carlo integration:  

$$
p(y | x) \approx \frac{1}{T} \sum_{t=1}^T f_m^{\theta_t}(x)
$$

where $ T $ is the number of stochastic forward passes. The uncertainty estimate $ \sigma_y $ for modality $ m $ will be derived from the variance across these T forward passes:  

$$
\sigma_y(m) = \frac{1}{T} \sum_{t=1}^T (f_m^{\theta_t}(x) - \hat{y}_m)^2
$$

This Bayesian uncertainty estimation ensures that the model dynamically identifies unreliable modalities, enabling adaptive decision-making in the fusion process.  

#### 2. Uncertainty-Aware Attention Fusion  

To integrate modality reliability into the fusion process, we introduce an uncertainty-driven attention mechanism that dynamically adjusts attention weights based on modality uncertainty. Let $ \mu_i $ and $ \sigma_i $ denote the mean prediction and uncertainty for modality $ i $. The attention weight $ \alpha_i $ for modality $ i $ is computed using a softmax function that incorporates uncertainty scaling:  

$$
\alpha_i = \text{softmax}(W \cdot (\mu_i - \lambda \sigma_i) + b)
$$

Here, $ W $ and $ b $ are learnable parameters that control the attention weighting, and $ \lambda $ is a hyperparameter that determines the influence of modality uncertainty on attention allocation. The final fused prediction $ \hat{y}_{fusion} $ is computed as a weighted combination of modality predictions:  

$$
\hat{y}_{fusion} = \sum_{i=1}^M \alpha_i \hat{y}_i
$$

By integrating uncertainty estimation directly into the attention scoring function, the framework ensures that less reliable modalities contribute proportionally less to the final prediction, enhancing robustness against modality degradation.  

#### 3. Training Strategy and Self-Supervised Auxiliary Task  

To further improve reliability estimation, we introduce a self-supervised auxiliary task that trains the model to recognize and adapt to modality corruption. During training, synthetic noise and partial data occlusions will be applied to the input modalities, and the model will be tasked with identifying corrupted instances. This auxiliary loss ensures that the model learns robust features and effectively distinguishes between reliable and unreliable modality inputs.  

#### 4. Data Collection, Datasets, and Baseline Comparisons  

The proposed framework will be validated on large-scale medical imaging and EHR datasets, including CheXpert, MIMIC-CXR, and The Cancer Genome Atlas (TCGA), which provide multi-modal data with diverse real-world degradation cases. These datasets allow for systematic evaluation of modality reliability estimation under varying conditions, including missing modalities, low-quality imaging, and incomplete EHR entries.  

The experimental evaluation will compare the proposed method against state-of-the-art fusion models such as MDA, DRIFA-Net, and HEALNet, using metrics such as accuracy, AUROC, AUPRC, and calibration errors (ECE and Brier score). Additionally, ablation studies will quantify the impact of uncertainty estimation by evaluating performance drops under synthetic modality degradation. Through this rigorous validation process, the proposed framework will demonstrate superior robustness and reliability in multi-modal healthcare applications.

### Expected Outcomes and Impact  

The proposed methodology aims to yield several key advancements in trustworthy multi-modal medical fusion, addressing critical limitations of existing fusion models. The first expected outcome is the development of a robust fusion framework that dynamically adapts to modality reliability. By incorporating Bayesian uncertainty estimation directly into attention mechanisms, the framework will effectively identify and down-weight unreliable modalities in real-time. This capability ensures that predictions remain reliable even in the presence of degraded or missing inputs, significantly improving model robustness without requiring predefined assumptions about modality availability. Quantitative evaluations will demonstrate superior performance over existing fusion techniques under varying levels of modality corruption, including synthetic noise injection, partial data occlusion, and domain shifts.  

Second, the methodology will enhance interpretability by providing uncertainty-aware attention maps that explicitly highlight modality contributions. Unlike conventional attention-based fusion models that may assign attention weights without regard for data quality, the proposed framework will ensure that the most reliable modalities receive the highest attention scores. Through visualization techniques, clinicians will be able to inspect which modalities contributed most to a given diagnosis, facilitating model transparency. This outcome is particularly crucial in high-stakes medical applications where decision-makers must understand the reliability of input sources. Comparative experiments will assess the interpretability improvements by evaluating the accuracy of attention heatmaps in identifying corrupted modalities and demonstrating how uncertainty influences attention allocation.  

Third, the framework’s reliability estimation component will enable uncertainty-aware prediction mechanisms that flag unreliable cases. By leveraging Bayesian neural networks, the system will not only provide point predictions but also quantify predictive confidences at both the modality and output levels. This capability ensures that clinicians receive calibrated risk estimates rather than uncalibrated probability scores, reducing the risk of overconfidence in low-confidence predictions. The evaluation will include calibration metrics such as expected calibration error (ECE) and Brier score, validating the framework’s ability to provide well-calibrated uncertainty estimates even in the presence of input degradation. Additionally, case studies will showcase how confidence scores can guide decision-making, for instance, by triggering secondary physician review or additional diagnostic tests for low-confidence cases.  

Finally, this research will contribute a practical benchmark for reliability-aware fusion, setting a foundation for future advancements in trustworthy medical ML. By introducing a training strategy that explicitly teaches models to recognize modality corruption, the framework will establish a standard for evaluating modality-robust medical fusion models. The experimental validation on benchmark datasets such as CheXpert, MIMIC-CXR, and TCGA will provide a comprehensive assessment of the framework’s generalization capabilities in real-world deployment scenarios. Through open-sourcing code and models, this work will encourage further developments in medical uncertainty estimation, attention-based fusion, and interpretable AI. The proposed methodology thus represents a significant step toward deployable, trustworthy multi-modal machine learning in healthcare.

### Broader Impact in Healthcare  

This research will significantly advance the deployment of trustworthy multi-modal machine learning in clinical settings by enhancing the reliability and interpretability of fusion models. By enabling dynamic modality reliability estimation, the framework ensures that predictions remain robust even when dealing with real-world data imperfections, such as missing or degraded modalities. This advancement will increase clinician confidence in AI-assisted decision-making, reducing the risk of overconfident or misleading predictions. Additionally, the uncertainty-aware attention mechanism provides interpretable modality contributions, ensuring that clinicians understand how different data sources influence diagnostic outcomes. Beyond improving model trustworthiness, this work also supports healthcare equity by mitigating biases that may arise from missing or unreliable data, ultimately fostering safer and more transparent AI adoption in medical practice.
# Causal Discovery through Multi-Omics Perturbation Analysis via Active Learning in Causal Graphical Models

## Introduction

Understanding the causal relationships underlying biological processes is crucial for drug discovery and personalized medicine. However, traditional observational studies are often confounded by external variables, leading to unreliable conclusions. Recent advancements in perturbation experiments, such as CRISPR and RNAi, and the availability of multimodal omics data (single-cell RNA, proteomics, spatial omics) offer new opportunities to infer causality. Despite these developments, there is a lack of efficient and interpretable methods to integrate these data and design experiments that maximize causal effect estimation.

The primary objective of this research is to develop a framework that combines causal graphical models with active learning to iteratively design perturbation experiments that maximize causal effect estimation. The proposed method will address the challenges of high-dimensional data, multimodal integration, interpretability, efficient experimental design, and uncertainty quantification. The expected outcomes include interpretable causal networks linking genes, proteins, and phenotypes, validated on synthetic benchmarks and real datasets such as LINCS and CRISPR screens. This approach could accelerate target discovery by prioritizing interventions with high causal confidence, reducing trial-and-error costs, and enhancing the robustness of drug development.

## Methodology

### Research Design

The proposed framework consists of three main components: (1) learning latent causal representations from multimodal data, (2) identifying causal relationships via interventional data through counterfactual reasoning, and (3) optimizing experimental design using active learning to select perturbations that reduce uncertainty in causal graphs.

#### 1. Learning Latent Causal Representations from Multimodal Data

We will use structured variational autoencoders (SVAEs) to learn latent causal representations from multimodal data. SVAEs extend variational autoencoders by incorporating structural constraints, such as causal graphs, to capture the underlying causal relationships in the data. The encoder $q(z|x)$ encodes the input data $x$ into a latent space $z$, while the decoder $p(x|z)$ reconstructs the data from the latent representation. The SVAE is trained to maximize the evidence lower bound (ELBO):

$$
\mathcal{L}_{\text{SVAE}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) || p(z))
$$

where $D_{\text{KL}}$ denotes the Kullback-Leibler divergence. To incorporate causal structure, we will use a causal graph $G = (V, E)$ where $V$ represents variables and $E$ represents directed edges. The encoder and decoder are designed to respect the causal graph, ensuring that the latent representation captures the causal dependencies in the data.

#### 2. Identifying Causal Relationships via Interventional Data through Counterfactual Reasoning

To identify causal relationships, we will use interventional data obtained from perturbation experiments. Counterfactual reasoning, which involves estimating the effect of interventions on the system, is a fundamental concept in causal inference. Given a causal graph $G$ and a set of interventions $A$, we can estimate the causal effect of $A$ on a target variable $Y$ using the do-calculus:

$$
P(Y|do(A)) = \sum_{X \in \text{parents}(Y)} P(Y|X, \text{parents}(X)) \prod_{X \in \text{parents}(Y)} P(X|do(A))
$$

where $\text{parents}(Y)$ denotes the parents of $Y$ in the causal graph. We will apply this formula to estimate the causal effect of each intervention on the target variable, updating the causal graph $G$ accordingly.

#### 3. Optimizing Experimental Design using Active Learning

To optimize experimental design, we will employ an active learning approach to iteratively select perturbations that reduce uncertainty in the causal graph. The active learning loop consists of the following steps:

1. **Query Selection**: Select a subset of variables $S$ from the causal graph $G$ that are most uncertain about their causal relationships.
2. **Perturbation Design**: Design perturbation experiments targeting the selected variables $S$.
3. **Data Collection**: Collect interventional data from the perturbation experiments.
4. **Causal Update**: Update the causal graph $G$ using the collected data and the counterfactual reasoning formula.
5. **Uncertainty Quantification**: Quantify the uncertainty in the updated causal graph $G$ and repeat the loop until convergence.

The uncertainty in the causal graph is quantified using a measure such as the conditional entropy or the mutual information between the selected variables and their parents. The active learning loop continues until the uncertainty in the causal graph is minimized, ensuring that the selected perturbations effectively reduce uncertainty.

### Evaluation Metrics

To evaluate the performance of the proposed framework, we will use the following metrics:

1. **Causal Discovery Accuracy**: Measure the accuracy of the inferred causal graph compared to the ground truth graph using metrics such as the Structural Hamming Distance (SHD) or the Area Under the Precision-Recall Curve (AUPRC).
2. **Causal Effect Estimation Accuracy**: Measure the accuracy of the estimated causal effects compared to the ground truth effects using metrics such as the Mean Absolute Error (MAE) or the Root Mean Squared Error (RMSE).
3. **Uncertainty Quantification**: Measure the accuracy of the uncertainty quantification in the inferred causal graph using metrics such as the Mean Absolute Error (MAE) or the Root Mean Squared Error (RMSE) between the estimated and true uncertainties.
4. **Experimental Efficiency**: Measure the efficiency of the active learning loop in reducing uncertainty in the causal graph using metrics such as the number of iterations required to achieve convergence or the total number of perturbation experiments conducted.

### Experimental Design

To validate the proposed framework, we will conduct experiments on both synthetic benchmarks and real datasets. The synthetic benchmarks will consist of generated causal graphs and interventional data, allowing us to evaluate the performance of the framework in controlled environments. The real datasets will include single-cell RNA, proteomics, and spatial omics data from sources such as LINCS and CRISPR screens, enabling us to assess the practical applicability of the framework in real-world scenarios.

## Expected Outcomes & Impact

The expected outcomes of this research include the development of an interpretable and efficient framework for causal discovery in genomics, validated on both synthetic benchmarks and real datasets. The proposed framework will enable biologists to identify causal relationships between genes, proteins, and phenotypes, accelerating target discovery and reducing trial-and-error costs in drug development. The integration of uncertainty quantification and experimental feedback loops will enhance the robustness of the framework, enabling scalable and hypothesis-driven drug development.

### Potential Applications

1. **Drug Target Identification**: By identifying causal relationships between genes and diseases, the proposed framework can prioritize drug targets with high causal confidence, reducing the failure rate of drug candidates in clinical trials.
2. **Personalized Medicine**: Understanding the causal mechanisms underlying individual phenotypes can enable the development of personalized treatment strategies, improving patient outcomes and reducing healthcare costs.
3. **Basic Biological Research**: The proposed framework can provide insights into the underlying mechanisms of biological processes, facilitating the discovery of new biological principles and the development of new therapeutic strategies.

### Limitations and Future Work

The proposed framework has several limitations that will be addressed in future work:

1. **Scalability**: The current implementation may not scale well to very large datasets or high-dimensional data. Future work will focus on developing more efficient algorithms and data structures to handle large-scale data.
2. **Interpretability**: While the proposed framework aims to provide interpretable causal graphs, further work is needed to ensure that the inferred causal relationships are easily understandable by biologists.
3. **Generalization**: The proposed framework may not generalize well to new or unseen datasets. Future work will focus on developing methods to improve the generalization performance of the framework.

In conclusion, the proposed framework combining causal graphical models with active learning offers a promising approach to causal discovery in genomics. By addressing the challenges of high-dimensional data, multimodal integration, interpretability, efficient experimental design, and uncertainty quantification, the framework has the potential to accelerate target discovery and enhance the robustness of drug development.
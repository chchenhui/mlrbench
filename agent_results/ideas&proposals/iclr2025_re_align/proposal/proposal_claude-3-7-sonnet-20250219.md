# Prototypical Contrastive Learning for Interpretable Brain-DNN Representational Alignment

## 1. Introduction

The alignment of representations between artificial and biological intelligence systems represents a fundamental challenge at the intersection of machine learning, neuroscience, and cognitive science. Both natural and artificial intelligences develop internal representations of the world that guide reasoning, decision-making, and communication. Understanding how these representations relate to each other is crucial for advancing our knowledge of intelligence broadly and for developing AI systems that better approximate human cognition.

Despite substantial progress in comparing deep neural network (DNN) representations with brain activity, existing methods suffer from several limitations. Current alignment metrics are largely post-hoc, applied after models are fully trained, offering limited insight into how to actively guide artificial systems toward more brain-like representations. Additionally, these metrics rarely provide interpretable anchors that can help explain what aspects of the representations are aligned or misaligned. Furthermore, most existing approaches lack direct intervention mechanisms to systematically influence the degree of alignment during model training.

The challenge of representational alignment spans multiple dimensions. First, we must determine the extent to which aligned representations indicate shared computational strategies between biological and artificial systems. Second, we need to develop more robust and generalizable measures of alignment that work across different domains and types of representations. Third, we must create methods to systematically increase or decrease representational alignment between systems. Finally, we need to understand the implications of varying degrees of alignment on behavioral performance, generalization, and other downstream tasks.

This research aims to address these challenges by introducing a novel framework for representational alignment based on prototypical contrastive learning. Our approach differs from existing methods by establishing a compact set of semantically meaningful "anchor" vectors (prototypes) shared across systems, which serve both as an interpretable alignment metric and as a mechanism for intervention during model training. By jointly clustering neural and DNN activations, we establish a common representational space defined by these prototypes, enabling more direct and meaningful comparison between the two systems.

The significance of this research is threefold. First, it advances our understanding of the computational principles shared between biological and artificial intelligence by providing interpretable anchors for alignment. Second, it offers a practical method for guiding DNN training toward more brain-like representations, potentially enhancing model performance on tasks that humans excel at. Third, it contributes to the broader goal of developing AI systems that better complement and augment human intelligence by sharing similar representational structures.

## 2. Methodology

Our proposed methodology consists of four major components: (1) data collection of paired DNN activations and neural responses, (2) joint prototype discovery through clustering, (3) development of a prototypical contrastive alignment loss, and (4) experimental evaluation across multiple dimensions of alignment.

### 2.1 Data Collection

We will collect paired data from both DNNs and human participants using a carefully selected stimulus set. For DNNs, we will extract intermediate layer activations from various architectures (e.g., ResNet, Vision Transformer, CLIP) in response to the stimulus set. For human participants, we will collect neural data using one or more of the following modalities:

1. **fMRI Data**: We will collect whole-brain fMRI data from participants (n=20) viewing the stimulus set, with particular focus on visual cortical areas (V1-V4, IT) and higher cognitive regions.

2. **EEG Data**: For higher temporal resolution, we will record 64-channel EEG data from participants (n=30) viewing the same stimuli.

3. **Behavioral Data**: We will supplement neural measurements with behavioral similarity judgments and feature importance ratings from a larger sample (n=100).

The stimulus set will consist of 1,000 natural images spanning diverse categories, attributes, and contexts, ensuring broad coverage of the visual domain. Each stimulus will be presented for 2 seconds with jittered inter-stimulus intervals of 0.5-1.5 seconds to minimize expectation effects.

### 2.2 Joint Prototype Discovery

To discover shared representational prototypes across human neural data and DNN activations, we will implement a joint clustering approach that aligns the two representational spaces while identifying semantically meaningful clusters. The process involves:

1. **Preprocessing**: We first normalize both neural and DNN activations to control for scale differences. For fMRI data, we apply standard preprocessing pipelines including motion correction, slice timing correction, and spatial normalization. For DNN activations, we apply PCA to reduce dimensionality while retaining 95% of variance.

2. **Representational Alignment**: Before clustering, we perform an initial alignment of the two spaces using Procrustes analysis to maximize correspondence:

$$\min_{R} \|XR - Y\|_F^2 \quad \text{subject to } R^TR = I$$

where $X$ is the matrix of DNN activations, $Y$ is the matrix of neural responses, and $R$ is the orthogonal transformation matrix.

3. **Joint Clustering**: We employ a modified prototype learning algorithm that simultaneously clusters both datasets while maximizing correspondence between the identified clusters:

$$\min_{\{c_k\}, \{P_k\}} \sum_{i=1}^N \left[ \lambda_1 d(x_i, c_{k_i}) + \lambda_2 d(y_i, P_{k_i}) + \lambda_3 d(c_{k_i}, P_{k_i}) \right]$$

where $\{c_k\}$ are the DNN prototypes, $\{P_k\}$ are the neural prototypes, $d(\cdot,\cdot)$ is a distance function (e.g., cosine distance), and $\lambda_1, \lambda_2, \lambda_3$ are weighting parameters.

4. **Prototype Refinement**: We iteratively refine the prototypes by optimizing both cluster assignment and prototype locations, gradually increasing alignment between corresponding prototypes from both domains.

The optimal number of prototypes $K$ will be determined through a combination of silhouette analysis, Davies-Bouldin index, and domain knowledge about the stimulus space, with expected values between 50-200 prototypes.

### 2.3 Prototypical Contrastive Alignment Loss

Using the discovered prototypes, we develop a novel prototypical contrastive alignment (PCA) loss function that serves both as an alignment metric and as an intervention mechanism during model training. The PCA loss consists of two main components:

1. **Prototype Attraction**: This component encourages DNN representations to align with their corresponding brain-derived prototypes:

$$\mathcal{L}_{\text{attract}}(x_i) = -\log \frac{\exp(s(f(x_i), c_{k_i})/\tau)}{\sum_{j=1}^K \exp(s(f(x_i), c_j)/\tau)}$$

where $f(x_i)$ is the DNN representation for input $x_i$, $c_{k_i}$ is the corresponding prototype, $s(\cdot,\cdot)$ is the cosine similarity, and $\tau$ is a temperature parameter.

2. **Prototype Contrast**: This component pushes representations away from non-matching prototypes, preserving the discriminative structure of the space:

$$\mathcal{L}_{\text{contrast}}(x_i) = -\log \frac{\exp(s(f(x_i), c_{k_i})/\tau)}{\sum_{j \neq k_i} \exp(s(f(x_i), c_j)/\tau)}$$

The complete PCA loss combines these components:

$$\mathcal{L}_{\text{PCA}}(x_i) = \alpha \mathcal{L}_{\text{attract}}(x_i) + (1-\alpha) \mathcal{L}_{\text{contrast}}(x_i)$$

where $\alpha$ is a balancing parameter between attraction and contrast.

For model training or fine-tuning, we combine this alignment loss with the standard task loss:

$$\mathcal{L}_{\text{total}}(x_i, y_i) = \mathcal{L}_{\text{task}}(x_i, y_i) + \beta \mathcal{L}_{\text{PCA}}(x_i)$$

where $\beta$ controls the strength of the alignment regularization.

### 2.4 Experimental Design and Evaluation

We will evaluate our method through a comprehensive set of experiments designed to test multiple aspects of representational alignment:

1. **Alignment Metric Evaluation**: We will compare our prototype-based alignment metric with existing methods (e.g., CKA, RSA, CCA) in terms of sensitivity, robustness to noise, and consistency across datasets. Specifically, we will compute:

   - **Prototype Alignment Score (PAS)**: $\text{PAS} = \frac{1}{N} \sum_{i=1}^N s(f(x_i), P_{k_i})$
   - **Prototype Distribution Alignment (PDA)**: $\text{PDA} = \text{KL}(\text{Dist}_{\text{DNN}}(P) || \text{Dist}_{\text{brain}}(P))$

2. **Intervention Effectiveness**: We will train multiple DNNs with varying levels of alignment regularization ($\beta$ from 0 to 1) and evaluate changes in:

   - Neural predictivity (measured by encoding model performance)
   - Representational similarity to brain data (using held-out stimuli)
   - Distribution of attention to semantic features

3. **Task Performance Impact**: We will assess how alignment affects model performance on:

   - Original task performance (e.g., image classification)
   - Transfer learning to new tasks
   - Out-of-distribution generalization
   - Human-like error patterns

4. **Interpretability Analysis**: We will conduct detailed analyses of the identified prototypes:

   - Visualization of prototype-activating stimuli
   - Semantic mapping of the prototype space
   - Correlation with human-interpretable features and categories

5. **Ablation Studies**: We will evaluate the impact of:

   - Number of prototypes (K)
   - Choice of neural data modality (fMRI vs. EEG)
   - Layer selection in the DNN
   - Different formulations of the contrastive loss

For each experiment, we will employ appropriate statistical tests (e.g., permutation tests for significance of alignment differences, bootstrap confidence intervals for metric reliability) and control for multiple comparisons when necessary.

### 2.5 Implementation Details

We will implement our method using PyTorch and conduct experiments on multiple model architectures:

1. **Vision Models**: ResNet-50, Vision Transformer (ViT-B/16), CLIP ViT-B/32
2. **Training Datasets**: ImageNet, Objects365, MS-COCO
3. **Computational Resources**: 4 NVIDIA A100 GPUs for model training, cloud storage for dataset management

The joint prototype discovery phase will be implemented using a combination of scikit-learn for initial dimensionality reduction and a custom clustering algorithm based on k-means but extended to support joint optimization across domains.

## 3. Expected Outcomes & Impact

This research is expected to yield several significant outcomes that will advance our understanding of representational alignment between biological and artificial systems:

### 3.1 Scientific Outcomes

1. **Interpretable Alignment Framework**: Our prototypical approach will provide a novel framework for measuring and interpreting representational alignment between brain and DNN representations. The resulting set of prototypes will serve as semantically meaningful anchors that help explain what aspects of cognition are shared across systems.

2. **Insights into Shared Computational Principles**: By analyzing which prototypes show strong alignment and which show divergence, we will gain deeper insights into the computational strategies that are shared between human and artificial vision systems. We expect to identify both domain-general principles (e.g., hierarchical feature extraction) and domain-specific processes that differ between systems.

3. **Quantitative Understanding of Alignment-Performance Relationships**: Our experiments will reveal how different degrees of representational alignment affect model performance on various tasks, clarifying the relationship between brain-like representations and functional capabilities.

### 3.2 Methodological Contributions

1. **Novel Alignment Metric**: The prototype-based alignment score will provide a more interpretable alternative to existing metrics, with stronger connections to semantically meaningful features.

2. **Intervention Mechanism**: The prototypical contrastive alignment loss will offer a practical method for guiding model training toward more brain-like representations, allowing for controlled experiments on the effects of alignment.

3. **Joint Prototype Discovery**: Our approach to jointly discovering prototypes across domains will provide a general methodology that could be applied to other cross-domain representation matching problems.

### 3.3 Practical Applications

1. **More Human-Compatible AI Systems**: Models trained with our alignment method may exhibit more human-like reasoning and error patterns, potentially making them more predictable and trustworthy for human users.

2. **Improved Brain-Computer Interfaces**: The prototype-based approach could enhance brain decoding applications by providing a more robust shared representational space between neural activity and computational models.

3. **Educational Applications**: Understanding how representations align could inform the development of AI-based educational tools that better complement human learning processes.

### 3.4 Broader Impact

This research contributes to the broader goal of developing AI systems that complement human intelligence by sharing similar representational structures. By providing interpretable anchors for alignment and methods to influence it, our work bridges the gap between post-hoc analysis and active intervention in representational learning.

Furthermore, our findings may inform ongoing discussions about AI interpretability and value alignment by providing concrete mechanisms to make AI systems more compatible with human cognitive processes. The prototypical framework offers a middle ground between complete mechanistic understanding and black-box modeling, providing semantically meaningful units of analysis that can guide future development.

In summary, our proposed prototypical contrastive alignment approach represents a significant advance in the field of representational alignment, offering both theoretical insights into shared computational principles and practical methods for developing more human-aligned AI systems.
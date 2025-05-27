**1. Title: Permutation-Invariant Transformer for Cross-Architecture Model Property Prediction from Weights**

**2. Introduction**

The proliferation of publicly available pre-trained neural network models, now numbering over a million on platforms like Hugging Face, presents a unique opportunity and a significant challenge. This abundance transforms neural network weights themselves into a rich source of information, a new data modality ripe for exploration (Workshop Overview). Understanding crucial model properties—such as robustness to adversarial attacks, fairness across demographic groups, or the generalization gap between training and unseen data—directly from these weights, without resorting to extensive and costly empirical evaluations, is a critical research direction. Such capability would dramatically accelerate model development, auditing, and selection processes.

However, neural network weights possess inherent symmetries, most notably permutation symmetry: the neurons within a given layer can be reordered, along with their corresponding weights, without altering the layer's input-output function. This characteristic makes the direct comparison or processing of raw weight vectors from different initializations, or even slightly different architectures, highly challenging for standard machine learning models. For instance, two models could be functionally identical but have vastly different weight representations due to neuron permutations, leading a naive model to perceive them as distinct.

Recent advancements in weight space learning have begun to tackle these challenges. Works like Neural Functional Transformers (NFTs) (Zhou et al., 2023), Universal Neural Functionals (UNFs) (Zhou et al., 2024), and equivariant architectures (Navon et al., 2023) have proposed methods to process neural network weights while respecting permutation symmetries, primarily for tasks like weight optimization or generation. However, the comprehensive prediction of diverse, high-level semantic properties of models across varying architectures directly from their weights remains a nascent area.

This research proposal outlines a plan to develop "WeightNet," a Transformer-based architecture specifically designed to ingest model weights and predict such high-level properties. The core innovation lies in effectively handling permutation symmetries and enabling generalization across diverse model architectures.

**Research Objectives:**

The primary objective of this research is:
*   To design and implement a novel Transformer-based model, WeightNet, capable of predicting a range of salient model properties (e.g., accuracy, generalization gap, robustness, fairness) directly from their raw or minimally processed weight parameters, while being invariant to neuron permutation symmetries.

Secondary objectives include:
1.  To investigate and compare different strategies for achieving permutation invariance within WeightNet, including (a) permutation-invariant attention mechanisms inspired by Set Transformers or NFTs, and (b) canonicalization preprocessing techniques for neuron ordering.
2.  To curate a large-scale, diverse "model zoo" dataset, comprising various neural network architectures and their corresponding weights, along with empirically measured functional properties.
3.  To rigorously evaluate WeightNet's predictive performance on diverse properties and its ability to generalize to unseen model architectures and initializations.
4.  To analyze the internal representations learned by WeightNet to gain insights into how specific weight patterns correlate with high-level model behaviors.
5.  To explore the model's capability to handle inputs from models with significantly different numbers of parameters and architectural types.

**Significance:**

This research directly addresses several key questions posed by the Workshop on Neural Network Weights as a New Data Modality, such as leveraging weight symmetries for learning, efficiently representing and manipulating weights for downstream tasks, and decoding model information from weights. The successful development of WeightNet would have significant impacts:
*   **Accelerated Model Development and Auditing:** Provide a tool for rapid estimation of model properties, reducing the reliance on time-consuming evaluations, thereby speeding up model iteration, selection, and pre-deployment auditing.
*   **Enhanced Understanding of Weight Space:** Contribute to a deeper understanding of the mapping from weight space to functional properties, potentially revealing novel insights into neural network generalization, robustness, and biases.
*   **Democratization of Model Analysis:** Offer a method for quickly assessing models from public repositories without needing to set up complex evaluation pipelines or access original training data.
*   **Enabling New Applications:** Pave the way for novel applications such as semantic search over model zoos (e.g., "find models with high robustness to X attack and low gender bias") or advanced components for Neural Architecture Search (NAS) that can predict properties of candidate architectures from initial weight estimates.

By tackling key challenges like permutation symmetry, high-dimensionality, and cross-architecture generalization, this research aims to make a substantial contribution to establishing neural network weights as a practical and valuable data modality.

**3. Methodology**

This section details the research design, including data collection, the proposed WeightNet architecture with a focus on permutation invariance, and the experimental plan for validation.

**3.1 Data Collection and Preparation: The "Model Zoo" Dataset**

A critical component of this research is the creation of a comprehensive dataset of neural network models and their associated properties.
*   **Model Sources:** We will gather models from diverse sources:
    *   Public model repositories like Hugging Face (Transformers, Diffusers, etc.), PyTorch Hub, TensorFlow Hub.
    *   Research publications that release pre-trained models (e.g., models from computer vision, NLP, robustness benchmarks).
    *   Programmatically trained models: We will train a large number of models on standard benchmark datasets (e.g., CIFAR-10/100, ImageNet subsets, GLUE tasks) with systematic variations in architectures (e.g., ResNet variants, ViT variants, MLP architectures), hyperparameters (learning rate, optimizer, weight decay), initialization seeds, and training epochs (storing intermediate checkpoints).
*   **Model Diversity:** The dataset will aim for diversity in:
    *   **Architectures:** Convolutional Neural Networks (CNNs), Transformers, Multilayer Perceptrons (MLPs), potentially Graph Neural Networks (GNNs) and Recurrent Neural Networks (RNNs).
    *   **Tasks:** Image classification, object detection, natural language understanding, etc.
    *   **Scale:** Models ranging from a few thousand to several million parameters.
*   **Target Properties (Labels):** For each model in the zoo, we will empirically measure and record a suite of properties:
    *   **Performance:** Accuracy, F1-score, perplexity, etc., on standard validation/test sets.
    *   **Generalization:** Generalization gap (difference between training and validation/test performance).
    *   **Robustness:**
        *   Accuracy under standard adversarial attacks (e.g., FGSM, PGD, AutoAttack).
        *   Performance on out-of-distribution or corrupted datasets (e.g., ImageNet-C, CIFAR-10-C).
    *   **Fairness:** (Where applicable and training data allows for disaggregation) Metrics like Demographic Parity Difference, Equalized Odds Difference using appropriate sensitive attributes.
    *   **Efficiency (Inherent):** FLOPs, parameter count (these are architectural but WeightNet can also be trained to predict them as a sanity check or for models where this isn't easily parsed).
    *   **Weight Characteristics:** Sparsity, spectral norm of weight matrices, singular value distributions (can be both input features and prediction targets).
*   **Weight Extraction and Storage:** For each model, we will extract all trainable parameters (weights and biases). These will be stored in a structured format, layer by layer, alongside metadata about the layer type, activation functions, and connections.

**3.2 Proposed Model: Permutation-Invariant Transformer ("WeightNet")**

We propose WeightNet, a Transformer-based model designed to map a sequence derived from a neural network's parameters to its functional properties.

**3.2.1 Input Representation and Preprocessing**
Given a neural network $\mathcal{M}$ with parameters $\theta = \{ (W_l, b_l) \}_{l=1}^L$ for $L$ layers.
*   **Flattening and Tokenization:** The parameters of $\mathcal{M}$ will be transformed into a sequence of input tokens for the Transformer. Several strategies will be considered:
    1.  **Global Flattening:** All parameters $(W_1, b_1, W_2, b_2, \dots)$ are concatenated into a single long vector. This vector is then chunked into tokens, or each parameter (or small group of parameters) becomes a token. This is the most direct interpretation of "flattened model weights."
    2.  **Hierarchical Tokenization (Neuron-centric):** For each neuron $j$ in layer $l$, its incoming weights $W_l[j,:]$ (or a column if $W_l[:,j]$) and its bias $b_l[j]$ are concatenated (and possibly projected) to form a "neuron token" $t_{l,j}$. The input sequence to WeightNet is then $[t_{1,1}, \dots, t_{1,N_1}, t_{2,1}, \dots, t_{L,N_L}]$, where $N_l$ is the number of neurons in layer $l$.
    3.  **Layer-centric Tokenization:** Each layer's entire weight matrix $W_l$ and bias vector $b_l$ are processed (e.g., by a small CNN or MLP, or a per-neuron equivariant operation similar to NFT's intra-layer attention) into one or more "layer tokens". The input sequence is then formed by these layer tokens.
    The choice will be guided by empirical performance and computational feasibility, especially for very large models. We will start with hierarchical/neuron-centric tokenization as it naturally interfaces with permutation symmetry strategies.
*   **Positional and Segment Encodings:** To provide context, especially for flattened representations, we will use:
    *   **Positional Embeddings:** Standard sinusoidal or learned embeddings to indicate the position of a token within the sequence.
    *   **Segment/Layer-Type Embeddings:** Additional embeddings to indicate which layer a parameter/neuron belongs to, the type of layer (e.g., Conv, Linear, Attention), its depth, and potentially other architectural metadata (e.g., kernel size, number of heads). This is crucial for cross-architecture generalization.

**3.2.2 Core Permutation-Invariant Transformer**
The sequence of tokens, augmented with positional and segment embeddings, is fed into a Transformer encoder.
Let the input token sequence be $X = [x_1, x_2, \dots, x_S]$. The initial embeddings are:
$$ H^{(0)} = \text{Embed}(X) + \text{PositionalEncode}(X) + \text{SegmentEncode}(X) $$
The Transformer consists of $K$ layers. Each layer $k$ computes:
$$ A^{(k)} = \text{PermutationInvariantMultiHeadAttention}(Q^{(k-1)}, K^{(k-1)}, V^{(k-1)}) $$
$$ H'^{(k)} = \text{LayerNorm}(H^{(k-1)} + A^{(k)}) $$
$$ H^{(k)} = \text{LayerNorm}(H'^{(k)} + \text{FeedForward}(H'^{(k)})) $$
where $Q^{(k-1)}, K^{(k-1)}, V^{(k-1)}$ are derived from $H^{(k-1)}$.

**Handling Permutation Symmetry:** This is the core challenge and innovation. We will explore two main approaches integrated into the `PermutationInvariantMultiHeadAttention`:

1.  **Permutation-Invariant Attention Mechanisms:**
    *   **Intra-Layer Symmetry:** For neuron-centric tokens belonging to the same hidden layer (e.g., an MLP layer), the attention mechanism over these tokens should be permutation-invariant. This can be achieved by ensuring that queries, keys, and values for neurons within the same layer are treated as elements of a set. For example, attention scores could be computed for all neuron pairs within a layer, and then aggregated.
    *   Inspired by Deep Sets (Zaheer et al., 2017) or Set Transformers (Lee et al., 2019): $\text{Attention}(Q,K,V)_{i} = \sum_{j \in \text{Layer}(i)} \text{softmax}_j(\frac{q_i^T k_j}{\sqrt{d_k}}) v_j$. If $q_i, k_j, v_j$ are processed consistently for all neurons $i,j$ in the same layer, and the output is pooled over the layer (or an equivalent operation is used), invariance can be approached.
    *   We will adapt ideas from NFTs (Zhou et al., 2023) which use attention across neuron axes within weight matrices and then across layers. For our sequence of neuron tokens, this would translate to self-attention mechanisms that pool information symmetrically from other neuron tokens belonging to the *same logical layer* before information is propagated across different layers (represented by different segments of tokens).

2.  **Canonicalization Preprocessing (Alternative/Complementary):**
    *   Before tokenization, attempt to transform the weights of each layer into a canonical form.
    *   **Activation-based sorting:** Probe the model with a fixed batch of data, record neuron activation statistics (e.g., mean, variance), and sort neurons within each layer based on these statistics. This makes the input order consistent. (Drawback: requires model inference).
    *   **Weight-based sorting:** Sort neurons based on a norm (e.g., L2 norm) of their incoming or outgoing weight vectors. This is simpler but might be less robust.
    *   **Optimal Transport (OT):** For each layer, align its weight matrix to a canonical template or to the previous layer's aligned weights using OT. This is computationally expensive but potentially powerful. Research will investigate its feasibility.
    If canonicalization is effective, the downstream Transformer might not need explicitly invariant attention, simplifying the architecture. A hybrid approach is also possible.

**3.2.3 Output Heads and Training**
*   **Aggregation:** The output hidden states $H^{(K)}$ from the Transformer encoder need to be aggregated into a fixed-size representation. This can be done using:
    *   A special [CLS] token approach: Prepend a classification token to the input sequence, and use its corresponding output hidden state $H^{(K)}_{[CLS]}$ as the aggregated representation.
    *   Mean/Max Pooling: Average or max-pool $H^{(K)}$ across the sequence dimension.
    *   Attention-based pooling.
*   **Prediction Heads:** The aggregated representation is then fed into one or more Multi-Layer Perceptron (MLP) heads to predict the target properties. Different properties might have dedicated heads or share a common MLP stem.
    *   For continuous properties (e.g., accuracy, robustness score): Regression head with a suitable loss like Mean Squared Error (MSE) or Mean Absolute Error (MAE). $L_{MSE} = \frac{1}{M} \sum_{i=1}^{M} (y_i - \hat{y}_i)^2$.
    *   For categorical properties (e.g., predicting if robustness is above a threshold): Classification head with Cross-Entropy loss.
*   **Training:** The model will be trained end-to-end using standard optimization algorithms like Adam or AdamW. The total loss will be a weighted sum of losses for all predicted properties.

**3.3 Experimental Design**

**3.3.1 Datasets and Model Zoos**
*   We will create several datasets for training and evaluation:
    *   **Single-Architecture Zoos:** E.g., a zoo of ResNet-18 models trained on CIFAR-10 with varying initializations, optimizers, and training durations. This will test the fundamental property prediction capability and permutation invariance robustly.
    *   **Cross-Architecture Zoos:**
        *   **Family-Specific:** E.g., various CNN architectures (ResNets, VGGs, MobileNets) for image classification. Various Transformer architectures (BERT-base, BERT-large, DistilBERT) for NLP tasks.
        *   **Highly Diverse:** A challenging zoo spanning CNNs, Transformers, MLPs across different tasks, testing the limits of cross-architecture generalization.
*   **Splits:** Data will be split into training, validation, and testing sets based on *model instances*. For cross-architecture generalization tests, entire architectural families present in the test set will be absent from the training set.

**3.3.2 Baselines**
WeightNet's performance will be compared against:
1.  **Simple MLP on Flattened Weights:** An MLP taking the globally flattened, naively ordered weight vector as input. This will highlight the benefit of the Transformer architecture and permutation handling.
2.  **MLP on Handcrafted Weight Features:** An MLP trained on features extracted from weights, such as mean, variance, L1/L2 norms of weights/biases per layer, spectral radius, principal component analysis (PCA) of weights.
3.  **Existing Weight-Space Models (Adapted):** If feasible, components from models like NFT or UNFs will be adapted for the property prediction task, though their original focus might differ (e.g., optimization, generation).
4.  **Zero-Shot Architectural Predictors:** For properties like FLOPs or parameter count, compare against direct calculation from architecture definitions.

**3.3.3 Evaluation Metrics**
*   **Regression Tasks (accuracy, generalization gap, robustness scores):**
    *   Mean Absolute Error (MAE): $\frac{1}{M} \sum_{i=1}^{M} |y_i - \hat{y}_i|$
    *   Root Mean Squared Error (RMSE): $\sqrt{\frac{1}{M} \sum_{i=1}^{M} (y_i - \hat{y}_i)^2}$
    *   Coefficient of Determination ($R^2$ score).
    *   Spearman Rank Correlation: To assess if the model can correctly rank models by property.
*   **Classification Tasks (if any):** Precision, Recall, F1-score, Accuracy.
*   **Generalization Assessment:** Performance drop when evaluated on unseen architectures or tasks compared to performance on seen ones.

**3.3.4 Ablation Studies**
To understand the contribution of different components of WeightNet:
1.  **Impact of Permutation Handling:** Compare models with invariant attention, canonicalization, and no explicit permutation handling.
2.  **Input Representation:** Evaluate different tokenization strategies (global, neuron-centric, layer-centric).
3.  **Transformer Architecture:** Vary depth, number of attention heads, embedding dimensions.
4.  **Effectiveness of Segment/Architectural Embeddings:** Assess their importance for cross-architecture generalization by removing them.
5.  **Dataset Size and Diversity:** Analyze how prediction performance scales with the size and diversity of the training model zoo.

**3.4 Addressing Key Challenges (from Literature Review)**
1.  **Permutation Symmetry Handling:** Directly addressed by the core design of WeightNet (invariant attention or canonicalization).
2.  **High-Dimensional Weight Spaces:** Transformers are well-suited for high-dimensional sequences. Neuron-centric or layer-centric tokenization combined with projections can manage dimensionality. Parameter-efficient Transformer variants (e.g., Linformer, Performer) will be considered if full attention becomes a bottleneck for extremely large models.
3.  **Generalization Across Architectures:** This is a primary experimental focus. The use of segment/architectural embeddings and a flexible tokenization strategy are designed to facilitate this. The evaluation on highly diverse model zoos will test this capability.
4.  **Efficient Training on Large Model Zoos:** Standard distributed training techniques will be employed. We may explore curriculum learning (e.g., starting with smaller models or simpler properties). The model zoo generation itself will be a significant but parallelizable effort.
5.  **Interpretability of Weight-Based Predictions:** We will use Transformer attention visualization techniques (analyzing attention weights) to identify which parts of the input model's weights (e.g., specific layers or types of parameters) are most salient for predicting certain properties. Sensitivity analysis (perturbing input weights and observing output changes) will also be performed.

**4. Expected Outcomes & Impact**

**Expected Outcomes:**
1.  **A Novel Predictive Model (WeightNet):** The primary outcome will be a robust and validated WeightNet model capable of accurately predicting multiple functional properties of neural networks directly from their weights, demonstrating invariance to permutation symmetries.
2.  **Comparative Analysis of Permutation Handling:** Empirical evidence and insights into the most effective strategies (invariant attention mechanisms vs. canonicalization techniques) for handling permutation symmetry in the context of weight-based property prediction.
3.  **Demonstrated Cross-Architecture Generalization:** Quantitative results showcasing WeightNet's ability to predict properties for neural network architectures not seen during its training phase, and an understanding of its limitations.
4.  **Superior Performance:** WeightNet is expected to significantly outperform baseline methods that do not explicitly account for weight space symmetries or lack the expressive power of Transformers for sequence modeling.
5.  **Insights into Weight-Property Mappings:** The research will contribute to understanding which features or patterns in weight space are most indicative of specific high-level behaviors (e.g., what distinguishes robust models from non-robust ones at the weight level). This may be gleaned from analyzing WeightNet's internal representations and attention patterns.
6.  **Publicly Available Resources:** We aim to release the curated model zoo dataset (weights, measured properties, and architectural metadata) and the WeightNet codebase to the research community, fostering further research and reproducibility.

**Impact:**

*   **Scientific Impact:**
    *   **Advancing Weight Space as a Modality:** This research will make a significant contribution to establishing neural network weights as a viable and informative data modality, moving beyond theoretical conceptualizations to practical applications.
    *   **New Analytical Tools:** WeightNet will provide a new class of tools for neural network analysis, complementing traditional empirical evaluation methods.
    *   **Understanding Model Internals:** By learning to map weights to behavior, WeightNet can contribute to the broader effort of interpreting and understanding the complex, high-dimensional manifolds of neural network weight spaces. It can help answer what information is encoded in weights about a model's training history, biases, and potential.
    *   **Bridging Research Areas:** This work will help bridge equivariant deep learning research with practical model evaluation and analysis, potentially inspiring new equivariant architectures for other weight space tasks.

*   **Practical Impact:**
    *   **Efficiency in MLOps:** Substantially reduce the computational cost and time associated with model validation and selection, particularly in scenarios involving large numbers of candidate models (e.g., NAS, large-scale hyperparameter optimization, model zoos).
    *   **Early Warning Systems:** Enable rapid, low-cost screening of models for potential issues like poor generalization, inherent biases, or vulnerability to attacks before full deployment.
    *   **Democratizing Model Assessment:** Allow researchers and practitioners with limited computational resources to quickly assess off-the-shelf models from repositories like Hugging Face.
    *   **Facilitating Model Governance and Trust:** By providing insights into model properties from weights, WeightNet could contribute to more transparent and trustworthy AI systems. For example, it could be used to quickly flag models that might have been "backdoored" if it can learn to detect tell-tale weight patterns.
    *   **Informing Model Design:** Long-term, understanding which weight configurations lead to desirable properties could inform better initialization strategies, regularization techniques, or even architectural design principles.

In conclusion, this research on a Permutation-Invariant Transformer for Cross-Architecture Model Property Prediction promises to deliver a powerful new tool for the machine learning community, advancing our ability to understand, analyze, and efficiently utilize the vast and growing landscape of neural network models.
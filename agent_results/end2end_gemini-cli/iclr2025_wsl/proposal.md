### **1. Title: Detecting Neural Backdoors via Permutation-Equivariant Graph Analysis of Weight Space**

---

### **2. Introduction**

#### **2.1. Background: The Proliferation of Models and the Rise of Backdoor Threats**
The machine learning landscape is undergoing a paradigm shift. Platforms like Hugging Face now host over a million pre-trained models, transforming neural networks from bespoke artifacts into readily accessible, off-the-shelf components. This democratization of AI, while fostering rapid innovation, introduces critical security vulnerabilities. A particularly insidious threat is the **backdoor attack**, or neural trojan, where malicious behavior is covertly embedded into a model during or after its training. A backdoored model performs as expected on benign inputs but executes a hidden, malicious task when a specific trigger (e.g., a small image patch, a particular phrase) is present in the input. These attacks pose a severe risk to AI-powered systems in sensitive domains such as autonomous driving, medical diagnosis, and content moderation.

The literature on backdoor defense and detection is rich and growing. However, a significant portion of existing methods (Zhu et al., 2023; Fu et al., 2023) exhibit critical limitations that hinder their practical deployment. Many approaches are **data-dependent**, requiring a set of clean data to perform analysis, which is often unavailable in real-world scenarios where users download models from untrusted sources. Others are **attack-specific**, designed to detect particular types of triggers or attack methodologies (Xiang et al., 2023), and may fail against novel or more subtle attacks (Han et al., 2023; Li et al., 2021). Consequently, there is a pressing need for a universal, data-free detection mechanism that can identify the presence of a backdoor by inspecting the model itself.

This proposal is situated within the emerging research area that treats **neural network weights as a new data modality**. We posit that the complex process of embedding a backdoor into a neural network leaves indelible structural artifacts—a "fingerprint"—within the model's parameter space. Just as a trained model's weights encode information about its primary task, we hypothesize they also encode information about its vulnerabilities and embedded malicious logic. Our goal is to develop a method to read and interpret these fingerprints directly from the weight space, thus creating a classifier for models themselves.

#### **2.2. Research Objectives**
The central aim of this research is to develop a novel, data-free framework for backdoor detection by leveraging the inherent graph structure of neural networks and the power of permutation-equivariant learning. Our approach rests on the key insight that the function of a neural network is invariant to the permutation of neurons within a hidden layer—a fundamental symmetry. By processing the model's computational graph with a Graph Neural Network (GNN), which naturally respects this symmetry, we can learn to identify canonical, architecture-agnostic signatures of backdoors.

We define the following specific research objectives:

1.  **Formalize Neural Networks as Attributed Graphs:** To develop a standardized procedure for representing any feed-forward neural network as a directed acyclic graph where neurons are nodes and synaptic weights are edge attributes, making them amenable to graph-based machine learning.

2.  **Design a Permutation-Equivariant Detector:** To design, implement, and train a specialized Graph Neural Network, which we term the **Backdoor-GraphNet (BD-GNN)**, capable of classifying an input model graph as either `clean` or `backdoored`.

3.  **Construct a Large-Scale Model Zoo Benchmark:** To create a diverse, large-scale dataset of neural network models—a "model zoo"—comprising thousands of both cleanly trained and backdoored instances across various architectures, training datasets, and attack types. This zoo will serve as the training and evaluation ground for the BD-GNN and will be released as a public resource.

4.  **Validate Performance and Generalizability:** To rigorously evaluate the BD-GNN's effectiveness in detecting backdoors and, crucially, its ability to generalize to unseen model architectures, tasks, and attack methodologies not encountered during its training.

#### **2.3. Significance**
This research promises significant contributions to both the theory and practice of AI security and deep learning. Its primary impact will be a practical, data-free tool for securing the AI supply chain, enabling users to vet third-party models without needing access to trusted data. Scientifically, this work serves as a pioneering case study for the "weights as data" paradigm, demonstrating that meta-learning over model populations can reveal complex properties like malicious behavior. By successfully leveraging the permutation symmetry of neural networks, our work addresses a fundamental challenge in weight space analysis and provides a blueprint for future research in model property inference, model editing, and meta-learning.

---

### **3. Methodology**

Our research plan is structured into four sequential phases: (1) Generation of a comprehensive model zoo dataset, (2) Formalization of the graph representation for neural networks, (3) Design and implementation of the BD-GNN detector, and (4) Rigorous experimental validation and analysis.

#### **3.1. Phase 1: Generation of the Model Zoo Dataset**

The foundation of our supervised learning approach is a large and diverse dataset of model weights. We will construct a "model zoo" containing thousands of trained neural networks, labeled as either `clean` or `backdoored`.

*   **Base Architectures:** To ensure architectural diversity, we will include a range of commonly used models:
    *   **Computer Vision:** VGG11/16, ResNet18/34, MobileNetV2.
    *   **Simple NLP:** Small-scale Transformer-based encoders and LSTMs for classification tasks.

*   **Training Tasks & Datasets:** Models will be trained on standard benchmark datasets to ensure they are realistic and functional.
    *   **Vision:** CIFAR-10, CIFAR-100, GTSRB (German Traffic Sign Recognition Benchmark).
    *   **NLP:** IMDB sentiment classification, AG News.

*   **Backdoor Attack Implementation:** For the `backdoored` subset of the zoo, we will implement a variety of established and recent backdoor attacks to promote detector generalization. This includes:
    *   **Patch-based Attacks (BadNets):** A small, fixed pixel pattern is associated with the target label.
    *   **Blended-Injection Attacks:** The trigger is a global, low-opacity noise pattern blended with the input image.
    *   **Clean-Label Attacks:** Poisoned training samples are correctly labeled but contain a trigger that enforces misclassification to a different target label at inference time.
    *   **Advanced Attacks:** We will investigate implementing more subtle attacks, such as those inspired by ReVeil (Alam et al., 2025) or natural triggers (Han et al., 2023), to test the limits of our detector.

*   **Data Generation Protocol:** We aim to generate approximately 10,000 models. For each combination of (architecture, dataset), we will train multiple `clean` instances with different random seeds. For each clean model, we will generate several `backdoored` counterparts using different attack types, trigger patterns, and target labels. All models will be trained until they reach a predetermined level of performance on the clean test set to ensure they represent viable, deployable models. The final dataset of models will be split into training (80%), validation (10%), and testing (10%) sets for developing the BD-GNN.

#### **3.2. Phase 2: Graph Representation of Neural Networks**

A neural network can be naturally viewed as a computational graph. We will formalize this representation to serve as the input for our GNN. A feed-forward network will be converted into a directed graph $G = (V, E, H, W)$, where:

*   **Nodes $V$**: Each neuron in the network corresponds to a node $v_i \in V$. This includes input, hidden, and output neurons.
*   **Edges $E$**: A directed edge $e_{ij} \in E$ exists from node $v_i$ to node $v_j$ if the output of neuron $i$ is an input to neuron $j$.
*   **Node Features $H$**: Each node $v_i$ is associated with a feature vector $h_i \in \mathbb{R}^{d_n}$. The initial node features will encode local information about the neuron. We will experiment with several feature designs:
    *   **Bias-based:** $h_i$ is simply the bias term $b_i$ of the neuron.
    *   **Structural:** $h_i$ is a one-hot encoding of the layer type (e.g., Conv, Linear, ReLU) and/or its depth in the network.
    *   **Learnable:** $h_i$ is a learnable embedding, conditioned on the layer type.
*   **Edge Features $W$**: Each edge $e_{ij}$ is associated with a feature vector representing the synaptic connection. For a standard dense or convolutional layer, this will simply be the scalar weight $w_{ji}$ connecting neuron $i$ to neuron $j$.

This graph representation is powerful because the adjacency structure is preserved regardless of how neurons within a layer are ordered. The GNN's message-passing mechanism, which operates based on this connectivity, will thus be inherently equivariant to neuron permutations.

#### **3.3. Phase 3: The Backdoor-GraphNet (BD-GNN) Architecture**

The core of our proposal is the BD-GNN, a GNN designed to perform graph-level classification on the model graphs.

*   **Message Passing Framework:** The BD-GNN will be composed of a stack of message-passing layers. In each layer $l$, the feature vector $h_i^{(l)}$ for each node $v_i$ is updated based on messages from its neighbors $\mathcal{N}(i)$. The general update rule is:
    $$
    h_i^{(l+1)} = \psi^{(l)} \left( h_i^{(l)}, \bigoplus_{j \in \mathcal{N}(i)} \phi^{(l)} \left( h_i^{(l)}, h_j^{(l)}, w_{ij} \right) \right)
    $$
    where:
    *   $h_i^{(l)}$ is the feature vector of node $i$ at layer $l$.
    *   $w_{ij}$ is the edge feature (weight) from node $j$ to $i$.
    *   $\phi^{(l)}$ is a learnable **message function** (e.g., an MLP) that computes a message from node $j$ to $i$.
    *   $\bigoplus$ is a permutation-invariant **aggregation function** (e.g., `sum`, `mean`, `max`) that combines incoming messages.
    *   $\psi^{(l)}$ is a learnable **update function** (e.g., an MLP or GRU-like cell) that combines the aggregated message with the node's previous state.

    This formulation, particularly the aggregation function $\bigoplus$, ensures that the learned representation is independent of the ordering of neighbors, directly addressing the permutation symmetry challenge. We will investigate variants of GNN layers, including GCN, GraphSAGE, and GIN, to find the most expressive architecture for this task.

*   **Graph Readout:** After $L$ layers of message passing, we obtain final node embeddings $h_i^{(L)}$. To produce a single vector representation for the entire graph (the entire model), we apply a graph pooling (readout) function:
    $$
    h_G = \text{READOUT} \left( \{ h_i^{(L)} \mid v_i \in V \} \right)
    $$
    We will experiment with global sum, mean, and attention-based pooling mechanisms.

*   **Classification Head:** The final graph embedding $h_G$ is passed through a multi-layer perceptron (MLP) with a sigmoid output layer to produce a probability $p \in [0, 1]$, indicating the likelihood that the input model is backdoored.

*   **Training:** The BD-GNN will be trained end-to-end using the binary cross-entropy loss function:
    $$
    \mathcal{L} = - \sum_{k=1}^{N} \left[ y_k \log(p_k) + (1 - y_k) \log(1 - p_k) \right]
    $$
    where $N$ is the number of models in the training set, $y_k \in \{0, 1\}$ is the ground-truth label (`clean`/`backdoored`), and $p_k$ is the BD-GNN's predicted probability for the $k$-th model graph. We will use the Adam optimizer with a learning rate schedule.

#### **3.4. Phase 4: Experimental Design and Validation**

A rigorous evaluation protocol is essential to validate our claims of robustness and generalization.

*   **Evaluation Metrics:** We will assess performance using standard classification metrics: Accuracy, Precision, Recall, F1-Score, and the Area Under the Receiver Operating Characteristic Curve (AUC-ROC), which is especially informative for imbalanced test sets and for setting detection thresholds.

*   **Baselines for Comparison:** Since data-free weight-space analysis is nascent, we will compare BD-GNN against several baselines:
    1.  **MLP on Flattened Weights:** An MLP that takes the flattened weight vector of a model as input. This will serve to demonstrate the necessity of a permutation-equivariant architecture.
    2.  **Statistical Baselines:** Methods based on analyzing the statistical properties of weight matrices, such as the norm distribution, singular value distribution, or spectral density.
    3.  **Adapted Interpretability Methods:** We will attempt to adapt the core logic of methods like TrojanInterpret (2024), which uses saliency maps, into a data-free heuristic for comparison.

*   **Generalization Experiments:** This is the most critical part of our evaluation. We will use the test split of our model zoo, which contains models and attacks held out from training.
    1.  **Cross-Architecture Generalization:** Train BD-GNN on models with ResNet architectures and evaluate its performance on VGG and MobileNetV2 models (and vice versa).
    2.  **Cross-Attack Generalization:** Train BD-GNN on models with only BadNets and blended attacks and evaluate its ability to detect unseen attack types like clean-label attacks.
    3.  **Cross-Task Generalization:** Train BD-GNN on models trained for CIFAR-10 and evaluate its performance on models trained for GTSRB.

*   **Ablation Studies and Analysis:** To understand what makes BD-GNN effective, we will conduct ablation studies on its components:
    *   The impact of different node and edge feature initializations.
    *   The contribution of message-passing depth (number of GNN layers).
    *   Comparative analysis of different aggregation and readout functions.
    *   We will also use GNN explanation techniques (e.g., GNNExplainer) on the trained BD-GNN to visualize which subgraphs (i.e., which parts of the neural network) are most indicative of a backdoor, providing valuable insights into the learned "fingerprints."

---

### **4. Expected Outcomes & Impact**

This research is expected to yield several significant outcomes with far-reaching impact.

*   **Expected Outcomes:**
    1.  **A Novel Backdoor Detection Framework (BD-GNN):** The primary output will be a fully implemented and validated system capable of detecting backdoored neural networks by analyzing their weights alone. The framework will be highly accurate and, most importantly, generalize across different model types and attack vectors.
    2.  **A Public Model Zoo Dataset:** We will release our large-scale dataset of over 10,000 clean and backdoored models. This will be an invaluable resource for the research community, enabling standardized evaluation of backdoor detection methods and fostering further research into weight space learning.
    3.  **Fundamental Insights into Backdoor Structures:** By analyzing the learned representations within the BD-GNN, we expect to uncover the structural properties in weight space that distinguish malicious models from benign ones. This moves beyond black-box detection to provide a deeper, qualitative understanding of how backdoors manifest in model parameters.

*   **Impact:**
    *   **Practical Impact on AI Security:** The BD-GNN framework can be deployed as a practical tool for securing the AI supply chain. Organizations and individuals using models from public repositories can use it as a "virus scanner" to mitigate the risk of deploying trojanized AI. This directly contributes to building a more trustworthy and secure AI ecosystem.
    *   **Scientific Impact on Weight Space Learning:** This project serves as a cornerstone for the emerging field of learning on neural network weights. It provides a compelling demonstration that it is possible to learn complex, functional properties of a model directly from its parameters. The success of our permutation-equivariant approach will establish a powerful methodology for other tasks in model analysis, such as predicting a model's generalization capabilities, identifying its training data properties, or even performing model merging and editing in a principled manner.
    *   **Fostering Interdisciplinary Research:** Our work lies at the intersection of AI security, graph representation learning, and meta-learning. By developing novel techniques and a public benchmark, we aim to bridge these communities and stimulate further research into the rich, unexplored frontier of the neural network weight space.
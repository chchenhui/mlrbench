# Reasoning Uncertainty Networks: Enhancing LLM Transparency Through Graph-Based Belief Propagation

## 1. Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities in text generation, reasoning, and problem-solving across diverse domains. However, as these models are increasingly deployed in high-stakes applications such as healthcare, legal systems, and autonomous vehicles, their tendency to generate confident-sounding but factually incorrect information—known as hallucinations—poses significant risks. The challenge is further compounded by LLMs' inability to transparently represent their uncertainty, making it difficult for users to discern when to trust model outputs and when human oversight is necessary.

Current approaches to uncertainty quantification (UQ) in LLMs typically employ post-hoc methods such as calibration techniques, ensemble methods, or sampling-based approaches like those seen in SelfCheckGPT (Manakul et al., 2023). While valuable, these methods treat uncertainty as an afterthought rather than as an integral component of the reasoning process itself. This limitation is particularly problematic when LLMs perform multi-step reasoning tasks, where errors can propagate and compound through the reasoning chain without any explicit signals of increasing uncertainty.

Recent work has begun to address these challenges from various angles. Multi-dimensional uncertainty quantification frameworks (Chen et al., 2025) integrate semantic and knowledge-aware similarity analysis to derive comprehensive uncertainty representations. Other approaches like HuDEx (Lee et al., 2025) focus on explanation-enhanced hallucination detection, while MetaQA (Yang et al., 2025) leverages metamorphic relations and prompt mutation for self-contained hallucination detection. However, these approaches still largely treat uncertainty estimation as separate from the reasoning process itself.

The objective of this research is to develop a novel framework called "Reasoning Uncertainty Networks" (RUNs) that represents LLM reasoning as a directed graph where uncertainty is explicitly modeled and propagated throughout the reasoning process. This approach aims to:

1. Make uncertainty an explicit, integral component of the reasoning chain rather than a post-hoc calculation
2. Provide fine-grained transparency into how confidence levels flow through complex reasoning steps
3. Enable automatic detection of potential hallucination points based on uncertainty thresholds
4. Create a computationally efficient method that operates at the semantic level rather than requiring multiple model inferences
5. Enhance explainability by allowing users to identify precisely where reasoning uncertainty originates

The significance of this research lies in its potential to address a critical gap in current LLM technology: the lack of transparent uncertainty representation throughout the reasoning process. By developing a framework that makes uncertainty explicit and interpretable at each step of reasoning, this work could substantially enhance the reliability and trustworthiness of LLMs in high-stakes domains. Furthermore, the graph-based approach provides a natural interface for human oversight, allowing domain experts to intervene precisely where reasoning becomes uncertain or potentially erroneous.

## 2. Methodology

### 2.1 System Architecture

The proposed Reasoning Uncertainty Networks (RUNs) framework consists of four main components:

1. **Reasoning Graph Constructor**: Transforms LLM-generated reasoning into a directed graph structure
2. **Uncertainty Initializer**: Assigns initial uncertainty distributions to graph nodes
3. **Belief Propagation Engine**: Updates uncertainty values across the graph using message passing
4. **Hallucination Detection Module**: Identifies potential hallucinations based on uncertainty thresholds

Figure 1 illustrates the overall architecture of the RUN framework.

### 2.2 Reasoning Graph Construction

The first step in our approach is to convert LLM reasoning into a structured graph representation. We prompt the LLM to generate reasoning in a step-by-step format, then parse this output to construct a directed graph $G = (V, E)$ where:

- Nodes $v_i \in V$ represent factual assertions or reasoning steps
- Edges $e_{ij} \in E$ represent logical dependencies between assertions (indicating that assertion $v_j$ depends on assertion $v_i$)

To construct this graph, we employ a two-stage process:

1. **Assertion Extraction**: We use a specialized prompt template to elicit structured reasoning from the LLM, where each reasoning step is clearly delineated. For example:
   ```
   Given [problem], please reason step by step:
   Step 1: [assertion]
   Step 2: [assertion]
   ...
   Conclusion: [final answer]
   ```

2. **Dependency Identification**: For each assertion, we prompt the LLM to identify which previous assertions it directly depends on. This creates the edges in our graph. For example:
   ```
   For the assertion "[assertion i]", which previous steps does this directly depend on?
   ```

Alternatively, for more complex reasoning, we can leverage existing work on chain-of-thought (CoT) parsing techniques to automatically extract the reasoning graph structure.

### 2.3 Uncertainty Representation and Initialization

Each node $v_i$ in the graph is associated with an uncertainty distribution $U_i$. We model uncertainty using a Beta distribution $\text{Beta}(\alpha_i, \beta_i)$, which naturally represents probabilities about probabilities and can express various shapes of uncertainty. The mean of this distribution, $\mu_i = \frac{\alpha_i}{\alpha_i + \beta_i}$, represents the expected confidence in the assertion, while the variance, $\sigma_i^2 = \frac{\alpha_i\beta_i}{(\alpha_i + \beta_i)^2(\alpha_i + \beta_i + 1)}$, represents the uncertainty about this confidence.

For initialization, we consider three sources of information:

1. **Direct LLM self-assessment**: We prompt the LLM to provide a confidence score for each assertion, mapping this to the mean $\mu_i$ of the Beta distribution.

2. **Semantic uncertainty estimation**: We implement a specialized uncertainty estimator based on embedding similarity. For each assertion $v_i$, we generate $n$ alternative formulations by prompting the LLM with slight variations. We then compute the average cosine similarity between the embeddings of these variations:

   $$S_i = \frac{1}{n(n-1)}\sum_{j=1}^{n}\sum_{k=j+1}^{n}\cos(\mathbf{e}_j, \mathbf{e}_k)$$

   where $\mathbf{e}_j$ and $\mathbf{e}_k$ are embeddings of alternative formulations. This similarity score is then mapped to parameters $\alpha_i$ and $\beta_i$ of the Beta distribution.

3. **Knowledge-grounded verification**: For factual assertions, we implement a retrieval-augmented verification step that compares the assertion with retrieved information from a knowledge base. The verification score contributes to the initial uncertainty estimation.

The parameters $\alpha_i$ and $\beta_i$ are set to ensure that the mean matches the confidence score from the LLM, and the variance reflects the consistency of alternative formulations and factual verification:

$$\alpha_i = \mu_i \cdot c_i$$
$$\beta_i = (1 - \mu_i) \cdot c_i$$

where $c_i$ is a concentration parameter derived from the semantic similarity and factual verification scores, with higher consistency leading to higher concentration (lower variance).

### 2.4 Belief Propagation Algorithm

The core of our approach is a belief propagation algorithm that updates uncertainty distributions across the graph based on the logical dependencies between assertions. Intuitively, if an assertion depends on uncertain premises, that uncertainty should propagate to the assertion itself.

We implement a message-passing algorithm where nodes exchange information about their uncertainty distributions. For each directed edge $e_{ij}$ from node $v_i$ to node $v_j$, a message $m_{ij}$ is passed that represents how the uncertainty in $v_i$ influences the uncertainty in $v_j$.

The belief propagation occurs in discrete time steps. At each step $t$:

1. Each node $v_i$ sends a message to its dependent nodes based on its current uncertainty distribution $U_i^{(t)}$
2. Each node $v_j$ updates its uncertainty distribution based on the messages received from its prerequisite nodes

The message from node $v_i$ to node $v_j$ at time $t$ is defined as:

$$m_{ij}^{(t)} = f_{\text{message}}(U_i^{(t)}, w_{ij})$$

where $w_{ij}$ represents the strength of the logical dependency between assertions $i$ and $j$, and $f_{\text{message}}$ is a function that transforms the uncertainty distribution based on this dependency strength.

The update rule for node $v_j$ at time $t+1$ is:

$$U_j^{(t+1)} = f_{\text{update}}(U_j^{(0)}, \{m_{ij}^{(t)} | v_i \in \text{parents}(v_j)\})$$

where $U_j^{(0)}$ is the initial uncertainty distribution for node $v_j$, and $f_{\text{update}}$ is a function that combines the initial uncertainty with the incoming messages.

For Beta distributions, these update functions can be implemented as:

$$m_{ij}^{(t)} = \text{Beta}(w_{ij} \cdot \alpha_i^{(t)}, w_{ij} \cdot \beta_i^{(t)})$$

$$U_j^{(t+1)} = \text{Beta}(\alpha_j^{(0)} + \sum_{v_i \in \text{parents}(v_j)} (w_{ij} \cdot \alpha_i^{(t)} - 1), \beta_j^{(0)} + \sum_{v_i \in \text{parents}(v_j)} (w_{ij} \cdot \beta_i^{(t)} - 1))$$

We iterate this message-passing algorithm until convergence or for a fixed number of steps.

### 2.5 Hallucination Detection

Based on the propagated uncertainty distributions, we implement a hallucination detection module that flags potential hallucinations using the following criteria:

1. **High uncertainty threshold**: Assertions with mean confidence below a threshold $\tau_{\text{conf}}$ (e.g., $\tau_{\text{conf}} = 0.7$) are flagged as potential hallucinations.

2. **Uncertainty increase detection**: Assertions where the uncertainty significantly increases after belief propagation (compared to the initial uncertainty) are flagged, as this indicates the assertion relies on uncertain premises.

3. **Logical consistency checking**: We implement a consistency checker that identifies potential contradictions within the reasoning graph. For assertions $v_i$ and $v_j$ that are determined to be contradictory, we flag the one with higher uncertainty as a potential hallucination.

For each flagged assertion, we compute a hallucination score $H_i$ given by:

$$H_i = (1 - \mu_i) \cdot (1 + \gamma \cdot \Delta\sigma_i^2) \cdot (1 + \delta \cdot C_i)$$

where $\mu_i$ is the mean confidence, $\Delta\sigma_i^2$ is the change in variance after belief propagation, $C_i$ is a measure of logical inconsistency with other assertions, and $\gamma$ and $\delta$ are weighting parameters.

### 2.6 Experimental Design

We will evaluate the RUN framework on the following tasks and datasets:

1. **Scientific reasoning**: Using the SciQ dataset and BIG-bench science tasks, we'll evaluate the framework's ability to detect hallucinations in scientific reasoning.

2. **Legal reasoning**: Using legal case analysis tasks where factual accuracy is crucial.

3. **Medical diagnosis reasoning**: Using medical case studies with established diagnoses.

For each task, we will construct test cases with deliberately introduced errors at different points in the reasoning chain to evaluate the system's ability to detect these errors through uncertainty propagation.

We will compare our approach against the following baselines:

1. **SelfCheckGPT**: A sampling-based approach that checks consistency across multiple samples.
2. **Multi-dimensional UQ**: The approach proposed by Chen et al. (2025) that integrates semantic and knowledge-aware similarity analysis.
3. **Calibration-based approaches**: Traditional methods that calibrate the model's output probabilities.
4. **HuDEx**: An explanation-enhanced hallucination detection model.
5. **MetaQA**: A metamorphic relation-based approach for hallucination detection.

### 2.7 Evaluation Metrics

We will evaluate our approach using the following metrics:

1. **Hallucination Detection Performance**:
   - Precision, Recall, and F1 score for identifying hallucinations
   - Area Under the Precision-Recall Curve (AUPRC)
   - False positive and false negative rates

2. **Uncertainty Calibration**:
   - Expected Calibration Error (ECE)
   - Brier score

3. **Computational Efficiency**:
   - Inference time compared to baseline approaches
   - Memory usage

4. **User Study Metrics**:
   - User trust assessment
   - Time to identify potential errors with and without the system
   - User confidence in system outputs

## 3. Expected Outcomes & Impact

### 3.1 Expected Outcomes

The proposed research is expected to yield several significant outcomes:

1. **A Novel Framework for Reasoning Uncertainty**: The development of Reasoning Uncertainty Networks will provide a new paradigm for modeling and propagating uncertainty throughout multi-step reasoning processes in LLMs. This framework will enable fine-grained tracking of confidence levels across complex reasoning chains.

2. **Enhanced Hallucination Detection**: By integrating uncertainty quantification directly into the reasoning process, we expect to achieve superior performance in detecting hallucinations compared to post-hoc methods. The system should be able to identify not just the presence of hallucinations but also their likely origin points in the reasoning chain.

3. **Improved Transparency and Explainability**: The graph-based representation will provide transparent and interpretable insights into model reasoning and uncertainty, making it easier for users to understand where and why an LLM might be uncertain or incorrect.

4. **Efficient Uncertainty Quantification**: By operating at the semantic level rather than requiring multiple model inferences, the approach should provide computational efficiency advantages over ensemble-based or sampling-based methods.

5. **New Benchmarks and Evaluation Metrics**: Through our experimental work, we will develop new benchmarks and evaluation methodologies for assessing uncertainty quantification and hallucination detection in reasoning tasks.

### 3.2 Potential Impact

The potential impact of this research spans several dimensions:

1. **Advancing AI Safety and Reliability**: By enabling LLMs to express uncertainty transparently and detect potential hallucinations, this work directly contributes to making AI systems safer and more reliable for deployment in high-stakes domains.

2. **Enabling New Applications**: Improved uncertainty quantification and hallucination detection could enable the responsible application of LLMs in domains that currently remain off-limits due to reliability concerns, such as medical diagnosis assistance, legal analysis, and scientific research.

3. **Enhancing Human-AI Collaboration**: The transparent representation of uncertainty enables more effective collaboration between humans and AI systems, allowing human experts to focus their attention where it's most needed and make informed decisions about when to trust AI outputs.

4. **Theoretical Contributions**: The mathematical formulation of belief propagation in reasoning graphs provides a theoretical foundation that bridges logical reasoning and probabilistic uncertainty, potentially informing future work on reasoning under uncertainty in AI systems.

5. **Practical Tooling**: The developed framework can serve as a practical tool for developers and users of LLMs, helping them to assess and mitigate risks associated with model hallucinations.

### 3.3 Limitations and Future Work

We recognize several potential limitations of our approach that would need to be addressed in future work:

1. **Dependence on Initial Graph Construction**: The quality of the reasoning graph construction directly impacts the effectiveness of uncertainty propagation. Future work could explore more robust methods for extracting reasoning structures from LLM outputs.

2. **Scalability to Very Complex Reasoning**: As reasoning chains become extremely complex, the graph representation may become unwieldy. Future work could explore hierarchical graph representations or other abstractions to manage complexity.

3. **Domain-Specific Adaptation**: While our approach aims to be domain-agnostic, optimal performance may require domain-specific adaptations, particularly in specialized fields like medicine or law.

4. **Integration with Retrieval-Augmented Generation**: Future work could explore tighter integration between our uncertainty propagation framework and retrieval-augmented generation approaches, potentially enabling dynamic retrieval based on identified high-uncertainty nodes.

In conclusion, the Reasoning Uncertainty Networks framework represents a significant step toward making uncertainty an explicit, integral component of LLM reasoning rather than an afterthought. By enabling transparent propagation of uncertainty through reasoning chains, this approach has the potential to substantially enhance the reliability, trustworthiness, and safety of large language models in high-stakes applications.
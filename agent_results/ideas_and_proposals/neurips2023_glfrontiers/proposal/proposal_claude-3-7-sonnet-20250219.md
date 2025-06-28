# GraphLang: A Unified Graph-Language Foundation Model for Interactive Graph Reasoning

## 1. Introduction

### Background
Graphs are ubiquitous data structures that capture relationships between entities across diverse domains: knowledge bases represent factual relationships, molecular graphs encode chemical structures, social networks map human connections, and protein interaction networks describe biological processes. Despite this prevalence, graph data remains largely inaccessible to non-specialists due to the technical expertise required for analysis and interpretation. Even for domain experts, extracting insights from complex graph structures often demands specialized programming skills and domain-specific tools.

Concurrently, large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding, generation, and reasoning. These models have revolutionized human-computer interaction by providing intuitive interfaces through natural language. However, as noted in recent research (Zhao et al., 2023; Tang et al., 2023), LLMs struggle with graph-structured data, particularly in tasks requiring structural reasoning, relationship inference, and graph manipulation.

Several recent approaches have attempted to bridge this gap. GraphText (Zhao et al., 2023) translates graphs into text representations to leverage LLMs' reasoning capabilities. GraphGPT (Tang et al., 2023) introduces instruction tuning to enhance LLMs with graph structural knowledge. GraphLLM (Chai et al., 2023) integrates graph learning models with LLMs to improve graph reasoning. GRAPHGPT-O (Fang et al., 2025) further extends this concept to multimodal graph understanding. Despite these advances, current approaches typically treat graphs and language as separate modalities with limited integration, often requiring complex processing pipelines or conversion strategies that can lose critical structural information.

### Research Objectives
This research proposes GraphLang, a unified graph-language foundation model that seamlessly integrates graph representation learning with natural language processing. GraphLang aims to:

1. Develop a multi-modal Transformer architecture that jointly processes graph structures and natural language, enabling bidirectional translation between these modalities.

2. Design and implement an effective pre-training framework using diverse graph-text paired data from knowledge graphs, molecular datasets, and scene graphs.

3. Create an instruction tuning methodology specifically tailored for graph reasoning tasks, enabling users to query, explain, and modify graphs through natural language.

4. Evaluate the model's performance on downstream tasks including graph question answering, subgraph retrieval, and language-guided graph editing across various domains.

5. Demonstrate GraphLang's capability to generalize to heterophilic graphs and unseen graph structures, addressing a key limitation identified in current graph neural networks (Luan et al., 2022, 2023).

### Significance
GraphLang represents a significant advancement in bridging graph learning and natural language processing for several reasons:

First, by creating a unified foundation model for graphs and language, GraphLang democratizes access to graph-structured data, enabling non-specialists to leverage graph insights through intuitive natural language interfaces. This has profound implications for scientific discovery, particularly in domains like drug discovery, materials science, and systems biology, where complex relationships are central but challenging to explore.

Second, GraphLang addresses the limitations of current approaches that either convert graphs to text (losing structural information) or require specialized graph neural networks for each task (limiting generalizability). Our unified architecture preserves the structural properties of graphs while leveraging the semantic richness of language.

Third, this research extends foundation models beyond text and images to encompass relational data, advancing the frontier of artificial intelligence to better model the inherently relational nature of the world. As noted in the GLFrontiers workshop call, this aligns with the vision of graph learning as "a universal language that can be used to describe the complex world."

Finally, GraphLang opens new possibilities for interactive graph reasoning, where users can engage in dialogue about graph structures, request explanations for graph properties, and guide graph modifications through natural language—capabilities that have significant implications for scientific discovery and knowledge work across domains.

## 2. Methodology

### 2.1 Model Architecture

GraphLang employs a dual-encoder architecture with a graph encoder, a text encoder, and a cross-modal fusion module:

#### 2.1.1 Graph Encoder
We propose a hierarchical graph encoder that processes graphs at multiple levels of granularity:

1. **Node-level encoding**: Each node $v_i$ with features $\mathbf{x}_i$ is initially encoded using a projection layer:
   $$\mathbf{h}_i^{(0)} = \text{MLP}(\mathbf{x}_i)$$

2. **Structure-aware encoding**: We employ a series of graph Transformer layers that capture both local and global graph structure:
   $$\mathbf{H}^{(l+1)} = \text{GraphTransformer}(\mathbf{H}^{(l)}, \mathcal{G})$$

   The GraphTransformer layer computes:
   $$\mathbf{H}^{(l+1)} = \text{MLP}(\text{LN}(\mathbf{H}^{(l)} + \text{MultiHeadAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathcal{A})))$$

   Where $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ are query, key, and value projections of $\mathbf{H}^{(l)}$, and $\mathcal{A}$ is the adjacency structure of graph $\mathcal{G}$.

3. **Subgraph pooling**: For handling subgraphs, we implement a differentiable pooling mechanism:
   $$\mathbf{g}_s = \text{Pool}(\{\mathbf{h}_i^{(L)} | v_i \in \mathcal{S}\})$$
   
   Where $\mathcal{S}$ is a subgraph and $\text{Pool}$ combines attention-based pooling with structure-preserving operations.

4. **Graph-level representation**: A special [GRAPH] token aggregates information across the entire graph:
   $$\mathbf{g} = \mathbf{h}_{[\text{GRAPH}]}^{(L)}$$

#### 2.1.2 Text Encoder
The text encoder leverages a Transformer-based architecture compatible with existing LLMs:

1. **Token embedding**: Text tokens are embedded using standard techniques:
   $$\mathbf{e}_t = \text{Embedding}(t)$$

2. **Contextual encoding**: Token embeddings are processed through Transformer layers:
   $$\mathbf{T} = \text{TransformerEncoder}(\mathbf{E})$$

3. **Text representation**: A [CLS] token provides a global text representation:
   $$\mathbf{t} = \mathbf{T}_{[\text{CLS}]}$$

#### 2.1.3 Cross-Modal Fusion Module
The fusion module enables bidirectional information flow between graph and text representations:

1. **Cross-attention mechanism**:
   $$\mathbf{H}_{\text{graph} \rightarrow \text{text}} = \text{CrossAttention}(\mathbf{T}, \mathbf{H}^{(L)}, \mathbf{H}^{(L)})$$
   $$\mathbf{H}_{\text{text} \rightarrow \text{graph}} = \text{CrossAttention}(\mathbf{H}^{(L)}, \mathbf{T}, \mathbf{T})$$

2. **Fusion layer**:
   $$\mathbf{F} = \text{FusionLayer}(\mathbf{H}_{\text{graph} \rightarrow \text{text}}, \mathbf{H}_{\text{text} \rightarrow \text{graph}})$$

3. **Output heads** for different tasks:
   - Graph prediction: $\hat{\mathcal{G}} = \text{GraphDecoder}(\mathbf{F})$
   - Text generation: $\hat{\mathbf{T}} = \text{TextDecoder}(\mathbf{F})$
   - Classification: $\hat{y} = \text{ClassificationHead}(\mathbf{F})$

### 2.2 Pre-training Framework

We propose a multi-task pre-training framework using diverse graph-text pairs:

#### 2.2.1 Data Sources
1. **Knowledge Graphs**: Entity-relation triples paired with textual descriptions (e.g., Wikidata, DBpedia)
2. **Molecular Graphs**: Molecular structures paired with descriptions from PubChem, ChEMBL
3. **Scientific Literature**: Citation networks paired with paper abstracts
4. **Scene Graphs**: Visual scene graphs paired with image captions
5. **Protein Interaction Networks**: Biological networks with functional annotations

#### 2.2.2 Pre-training Tasks

1. **Masked Node/Edge Prediction**:
   - Randomly mask nodes and edges in the input graph
   - Train the model to reconstruct the masked elements:
     $$\mathcal{L}_{\text{mask}} = -\sum_{v_i \in \mathcal{M}_v} \log p(v_i | \mathcal{G}_{\backslash \mathcal{M}}) - \sum_{e_{ij} \in \mathcal{M}_e} \log p(e_{ij} | \mathcal{G}_{\backslash \mathcal{M}})$$
   Where $\mathcal{M}_v$ and $\mathcal{M}_e$ are the sets of masked nodes and edges.

2. **Graph-to-Text Generation**:
   - Given a graph, train the model to generate its textual description:
     $$\mathcal{L}_{\text{g2t}} = -\sum_{t=1}^{T} \log p(y_t | y_{<t}, \mathcal{G})$$

3. **Text-to-Graph Generation**:
   - Given a textual description, train the model to generate the corresponding graph:
     $$\mathcal{L}_{\text{t2g}} = -\log p(\mathcal{G} | \mathbf{T})$$

4. **Contrastive Alignment**:
   - Align graph and text representations in a shared embedding space:
     $$\mathcal{L}_{\text{align}} = -\log \frac{\exp(\text{sim}(\mathbf{g}, \mathbf{t})/\tau)}{\sum_{j=1}^{B} \exp(\text{sim}(\mathbf{g}, \mathbf{t}_j)/\tau)}$$
   Where $\mathbf{g}$ and $\mathbf{t}$ are the graph and text representations, and $\tau$ is a temperature parameter.

5. **Structure-preserving objective**:
   - Ensure that the model preserves important graph properties even after translation:
     $$\mathcal{L}_{\text{struct}} = \| \Phi(\mathcal{G}) - \Phi(\hat{\mathcal{G}}) \|^2$$
   Where $\Phi$ extracts structural properties (e.g., clustering coefficient, node centrality).

The total pre-training loss is a weighted combination:
$$\mathcal{L}_{\text{pretrain}} = \alpha \mathcal{L}_{\text{mask}} + \beta \mathcal{L}_{\text{g2t}} + \gamma \mathcal{L}_{\text{t2g}} + \delta \mathcal{L}_{\text{align}} + \epsilon \mathcal{L}_{\text{struct}}$$

### 2.3 Instruction Tuning

Following pre-training, we finetune GraphLang on instructional data to enhance its reasoning and interaction capabilities:

#### 2.3.1 Instruction Data Generation
We create a diverse set of graph reasoning instructions through:

1. **Template-based generation**: Using templates to create queries about graph properties, paths, and substructures.
2. **LLM-assisted synthesis**: Employing existing LLMs to generate complex graph reasoning scenarios.
3. **Human-in-the-loop curation**: Expert verification and refinement of synthetic instructions.
4. **Bootstrapping from model outputs**: Using model-generated explanations to create new instructions.

#### 2.3.2 Instruction Categories

1. **Graph Questioning**: "What is the shortest path between nodes A and B?"
2. **Subgraph Retrieval**: "Find all proteins that interact with kinase inhibitors."
3. **Graph Explanation**: "Explain why these two compounds have similar properties."
4. **Graph Editing**: "Add a hydroxyl group to this molecule at position 3."
5. **Counterfactual Reasoning**: "How would the network structure change if node X were removed?"

#### 2.3.3 Instruction Tuning Objective
We employ a standard instruction tuning objective:
$$\mathcal{L}_{\text{inst}} = -\sum_{t=1}^{T} \log p(y_t | y_{<t}, \mathbf{x}, \mathcal{G})$$

Where $\mathbf{x}$ is the instruction text and $y$ is the target response.

### 2.4 Experimental Design

We design a comprehensive evaluation framework to assess GraphLang's capabilities:

#### 2.4.1 Datasets
1. **OGB (Open Graph Benchmark)**: For standard graph learning tasks
2. **MoleculeNet**: For molecular property prediction and generation
3. **KGQA datasets**: WebQuestionsSP, MetaQA for knowledge graph QA
4. **Scientific graph datasets**: Citation networks, protein-protein interaction graphs
5. **Synthetic heterophilic graphs**: To evaluate performance on low-homophily scenarios

#### 2.4.2 Baselines
1. **GNN-based models**: GCN, GAT, GIN
2. **Graph-language models**: GraphText, GraphGPT, GraphLLM
3. **Text-only LLMs**: GPT-4, LLaMA, PaLM
4. **Specialized domain models**: MolGPT for molecules, BioLLM for biological data

#### 2.4.3 Evaluation Tasks and Metrics

1. **Graph Question Answering**:
   - Metrics: Accuracy, F1 score, BLEU score
   - Example: "Which proteins in this network are associated with Alzheimer's disease?"

2. **Subgraph Retrieval**:
   - Metrics: Precision@k, Recall@k, Mean Average Precision
   - Example: "Find all molecules that contain a benzene ring connected to a carboxyl group."

3. **Graph-to-Text Generation**:
   - Metrics: BLEU, ROUGE, BERTScore
   - Example: "Describe this molecular structure in technical terms."

4. **Text-to-Graph Generation**:
   - Metrics: Graph Edit Distance, Graph Matching Score
   - Example: "Generate a molecule that acts as a selective serotonin reuptake inhibitor."

5. **Graph Editing**:
   - Metrics: Edit Success Rate, Validity Rate, Property Preservation
   - Example: "Modify this molecule to increase its solubility while maintaining its binding affinity."

6. **Zero-shot Transfer**:
   - Evaluate on unseen graph types and domains
   - Assess performance degradation compared to in-domain tasks

7. **Heterophilic Graph Reasoning**:
   - Evaluate on graphs with varying homophily levels
   - Compare with specialized heterophilic GNNs

#### 2.4.4 Ablation Studies

1. **Architecture components**: Remove or modify the graph encoder, cross-attention mechanism
2. **Pre-training tasks**: Evaluate the importance of each pre-training objective
3. **Data sources**: Assess the impact of diverse data types on performance
4. **Instruction tuning**: Compare different instruction sets and tuning strategies

## 3. Expected Outcomes & Impact

### 3.1 Expected Technical Outcomes

1. **State-of-the-art performance** on graph-language tasks across domains, particularly in complex reasoning scenarios requiring both structural understanding and semantic knowledge.

2. **Zero-shot capabilities** for graph understanding tasks, where GraphLang can interpret and reason about previously unseen graph types without additional training.

3. **Enhanced performance on heterophilic graphs**, addressing a key limitation of current graph neural networks identified in recent literature (Luan et al., 2022, 2023).

4. **Efficient knowledge transfer** between graph and language modalities, enabling more effective knowledge extraction and reasoning across representations.

5. **Interactive graph exploration system** that allows users to progressively refine their understanding of complex graph structures through natural dialogue.

6. **Open-source implementation** of GraphLang, including pre-trained models, instruction tuning datasets, and evaluation benchmarks to facilitate further research.

### 3.2 Scientific and Societal Impact

**Scientific Discovery**: GraphLang has the potential to accelerate scientific discovery by making complex relational data more accessible. In drug discovery, researchers could use natural language to explore molecular graphs and identify promising candidates. In systems biology, scientists could query protein interaction networks to generate hypotheses about disease mechanisms.

**Knowledge Democratization**: By providing intuitive natural language interfaces to graph data, GraphLang will democratize access to complex relational information, enabling domain experts without programming skills to leverage insights from graph data.

**Enhanced Decision Support**: In domains like financial systems, supply chains, and cybersecurity, GraphLang could provide decision makers with natural language interfaces to complex network data, enabling more informed risk assessment and strategic planning.

**Education and Explainability**: GraphLang's ability to explain graph properties and relationships in natural language makes it a valuable educational tool for teaching complex network concepts and improving the explainability of graph-based AI systems.

**Cross-domain Knowledge Integration**: By unifying graph and language modalities, GraphLang facilitates knowledge transfer across domains, potentially uncovering novel connections between seemingly disparate fields through their relational structures.

### 3.3 Long-term Vision

The long-term vision for GraphLang is to serve as a foundation model for relational reasoning across domains. Just as large language models have transformed text-based interactions and diffusion models have revolutionized image generation, GraphLang aims to transform how we interact with, reason about, and generate complex relational structures.

By bridging the gap between the structured world of graphs and the intuitive interface of natural language, GraphLang represents a significant step toward more natural and effective human-AI collaboration for understanding complex systems. This aligns with the GLFrontiers workshop's goal of expanding "the impact of graph learning beyond the current boundaries" and establishing graph learning as "a generic tool for learning and understanding any type of (structured) data."
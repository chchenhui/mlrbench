# Knowledge-Guided Self-Explainable Models for Biomedical Discovery

## Introduction

### Background
The rapid adoption of machine learning (ML) in healthcare has demonstrated transformative potential in diagnosing diseases, predicting treatment outcomes, and discovering novel biomarkers. However, the "black-box" nature of state-of-the-art models like deep neural networks poses significant barriers to clinical trust and scientific utility. Post-hoc explanation methods, while widely used, often lack fidelity to the original model's decision-making process and may produce inconsistent or misleading interpretations. This challenge is particularly acute in biomedical discovery, where mechanistic understanding of predictions—such as identifying causal gene interactions or drug-target relationships—is critical for advancing precision medicine and therapeutic innovation.

Recent advances in eXplainable AI (XAI) have emphasized *a priori* (ante-hoc) interpretability, where models are inherently transparent by design. Techniques like self-explainable neural networks and interpretable graph neural networks (GNNs) have shown promise in domains requiring high-stakes decision-making. For instance, BrainNNExplainer (Cui et al., 2021) and IA-GCN (Kazi et al., 2021) demonstrate how attention mechanisms over brain connectivity graphs can reveal disease-specific biomarkers. Similarly, Ma & Zhang (2019) show that integrating Gene Ontology hierarchies into factor graph networks improves both predictive accuracy and biological plausibility in cancer genomics. These works highlight the potential of knowledge-guided architectures to bridge the gap between predictive performance and scientific insight.

### Research Objectives
This proposal aims to develop **knowledge-guided self-explainable models** that:
1. **Embed biomedical ontologies** (e.g., gene regulatory networks, drug-target interaction pathways) into GNNs and additive models to enforce biological plausibility.
2. **Enable end-to-end discovery** of mechanistic insights (e.g., subpopulation-specific oncogenic pathways) through attention-driven subgraph identification.
3. **Validate model-derived hypotheses** via collaboration with domain experts using wet-lab experiments or clinical trials.
4. **Establish a hybrid evaluation framework** that quantifies both predictive performance (e.g., survival analysis accuracy) and explainability (e.g., alignment with known biology).

### Significance
By directly addressing the challenge of balancing interpretability and performance, this work will:
- **Accelerate biomedical discovery** by transforming ML models into collaborative tools for hypothesis generation.
- **Enhance clinical trust** through transparent, ontology-aligned decision-making pathways.
- **Enable precision medicine** by identifying actionable biomarkers and synergistic drug targets tailored to patient subpopulations.

---

## Methodology

### Data Collection
We will curate multi-omics datasets spanning cancer genomics, pharmacogenomics, and clinical outcomes:
- **Cancer Genomics**: TCGA (The Cancer Genome Atlas) for gene expression, mutation, and survival data across 33 cancer types.
- **Drug Response**: GDSC (Genomics of Drug Sensitivity in Cancer) and PharmOmics for drug-target interactions and IC50 values.
- **Biomedical Ontologies**: 
  - **Gene Ontology (GO)** and **KEGG pathways** for functional gene interactions.
  - **STRING database** for protein-protein interaction networks.
  - **DrugBank** for pharmacokinetic pathways.

Data will be preprocessed to construct heterogeneous graphs where nodes represent genes, drugs, or clinical features, and edges encode known interactions or co-expression relationships.

### Model Design
Our architecture combines **ontology-driven GNNs** with **additive models** for hierarchical interpretability (Figure 1):

#### 1. Knowledge-Integrated Graph Construction
Let $ \mathcal{G} = (\mathcal{V}, \mathcal{E}) $ denote a graph where $ \mathcal{V} $ includes biomedical entities (genes, drugs) and $ \mathcal{E} $ encodes interactions from ontologies. Node features $ \mathbf{x}_v \in \mathbb{R}^d $ are initialized using omics data (e.g., gene expression levels). Ontology edges are weighted by confidence scores from databases like STRING.

#### 2. Self-Explainable GNN with Hierarchical Attention
We extend the Graph Attention Network (GAT) framework to incorporate ontology-guided inductive bias:
- **Ontology-Driven Message Passing**: For each node $ v \in \mathcal{V} $, aggregate features from neighbors $ u \in \mathcal{N}(v) $:
  $$
  \mathbf{h}_v^{(l+1)} = \sigma\left( \sum_{u \in \mathcal{N}(v)} \alpha_{vu} \mathbf{W}^{(l)} \mathbf{h}_u^{(l)} \right)
  $$
  where $ \alpha_{vu} $ is the attention coefficient computed as:
  $$
  \alpha_{vu} = \text{softmax}\left( \text{LeakyReLU}\left( \mathbf{a}^T [\mathbf{W}\mathbf{h}_v || \mathbf{W}\mathbf{h}_u] \right) \right)
  $$
  Here, $ || $ denotes concatenation, and $ \mathbf{a}, \mathbf{W} $ are learnable parameters.
- **Pathway Attention Module**: Introduce a hierarchical attention layer to prioritize subgraphs corresponding to known pathways:
  $$
  \beta_k = \text{sigmoid}\left( \mathbf{w}_k^T \frac{1}{|\mathcal{V}_k|} \sum_{v \in \mathcal{V}_k} \mathbf{h}_v \right)
  $$
  where $ \beta_k \in [0,1] $ weights the contribution of pathway $ k $, and $ \mathcal{V}_k \subseteq \mathcal{V} $ are nodes in the pathway.

#### 3. Additive Model for Feature Attribution
To decompose predictions into interpretable components, we use a generalized additive model (GAM) over pathway-level embeddings:
$$
y = \sigma\left( \sum_{k=1}^K \phi_k(\mathbf{z}_k) \right)
$$
where $ \mathbf{z}_k \in \mathbb{R}^{d'} $ is the pathway embedding from GNN outputs, $ \phi_k $ is a learnable shape function (implemented via a small neural network), and $ \sigma $ is the sigmoid function for binary outcomes (e.g., drug response).

### Training Protocol
- **Loss Function**: Multi-task objective combining prediction loss $ \mathcal{L}_{\text{pred}} $ and interpretability regularization $ \mathcal{L}_{\text{int}} $:
  $$
  \mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda \mathcal{L}_{\text{int}}
  $$
  - $ \mathcal{L}_{\text{pred}} $: Cox proportional hazards loss for survival analysis or cross-entropy for classification.
  - $ \mathcal{L}_{\text{int}} $: Encourages sparse attention weights via $ \ell_1 $-regularization on $ \beta_k $ and alignment of top-ranked pathways with known disease-associated pathways using Jaccard index maximization.
- **Optimization**: Adam optimizer with learning rate 0.001, batch size 32, and 5-fold cross-validation.

### Experimental Design
#### Tasks
1. **Survival Prediction**: Predict 5-year survival in TCGA cohorts.
2. **Drug Response Prediction**: Classify IC50 thresholds in GDSC.
3. **Biomarker Discovery**: Identify genes whose pathway attention scores correlate with treatment resistance.

#### Baselines
- Post-hoc methods: GNNExplainer, Integrated Gradients.
- Self-explainable models: GAMI-Net (additive models), FGNN (factor graph neural networks).

#### Evaluation Metrics
| **Category**          | **Metrics**                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| Predictive Performance| Concordance Index (C-index), AUC-ROC, F1-score                              |
| Explainability        | Jaccard index with known pathways, human expert validation score (1-5 scale)|
| Novelty               | Number of newly identified biomarkers validated in literature post-2023      |

#### Ablation Studies
- Effect of ontology integration: Compare performance with/without STRING edges.
- Impact of attention hierarchy: Evaluate pathway vs. node-level explanations.

---

## Expected Outcomes & Impact

### Technical Outcomes
1. **State-of-the-Art Performance**: Achieve C-index ≥ 0.75 on TCGA survival tasks, outperforming post-hoc methods by ≥5% in AUC.
2. **Interpretable Architectures**: Release open-source implementations of ontology-driven GNNs and pathway attention modules.
3. **Biomarker Discovery**: Identify ≥3 novel gene-drug interactions validated in independent cohorts (e.g., METABRIC).

### Scientific Impact
- **Mechanistic Insights**: Uncover subpopulation-specific resistance mechanisms in breast cancer (e.g., TP53-PIK3CA co-mutation networks).
- **Drug Repurposing**: Propose novel indications for FDA-approved drugs based on shared pathway activations.
- **Clinical Translation**: Collaborate with oncologists to design basket trials targeting model-identified biomarkers.

### Societal Impact
- **Trust in AI**: Demonstrate how self-explainable models align with the "right to explanation" in healthcare AI regulations.
- **Open Science**: Curate and share knowledge graphs integrating TCGA, DrugBank, and KEGG for community use.

---

## Conclusion
This proposal bridges the gap between ML predictivity and scientific discovery in healthcare by embedding biomedical knowledge into self-explainable models. Through rigorous validation and collaboration with domain experts, our work will advance precision oncology while setting a precedent for trustworthy AI in high-stakes scientific applications.
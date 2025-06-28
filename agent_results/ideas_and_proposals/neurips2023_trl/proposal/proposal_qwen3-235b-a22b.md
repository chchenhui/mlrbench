# Structure-Aware Dual-Stream Pretraining for Robust Tabular Language Models  

## Introduction  

### Background  
Structured tabular data remains a cornerstone of enterprise workflows, scientific analysis, and data-driven decision-making systems. Despite its ubiquity in applications like relational databases and spreadsheet software, tabular data has been historically underrepresented in foundation model development compared to unstructured modalities like text and images. Recent advancements in pretraining strategies for tabular data—such as TURL’s column type annotation, TableFormer’s structural invariance, and TabTreeFormer’s hybrid tree-transformer—have demonstrated that incorporating table-specific inductive biases improves NLP tasks involving tables. However, these approaches often prioritize local cell relationships or abstracted structural properties over explicit modeling of hierarchical schema metadata (e.g., nested headers, primary keys, inter-column dependencies).  

Large language models (LLMs) trained on unstructured corpora fail to accurately interpret nested or relational table topologies, leading to suboptimal performance in critical tasks like text-to-SQL generation and semantic table QA. For example, even state-of-the-art models like TaBERT and TableFormer exhibit limitations in handling sparse relations or inconsistent column alignment, particularly in enterprise settings where table schemas evolve dynamically. This gap highlights the need for a novel pretraining paradigm that explicitly decouples and jointly learns from *content* (cell values, textual descriptions) and *structure* (schema graph topology, data types).  

### Research Objectives  
This research proposes **Structure-Aware Dual-Stream Pretraining (SAD)**, a framework that:  
1. Separates content and structural metadata into distinct encoding streams.  
2. Integrates hierarchical schema graphs into transformer architectures.  
3. Introduces cross-stream alignment pretraining tasks.  
By explicitly modeling table structure and content in parallel pathways, SAD aims to:  
- Improve robustness to heterogeneous schemas (e.g., nested/multi-index headers).  
- Enable precise reasoning over sparse relational dependencies (e.g., primary-foreign key associations).  
- Achieve state-of-the-art results on downstream tabular NLP tasks.  

### Significance  
The proposed framework addresses three central limitations of current tabular LLMs:  
1. **Schema-Aware Generalization**: Explicit modeling of structural metadata (e.g., data types, foreign keys) enables adaptation to domain-specific table varieties without task-specific fine-tuning.  
2. **Improved Cross-Modal Alignment**: Joint training of content and structure streams enhances the understanding of complex queries requiring both cell-value lookup and relational reasoning (e.g., “Compare revenue across regions in 2023”—which demands numeric aggregation *and* column-to-column mapping).  
3. **Production Readiness**: Explicit schema graph representation facilitates real-world deployment by enabling model debugging (e.g., tracing prediction errors to schema misalignments) and dynamic schema updates.  

## Methodology  

### Data Collection and Preprocessing  
**Corpus Curation:**  
The pretraining dataset will aggregate 18 million structured tables from:  
- **Web Sources**: Wikipedia tables, WikiSQL, and HTML exports from open-data portals (e.g., data.world, Kaggle datasets).  
- **Synthetic Generators**: Modified TabTreeFormer for generating controlled tables with nested headers and relational dependencies.  
- **Enterprise Logs**: Anonymized transactional/balance sheet data from public corpora (CC-24-EnterNet).  

**Schema Metadata Extraction:**  
For each table, extract:  
1. **Content Data**: Tokenized cell values (text/numeric) with positional embeddings for row/column indices.  
2. **Structural Metadata**:  
   - Header hierarchy (flat: “Sales/Q4”; nested: “Region > (North, South)” in multi-index tables)  
   - Data types (categorical, continuous, datetime)  
   - Inter-column relations (primary keys, foreign keys).  

### Model Architecture  

#### Dual-Stream Transformer  
The framework consists of two parallel encoders:  

1. **Content Stream**  
   - Tokenizes cell values using a hybrid byte-level and domain-specific tokenizer (e.g., numeric tokenizer for financial data).  
   - Applies positional embeddings for row/column indices:  
     $$E_{\text{content}} = W_{\text{token}} \cdot x + P_{\text{row}}^{pos} + P_{\text{col}}^{pos}$$  
     where $x$ is the token embedding and $P_{\text{row}}, P_{\text{col}}$ are learnable positional embeddings.  

2. **Structure Stream**  
   - Encodes the schema graph $G = (V, E)$, where $V$ contains column/row nodes and $E$ represents dependencies (e.g., “Region” → “Sales”).  
   - Uses Graph Attention Networks (GAT) for node embeddings:  
     $$s_i^{(l)} = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W^{(l)} s_j^{(l-1)} \right),$$  
     where $\alpha_{ij}$ measures attention between node $i$ and neighbor $j$, and $s_i^{(l)}$ is the node embedding at layer $l$.  

**Cross-Stream Attention:**  
To align content and structure, we introduce cross-attention layers where queries ($Q$) come from the content stream and keys ($K$)/values ($V$) from the structure stream:  
$$ \text{Att}_{c \to s} = \text{softmax}\left(\frac{Q^{\text{content}}(K^{\text{structure}})^\top}{\sqrt{d_k}}\right)V^{\text{structure}}. $$  
This mechanism enables the model to attend to structural relations when decoding cell values and vice versa.  

### Pretraining Tasks  

1. **Masked Cell Recovery (MCR)**  
   - Randomly mask 15% of cell tokens and reconstruct them using the content stream:  
     $$\mathcal{L}_{\text{MCR}} = -\sum_{i \in \text{mask}} \log p_{\text{content}}(x_i \mid X_{\text{others}}, G).$$  

2. **Schema Relation Prediction (SRP)**  
   - Predict edge relations (e.g., “foreign key → column”):  
     $$\mathcal{L}_{\text{SRP}} = -\sum_{(i, j, r) \in \mathcal{E}} \log p_{\text{structure}}(r \mid s_i, s_j).$$  

3. **Cross-Stream Alignment (CSA)**  
   - Align natural language queries to schema subgraphs (e.g., map “average salary” → “Employee” table’s “Salary” column):  
     $$\mathcal{L}_{\text{CSA}} = -\log \frac{\exp(\text{sim}(Q_{\text{query}}, G_{\text{sub}})/\tau)}{\sum_{G_{\text{neg}}} \exp(\text{sim}(Q_{\text{query}}, G_{\text{neg}})/\tau)},$$  
     where $\text{sim}$ is cosine similarity, and $G_{\text{neg}}$ is a negative schema graph.  

Total pretraining loss:  
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{MCR}} + \lambda_2 \mathcal{L}_{\text{SRP}} + \lambda_3 \mathcal{L}_{\text{CSA}},$$  
with coefficients $\lambda_1, \lambda_2, \lambda_3$ tuned via grid search.  

### Experimental Design  

**Baselines:** Evaluate against:  
- Single-stream models: TURL, TableFormer, TabNet  
- Multi-stream models: TaBERT, TAPAS  

**Downstream Tasks:**  
1. **Text-to-SQL (Spider Benchmark)**: Measure exact match accuracy (EM) and execution accuracy.  
2. **Table QA (WikiTableQuestions)**: Use accuracy and F1 score.  
3. **Schema Matching (CC-24-EnterNet)**: Compute precision/recall for inter-column relation recovery.  

**Ablation Studies:**  
- Impact of dual-stream vs. single-stream design.  
- Effectiveness of each pretraining task (e.g., ablate SRP or CSA).  
- Robustness to corrupted tables (e.g., added sparse cells or invalid headers).  

**Evaluation Metrics:**  
- **Content Accuracy**: Cell value recovery F1 for numeric/categorical data.  
- **Structural Precision**: Edge-type prediction for schema graphs.  
- **Task Performance**: Benchmarks’ official metrics (e.g., Spider’s execution accuracy).  

### Hyperparameter Settings  
- **Optimizer**: AdamW with cosine learning rate decay.  
- **Batch Size**: 256 for pretraining; 16 for fine-tuning.  
- **Model Size**: 125M parameters (comparable to BERT-base).  

## Expected Outcomes & Impact  

### Expected Outcomes  
1. **Architectural Innovation**: Demonstration of dual-stream transformers’ superiority over single-stream approaches, with >15% improvement in structural relation prediction accuracy over TableFormer on the TURL dataset.  
2. **Benchmark Performance**: Achieve ≥68% execution accuracy on Spider (vs. current SOTA of 63%) by better resolving column-pointer ambiguities via schema-aware attention.  
3. **Robustness Validation**: Show ≤5% performance degradation under synthetic schema corruption (e.g., 20% missing headers), proving structural stream resilience.  
4. **Open-Source Release**: Publish model weights, pretraining code, and synthetic dataset generation toolkit.  

### Broader Impact  
1. **Industry Applications**: Enhanced text-to-SQL systems will reduce enterprise reliance on manual query writing, lowering barriers to data access for non-technical users.  
2. **Research Advancements**: Introduce schema graphs as a standard structural bias for tabular LLMs, influencing multi-modal learning where tables integrate with code/images.  
3. **Community Benefits**: Foster NLP-ML cross-pollination through open benchmarks and collaborative tooling, aligning with the TRL workshop’s goal of unifying disparate communities (NLP, databases, IR).  
4. **Ethical Considerations**: Address data privacy in tabular generation through metadata-aware synthetic data synthesis, ensuring schema-preserving anonymization.  

### Limitations and Future Work  
- **Computational Overhead**: Dual-stream training may increase latency; future work includes distilling the schema stream into compact embeddings.  
- **Schema Parsing Errors**: Manual schema extraction (e.g., foreign key detection) could degrade pretraining quality; semi-supervised learning over weakly labeled tables is a mitigation.  
- **Multimodal Extensions**: Adapt SAD’s structural stream for joint table-image/video processing by replacing schema graphs with heterogeneous graphs.  

This proposal advances tabular representation learning by treating structure not as an afterthought but as a first-class component of pretraining—a critical step toward making LLMs effective in real-world structured data ecosystems.
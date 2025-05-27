Title  
Structure-Aware Dual-Stream Pretraining for Robust Tabular Language Models  

Introduction  
Background  
Tabular data—spreadsheets, relational tables, and semi‐structured logs—underpins critical decision‐making in finance, healthcare, scientific research, and web services. Despite its ubiquity, structured data has historically received less attention from the representation‐learning community than text and images. Recent advances (TaBERT, TAPAS, TableFormer, UniTabE, XTab, TabTreeFormer) demonstrate that modeling both table content and its structural metadata can substantially improve downstream tasks such as text‐to‐SQL, table‐question answering (TableQA), column‐type annotation, and table‐based classification/regression. However, current approaches often fuse content and structure implicitly or focus largely on content encoding, limiting performance on tables with complex schemas, nested or hierarchical headers, and sparse inter‐column relations.  

Research Objectives  
We propose “Structure‐Aware Dual‐Stream Pretraining” (SADS), a novel transformer framework that:  
1. Separately encodes cell content and schema structure via two interacting transformer streams.  
2. Introduces pretraining objectives explicitly targeting both masked‐value recovery (content), schema‐relation prediction (structure), and cross‐stream alignment.  
3. Demonstrates state‐of‐the‐art performance on diverse benchmarks (Spider, WikiTableQuestions, TabFact).  

Significance  
By modeling table topology explicitly, SADS will:  
• Improve generalization across heterogeneous schemas.  
• Enhance reasoning over nested headers and foreign‐key relationships.  
• Provide a modular foundation for multimodal extensions (text+table, code+table).  
This work will advance the TRL workshop goals by motivating structured data as a primary modality and showcasing how explicit structure modeling elevates end‐to‐end NLP and data‐analysis pipelines.  

Methodology  
Overview  
Given a table $T$ with $m$ rows, $n$ columns, cell values $\{v_{i,j}\}$, header hierarchy $H$, and schema graph $G=(V, E)$ capturing column‐level relations (e.g., primary/foreign keys), we learn two streams:  
1. Content Stream $f_c$: encodes tokenized cell values with positional and column‐header embeddings.  
2. Structure Stream $f_s$: encodes a graph $G$ of schema nodes using a graph‐transformer.  
“Cross‐Stream Interaction” layers allow the two streams to exchange contextual signals. Pretraining jointly minimizes three losses: masked cell recovery $L_{\mathrm{mask}}$, schema‐relation classification $L_{\mathrm{schema}}$, and cross‐stream alignment $L_{\mathrm{align}}$.  

Data Collection and Preprocessing  
We aggregate large corpora of diverse tables:  
• WikiTables (∼2.5M tables)  
• Spider’s database schemas (∼200)  
• VizNet spreadsheet corpus (∼1M tables)  
• TabFact (∼10K tables + proofs)  
Preprocessing steps:  
1. Normalize cell values: tokenize numbers (e.g., “3,400”→“3400”), dates, and categorical labels.  
2. Flatten nested headers into hierarchical paths (e.g., “Region → Country → City”).  
3. Construct schema graph $G$ where each column node $c_i$ connects to parent headers and to other columns via foreign‐key edges.  
4. Encode data types ($\mathrm{int}$, $\mathrm{float}$, $\mathrm{string}$, $\mathrm{date}$) as one‐hot vectors.  

Model Architecture  
Content Stream  
We represent each cell value $v_{i,j}$ as a sequence of subword tokens $[w_1, w_2, …, w_k]$ (via BPE). Token $w_t$ is embedded as  
$$e_t = W_{\mathrm{token}}[w_t] + P_{\mathrm{cell}}(i,j) + H_{\mathrm{col}}(j),$$  
where $P_{\mathrm{cell}}$ is a learnable 2D positional embedding for row $i$, column $j$, and $H_{\mathrm{col}}(j)$ is a header‐embedding of column $j$ derived from header‐path tokens. The sequence for all values in $T$ is concatenated with special separators. The content stream is a Transformer with $L_c$ layers, hidden size $d_c$, and $h_c$ attention heads.  

Structure Stream  
We treat the schema graph $G=(V,E)$ with $|V|=n$ column/header nodes. Each node $v_i$ has initial features  
$$x_i = W_{\mathrm{type}}\cdot \mathrm{onehot}(\text{data\_type}_i) + W_{\mathrm{emb}}\cdot H_{\mathrm{col}}(i).$$  
We apply a graph‐transformer (or graph‐attention‐network) with $L_s$ layers. At layer $\ell$, node features update via  
$$x_i^{(\ell+1)} = \mathrm{LayerNorm}\Bigl(x_i^{(\ell)} + \sum_{j\in \mathcal{N}(i)} \alpha_{ij} W_V x_j^{(\ell)}\Bigr)$$  
where attention coefficients  
$$\alpha_{ij} = \frac{\exp\bigl(\mathrm{LeakyReLU}\,(W_Q x_i^{(\ell)} + W_K x_j^{(\ell)})\bigr)}{\sum_{k\in \mathcal{N}(i)} \exp\bigl(\mathrm{LeakyReLU}\,(W_Q x_i^{(\ell)} + W_K x_k^{(\ell)})\bigr)}\,. $$  

Cross‐Stream Interaction  
Every $p$ layers, we align content and structure via cross‐attention: let $C\in \mathbb{R}^{N_c\times d_c}$ be content‐stream token outputs, $S\in \mathbb{R}^{n\times d_s}$ structure‐stream node embeddings. We project $C\to Q_C,K_C,V_C$ and $S\to K_S,V_S$. Then update content tokens:  
$$C' = \mathrm{Softmax}\!\bigl(Q_CK_S^\top/\sqrt{d_k}\bigr)V_S + C$$  
and symmetrically update $S$ via queries from structure to content. This allows semantic signals to flow across modalities.  

Pretraining Objectives  
1. Masked Cell Recovery ($L_{\mathrm{mask}}$)  
Randomly mask a fraction $\rho$ of subword tokens in cell sequences. Predict masked tokens via cross‐entropy:  
$$L_{\mathrm{mask}} = -\sum_{t\in \mathcal{M}} \log p(w_t \mid C_{\setminus t}).$$  

2. Schema Relation Prediction ($L_{\mathrm{schema}}$)  
For each pair $(i,j)$ of schema‐graph nodes, predict the relation label $r_{ij}\in\{\mathrm{PK\mbox{-}FK},\mathrm{parent\mbox{-}child},\mathrm{none}\}$ via a softmax head on their final embeddings $x_i^{(L_s)}, x_j^{(L_s)}$. Use cross‐entropy loss:  
$$L_{\mathrm{schema}} = -\sum_{i<j} \log p(r_{ij}\mid x_i, x_j).$$  

3. Cross‐Stream Alignment ($L_{\mathrm{align}}$)  
We anchor natural language questions or generated SQL snippets to schema subgraphs: given a question $q$ pairing with a table, we embed $q$ via an LLM encoder $E(q)$ and encourage high dot‐product similarity with the joint content‐structure contextual embedding of the relevant columns. We use a contrastive InfoNCE loss:  
$$L_{\mathrm{align}} = -\sum_q \log\frac{\exp\bigl(E(q)^\top f_{c,s}(T)\bigr)}{\sum_{q'} \exp\bigl(E(q')^\top f_{c,s}(T)\bigr)}\,. $$  

Overall Loss  
$$L = \lambda_1 L_{\mathrm{mask}} + \lambda_2 L_{\mathrm{schema}} + \lambda_3 L_{\mathrm{align}},$$  
with weights $(\lambda_1,\lambda_2,\lambda_3)$ tuned on a held‐out validation set.  

Training Details  
• Optimizer: AdamW, learning rate $2\mathrm{e}{-4}$, weight decay $0.01$, linear warmup → decay.  
• Batch: 512 tables per GPU, gradient accumulation for effective batch 2048.  
• Pretrain for 100K steps on 128 A100 GPUs (∼1 week).  
• Hyperparameters: $L_c=12$, $L_s=6$, $d_c=d_s=768$, $h_c=12$, attention dropout 0.1.  

Experimental Design and Evaluation  
Baselines  
• TaBERT, TAPAS, TableFormer, UniTabE, XTab, TabTreeFormer.  

Datasets & Tasks  
1. Text-to-SQL (Spider): Evaluate exact match (EM) and execution accuracy.  
2. TableQA (WikiTableQuestions): Measure F1 score on question answering.  
3. TabFact: Logical consistency and entailment accuracy.  
4. Column‐Type Annotation: Classification accuracy.  
5. Cross‐Domain Transfer: Pretrain on one domain (e.g. sports tables) and fine‐tune on another (e.g. financial tables).  

Ablation Studies  
• Remove structure stream (content only).  
• Remove cross‐stream interaction.  
• Vary $\lambda$ weights.  
• Replace graph‐transformer with plain MLP.  

Evaluation Metrics  
• EM, F1 for QA and SQL tasks.  
• Macro/micro‐precision‐recall on column‐type tasks.  
• Pretraining convergence speed and sample efficiency.  
• Model size vs. performance trade‐off.  
• Robustness to schema perturbations (randomly shuffle columns, add dummy headers).  

Expected Outcomes & Impact  
We anticipate SADS will:  
1. Achieve ≥5% relative improvement in EM on Spider over TableFormer, and ≥3% relative F1 gain on WikiTableQuestions.  
2. Demonstrate superior cross‐domain generalization: ↗ 10% accuracy when transferring to unseen schemas.  
3. Show ablations that explicit structure modeling and cross‐stream alignment are critical for performance gains.  

Impact  
Immediate beneficiaries include NLP systems performing semantic parsing, interactive data‐analysis assistants, BI tools, and anyone requiring precise table‐understanding. By releasing pretrained SADS checkpoints and code, we will democratize access to robust table models. Long‐term, our dual‐stream paradigm can extend to multimodal scenarios: linking tables with charts, code, or knowledge graphs.  

Broader Contributions  
• A reproducible dual‐stream transformer framework for tabular pretraining.  
• A unified set of pretraining tasks that balance content, structure, and alignment objectives.  
• Rigorous benchmarks and analyses on robustness to schema complexity and heterogeneity.  
• Insights into how explicit structural biases improve representation learning in structured modalities.  

This research will propel table representation learning from ad‐hoc fusion strategies toward principled, structure‐aware deep learning, unlocking richer data‐driven insights in NLP and beyond.
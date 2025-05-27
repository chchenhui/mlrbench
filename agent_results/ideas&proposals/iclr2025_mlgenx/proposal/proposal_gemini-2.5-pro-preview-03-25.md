Okay, here is a research proposal based on the provided task description, research idea, and literature review.

---

**1. Title:** Genomic Circuit Foundation Model (GCFM): Integrating Multi-Scale Attention and Graph Neural Networks for Regulatory Network Inference and Perturbation Prediction

**2. Introduction**

**2.1 Background**
The intricate network of gene regulatory interactions, often referred to as genomic circuits, dictates cellular identity, function, and response to stimuli. Understanding these circuits is paramount for deciphering disease mechanisms and identifying effective therapeutic targets (Lander, 2011; Levine & Tjian, 2003). However, mapping these complex regulatory landscapes remains a formidable challenge. Gene regulation involves interactions spanning vast genomic distances, context-specific elements (e.g., enhancers active only in certain cell types), and dynamic feedback loops. This complexity results in a significant bottleneck in drug discovery, where a lack of mechanistic understanding contributes to high failure rates in clinical trials (Wong et al., 2019).

Recent advancements in high-throughput genomics, including sequencing (RNA-seq, ChIP-seq, ATAC-seq) and perturbation technologies (CRISPR screens, chemical screens), have generated unprecedented volumes of data (Consortium, 2012; Consortium, 2015; GTEx Consortium, 2017). These datasets offer a rich substrate for machine learning (ML) approaches to model gene regulation. While progress has been made, existing methods often struggle with several key limitations identified in recent literature (Zhang et al., 2023; Ke et al., 2023; Otal et al., 2024): high noise levels in biological data, difficulty capturing long-range dependencies and complex interactions, challenges in integrating multimodal data effectively, scalability issues with large datasets, and the need for interpretable models that yield biological insights.

Foundation models, pre-trained on broad data to capture fundamental patterns, have shown immense promise in various domains, including natural language processing and computerision (Bommasani et al., 2021). Their application to genomics is emerging as a powerful paradigm for learning the 'grammar' of biological sequences and regulatory logic (Ji et al., 2021; Avsec et al., 2021). This workshop, focusing on ML for genomics explorations, target identification, and new drug modalities, provides the ideal venue to explore the potential of foundation models tailored for deciphering genomic regulatory circuits.

**2.2 Problem Statement**
The core problem is the inadequate modeling of complex, multi-scale, and context-specific gene regulatory networks from genomic data. Current computational approaches often analyze sequence motifs locally, fail to integrate diverse data modalities coherently, or struggle to infer the causal structure and dynamic behavior of regulatory circuits, especially under perturbation. Specifically, we lack models that can simultaneously:
1.  Capture both short-range sequence motifs (e.g., transcription factor binding sites - TFBS) and long-range interactions (e.g., enhancer-promoter loops) effectively.
2.  Explicitly model the network structure of gene-gene and element-gene interactions.
3.  Integrate sequence information with functional genomics data (epigenomics, transcriptomics) across diverse cellular contexts.
4.  Predict the downstream consequences of genetic or chemical perturbations on the regulatory network and cellular state.
5.  Achieve robustness against noise and generalize across different cell types and conditions.

**2.3 Proposed Solution: Genomic Circuit Foundation Model (GCFM)**
We propose the development of a novel **Genomic Circuit Foundation Model (GCFM)**, designed specifically to learn the principles of gene regulation from large-scale, diverse genomic datasets. GCFM leverages a hybrid architecture integrating multi-scale attention mechanisms, inspired by successes in sequence modeling (Vaswani et al., 2017; Avsec et al., 2021), with Graph Neural Networks (GNNs), which excel at capturing relational information (Kipf & Welling, 2017; Veličković et al., 2018).

Our model incorporates three key innovations:
1.  **Multi-Scale Sequence Representation:** Utilizes attention mechanisms operating over different genomic ranges to learn both local motifs and distal regulatory element interactions directly from DNA sequences, complemented by epigenetic features.
2.  **Inductive Regulatory Graph Learning:** Employs a GNN module to explicitly model and infer the structure of regulatory interactions (e.g., TF-gene, enhancer-gene), treating genes and regulatory elements as nodes and their interactions as edges whose properties are learned from data.
3.  **Integrated Perturbation Prediction:** Incorporates a module designed to predict changes in gene expression profiles resulting from simulated genetic (e.g., gene knockout/knockdown) or chemical perturbations, enabling *in silico* hypothesis testing and target prioritization.

GCFM will be pre-trained on a comprehensive collection of public genomic datasets (e.g., ENCODE, Roadmap Epigenomics, GTEx) covering diverse cell types, tissues, and experimental assays. This pre-training phase aims to instill a fundamental understanding of regulatory grammar. The pre-trained model can then be fine-tuned for specific downstream tasks, such as predicting cell-type-specific enhancer-gene links, inferring detailed GRNs, or forecasting drug responses.

**2.4 Research Objectives**
The primary objectives of this research are:
1.  **Develop the GCFM architecture:** Design and implement the hybrid multi-scale attention and GNN architecture capable of processing sequence, epigenomic, and transcriptomic data.
2.  **Pre-train the GCFM:** Train the model on large-scale public genomic datasets to learn foundational representations of gene regulation across diverse biological contexts.
3.  **Validate GCFM performance:** Evaluate the model's ability to perform key regulatory genomics tasks, including prediction of regulatory elements (enhancers, TFBS), inference of regulatory interactions (GRNs), and prediction of gene expression. Benchmark against state-of-the-art methods.
4.  **Demonstrate perturbation prediction capabilities:** Assess the model's accuracy in predicting gene expression changes following simulated genetic or chemical perturbations, comparing against experimental perturbation data where available.
5.  **Evaluate model interpretability and biological relevance:** Employ interpretability techniques to understand the regulatory features learned by GCFM and validate key predictions against known biological pathways and interactions.

**2.5 Significance**
This research holds significant potential for advancing both machine learning and genomics. By developing a powerful foundation model for genomic circuits, we aim to:
*   Provide a more accurate and comprehensive understanding of gene regulatory mechanisms, including complex long-range and context-specific interactions.
*   Accelerate drug discovery by enabling more effective *in silico* target identification and prioritization through accurate perturbation prediction.
*   Offer a versatile tool for the research community, adaptable to various specific biological questions and datasets.
*   Bridge the gap between ML and genomics, directly addressing the core themes of the workshop by fostering interdisciplinary innovation in understanding disease and developing novel therapeutics.
*   Address key challenges highlighted in recent literature, particularly regarding modeling complexity, integrating multimodality, and improving robustness.

**3. Methodology**

**3.1 Data Collection and Preprocessing**
We will leverage large, publicly available datasets:
1.  **ENCODE Project & Roadmap Epigenomics Project:** Provide extensive data on chromatin accessibility (DNase-seq, ATAC-seq), histone modifications (ChIP-seq), transcription factor binding (ChIP-seq), and gene expression (RNA-seq) across hundreds of human cell types and tissues.
2.  **GTEx Project:** Offers gene expression data (RNA-seq) across a wide range of human tissues, valuable for learning tissue-specific regulation.
3.  **Reference Genome:** Human genome sequence (e.g., hg38) as the backbone for sequence-based analysis.
4.  **Perturbation Datasets (for validation/fine-tuning):** Datasets like Connectivity Map (CMap L1000) for chemical perturbations (Subramanian et al., 2017), DepMap for genetic perturbations (CRISPR screens) (Tsherniak et al., 2017), and potentially specific CRISPRi/a screen datasets relevant to enhancer function (e.g., Gasperini et al., 2019).

**Preprocessing Steps:**
*   **Sequence Data:** DNA sequences will be extracted corresponding to gene promoters, gene bodies, and potential regulatory regions (identified via chromatin accessibility peaks or predicted enhancer databases). Sequences will be one-hot encoded.
*   **Epigenomic Data:** ChIP-seq and DNase/ATAC-seq signals will be processed (e.g., read counts, fold-change over control) and aligned to the reference genome. Signals will be aggregated over genomic bins or associated with specific genomic elements (genes, enhancers). Normalization techniques (e.g., quantile normalization, CPM) will be applied.
*   **Transcriptomic Data:** RNA-seq data will be processed to obtain gene expression levels (e.g., TPM, FPKM, or counts). Data will be log-transformed (e.g., $log_2(x+1)$) and normalized across samples.
*   **Data Integration:** Data modalities will be aligned based on genomic coordinates. For a given gene or region, we will assemble input vectors containing sequence information, relevant epigenetic marks within a defined window (e.g., +/- 500kb), and baseline expression levels where applicable.

**3.2 Model Architecture: GCFM**
The GCFM architecture comprises three main modules: (A) Multi-Scale Sequence & Epigenome Encoder, (B) Regulatory Graph Interaction Module, and (C) Perturbation Prediction Head.

**A. Multi-Scale Sequence & Epigenome Encoder:**
*   **Input:** One-hot encoded DNA sequence ($S$), associated epigenetic features ($E$) across a potentially long genomic window (e.g., 1Mb).
*   **Architecture:** We will adapt concepts from efficient Transformer architectures (e.g., Performer, Longformer) or use convolutional layers with increasing dilation rates followed by attention layers. This module aims to capture features at multiple scales:
    *   *Local Features:* Small convolutional kernels or attention windows capture motifs like TFBS.
    *   *Regional Features:* Larger receptive fields or longer attention spans capture enhancer structures and local chromatin context.
    *   *Long-Range Features:* Global attention mechanisms or dilated convolutions spanning hundreds of kilobases capture distal enhancer-promoter interactions.
*   **Mathematical Sketch:** Let $X = [S; E]$ be the concatenated input sequence and epigenetic features. The encoder $f_{enc}$ produces context-aware embeddings $H = f_{enc}(X)$.
    $$ H = Attention_{MultiScale}(Conv_{Dilated}(X)) $$
    Where $Conv_{Dilated}$ represents layers of dilated convolutions and $Attention_{MultiScale}$ represents attention mechanisms designed to handle long sequences efficiently. The output $H$ contains rich embeddings for different genomic positions.

**B. Regulatory Graph Interaction Module:**
*   **Input:** Embeddings $H$ from the encoder, potentially focused on embeddings corresponding to gene TSSs ($H_{gene}$) and candidate regulatory elements ($H_{cre}$).
*   **Architecture:** A Graph Neural Network (GNN), likely a Graph Attention Network (GAT) variant (Veličković et al., 2018; Zhang et al., 2023), will model interactions.
    *   *Nodes:* Represent genes and candidate regulatory elements (CREs). Initial node features derived from $H_{gene}$ and $H_{cre}$.
    *   *Edges:* Represent potential regulatory interactions. Initial edges could be based on genomic proximity (e.g., CREs within X kb of a gene TSS) or predicted TF binding. The GNN learns to weight these edges or even predict new edges.
*   **Graph Induction:** We propose an *inductive* approach where the graph structure is learned or refined. This could involve:
    1.  Learning edge weights representing interaction strength via attention within the GAT.
    2.  An auxiliary edge prediction task during pre-training, predicting known enhancer-gene links or co-expression patterns.
*   **Mathematical Sketch:** Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be the graph with nodes $v \in \mathcal{V}$ (genes, CREs) and edges $e \in \mathcal{E}$. Node features are initialized from $H$. The GNN updates node representations $h_v^{(l)}$ iteratively:
    $$ h_i^{(l+1)} = \sigma(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}^{(l)} W^{(l)} h_j^{(l)}) $$
    where $\alpha_{ij}^{(l)}$ are learned attention weights (from GAT) between nodes $i$ and $j$ at layer $l$, $W^{(l)}$ is a learnable weight matrix, and $\sigma$ is an activation function. Final node embeddings $h_v^{(L)}$ capture network context.

**C. Perturbation Prediction Head:**
*   **Input:** Final gene embeddings $h_{gene}^{(L)}$ from the GNN module, representing the baseline cellular state, and an embedding representing the perturbation $p$ (e.g., one-hot vector for gene knockout, embedding for chemical structure or target).
*   **Architecture:** A feed-forward network $f_{pert}$ predicts the change in gene expression $\Delta Y$ or the final expression state $Y_{pert}$.
*   **Mathematical Sketch:**
    $$ \Delta Y = f_{pert}([h_{gene}^{(L)}; p]) $$
    or
    $$ Y_{pert} = f_{pert}([h_{gene}^{(L)}; p]) $$
    Where $[ ; ]$ denotes concatenation. The model learns to associate specific perturbations with changes in the network state captured by $h_{gene}^{(L)}$.

**3.3 Training Strategy**
*   **Pre-training:** We will employ a multi-task self-supervised learning objective on the large corpus of ENCODE, Roadmap, and GTEx data. Potential objectives include:
    1.  Masked Sequence Modeling: Predict masked nucleotides in the DNA sequence.
    2.  Masked Epigenomic Signal Prediction: Predict masked epigenetic signals (e.g., histone modification levels) based on surrounding sequence and other epigenetic marks.
    3.  Gene Expression Prediction (Self-Supervised): Predict the expression level of a gene based on its surrounding genomic context (sequence and epigenome).
    4.  Contrastive Learning: Encourage similar representations for the same gene/region across similar cell types/conditions and dissimilar representations otherwise.
    5.  Regulatory Link Prediction (optional): If incorporating known links (e.g., from Hi-C or eQTLs), predict these links as an auxiliary task.
    *   The overall pre-training loss $\mathcal{L}_{pretrain}$ will be a weighted sum of these individual losses.
*   **Fine-tuning:** The pre-trained GCFM will be fine-tuned on specific downstream tasks using task-specific labeled data (if available) or task-specific objectives on unlabeled data.
    *   *Gene Expression Prediction:* Fine-tune using measured RNA-seq data, minimizing Mean Squared Error (MSE) or Mean Absolute Error (MAE) between predicted ($Y_{pred}$) and actual ($Y_{true}$) expression: $$\mathcal{L}_{expr} = \frac{1}{N} \sum_{i=1}^N (Y_{pred, i} - Y_{true, i})^2$$.
    *   *Regulatory Element Prediction:* Fine-tune to predict enhancer activity or TFBS presence, using binary cross-entropy loss.
    *   *Perturbation Prediction:* Fine-tune on perturbation datasets (CMap, DepMap), minimizing MSE between predicted and observed expression changes ($\Delta Y$).
*   **Optimization:** We will use the AdamW optimizer (Loshchilov & Hutter, 2019) with a learning rate schedule (e.g., linear warm-up followed by cosine decay). Training will be performed on high-performance computing clusters with GPUs.

**3.4 Experimental Design and Validation**
We will evaluate GCFM through rigorous experiments and comparisons:

*   **Task 1: Gene Expression Prediction:** Predict held-out gene expression levels in specific cell types/tissues. Evaluate using Pearson/Spearman correlation, R-squared, MSE, MAE. Compare against baseline models (e.g., linear regression using epigenetic features) and state-of-the-art sequence-based models (e.g., Enformer, Basenji). We will use cross-validation across cell types or chromosomes.
*   **Task 2: Regulatory Element Function Prediction:** Predict enhancer activity scores or TFBS locations using held-out data from ENCODE/Roadmap. Evaluate using Area Under the Receiver Operating Characteristic curve (AUC-ROC) and Area Under the Precision-Recall curve (AUPR). Compare against models like DeepSEA, GCBLANE (Ferrao et al., 2025).
*   **Task 3: Regulatory Interaction Inference:** Evaluate the learned graph component. Assess predicted enhancer-gene links against experimentally validated links (e.g., from CRISPRi screens, eQTL databases). Evaluate inferred TF-gene interactions against known motifs (JASPAR) and ChIP-seq binding evidence. Metrics: AUPR, edge prediction accuracy. Compare against GRN inference methods like Q-GAT (Zhang et al., 2023) and DiscoGen (Ke et al., 2023), adapting input data as needed.
*   **Task 4: Perturbation Effect Prediction:** Predict gene expression changes upon perturbation using withheld data from CMap or DepMap. Evaluate using correlation between predicted and observed expression changes, MSE. Compare against baseline models predicting based on pathway databases or simpler regression models.
*   **Task 5: Ablation Studies:** Systematically remove or replace components of GCFM (multi-scale attention, GNN module) to assess the contribution of each innovation.
*   **Task 6: Interpretability Analysis:** Use methods like attention map visualization, integrated gradients, or SHAP values to understand which sequence features, epigenetic marks, or graph interactions drive predictions. Validate findings against known biology (e.g., do important features correspond to known motifs or enhancers?).

**Evaluation Metrics:**
Standard metrics relevant to each task will be used:
*   Regression: R-squared, Pearson/Spearman correlation, Mean Squared Error (MSE), Mean Absolute Error (MAE).
*   Classification/Link Prediction: AUC-ROC, AUPR, F1-score.
*   Biological Validation: Overlap with known databases (TRRUST, JASPAR, EnhancerAtlas), pathway enrichment analysis of predicted targets/interactions.

**4. Expected Outcomes & Impact**

**4.1 Expected Outcomes**
1.  **A Novel Foundation Model (GCFM):** A pre-trained model capturing fundamental patterns of gene regulation from sequence, epigenome, and transcriptome data, incorporating multi-scale attention and graph neural networks.
2.  **State-of-the-Art Performance:** Demonstrated superior or competitive performance on key benchmark tasks: gene expression prediction, regulatory element identification, GRN inference, and perturbation effect prediction compared to existing methods.
3.  **New Biological Insights:** Identification of potentially novel regulatory elements, long-range interactions, or context-specific regulatory logic through model predictions and interpretability analysis. Validation of key model predictions against biological databases and literature.
4.  **Validated Perturbation Prediction Framework:** A tool capable of *in silico* screening of genetic and chemical perturbations, predicting their impact on cellular gene expression programs.
5.  **Open-Source Contribution:** Release of the GCFM model architecture, pre-trained weights, and associated code to the research community to facilitate reproducibility and further development.

**4.2 Impact**
*   **Scientific Impact:** GCFM will provide a more powerful computational lens for studying gene regulation, enabling deeper insights into the mechanisms governing cell behavior in health and disease. It pushes the boundary of applying foundation models to complex biological systems involving sequence, graph structures, and multi-modal data.
*   **Translational Impact:** By improving the ability to predict the effects of perturbations, GCFM can significantly accelerate drug discovery. It allows for rapid *in silico* hypothesis testing, identification of novel drug targets (genes or regulatory elements), and prediction of potential drug efficacy or off-target effects, aligning directly with the workshop's focus on target identification and advanced therapeutics.
*   **Community Impact:** This work will provide a valuable resource for the genomics and ML communities. It addresses several key challenges outlined in recent literature (noise robustness via pre-training, complexity via hybrid architecture, scalability via efficient implementations, multimodality integration, interpretability analysis), offering a platform for further research. It directly contributes to the workshop's goal of fostering interdisciplinary collaboration and advancing ML applications in genomics. By demonstrating the power of integrating sophisticated sequence modeling with graph-based reasoning, GCFM can inspire new approaches to modeling complex biological systems.

**5. References**

*   Avsec, Ž., Agarwal, V., Visentin, D., Ledsam, J. R., Grnarova, P., kneel, J., ... & Kelley, D. R. (2021). Effective gene expression prediction from sequence by integrating long-range interactions. *Nature Methods*, *18*(10), 1196-1203.
*   Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., ... & Liang, P. (2021). On the opportunities and risks of foundation models. *arXiv preprint arXiv:2108.07258*.
*   Consortium, E. P. (2012). An integrated encyclopedia of DNA elements in the human genome. *Nature*, *489*(7414), 57-74.
*   Consortium, R. E. P., Kundaje, A., Meuleman, W., Ernst, J., Bilenky, M., Yen, A., ... & Kellis, M. (2015). Integrative analysis of 111 reference human epigenomes. *Nature*, *518*(7539), 317-330.
*   Ferrao, J. C., Dias, D., Morajkar, S., & Dessai, M. G. F. (2025). GCBLANE: A graph-enhanced convolutional BiLSTM attention network for improved transcription factor binding site prediction. *arXiv preprint arXiv:2503.12377*. (Note: Year adjusted based on typical preprint timelines, assuming 2024/2025 is intended)
*   Gasperini, M., Hill, A. J., McFaline-Figueroa, J. L., Martin, B., Kim, S., Zhang, M. D., ... & Shendure, J. (2019). A genome-wide framework for mapping gene regulation via cellular genetic screens. *Cell*, *176*(1-2), 377-390.
*   GTEx Consortium. (2017). Genetic effects on gene expression across human tissues. *Nature*, *550*(7675), 204-213.
*   Ji, Z., Zhou, Z., Xu, K., & Poggio, T. (2021). The genome transformer: Understanding and predicting protein function using large language models. *bioRxiv*.
*   Ke, N. R., Dunn, S. J., Bornschein, J., Chiappa, S., Rey, M., Lespiau, J. B., ... & Rezende, D. (2023). DiscoGen: Learning to Discover Gene Regulatory Networks. *arXiv preprint arXiv:2304.05823*.
*   Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *arXiv preprint arXiv:1609.02907*.
*   Lander, E. S. (2011). Initial impact of the sequencing of the human genome. *Nature*, *470*(7333), 187-197.
*   Levine, M., & Tjian, R. (2003). Transcription regulation and animal diversity. *Nature*, *424*(6945), 147-151.
*   Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*.
*   Otal, H. T., Subasi, A., Kurt, F., Canbaz, M. A., & Uzun, Y. (2024). Analysis of Gene Regulatory Networks from Gene Expression Using Graph Neural Networks. *arXiv preprint arXiv:2409.13664*. (Note: Year adjusted based on typical preprint timelines, assuming 2023/2024 is intended)
*   Subramanian, A., Narayan, R., Corsello, S. M., Peck, D. D., Natoli, T. E., Lu, X., ... & Golub, T. R. (2017). A next generation connectivity map: L1000 platform and the first 1,000,000 profiles. *Cell*, *171*(6), 1437-1452.
*   Tsherniak, A., Vazquez, F., Montgomery, P. G., Weir, B. A., Kryukov, G., Cowley, G. S., ... & Hahn, W. C. (2017). Defining a cancer dependency map. *Cell*, *170*(3), 564-576.
*   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems*, *30*.
*   Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph attention networks. *arXiv preprint arXiv:1710.10903*.
*   Wong, C. H., Siah, K. W., & Lo, A. W. (2019). Estimation of clinical trial success rates and related parameters. *Biostatistics*, *20*(2), 273-286.
*   Zhang, H., An, X., He, Q., Yao, Y., Zhang, Y., Fan, F. L., & Teng, Y. (2023). Quadratic Graph Attention Network (Q-GAT) for Robust Construction of Gene Regulatory Networks. *arXiv preprint arXiv:2303.14193*.

---
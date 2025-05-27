Okay, here is a research proposal structured according to your requirements, building upon the provided task description, research idea, and literature review.

---

**1. Title:** **Knowledge-Guided Self-Explainable Models for Biomedical Discovery**

**2. Introduction**

**2.1 Background**
Machine learning (ML) models have demonstrated remarkable capabilities across diverse scientific domains, including healthcare and biology, offering potential breakthroughs in disease diagnosis, prognosis, drug discovery, and understanding complex biological systems (Esteva et al., 2019; Ching et al., 2018). However, the widespread adoption of sophisticated models, particularly deep learning architectures often characterized as "black boxes," faces significant hurdles in high-stakes fields like medicine. The lack of transparency regarding their internal decision-making processes hinders clinical trust, complicates regulatory approval, and limits their utility as tools for generating new scientific insights (Rudin, 2019). This opacity can lead to unexpected failures, potential biases, and an inability to verify whether the model has learned genuinely meaningful biological patterns or merely exploited spurious correlations in the data.

Explainable AI (XAI) aims to address this challenge by developing methods to make ML models more understandable to humans. Current XAI approaches often fall into two categories: post-hoc methods (e.g., LIME, SHAP) that attempt to explain predictions of an already trained black-box model, and ante-hoc (interpretable-by-design) methods that build transparency directly into the model architecture (Adadi & Berrada, 2018). While post-hoc methods are versatile, their explanations can sometimes lack fidelity to the original model's reasoning or be unstable (Rudin, 2019). Conversely, inherently interpretable models often involve a trade-off, potentially sacrificing predictive performance for transparency.

The XAI4Science workshop highlights a critical aspiration: moving beyond simply understanding *why* a model makes a prediction towards using that understanding to *discover new scientific knowledge*. This requires models that not only perform well but whose internal workings reflect and respect the underlying principles of the domain they model. In biomedicine, this means leveraging the vast stores of existing biological knowledge (e.g., gene interactions, metabolic pathways, pharmacological mechanisms) not just as input features, but as structural priors or constraints within the model itself. Integrating domain knowledge can guide the learning process towards biologically plausible solutions, enhance generalization, and critically, yield explanations grounded in established biological entities and relationships.

**2.2 Research Objectives**
This research proposes the development and validation of **Knowledge-Guided Self-Explainable Models (KG-SEM)**, a novel framework specifically designed for biomedical discovery. Our primary goal is to bridge the gap between high-performance predictive modeling and interpretable, actionable scientific insight generation. We aim to create models that are inherently transparent by virtue of their architecture, directly incorporating biomedical domain knowledge to structure their computations and ensure explanations are scientifically meaningful.

The specific objectives are:

1.  **Develop the KG-SEM Framework:** Design and implement a flexible model architecture that synergistically integrates biomedical knowledge graphs/ontologies (e.g., protein-protein interaction networks, pathway databases like KEGG or Reactome, Gene Ontology) with advanced ML components like Graph Neural Networks (GNNs) and generalized additive models (GAMs). The architecture will be modular, allowing different knowledge sources and ML techniques to be combined based on the specific biomedical problem.
2.  **Incorporate Knowledge for Interpretability:** Engineer mechanisms within the KG-SEM framework (e.g., knowledge-guided attention, pathway-constrained GNN layers, interpretable latent representations linked to biological concepts) to ensure that model components and their interactions directly correspond to recognizable biological entities (genes, proteins, pathways, cell types, drugs) and processes.
3.  **Achieve High Predictive Accuracy:** Ensure that the inherent interpretability of KG-SEM does not unduly compromise predictive performance on challenging biomedical tasks, such as disease subtyping, treatment response prediction, and biomarker identification.
4.  **Generate Actionable Scientific Insights:** Demonstrate that the explanations derived from KG-SEM (e.g., identified influential genes/pathways, predicted synergistic drug effects, discovered patient subgroups with distinct molecular signatures) are biologically plausible, align with existing knowledge where applicable, and can generate novel, verifiable hypotheses for further experimental or clinical investigation.
5.  **Establish a Hybrid Evaluation Protocol:** Develop and apply a comprehensive evaluation strategy that assesses both the predictive accuracy (using standard ML metrics) and the quality of explanations (using quantitative alignment metrics and qualitative expert review) of the KG-SEM framework.

**2.3 Significance**
This research directly addresses the core themes of the XAI4Science workshop by focusing on leveraging XAI not just for model understanding but for tangible knowledge discovery in healthcare. The significance lies in several key areas:

*   **Advancing Interpretable AI:** It contributes a novel approach to ante-hoc interpretability, moving beyond generic interpretable models by deeply integrating domain-specific knowledge into the model's structure. This tackles the challenge of creating explanations that are not just simple feature attributions but reflect complex biological mechanisms.
*   **Enhancing Biomedical Discovery:** By generating biologically grounded explanations, KG-SEM can act as a powerful hypothesis generation engine, potentially uncovering novel disease mechanisms, identifying new biomarkers, suggesting synergistic drug combinations, or refining patient stratification strategies. This directly supports the translation of ML insights into actionable scientific and clinical advances.
*   **Building Trust and Facilitating Adoption:** Inherently transparent models whose reasoning aligns with established biological principles are more likely to gain the trust of clinicians and researchers. This can accelerate the responsible adoption of AI tools in clinical decision support and biomedical research pipelines.
*   **Addressing Limitations of Current Methods:** KG-SEM aims to overcome the fidelity issues associated with post-hoc explanations and the potential performance limitations of simpler interpretable models. It seeks to find a "sweet spot" combining state-of-the-art performance with meaningful, built-in transparency.
*   **Cross-Disciplinary Impact:** While focused on biomedicine, the principles of integrating domain knowledge into self-explainable architectures could be adapted to other scientific fields featured in the workshop (e.g., climate science, material science) where complex systems and established scientific knowledge coexist.

By successfully developing and validating KG-SEM, this research aims to demonstrate a powerful paradigm where AI models serve not just as predictive tools, but as collaborative partners in the scientific discovery process, helping to unravel the complexities of biology and improve human health.

**3. Methodology**

**3.1 Research Design**
This research employs a constructive and empirical research design. We will first design and implement the novel KG-SEM framework. Then, we will instantiate this framework for specific biomedical tasks and datasets. Finally, we will rigorously evaluate the framework's predictive performance and the quality/utility of its explanations using a hybrid quantitative and qualitative approach, comparing it against relevant baseline methods.

**3.2 Data Collection and Preprocessing**
We will utilize publicly available large-scale biomedical datasets, potentially including:

*   **Cancer Genomics:** The Cancer Genome Atlas (TCGA), International Cancer Genome Consortium (ICGC) – providing multi-omics data (gene expression, mutation, copy number variation, methylation) and clinical information (survival, treatment response, subtypes).
*   **Biomedical Knowledge Graphs/Ontologies:** Gene Ontology (GO), KEGG Pathway Database, Reactome, DrugBank, STRING database (protein-protein interactions), Human Phenotype Ontology (HPO). These provide structured knowledge about genes, proteins, drugs, diseases, pathways, and their relationships.
*   **Clinical Trial Data:** Publicly accessible anonymized data from relevant clinical trials (where available) focusing on treatment outcomes.
*   **Other Disease Cohorts:** Depending on the specific application, data from cohorts like the UK Biobank or specialized disease consortia (e.g., ADNI for Alzheimer's).

**Preprocessing Steps:**
1.  **Omics Data:** Normalization (e.g., TPM for RNA-Seq, log-transformation), batch effect correction (e.g., ComBat), feature selection (variance filtering, pathway-guided selection), missing value imputation.
2.  **Clinical Data:** Standardization, encoding of categorical variables, handling missing data.
3.  **Knowledge Integration:** Constructing graph representations from ontologies and interaction databases. Nodes will typically represent biological entities (genes, proteins, drugs), and edges will represent known relationships (interactions, pathway membership, regulatory links). Node/edge features can encode additional information (e.g., gene function annotations, drug properties).

**3.3 Proposed Model Architecture: KG-SEM**
The KG-SEM framework will integrate domain knowledge structurally, primarily using GNNs operating on knowledge graphs and potentially combining their outputs within an additive framework for final prediction and explanation.

**Core Idea:** Model the biological system explicitly. For instance, in predicting drug response from gene expression, the model might use a GNN over a protein-protein interaction network modulated by gene expression levels, combined with a module representing drug properties and their known targets, feeding into an additive model where each term represents the contribution of a specific pathway or biological process.

**Mathematical Formulation (Illustrative Example):**

Consider predicting a clinical outcome $y$ (e.g., patient survival probability) based on multi-omics features $X_{omics}$ and a biological knowledge graph $G = (V, E)$, where $V$ are biological entities (e.g., genes) and $E$ are relationships (e.g., interactions). $X_{omics}$ may provide initial node features $h_v^{(0)}$ for $v \in V$.

**1. Knowledge-Guided GNN Module:**
We can use a GNN to learn representations $h_v^{(L)}$ for each entity $v$ after $L$ layers, incorporating biological context. An attention mechanism guided by known pathways (e.g., KEGG) could be used. A layer update might look like:
$$ h_v^{(l+1)} = \sigma \left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \alpha_{vu}^{(l)} W^{(l)} h_u^{(l)} \right) $$
where $\mathcal{N}(v)$ is the neighborhood of $v$ in $G$, $W^{(l)}$ is a learnable weight matrix, $\sigma$ is an activation function, and $\alpha_{vu}^{(l)}$ are attention coefficients. Crucially, $\alpha_{vu}^{(l)}$ could be designed to reflect biological importance or pathway constraints:
$$ \alpha_{vu}^{(l)} = \text{softmax}_u (e_{vu}^{(l)}) = \frac{\exp(\text{LeakyReLU}(a^T [W^{(l)}h_v^{(l)} || W^{(l)}h_u^{(l)} || K_{vu}]))}{\sum_{k \in \mathcal{N}(v) \cup \{v\}} \exp(\text{LeakyReLU}(a^T [W^{(l)}h_v^{(l)} || W^{(l)}h_k^{(l)} || K_{vk}]))} $$
Here, $||$ denotes concatenation, $a$ is a learnable attention vector, and $K_{vu}$ is a feature vector derived from domain knowledge representing the known relationship between $v$ and $u$ (e.g., type of interaction, pathway co-membership score). This explicitly injects knowledge into the information propagation process.

**2. Interpretable Aggregation/Prediction Module:**
The final node representations $\{h_v^{(L)}\}$ can be aggregated, potentially using another attention mechanism or pooling, to form a graph-level representation $h_G$. Alternatively, specific node representations corresponding to key biological entities (e.g., known disease genes, drug targets) could be selected.

To enhance interpretability, we can structure the prediction function $f(\cdot)$ as a form of generalized additive model (GAM):
$$ \hat{y} = g_0 + \sum_{p \in \mathcal{P}} g_p(Z_p) $$
where $g_0$ is an intercept, $\mathcal{P}$ is a set of interpretable components (e.g., pathways, functional modules identified via clustering on the graph), $Z_p$ is a representation derived from the GNN outputs relevant to component $p$ (e.g., pooled representations of nodes in pathway $p$), and $g_p$ is an interpretable function (e.g., linear function, small neural network, or spline) whose contribution to the final prediction $\hat{y}$ can be easily assessed. The GNN module learns the underlying representations $Z_p$, while the additive structure provides high-level interpretability.

**3. Self-Explainability:**
The explanation arises directly from the model structure:
*   **Component Importance:** The magnitude or contribution of each term $g_p(Z_p)$ indicates the importance of biological component $p$ for the prediction.
*   **Intra-Component Insights:** Analyzing the attention weights $\alpha_{vu}^{(l)}$ within the GNN can reveal which specific interactions or entities within a component were most influential.
*   **Feature Importance:** If the initial features $h_v^{(0)}$ are interpretable (e.g., gene expression), their influence can be traced through the network.

**4. Loss Function:**
The model will be trained end-to--end using a task-specific loss (e.g., cross-entropy for classification, mean squared error for regression) possibly augmented with regularization terms to encourage desired properties:
$$ \mathcal{L} = \mathcal{L}_{task}(\hat{y}, y) + \lambda_1 \mathcal{R}_{sparsity} + \lambda_2 \mathcal{R}_{knowledge} $$
where $\mathcal{R}_{sparsity}$ could encourage sparsity in attention or component contributions for clearer explanations, and $\mathcal{R}_{knowledge}$ could add a soft constraint encouraging alignment with known biological priors not already hard-coded in the architecture (e.g., promoting activity in pathways known to be relevant to the disease).

**3.4 Implementation Details**
We plan to implement the KG-SEM framework using Python with standard ML libraries such as PyTorch or TensorFlow. GNN components will leverage specialized libraries like PyTorch Geometric (PyG) or Deep Graph Library (DGL). We will utilize publicly available implementations of baseline methods for comparison. Code developed for the KG-SEM framework will be made open-source upon publication to ensure reproducibility.

**3.5 Experimental Design and Validation**

*   **Tasks & Datasets:** We will select 2-3 specific biomedical prediction tasks for thorough evaluation, such as:
    *   *Cancer Subtype Classification:* Using TCGA multi-omics data and clinical subtype labels.
    *   *Cancer Treatment Response Prediction:* Using TCGA data or specific clinical trial datasets with treatment outcome labels.
    *   *Biomarker Discovery for Disease Presence/Severity:* Using datasets like ADNI (Alzheimer's) or UK Biobank (various conditions).
*   **Baseline Models:**
    *   *Non-interpretable High-Performance Models:* XGBoost, Random Forest, standard non-interpretable Deep Neural Networks (DNNs), standard GNNs (like GCN, GAT) without explicit knowledge integration beyond graph structure.
    *   *Post-hoc Explanations:* Applying methods like SHAP or LIME to the best-performing black-box baselines.
    *   *Ante-hoc Interpretable Models:* Relevant models from the literature review (e.g., conceptually similar works like Factor Graph Neural Networks [Ma & Zhang, 2019] if applicable to the task), potentially simpler interpretable models like logistic regression with pathway features.
*   **Evaluation Metrics:**
    *   *Predictive Performance:* Standard metrics relevant to the task (e.g., Accuracy, AUC-ROC, Precision, Recall, F1-score for classification; MSE, MAE, R-squared for regression; Concordance Index for survival analysis). Performance will be assessed using rigorous cross-validation protocols (e.g., 5-fold or 10-fold CV).
    *   *Explainability & Interpretability:*
        *   *Quantitative Alignment:* Measure the overlap between top features/pathways/components identified by KG-SEM and known disease genes/pathways from gold-standard databases (e.g., OMIM, KEGG Disease, DisGeNET). This can involve calculating enrichment scores (e.g., using hypergeometric tests) or overlap coefficients (e.g., Jaccard index).
        *   *Sparsity/Complexity:* Measure the complexity of explanations (e.g., number of active components/features).
        *   *Stability:* Assess the consistency of explanations across different data folds or minor perturbations (e.g., using Jaccard index on identified feature sets).
        *   *Qualitative Evaluation:* Conduct case studies on specific predictions. Visualize attention maps, component contributions, and identified subgraphs. Present these explanations to domain experts (biologists, clinicians) for assessment of biological plausibility, novelty, and potential actionability. A structured questionnaire or interview protocol will be used to collect expert feedback.
*   **Validation of Discovered Insights:** While full wet-lab validation is beyond the scope of a typical ML project, we will prioritize identifying insights that are strongly supported by literature or amenable to *in silico* validation (e.g., checking predicted drug synergies against known interaction databases, analyzing identified patient subgroups for distinct molecular pathway enrichments). We will clearly state generated hypotheses suitable for future experimental validation.

**4. Expected Outcomes & Impact**

**4.1 Expected Outcomes**
1.  **A Novel KG-SEM Framework:** A well-documented, implemented, and validated software framework for building knowledge-guided self-explainable models in biomedicine, adaptable to various data types and knowledge sources.
2.  **High-Performing Interpretable Models:** Demonstration that KG-SEM instances can achieve predictive performance comparable or superior to state-of-the-art black-box models on selected biomedical tasks, while providing inherent interpretability.
3.  **Biologically Plausible Explanations:** Evidence through quantitative metrics and expert evaluation that the explanations generated by KG-SEM align well with existing biological knowledge and are considered plausible and potentially insightful by domain experts.
4.  **Generation of Novel Hypotheses:** Identification of potentially novel biological insights (e.g., previously unappreciated gene interactions in a disease context, molecular drivers of treatment resistance in specific patient subgroups, novel biomarkers) derived directly from the model's interpretable components. These hypotheses will be formulated clearly for potential future validation.
5.  **Publications and Dissemination:** Peer-reviewed publications in leading ML and bioinformatics/computational biology journals/conferences (e.g., NeurIPS, ICML, ISMB, RECOMB). Open-source release of the KG-SEM codebase and potentially trained models for benchmark datasets. Presentations at relevant workshops and conferences, including potentially the XAI4Science workshop.

**4.2 Impact**
This research is poised to make significant impacts:

*   **Methodological Impact:** It will advance the field of XAI by providing a robust methodology for integrating domain knowledge into complex models to achieve *ante-hoc*, scientifically meaningful interpretability. This directly addresses key challenges identified in the literature regarding the trade-off between accuracy and interpretability, and the limitations of post-hoc methods.
*   **Scientific Impact:** By enabling ML models to generate interpretable and biologically grounded insights, KG-SEM will facilitate a deeper understanding of complex disease mechanisms. It offers a computational tool to accelerate hypothesis generation and guide experimental research, ultimately contributing to the broader goal of using AI for scientific discovery as championed by the XAI4Science workshop.
*   **Clinical Impact:** In the long term, trustworthy and interpretable models like KG-SEM have the potential to improve clinical decision-making. For example, identifying patient subgroups who will benefit most from a specific therapy based on interpretable molecular signatures, or suggesting personalized drug combinations based on predicted synergistic effects grounded in pathway analysis, could significantly advance precision medicine.
*   **Building Trust:** By making the reasoning process transparent and aligned with biological knowledge, this work aims to foster greater trust in AI among biomedical researchers and clinicians, overcoming a major barrier to the adoption of advanced ML tools in healthcare.
*   **Broader Applicability:** The core principles of KG-SEM — structuring models around known domain entities and relationships to achieve self-explainability — can potentially be generalized to other scientific domains characterized by complex systems and rich theoretical or empirical knowledge (e.g., material science, climate modeling), furthering the broader XAI4Science agenda.

In conclusion, the proposed research on Knowledge-Guided Self-Explainable Models offers a promising path towards AI systems that not only predict accurately but also contribute meaningfully to scientific understanding and discovery in the critical domain of biomedicine.

**References** (Illustrative - a full proposal would include a comprehensive list)

*   Adadi, A., & Berrada, M. (2018). Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI). *IEEE Access*, 6, 52138–52160.
*   Ching, T., Himmelstein, D. S., Beaulieu-Jones, B. K., Kalinin, A. A., Do, B. T., Way, G. P., ... & Greene, C. S. (2018). Opportunities and obstacles for deep learning in biology and medicine. *Journal of The Royal Society Interface*, 15(141), 20170387.
*   Cui, H., Dai, W., Zhu, Y., Li, X., He, L., & Yang, C. (2022). Interpretable Graph Neural Networks for Connectome-Based Brain Disorder Analysis. *arXiv preprint arXiv:2207.00813*.
*   Esteva, A., Robicquet, A., Ramsundar, B., Kuleshov, V., DePristo, M., Chou, K., ... & Dean, J. (2019). A guide to deep learning in healthcare. *Nature Medicine*, 25(1), 24–29.
*   Ma, T., & Zhang, A. (2019). Incorporating Biological Knowledge with Factor Graph Neural Network for Interpretable Deep Learning. *arXiv preprint arXiv:1906.00537*.
*   Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206–215.

---
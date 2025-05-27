Okay, here is a detailed research proposal based on the provided task description, research idea, and literature review.

---

**1. Title:**

DKG-Adapter: Dynamic Knowledge Graph-Infused Adapters for Scalable Continual Learning of Foundation Models

**2. Introduction**

**2.1 Background**
Foundation models (FMs), such as large language models (LLMs) and vision transformers (ViTs), have demonstrated remarkable capabilities across a wide range of tasks (Brown et al., 2020; Dosovitskiy et al., 2020). However, their standard training paradigm relies on massive, static datasets. This leads to several fundamental limitations in the context of real-world deployment: 1) **Static Knowledge:** The knowledge encoded within these models becomes outdated as the world changes. 2) **Knowledge Saturation:** Continuously expanding static datasets for retraining becomes computationally infeasible and potentially less effective. 3) **Compute Waste:** Retraining or even fully fine-tuning these enormous models from scratch or near-scratch for updates is extremely resource-intensive (compute, time, energy).

Continual Learning (CL) offers a promising framework to address these limitations by enabling models to learn sequentially from evolving data streams while retaining previously acquired knowledge (Parisi et al., 2019; De Lange et al., 2021). However, scaling CL methods to the size of modern FMs presents significant challenges. Existing CL techniques often struggle with **catastrophic forgetting**—the tendency to lose performance on past tasks when learning new ones—especially when new data is smaller or less diverse than the original pretraining data. Furthermore, effectively managing domain shifts and long-tailed data distributions, common in real-world scenarios, remains an open problem for CL at scale (Workshop Theme).

Parameter-Efficient Fine-Tuning (PEFT) methods, particularly **adapters** (Houlsby et al., 2019; Rebuffi et al., 2017), have emerged as a way to adapt FMs with minimal parameter updates, reducing computational cost and potentially mitigating forgetting by freezing the core FM backbone. However, standard adapter approaches may still suffer from forgetting within the adapter parameters themselves or lack mechanisms to explicitly leverage prior structured knowledge for robust adaptation. Recent works like K-Adapter (Wang et al., 2020) have shown the potential of injecting static knowledge via adapters, while others like Linked Adapters (Chandra et al., 2024) and I2I (Srinivasan et al., 2023) explore inter-adapter knowledge transfer. Concurrently, research on continual knowledge graph embedding (Liu et al., 2024) highlights the possibility of dynamically updating structured knowledge bases.

There is a clear need, highlighted by the workshop themes, for CL methods that scale to FMs, effectively combat catastrophic forgetting particularly on smaller/diverse datasets, handle real-world data distributions, and potentially integrate structured knowledge sources. Our research idea directly addresses this gap by proposing a novel approach that synergizes the efficiency of adapters with the structured, dynamic knowledge representation capabilities of knowledge graphs (KGs).

**2.2 Research Objectives**
This research aims to develop and evaluate a novel framework, **Dynamic Knowledge Graph-Infused Adapters (DKG-Adapter)**, for scalable and efficient continual learning of foundation models. The primary objectives are:

1.  **Develop the DKG-Adapter Framework:** Design and implement lightweight adapter modules augmented with a mechanism to dynamically integrate relevant knowledge from an incrementally updated Knowledge Graph (KG). This includes defining the KG update process, the KG-adapter interaction mechanism (via cross-attention), and strategies for efficient knowledge retrieval and KG size management.
2.  **Evaluate Performance on CL Benchmarks:** Assess the effectiveness of DKG-Adapter in mitigating catastrophic forgetting and enabling positive knowledge transfer across sequential tasks, particularly those involving domain shifts and long-tailed distributions, using relevant language and multimodal benchmarks.
3.  **Quantify Scalability and Efficiency:** Measure the computational and memory efficiency of DKG-Adapter compared to baseline methods like full fine-tuning, standard adapter tuning, and existing CL techniques. This includes evaluating training time, inference time, parameter overhead, and KG storage costs.
4.  **Analyze the Role of Dynamic Knowledge:** Investigate how the dynamic KG component contributes to performance improvements, knowledge retention, and adaptation to new concepts or domains.

**2.3 Significance**
This research holds significant potential for advancing the field of scalable continual learning for foundation models:

*   **Addressing Core CL Challenges:** It directly tackles catastrophic forgetting, efficient knowledge transfer, and scalability—key challenges identified in the literature and the workshop call.
*   **Enabling Lifelong FMs:** By providing a mechanism for efficient and effective updates, DKG-Adapter could pave the way for FMs that continuously learn and adapt to new information without costly retraining cycles, making them truly lifelong learning systems.
*   **Harnessing Structured Knowledge:** It proposes a novel way to integrate dynamic structured knowledge (KGs) with the implicit knowledge of FMs, potentially leading to more robust, interpretable, and factually grounded models. This directly addresses a key topic of interest in the workshop.
*   **Reducing Computational Cost:** The focus on lightweight adapters and sparse KG retrieval aims to significantly reduce the computational resources required for adapting large FMs, contributing to more sustainable AI development.
*   **Broad Applicability:** The proposed framework is designed to be applicable to various modalities (language, vision, multimodal) and foundation model architectures, potentially impacting diverse application domains requiring up-to-date knowledge and adaptation, such as personalized assistants, dynamic recommender systems, and autonomous agents operating in evolving environments.

**3. Methodology**

**3.1 Overall Framework**
The proposed DKG-Adapter framework integrates a pre-trained foundation model (FM) backbone (e.g., BERT, RoBERTa, ViT, CLIP), lightweight adapter modules inserted within the FM layers, and a dynamically evolving Knowledge Graph (KG). The FM backbone parameters remain frozen during continual learning. For each new task or data stream $D_t$, the following steps occur:

1.  **KG Update:** New entities and relations relevant to $D_t$ are identified and used to incrementally update the global KG, $G_t$.
2.  **Task Learning:** The model processes data from $D_t$. For each input instance (or batch), relevant facts are retrieved from $G_t$.
3.  **KG-Infused Adaptation:** The retrieved KG information modulates the behavior of the adapter modules via a cross-attention mechanism. Only the adapter parameters $\theta_{adapter}$ are updated based on the task-specific loss.
4.  **KG Consolidation:** Periodically, the KG $G_t$ undergoes a consolidation process to manage its size and complexity.

**3.2 Data Collection and Benchmarks**
To validate DKG-Adapter, we will utilize established and potentially newly curated datasets designed to test continual learning capabilities, particularly focusing on scalability, domain shifts, and long-tailed distributions.

*   **Language Tasks:**
    *   *Sequence of Domain-Specific Datasets:* E.g., news articles from different time periods or topics (Reuters, NYT Annotated Corpus), sequences of product reviews from different categories (Amazon Reviews), evolving scientific literature datasets (PubMed).
    *   *Standard CL NLP Benchmarks:* Potentially adapt benchmarks like CLiMB (Srinivasan et al., 2023) or create sequences from GLUE/SuperGLUE tasks if appropriate domain shifts can be simulated.
*   **Multimodal Tasks:**
    *   *Sequential Visual Question Answering (VQA):* Use datasets like VQA v2, GQA, potentially creating sequences based on question types, image domains, or evolving knowledge requirements (e.g., VQA related to current events).
    *   *Image Classification with Domain Shifts:* Sequences based on datasets like DomainNet or Office-Home, potentially focusing on long-tailed recognition within domains.
*   **Data Characteristics:** We will prioritize benchmarks exhibiting explicit domain shifts, the introduction of new concepts/entities over time, and potentially long-tailed distributions within tasks, reflecting real-world challenges.

**3.3 Dynamic Knowledge Graph Construction and Update**
The KG, $G = (E, R, F)$, consists of entities $E$, relations $R$, and facts $F \subseteq E \times R \times E$.

*   **Incremental Update:** When a new data stream $D_t$ arrives, we first identify new entities and relations. This can be done using:
    *   Off-the-shelf Named Entity Recognition (NER) and Relation Extraction (RE) tools (e.g., spaCy, OpenNRE).
    *   Fine-tuning smaller, efficient models for NER/RE specifically on the incoming data context if needed.
    *   Extracting concepts related to the task domain (e.g., class labels in classification, key concepts in QA).
*   **Subgraph Creation:** New facts involving these entities/relations form a task-specific subgraph $\Delta G_t$.
*   **Integration:** The global KG is updated: $G_t = G_{t-1} \cup \Delta G_t$. Entity and relation embeddings within the KG can be initialized or updated using standard KG embedding methods like TransE, RotatE (Bordes et al., 2013; Sun et al., 2019), potentially adapted for incremental updates similar to Liu et al. (2024). We denote the embedding for an entity $e$ or relation $r$ as $\mathbf{e}$ and $\mathbf{r}$ respectively.

**3.4 KG-Infused Adapter Architecture (DKG-Adapter Module)**
We will adopt a standard adapter architecture (e.g., bottleneck adapters) inserted after the feed-forward networks in the transformer layers of the FM. Let the input hidden state to the adapter at layer $l$ be $\mathbf{h}_l$. The adapter computes $\Delta \mathbf{h}_l = Adapter(\mathbf{h}_l)$. The key innovation lies within the adapter's computation.

*   **KG Retrieval:** For an input instance $x$ (or batch), a sparse retrieval mechanism (Sec 3.5) identifies a relevant subgraph $g_x \subset G_t$. Let the embeddings of facts (e.g., concatenated head entity, relation, tail entity embeddings: $[\mathbf{e}_h; \mathbf{r}; \mathbf{e}_t]$) in $g_x$ be denoted as $\{\mathbf{f}_1, \mathbf{f}_2, ..., \mathbf{f}_k\}$. Let $\mathbf{F}_x = [\mathbf{f}_1, ..., \mathbf{f}_k]^T$ be the matrix of retrieved fact embeddings.
*   **Cross-Attention Mechanism:** Inside the adapter, we introduce a cross-attention layer where the adapter's internal representation acts as the query, and the retrieved KG fact embeddings act as keys and values.
    *   Let $\mathbf{q}$ be an intermediate representation within the adapter derived from $\mathbf{h}_l$.
    *   The context vector $\mathbf{c}$ from the KG is computed as:
        $$ \text{Attention}(\mathbf{q}, \mathbf{F}_x, \mathbf{F}_x) = \text{softmax}\left(\frac{\mathbf{q} (\mathbf{W}_K \mathbf{F}_x)^T}{\sqrt{d_k}}\right) (\mathbf{W}_V \mathbf{F}_x) $$
        where $\mathbf{W}_K$ and $\mathbf{W}_V$ are learnable projection matrices, and $d_k$ is the dimension of the keys. Multiple attention heads can be used.
*   **Modulation:** The KG context vector $\mathbf{c}$ is then integrated into the adapter's computation, for example, by concatenation or addition before a final projection layer:
    $$ \Delta \mathbf{h}_l = \mathbf{W}_{out} (\text{activation}(\mathbf{W}_{down}(\mathbf{h}_l) + \mathbf{c})) + \mathbf{h}_l $$
    (This shows adding context $\mathbf{c}$ after the down-projection, other integration points are possible).
*   **Trainable Parameters:** Only the adapter parameters (including $\mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_{down}, \mathbf{W}_{up}, \mathbf{W}_{out}$, etc.) are updated during continual learning. The FM backbone and KG embedding generation process (if pre-trained) remain frozen.

**3.5 Sparse Retrieval Mechanism**
To ensure scalability and efficiency, retrieving the entire KG for every input is infeasible. We will implement a sparse retrieval mechanism:

1.  **Candidate Generation:** Identify potential relevant entities/concepts in the input $x$ (e.g., using lightweight NER or keyword extraction).
2.  **Subgraph Expansion:** Retrieve a small neighborhood (1-hop or 2-hop) around these candidate entities in the global KG $G_t$. This forms the initial candidate subgraph $g'_{x}$.
3.  **Relevance Scoring:** Score the relevance of facts $f \in g'_{x}$ to the current input $x$ or its intermediate representation $\mathbf{h}_l$. Relevance $S_{rel}(f, x)$ could be based on:
    *   Semantic similarity between fact embeddings $\mathbf{f}$ and input embeddings (e.g., using sentence transformers for text or CLIP embeddings for images/text).
    *   Task-specific relevance derived from attention scores during a preliminary pass.
4.  **Selection:** Select the top-k most relevant facts or facts above a certain relevance threshold to form the final subgraph $g_x$ used in the cross-attention mechanism.

**3.6 Graph Consolidation Strategy**
To prevent unbounded growth of the KG $G_t$, especially over long learning lifetimes, we will implement a periodic consolidation strategy:

1.  **Redundancy Detection:** Identify potentially redundant entities or relations based on embedding similarity (e.g., $\|\mathbf{e}_i - \mathbf{e}_j\| < \epsilon$) or canonical representation matching.
2.  **Merging:** Merge highly similar nodes/relations, updating associated facts accordingly. This might involve averaging embeddings or selecting a canonical representation.
3.  **Pruning:** Optionally prune entities or facts that have not been retrieved or deemed relevant for a long period, possibly employing a Least Recently Used (LRU) policy or relevance-based decay.
This process will be performed offline periodically (e.g., after processing a certain number of tasks or data streams) to balance KG compactness and information fidelity.

**3.7 Continual Learning Algorithm**
Input: Foundation Model $M_{\Phi}$ (frozen parameters $\Phi$), sequence of data streams $D_1, D_2, ..., D_T$.
Initialize: Empty KG $G_0 = \emptyset$, empty set of adapter parameters $\Theta_{adapter} = \emptyset$.

For $t = 1$ to $T$:
1.  Receive new data stream $D_t$ for task $\mathcal{T}_t$.
2.  **KG Update:** Identify new entities/relations in $D_t$, create $\Delta G_t$. Update $G_t = G_{t-1} \cup \Delta G_t$. Update/initialize embeddings for new elements in $\Delta G_t$.
3.  Initialize new adapter parameters $\theta_{adapter}^{(t)}$ (e.g., randomly or using initialization strategies like I2I (Srinivasan et al., 2023)). Add $\theta_{adapter}^{(t)}$ to $\Theta_{adapter}$.
4.  **Train Adapters:** For number of epochs or iterations:
    *   For each batch $b \in D_t$:
        a.  Perform forward pass through $M_{\Phi}$ with adapters $\Theta_{adapter}$.
        b.  For relevant layers $l$:
            i.  Retrieve relevant KG subgraph $g_b \subset G_t$ using sparse retrieval (Sec 3.5).
            ii. Compute KG context $\mathbf{c}$ using cross-attention (Sec 3.4).
            iii. Compute adapter output $\Delta \mathbf{h}_l$ incorporating $\mathbf{c}$.
        c.  Compute task loss $\mathcal{L}_{\mathcal{T}_t}$ on the batch output.
        d.  Optionally add regularization terms $\mathcal{L}_{reg}$ (e.g., L2 on adapter weights, sparsity regularization on retrieval). Total Loss $\mathcal{L} = \mathcal{L}_{\mathcal{T}_t} + \lambda \mathcal{L}_{reg}$.
        e.  Compute gradients $\nabla_{\theta_{adapter}^{(t)}} \mathcal{L}$.
        f.  Update $\theta_{adapter}^{(t)}$ using an optimizer (e.g., Adam). *Crucially, only $\theta_{adapter}^{(t)}$ might be updated, or strategies allowing updates to previous adapters (like Linked Adapters' attention) could be explored as variants.*
5.  **Evaluation:** Evaluate performance on $\mathcal{T}_t$ and all previous tasks $\mathcal{T}_1, ..., \mathcal{T}_{t-1}$.
6.  **Optional KG Consolidation:** If triggered (e.g., based on KG size or task count), perform KG consolidation (Sec 3.6).

**3.8 Experimental Design**
*   **Baselines:**
    *   Full Fine-tuning (FT): Fine-tune the entire FM on each task sequentially. (Upper bound for task performance, lower bound for forgetting).
    *   Standard Adapter Tuning (AT): Train separate adapters for each task, freezing the FM backbone.
    *   Existing CL Methods:
        *   Regularization-based: Elastic Weight Consolidation (EWC) (Kirkpatrick et al., 2017).
        *   Rehearsal-based: Experience Replay (ER) (Rolnick et al., 2019) (with limited buffer size).
        *   Parameter Isolation / Architecture-based: Standard Adapters (AT), potentially Linked Adapters (Chandra et al., 2024) or I2I (Srinivasan et al., 2023) if implementations are comparable.
        *   Static KG baseline: K-Adapter-like approach with a static, pre-compiled KG relevant to all tasks.
*   **Evaluation Protocol:** We will follow standard CL evaluation protocols, primarily focusing on the **Task-Incremental Learning (Task-IL)** or **Domain-Incremental Learning (Domain-IL)** scenarios, where the model needs to infer the task ID or domain context at test time (often implicitly handled by task-specific adapters).
*   **Metrics:**
    *   **Accuracy/Performance:** Average accuracy (or F1-score, task-specific metric) across all tasks seen so far after learning task $t$: $A_t = \frac{1}{t} \sum_{i=1}^{t} R_{t,i}$, where $R_{t,i}$ is the performance on task $i$ after training on task $t$.
    *   **Forgetting:** Backward Transfer (BWT) (Lopez-Paz & Ranzato, 2017): $BWT = \frac{1}{T-1} \sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})$. Measures average drop in performance on past tasks. Aim for BWT close to 0 or positive.
    *   **Forward Transfer (FWT):** $FWT = \frac{1}{T-1} \sum_{i=2}^{T} (R_{i-1, i} - b_i)$, where $b_i$ is the accuracy of a randomly initialized model on task $i$. Measures influence on future tasks.
    *   **Computational Cost:** Training time per task, FLOPs required for adaptation, inference latency.
    *   **Memory Cost:** Number of trainable parameters (adapter size), total model size, KG storage size (number of nodes/edges, embedding storage).

**4. Expected Outcomes & Impact**

**4.1 Expected Outcomes**
We anticipate the following outcomes from this research:

1.  **A Functional DKG-Adapter Framework:** A robust implementation of the proposed method, applicable to standard transformer-based FMs for language and multimodal tasks.
2.  **Superior CL Performance:** DKG-Adapter is expected to significantly outperform baseline methods (especially Full FT, standard AT, EWC) on CL benchmarks involving domain shifts and long tails, demonstrating higher average accuracy ($A_T$) and substantially lower forgetting (higher BWT).
3.  **Demonstrated Scalability and Efficiency:** Quantitative results showing that DKG-Adapter achieves competitive performance with significantly reduced computational cost (training time, FLOPs) and parameter overhead compared to full fine-tuning, and manageable KG storage costs due to sparse retrieval and consolidation.
4.  **Improved Knowledge Retention and Integration:** Analysis revealing how the dynamic KG helps retain specific factual knowledge relevant to past tasks and effectively integrates new knowledge for current tasks, potentially leading to more grounded and accurate predictions.
5.  **Insights into KG-FM Synergy:** A deeper understanding of how structured knowledge (KG) and implicit knowledge (FM) can be dynamically combined for effective lifelong learning, including the sensitivity analysis of components like retrieval sparsity, consolidation frequency, and cross-attention design.

**4.2 Impact**
This research has the potential for significant impact:

*   **Advancing Scalable CL:** Contributes a novel and potentially highly effective method to the arsenal of CL techniques specifically designed for the challenges posed by large-scale Foundation Models, directly addressing the core theme of the workshop.
*   **Enabling Practical Lifelong Learning Systems:** Moves FMs closer to becoming truly adaptable systems capable of operating continuously in dynamic environments, crucial for applications like chatbots that need to stay current, personalized education tools, and autonomous systems.
*   **Promoting Efficient AI:** Offers a more sustainable approach to updating large models, reducing the immense computational and energy costs associated with frequent retraining.
*   **Bridging Symbolic and Sub-symbolic AI:** Provides a concrete mechanism for integrating structured, symbolic knowledge (KGs) with powerful, sub-symbolic deep learning models (FMs) in a dynamic setting, contributing to the broader goal of more robust and interpretable AI.
*   **Stimulating Future Research:** The framework and findings could inspire further research into dynamic knowledge integration, efficient retrieval mechanisms for CL, and the development of more comprehensive benchmarks for evaluating lifelong learning in FMs.

**5. References**

*  Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating embeddings for modeling multi-relational data. In *Advances in Neural Information Processing Systems (NIPS)*.
*  Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. In *Advances in Neural Information Processing Systems (NeurIPS)*.
*  Chandra, D. S., Srijith, P. K., Rezazadegan, D., & McCarthy, C. (2024). Linked Adapters: Linking Past and Future to Present for Effective Continual Learning. *arXiv preprint arXiv:2412.10687*. (Note: Year adjusted based on usual arXiv practice, hypothetical future date used in prompt).
*  De Lange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A., ... & Tuytelaars, T. (2021). A continual learning survey: Defying forgetting in classification tasks. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
*  Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*.
*  Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., ... & Gelly, S. (2019). Parameter-efficient transfer learning for NLP. In *International Conference on Machine Learning (ICML)*.
*  Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences (PNAS)*.
*  Liu, J., Ke, W., Wang, P., Wang, J., Gao, J., Shang, Z., ... & Li, Y. (2024). Fast and Continual Knowledge Graph Embedding via Incremental LoRA. *arXiv preprint arXiv:2407.05705*.
*  Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. In *Advances in Neural Information Processing Systems (NeurIPS)*.
*  Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. *Neural Networks*.
*  Rebuffi, S. A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017). iCaRL: Incremental classifier and representation learning. In *Conference on Computer Vision and Pattern Recognition (CVPR)*.
*  Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience replay for continual learning. In *Advances in Neural Information Processing Systems (NeurIPS)*.
*  Srinivasan, T., Jia, F., Rostami, M., & Thomason, J. (2023). I2I: Initializing Adapters with Improvised Knowledge. *arXiv preprint arXiv:2304.02168*.
*  Sun, Z., Deng, Z. H., Nie, J. Y., & Tang, J. (2019). RotatE: Knowledge graph embedding by relational rotation in complex space. In *International Conference on Learning Representations (ICLR)*.
*  Wang, R., Tang, D., Duan, N., Wei, Z., Huang, X., Ji, J., ... & Zhou, M. (2020). K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters. *arXiv preprint arXiv:2002.01808*.

---
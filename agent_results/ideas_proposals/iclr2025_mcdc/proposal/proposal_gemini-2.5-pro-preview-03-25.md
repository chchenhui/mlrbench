**1. Title: Decentralized Modular Knowledge Distillation (DMKD) for Sustainable Continual Learning**

**2. Introduction**

*   **Background:** The rapid advancement of deep learning has been largely driven by scaling models and datasets, often adhering to a "bigger is better" philosophy. While successful, this approach faces significant sustainability and scalability challenges (Workshop Summary). The computational cost of training and deploying enormous monolithic models is becoming prohibitive. Furthermore, the prevailing development lifecycle, where models are trained from scratch and discarded upon deprecation, represents a colossal waste of resources and accumulated knowledge. This practice stems from the black-box, entangled nature of large models, where modifying specific functionalities without causing detrimental side effects, such as catastrophic forgetting in continual learning scenarios, is notoriously difficult (Workshop Summary; Chen et al., 2023). Catastrophic forgetting, the tendency of models to forget previously learned tasks when trained sequentially on new ones, remains a critical bottleneck for deploying AI in dynamic, real-world environments (Roy et al., 2023).

    In contrast, principles like modularity, functional specialization, and reusability are cornerstones of robust systems in both software engineering and biology (Workshop Summary). Applying these principles to deep learning offers a promising path towards more sustainable, adaptable, and collaborative AI development. Modular architectures, such as Mixture-of-Experts (MoE), allow for specialized components (experts) that can potentially be trained, updated, and reused independently (Doe & Smith, 2023). Decentralized learning paradigms further offer ways to train models collaboratively across distributed resources, reducing reliance on centralized computation and potentially improving communication efficiency (Saadati et al., 2024). Combining modularity with techniques like knowledge distillation (KD), where knowledge from a larger "teacher" model is transferred to a smaller "student," provides a mechanism for efficient knowledge transfer and preservation (Lo et al., 2024). This confluence of ideas – modularity, decentralization, knowledge distillation, and continual learning – motivates our proposed research.

*   **Problem Statement:** The core problems addressed by this research are:
    1.  **Unsustainable Model Development:** The current paradigm of training large, monolithic models from scratch and discarding them leads to immense computational waste and hinders the accumulation of knowledge across model generations.
    2.  **Catastrophic Forgetting in Continual Learning:** Standard deep learning models struggle to learn new tasks sequentially without significantly degrading performance on previously learned tasks.
    3.  **Inefficiency in Knowledge Reuse:** Existing frameworks lack effective mechanisms to identify, preserve, and systematically reuse valuable knowledge embedded within pre-trained or deprecated models in a modular fashion.
    4.  **Scalability and Collaboration Barriers:** Training and adapting massive models requires centralized resources, hindering collaborative development and deployment in decentralized settings.

*   **Proposed Solution:** We propose a **Decentralized Modular Knowledge Distillation (DMKD)** framework designed explicitly for continual learning. The core idea is to represent complex knowledge not as a single monolithic model, but as a dynamic network of smaller, specialized expert modules. These modules are trained collaboratively in a decentralized manner. Knowledge from larger, potentially pre-existing or periodically trained "teacher" models (or even ensembles of previous modules) is distilled into these specialized expert modules. A dynamic routing mechanism selects the most relevant subset of modules for a given input or task, promoting sparsity and computational efficiency (Johnson & Lee, 2023). Crucially, we introduce a novel "knowledge preservation protocol" to extract and transfer valuable learned parameters (knowledge) from modules being updated or deprecated into new or evolving modules, ensuring knowledge continuity (White & Black, 2023). Module specialization is quantified using an entropy-based metric, which informs both the routing mechanism and the knowledge preservation process (Green & Brown, 2023). This creates an evolving ecosystem of reusable, specialized components, mitigating catastrophic forgetting, reducing redundancy, and enabling efficient, collaborative, and continual learning.

*   **Research Objectives:**
    1.  **Develop the DMKD Framework:** Design and implement the core architecture for decentralized training of modular expert networks using knowledge distillation.
    2.  **Design the Knowledge Preservation Protocol:** Formulate and validate a method for identifying and transferring salient parameters (knowledge) between modules across time or updates.
    3.  **Implement Entropy-Guided Dynamic Routing:** Develop a routing mechanism that leverages module specialization, quantified by entropy, to efficiently select and combine expert modules for incoming tasks.
    4.  **Evaluate Continual Learning Performance:** Rigorously evaluate the DMKD framework on standard continual learning benchmarks, measuring its ability to mitigate catastrophic forgetting while effectively learning new tasks.
    5.  **Analyze Efficiency and Scalability:** Quantify the computational and potential communication efficiency gains of DMKD compared to monolithic baselines and other modular/continual learning approaches, particularly in decentralized settings.
    6.  **Investigate Collaborative Aspects:** Explore how the decentralized nature of DMKD facilitates collaborative model building and knowledge sharing across different nodes or agents.

*   **Significance:** This research directly addresses the critical limitations of current deep learning paradigms highlighted in the workshop call. By promoting modularity, reusability, and efficient knowledge transfer, DMKD offers a path towards more sustainable AI development, significantly reducing the computational cost and environmental impact associated with repeated large-scale training. It tackles catastrophic forgetting, a fundamental challenge in continual learning, by enabling targeted updates and knowledge preservation within a modular structure. The framework's decentralized nature aligns with the growing need for collaborative AI and edge computing scenarios (Blue & Red, 2023). Success in this research would provide a practical and principled approach to building adaptable, evolving, and longevous AI systems, contributing significantly to the fields of continual learning, modular deep learning, and decentralized AI. It directly addresses workshop topics including MoE architectures, routing, upcycling (via knowledge preservation), decentralized training, and adaptive architectures.

**3. Methodology**

*   **Overall Framework:** The DMKD framework consists of a network of $N$ expert modules $\{M_1, M_2, ..., M_N\}$, each potentially specialized for different data distributions, tasks, or feature types. A routing mechanism $R$ determines which subset of modules processes a given input $x$. The training process is decentralized, assuming multiple participating nodes (agents), each potentially holding a subset of modules and data. Knowledge distillation occurs periodically, potentially from a conceptual global "teacher" model (which could be an aggregate of modules, a larger pre-trained model, or derived from global data statistics) or peer-to-peer between modules/nodes.

*   **Data Collection and Preparation:** We will utilize standard continual learning benchmark datasets to evaluate DMKD. Initial experiments will focus on image classification tasks:
    1.  **Split CIFAR-10/100:** Standard benchmarks involving splitting classes into sequential tasks (Class-Incremental Learning).
    2.  **Split Tiny-ImageNet / ImageNet Subsets:** Larger-scale benchmarks providing more complex visual features and task sequences. (Lo et al., 2024; Roy et al., 2023).
    3.  **Domain-Incremental Scenarios:** Datasets like PACS or OfficeHome, where tasks differ by data domain (e.g., photos, sketches, paintings) rather than classes.

    Data will be partitioned sequentially to simulate the continual learning stream. For decentralized experiments, data will be distributed across simulated nodes, considering both IID (Independent and Identically Distributed) and Non-IID settings to evaluate robustness (Saadati et al., 2024).

*   **Algorithmic Steps:**

    1.  **Initialization:**
        *   Initialize $N$ expert modules $\{M_i\}_{i=1}^N$, potentially with diverse architectures or random weights. Alternatively, initialize via distillation from a pre-trained foundation model.
        *   Initialize the router $R$.
        *   Distribute modules and initial data partitions across decentralized nodes.

    2.  **Task Arrival (Task $t$):**
        *   Nodes receive data $D_t$ corresponding to the new task.

    3.  **Decentralized Training Loop (within each node $k$):**
        *   For input $x \in D_t$:
            *   **Routing:** The router $R$ (potentially a lightweight network, possibly duplicated or coordinated across nodes) computes gating scores $\{g_i(x)\}_{i=1}^N$ indicating the relevance of each module $M_i$ for input $x$. A subset of $K$ top-scoring modules is selected (sparse activation).
                $$ g(x) = \text{Softmax}(\text{Linear}(f_{enc}(x))) $$
                where $f_{enc}(x)$ is an encoding of the input. The final output can be a weighted sum based on $g_i(x)$.
            *   **Forward Pass:** Process $x$ through the selected modules $\{M_{i_1}, ..., M_{i_K}\}$. The final output $y_{pred}$ is computed by combining module outputs (e.g., weighted averaging based on router gates).
            *   **Loss Calculation:** Compute the task-specific loss $L_{task}(y_{pred}, y_{true})$.
            *   **Distillation Loss:** Compute the knowledge distillation loss $L_{KD}$ between the output of the current module configuration and a "teacher" signal. This teacher could be:
                *   A snapshot of the model from a previous task.
                *   An aggregated model from other nodes (requiring communication).
                *   A larger, static pre-trained model.
                We may use KL divergence between softened outputs:
                $$ L_{KD} = \sum_i D_{KL}(\sigma(z_{teacher}/\tau) || \sigma(z_{student}/\tau)) $$
                where $z$ are logits and $\tau$ is the temperature. We will explore module-to-module KD inspired by m2mKD (Lo et al., 2024).
            *   **Combined Loss:** $L_{total} = L_{task} + \lambda_{KD} L_{KD} + \lambda_{reg} L_{reg}$ (where $L_{reg}$ includes regularization terms, potentially encouraging module diversity or sparsity).
            *   **Backward Pass & Parameter Update:** Update parameters of the activated modules $M_{i_j}$ and the router $R$ using gradients from $L_{total}$. Updates are local to the node.

    4.  **Knowledge Preservation (Periodically or at task boundaries):**
        *   **Identify Salient Parameters:** For modules being updated or potentially replaced, identify critical parameters using methods like:
            *   Magnitude pruning: Identify parameters with large absolute values.
            *   Fisher Information: Estimate parameter importance based on its contribution to the likelihood ($F_{ii} = E[(\frac{\partial \log p(y|x, \theta)}{\partial \theta_i})^2]$).
        *   **Transfer Knowledge:** Transfer these salient parameters or their learned representations (e.g., subspace representations inspired by Roy et al., 2023) to corresponding new or updated modules. This could involve direct weight injection, initialization strategies, or using them as regularization targets during the fine-tuning of the recipient module (White & Black, 2023).

    5.  **Module Specialization Assessment (Periodically):**
        *   For each module $M_i$, estimate its activation distribution over a representative dataset (e.g., validation set from previous tasks or current task).
        *   Calculate an entropy-based metric for each module. A simpler metric could be the entropy of the average router gate activations assigned to module $M_i$ over the dataset $D$: $H(M_i) = -\sum_{c \in \mathcal{C}} p(c|M_i) \log p(c|M_i)$, where $p(c|M_i)$ is the probability distribution over classes/concepts that activate module $M_i$. Lower entropy suggests higher specialization (Green & Brown, 2023).
        *   Use $H(M_i)$ to potentially guide the router (e.g., prioritize specialized modules), inform module pruning/addition decisions, or guide the knowledge preservation protocol (e.g., preserve parameters from highly specialized modules more carefully).

    6.  **Decentralized Coordination (Periodically):**
        *   Nodes communicate updates or aggregated module information. We will explore strategies inspired by DIMAT (Saadati et al., 2024), such as periodic averaging of module weights or router parameters, potentially focusing on communication-efficient methods like exchanging only important parameters identified by the preservation protocol or utilizing federated averaging variants. The frequency and content of communication will be key parameters.

    7.  **Repeat steps 2-6 for subsequent tasks.**

*   **Mathematical Formulas:**
    *   Router Gating: $g(x) = \text{Softmax}(W_{route} f_{enc}(x) + b_{route})$
    *   Combined Output (Example: Weighted Average): $y_{pred} = \sum_{i=1}^N g_i(x) M_i(x)$ (assuming sparse selection, sum is over selected K modules).
    *   Knowledge Distillation Loss (KL Divergence): $$ L_{KD} = \tau^2 \sum D_{KL}(\sigma(z_{student}/\tau) || \sigma(z_{teacher}/\tau)) $$
    *   Module Specialization Entropy (Conceptual): $$ H(M_i) \approx -\sum_{x \in D_{val}} p(M_i|x) \log p(M_i|x) $$ where $p(M_i|x)$ is approximated by the router gate $g_i(x)$.
    *   Fisher Information (Diagonal Approx): $$ F_{ii} \approx \frac{1}{|D|} \sum_{(x,y) \in D} (\frac{\partial L(y, M(x;\theta))}{\partial \theta_i})^2 $$

*   **Experimental Design & Validation:**
    *   **Baselines:**
        *   *Monolithic Models:* Fine-tuning (lower bound), Joint Training (upper bound, requires all data).
        *   *Standard CL Methods:* EWC (Elastic Weight Consolidation), LwF (Learning without Forgetting), SI (Synaptic Intelligence), ER (Experience Replay).
        *   *Modular/KD CL Methods:* m2mKD (adapted for CL), Methods from (Chen et al., 2023), (Roy et al., 2023), potentially simple MoE baselines adapted for CL.
        *   *Decentralized Baselines:* Federated Averaging (FedAvg) applied sequentially, DIMAT baseline adapted for CL.
    *   **Evaluation Metrics:**
        *   *Accuracy:* Average accuracy across all tasks learned so far ($\text{AvgAcc} = \frac{1}{T} \sum_{i=1}^T A_{T,i}$, where $A_{T,i}$ is accuracy on task $i$ after training on task $T$). Final accuracy on all tasks.
        *   *Forgetting:* Average Forgetting ($F = \frac{1}{T-1} \sum_{i=1}^{T-1} \max_{t' \in \{1,..,T-1\}} (A_{t',i} - A_{T,i})$). Measures performance drop on previous tasks.
        *   *Backward Transfer (BWT):* Average accuracy improvement on previous tasks after learning a new task (often negative, equals -Forgetting).
        *   *Forward Transfer (FWT):* Influence of learning previous tasks on the performance of future tasks.
        *   *Computational Cost:* FLOPs per inference/update, total training time, number of trainable parameters.
        *   *Communication Cost (for decentralized):* Total data transferred between nodes during training.
        *   *Memory Footprint:* Peak memory usage during training and inference.
    *   **Ablation Studies:** We will systematically disable or modify components of DMKD to understand their individual contributions:
        *   No Knowledge Distillation vs. Different KD variants.
        *   No Knowledge Preservation vs. Different preservation strategies.
        *   Static Routing vs. Dynamic Entropy-Guided Routing.
        *   Centralized Training vs. Decentralized Training (varying communication protocols).
        *   Impact of the number of modules ($N$) and selected experts ($K$).
        *   Effectiveness of the entropy metric for specialization.
    *   **Analysis:** We will analyze module activations and specialization scores (entropy) to verify if modules indeed specialize as expected. We will visualize parameter importance (e.g., Fisher info) used in the preservation protocol. Convergence speed and stability in decentralized settings (IID vs. Non-IID) will be assessed.

**4. Expected Outcomes & Impact**

*   **Expected Outcomes:**
    1.  **State-of-the-Art Continual Learning Performance:** We expect DMKD to significantly outperform standard CL baselines in terms of average accuracy and reduced forgetting, particularly on longer task sequences and more complex datasets.
    2.  **Demonstrated Knowledge Preservation:** The knowledge preservation protocol is expected to demonstrably retain critical information from earlier tasks, quantifiable through reduced forgetting rates and potentially positive backward transfer compared to methods without explicit preservation.
    3.  **Efficient Resource Utilization:** DMKD should exhibit lower computational costs (FLOPs, training time) compared to retraining monolithic models and potentially competitive or lower costs compared to dense CL methods, due to sparse module activation and reuse. Parameter count can be controlled by the number and size of modules.
    4.  **Effective Module Specialization:** We anticipate observing clear specialization patterns in modules, validated by the entropy metric and activation analysis, leading to efficient task handling via the dynamic router.
    5.  **Scalable Decentralized Learning:** The framework is expected to function effectively in simulated decentralized environments, demonstrating tolerance to Non-IID data distributions and showcasing the potential for communication efficiency compared to naive federated learning approaches, by leveraging modularity and potentially sparse updates based on the preservation protocol.
    6.  **A Functional Framework:** The primary outcome will be a well-documented and empirically validated framework (code and results) for decentralized, modular continual learning.

*   **Impact:**
    1.  **Sustainability in AI:** By enabling knowledge reuse and reducing the need for complete retraining, DMKD offers a concrete step towards more sustainable AI development practices, aligning with the workshop's core theme of moving beyond disposable models.
    2.  **Advancing Continual Learning:** This research directly tackles catastrophic forgetting and the stability-plasticity dilemma by integrating modularity, distillation, and knowledge preservation, potentially setting a new direction for CL research.
    3.  **Enabling Collaborative AI:** The decentralized nature promotes scenarios where multiple agents or institutions can collaboratively build and improve AI models without sharing raw data or requiring massive centralized infrastructure, fostering distributed expertise.
    4.  **Bridging Theory and Practice:** The proposed framework combines theoretical concepts (information theory via entropy, KD theory) with practical mechanisms (routing, parameter transfer, decentralized protocols), aiming for a system applicable to real-world scenarios.
    5.  **Addressing Key Challenges:** This work directly confronts several key challenges identified in the literature review: provides a novel approach to optimize modular architectures via guided distillation and routing; explicitly balances stability (preservation) and plasticity (new task learning); aims to mitigate decentralized overheads through modular communication; directly targets catastrophic forgetting; and offers a specific mechanism for efficient knowledge transfer and preservation.
    6.  **Contribution to Workshop Themes:** The project aligns perfectly with the workshop's focus on modularity (MoE, routing, upcycling via preservation), collaborative/decentralized training, and applications in continual learning. The findings will contribute valuable insights and potential solutions to the workshop's discussions.
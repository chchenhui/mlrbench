## 1. Title

**MeLPA: Meta-Learned Personalized Adapters for Efficient Continual Adaptation of Foundation Models**

## 2. Introduction

Foundation Models (FMs), encompassing large language models (LLMs), vision transformers, and multimodal architectures, have demonstrated remarkable capabilities across a wide array of tasks. Their success, however, often comes with substantial computational demands, particularly when adapting them to specific downstream applications or individual user needs. The rapidly evolving AI landscape necessitates scalable optimization methods to yield FMs that are not only powerful but also efficient and adaptive, especially for inference services. As highlighted by the workshop's scope, key challenges include enabling continual weight updates, compute- and memory-efficient fine-tuning, and personalized adaptation while handling long contexts and evolving knowledge.

Current paradigms for specializing FMs often involve full model fine-tuning, which is resource-intensive and impractical for scenarios requiring rapid or frequent adaptation to numerous users or tasks. While parameter-efficient fine-tuning (PEFT) methods, such as adapter modules (Houlsby et al., 2019; Pfeiffer et al., 2020), LoRA (Hu et al., 2021), and prompt tuning (Lester et al., 2021), have reduced the number of trainable parameters, they still face hurdles in the context of continual learning and deep personalization. Specifically, these models can suffer from "catastrophic forgetting" – the loss of previously learned knowledge when acquiring new information – and may not possess optimal strategies for initializing or updating themselves for novel data streams (Key Challenge 1, 2).

The challenge is to endow FMs with the ability to continuously learn from individual user data streams or sequences of tasks, tailoring their behavior without incurring prohibitive retraining costs or forgetting past knowledge. This requires a delicate balance between stability (retaining general knowledge from the pre-trained FM and past adaptations) and plasticity (quickly learning new, user-specific information). Existing continual learning approaches for FMs, such as ConPET (Song et al., 2023) and SAFE (Zhao et al., 2024), employ various strategies like regularization or dynamic architectures. However, finding universally optimal adaptation strategies that generalize across diverse users and tasks remains an open problem (Key Challenge 4, 5). Meta-learning, or learning to learn, offers a promising avenue to address these challenges by learning an efficient adaptation process itself from a multitude of related tasks. Works like OMLA (Zhu et al., 2025) showcase meta-learned adapters in robotics, and Qin et al. (2023) explore meta-learning for prompt tuning initialization, indicating the potential of meta-strategies.

**Research Objectives:**

This research proposes **MeLPA (Meta-Learned Personalized Adapters)**, a novel framework that leverages meta-learning to optimize the initialization and update dynamics of compact, personalized adapter modules for FMs. The core FM remains frozen, ensuring scalability and preserving its general capabilities. The primary objectives are:

1.  **To design and implement a meta-learning algorithm that learns an optimal initialization strategy for personalized adapter modules.** This aims to provide a strong starting point for adaptation, significantly reducing the fine-tuning effort required for new users or tasks.
2.  **To develop a meta-learned update rule for adapter modules.** This rule will be optimized to facilitate rapid, stable, and efficient fine-tuning on new data streams, inherently promoting knowledge retention and plasticity.
3.  **To evaluate MeLPA's effectiveness in mitigating catastrophic forgetting in continual learning scenarios.** The framework's capacity to maintain performance on previously learned tasks while adapting to new ones will be rigorously assessed.
4.  **To demonstrate superior parameter efficiency, reduced computational overhead, and faster adaptation speeds for MeLPA compared to existing personalization and continual learning techniques for FMs.**
5.  **To validate the framework's ability to achieve strong personalization performance across diverse simulated user tasks and data streams in language and potentially vision domains.**

**Significance:**

MeLPA directly addresses the workshop's call for scalable optimization methods for efficient and adaptive FMs. By focusing on meta-learning for personalized adapters, this research aims to make significant contributions:

*   **Enhanced Personalization:** Enabling FMs to quickly and efficiently adapt to individual user preferences, data, and evolving needs, leading to more relevant and engaging AI experiences.
*   **Efficient Continual Learning:** Providing a robust solution to catastrophic forgetting in continual adaptation scenarios, allowing FMs to continuously integrate new knowledge without extensive retraining.
*   **Reduced Computational Burden:** Drastically lowering the computational and memory costs associated with personalizing and continually updating FMs, making advanced AI more accessible.
*   **Advancement in Meta-Learning for FMs:** Pioneering a novel application of meta-learning for optimizing both the initialization and the learning process of PEFT modules, offering new insights into building truly adaptive AI systems.

This research holds the potential to transform how FMs are deployed and maintained, particularly in dynamic environments requiring ongoing adaptation, such as personalized news feeds, lifelong learning assistants, or continually updated knowledge retrieval systems.

## 3. Methodology

The proposed MeLPA framework integrates a frozen pre-trained foundation model with lightweight adapter modules. The key innovation lies in using a meta-learning approach to learn how to best initialize these adapters and how to update them efficiently when presented with new, user-specific data streams in a continual learning setting.

**3.1. Overall Framework Architecture**

The MeLPA framework consists of three main components:

1.  **Frozen Foundation Model ($FM_{W_{core}}$):** A pre-trained FM (e.g., a Transformer-based LLM like Llama-2 or a Vision Transformer like ViT) whose core weights $W_{core}$ remain frozen during meta-training and subsequent adaptation. This preserves the general knowledge encoded in the FM and ensures computational efficiency.
2.  **Personalized Adapter Modules ($A_{\theta_{adapter}}$):** Lightweight neural network modules (e.g., bottleneck feed-forward layers as in Pfeiffer et al., 2020, or LoRA-style matrices) with parameters $\theta_{adapter}$. These adapters are inserted into the layers of the frozen FM. Each user or distinct continual learning sequence will have its own dedicated set of adapter parameters.
3.  **Meta-Learner:** The meta-learner optimizes two components:
    *   **An Initialization Network ($f_{init}(\cdot; \Theta_{init})$):** A small neural network parameterized by $\Theta_{init}$ that, given contextual information about a new task or user (optional, or can be a universal initializer), outputs an effective initial set of weights $\theta_{adapter}^{(0)}$ for the personalized adapter.
    *   **An Update Mechanism ($U(\cdot; \Phi_{update})$):** This mechanism dictates how the adapter parameters $\theta_{adapter}$ are updated. It could be a set of meta-learned hyperparameters (e.g., learning rates, regularization strengths) for a standard optimizer, or a more sophisticated learned optimizer (e.g., an RNN that outputs parameter updates) parameterized by $\Phi_{update}$.

**3.2. Data Collection and Simulation**

*   **Meta-Training Data ($\mathcal{D}_{meta-train}$):** A large collection of diverse "adaptation tasks" or "episodes" will be curated or simulated. Each task $T_i = (D_i^{supp}, D_i^{query})$ will consist of a support set $D_i^{supp}$ for learning/adapting the adapter, and a query set $D_i^{query}$ for evaluating the adaptation.
    *   For NLP tasks, $D_i^{supp}$ and $D_i^{query}$ could be drawn from different text classification datasets (e.g., subsets of GLUE, SuperGLUE), summarization tasks, or question-answering datasets, simulating adaptation to various domains or specific user interests.
    *   For continual learning simulation, tasks will be presented sequentially, mirroring a user's evolving data or a sequence of related sub-tasks. For instance, data could be split by topic, time, or domain.
*   **Meta-Testing Data ($\mathcal{D}_{meta-test}$):** A separate set of tasks, unseen during meta-training, will be used to evaluate the generalization capability of the learned initialization and update strategies. These tasks will simulate new users or new continual learning scenarios.

**3.3. Meta-Learning Process (Outer Loop)**

The goal of the meta-learner is to find optimal parameters $\Theta_{init}^*$ and $\Phi_{update}^*$ that enable rapid and effective adaptation of personalized adapters across a distribution of tasks.

The meta-training objective can be formulated as minimizing the expected loss on query sets after adaptation:

$$
\min_{\Theta_{init}, \Phi_{update}} \mathbb{E}_{T_i \sim \mathcal{D}_{meta-train}} \left[ \mathcal{L}_{query} \left( FM_{W_{core}}(X_i^{query}, A_{\theta_{adapter,i}^{(K)}}), Y_i^{query} \right) \right]
$$

where:
*   $T_i = ( (X_i^{supp}, Y_i^{supp}), (X_i^{query}, Y_i^{query}) )$ is a sampled task.
*   Adapter initialization: $\theta_{adapter,i}^{(0)} = f_{init}(c_i; \Theta_{init})$, where $c_i$ is optional task context.
*   Adapter adaptation: $\theta_{adapter,i}^{(K)}$ are the adapter parameters after $K$ update steps on $D_i^{supp}$ using the update mechanism $U(\cdot; \Phi_{update})$. The update for step $k$ is:
    $$
    \theta_{adapter,i}^{(k+1)} = \theta_{adapter,i}^{(k)} - U(\nabla_{\theta_{adapter,i}^{(k)}} \mathcal{L}_{supp}(FM_{W_{core}}(X_i^{supp}, A_{\theta_{adapter,i}^{(k)}}), Y_i^{supp}); \Phi_{update})
    $$
    Here, $\mathcal{L}_{supp}$ is the loss on the support set. The update mechanism $U$ could be a simple meta-learned learning rate $\alpha_{meta}$ (where $\Phi_{update} = \{\alpha_{meta}\}$), or a more complex function $g_{update}(\nabla, \theta^{(k)}; \Phi_{update})$ that computes the update.
*   $\mathcal{L}_{query}$ is the loss on the query set, evaluating the quality of the adapted adapter.

This meta-optimization will likely be performed using a gradient-based meta-learning algorithm like MAML (Finn et al., 2017), Reptile (Nichol et al., 2018), or an analogous algorithm suitable for learning initializations and update rules. The choice will depend on computational constraints and stability.

**3.4. Personalized Continual Adaptation (Inner Loop / Deployment)**

Once $\Theta_{init}^*$ and $\Phi_{update}^*$ are learned, MeLPA can be deployed for a new user $j$ or a new sequence of continual learning tasks $S_j = (task_1, task_2, \ldots, task_M)$:

1.  **Initialization:** For the first task (or new user), the personalized adapter parameters $\theta_{adapter,j}^{(0)}$ are initialized using $f_{init}(\cdot; \Theta_{init}^*)$.
2.  **Fine-tuning:** The adapter $\theta_{adapter,j}$ is then fine-tuned on the user's specific data stream $D_j$ (or $task_1, task_2, \ldots$) using the meta-learned update mechanism $U(\cdot; \Phi_{update}^*)$. Only $\theta_{adapter,j}$ is updated; $W_{core}$, $\Theta_{init}^*$, and $\Phi_{update}^*$ remain fixed.
    $$
    \theta_{adapter,j,t}^{(k+1)} = \theta_{adapter,j,t}^{(k)} - U(\nabla_{\theta_{adapter,j,t}^{(k)}} \mathcal{L}_{task_t}(FM_{W_{core}}(X_{j,t}, A_{\theta_{adapter,j,t}^{(k)}}), Y_{j,t}); \Phi_{update}^*)
    $$
    where $\mathcal{L}_{task_t}$ is the loss for the current data chunk or task $t$ from user $j$.
3.  **Knowledge Retention:** The meta-learned update mechanism $U$ is expected to implicitly handle the stability-plasticity dilemma. For instance, if $U$ is a learned optimizer, it might learn to make smaller updates to parameters deemed important for past tasks or regularize updates in a task-aware manner. We may also explore explicit regularization terms in $\mathcal{L}_{supp}$ during meta-training that penalize deviation from initial (meta-learned) parameters or forgetting, inspired by methods like LwF or EWC, but with meta-learned importance weights.

**3.5. Mitigating Catastrophic Forgetting**

MeLPA addresses catastrophic forgetting (Key Challenge 1) through several mechanisms:

*   **Frozen Core Model:** Preserves general knowledge.
*   **Isolated Personalized Adapters:** Each user or long-term task sequence has its own adapter, preventing direct interference between unrelated adaptation processes.
*   **Meta-Learned Update Rule:** The update rule $U(\cdot; \Phi_{update}^*)$ is trained over many sequential adaptation episodes. It is hypothesized that this process will lead $U$ to learn update strategies that inherently balance retaining past information with learning new information effectively. For example, it might learn to identify and protect critical adapter parameters or learn an implicit form of dynamic task-specific regularization.
*   **Meta-Learned Initialization:** A good initialization $f_{init}(\cdot; \Theta_{init}^*)$ positions the adapter in a parameter space region that is amenable to learning new tasks without drastically overwriting features crucial for a broader class of tasks encountered during meta-training.

**3.6. Experimental Design and Validation**

*   **Foundation Models:** We plan to use publicly available FMs such as GPT-2/DistilGPT-2 or T5-small/base for language tasks, and ViT-B/16 or ResNet variants for vision tasks, depending on computational resources.
*   **Adapter Architecture:** We will start with Pfeiffer adapters (Pfeiffer et al., 2020) which add small bottleneck layers, and also explore LoRA-style adaptations.
*   **Datasets:**
    *   **Meta-Training:**
        *   NLP: A diverse collection of text classification tasks (e.g., subsets of GLUE, Amazon reviews by category, news topics from different sources).
        *   Vision (if pursued): miniImageNet (Vinyals et al., 2016), tieredImageNet (Ren et al., 2018) for task diversity, or CelebA (Liu et al., 2015) for attribute-based personalization tasks.
    *   **Evaluation (Personalization & Continual Learning):**
        *   NLP:
            *   **Personalization:** Simulated user profiles with distinct writing styles or topic preferences, e.g., adapting a language model to generate text in the style of different authors or respond based on specific personas.
            *   **Continual Learning:** Benchmarks like CLUE (Ke et al., 2022) or constructing sequences of tasks from datasets like ARWU (Song et al., 2023) or 20 Newsgroups (sequential topic learning).
        *   Vision (if pursued):
            *   **Continual Learning:** Split CIFAR-10/100, Permuted MNIST, or sequences of object recognition/classification tasks from CORe50 (Lomonaco & Maltoni, 2017).
*   **Baselines:**
    *   **Full Fine-tuning (FT):** Fine-tuning the entire FM (computationally expensive, upper bound on task performance).
    *   **Standard Adapter Tuning (AT):** Fine-tuning adapters from scratch with a fixed optimizer (e.g., Adam with tuned LR).
    *   **LoRA:** Standard LoRA fine-tuning.
    *   **Existing Continual Learning Methods:** EWC (Kirkpatrick et al., 2017), LwF (Li & Hoiem, 2017), ConPET (Song et al., 2023), SAFE (Zhao et al., 2024), SEMA (Wang et al., 2024) adapted for personalized adapter setting where possible.
    *   **Meta-Learning Baselines:** MAML applied directly to adapter tuning, methods like OMLA (Zhu et al., 2025) if comparable task setups can be established.
*   **Evaluation Metrics:**
    *   **Task Performance:** Accuracy, F1-score, Perplexity, BLEU score, etc., depending on the task.
    *   **Adaptation Speed:** Number of gradient updates or wall-clock time to reach a target performance level on new tasks/user data.
    *   **Parameter Efficiency:** Total number of trainable parameters per user/task sequence (should be only adapter parameters for MeLPA).
    *   **Computational Cost:** FLOPs required for adaptation and inference.
    *   **Continual Learning Metrics:**
        *   Average Accuracy (ACC): $\frac{1}{N} \sum_{i=1}^{N} R_{N,i}$, where $R_{t,i}$ is the performance on task $i$ after learning up to task $t$.
        *   Backward Transfer (BWT): $BWT = \frac{1}{N-1} \sum_{i=1}^{N-1} (R_{N,i} - R_{i,i})$. A less negative BWT indicates less forgetting.
        *   Forward Transfer (FWT): $FWT = \frac{1}{N-1} \sum_{i=2}^{N} (R_{i-1,i} - R_{base,i})$, where $R_{base,i}$ is the performance on task $i$ by a model trained only on task $i$.
*   **Ablation Studies:**
    *   Effectiveness of meta-learned initialization ($f_{init}$) vs. random/zero initialization.
    *   Effectiveness of meta-learned update rule ($U$) vs. standard optimizers (Adam, SGD) with tuned hyperparameters.
    *   Impact of the complexity/architecture of $f_{init}$ and $U$.
    *   Sensitivity to adapter size/architecture.
    *   Influence of the number of meta-training tasks and inner loop adaptation steps ($K$).

## 4. Expected Outcomes & Impact

This research on Meta-Learned Personalized Adapters (MeLPA) is poised to deliver significant advancements in making foundation models more efficient, adaptive, and personalized.

**Expected Outcomes:**

1.  **A Novel and Robust MeLPA Framework:** We expect to develop a fully functional meta-learning framework capable of learning optimal initialization strategies and update mechanisms for personalized adapter modules integrated with frozen FMs. This framework will be demonstrated across representative language and potentially vision tasks.
2.  **Drastically Improved Adaptation Efficiency:** MeLPA is anticipated to achieve significantly faster adaptation to new user data streams or downstream tasks compared to conventional adapter tuning methods. The meta-learned initialization should provide a much better starting point, and the meta-learned update rule should guide the parameters towards optimal solutions more directly, requiring fewer gradient steps and less data (addresses Key Challenge 3, 5).
3.  **Substantial Reduction in Catastrophic Forgetting:** By training the meta-learner across numerous simulated continual adaptation episodes, the learned update mechanism is expected to inherently balance plasticity and stability. This will lead to significantly better knowledge retention (higher BWT, less negative BWT) on previous tasks when adapting to new ones, outperforming baseline continual learning methods that do not leverage meta-learned adaptation strategies (addresses Key Challenge 1, 2).
4.  **Superior Parameter and Computational Efficiency:** The proposed approach will maintain the high parameter efficiency of adapter-based methods (only a small fraction of FM parameters are tuned per user). Moreover, by accelerating adaptation, MeLPA will reduce the overall computational cost (FLOPs, time) for personalization and continual updates.
5.  **State-of-the-Art Personalized Performance:** We expect MeLPA to achieve competitive or superior performance on user-specific tasks compared to baselines, demonstrating its ability to effectively tailor FM behavior to individual needs while preserving general capabilities.
6.  **Valuable Insights into Meta-Learning for FM Adaptation:** The research will provide deeper understanding of how meta-learning can be structured to optimize the fine-tuning process itself, particularly the interplay between initialization and update rules for modular components like adapters. Ablation studies will pinpoint the contributions of each meta-learned component.

**Potential Impact:**

The successful development and validation of MeLPA will have a broad and significant impact:

*   **Paving the Way for Truly Adaptive AI:** MeLPA can enable FMs that dynamically and efficiently evolve with user interactions and changing environments. This is crucial for applications like personalized virtual assistants, adaptive educational platforms, continuously updated recommender systems, and personalized healthcare tools.
*   **Democratizing FM Personalization:** By significantly reducing the computational resources and data needed for effective personalization and continual learning, MeLPA can make advanced, tailored AI accessible to a wider range of users and organizations, including those with limited computational budgets.
*   **Advancing Scalable Optimization for FMs:** This research directly addresses the core themes of the workshop. It offers a scalable solution for making FMs adaptive and efficient for inference, contributing to the development of next-generation AI systems that are both powerful and practical.
*   **New Avenues for Continual Learning Research:** MeLPA introduces a novel perspective on tackling catastrophic forgetting by meta-learning the adaptation process itself. This could inspire new research directions in continual learning, moving beyond task-specific regularization or replay mechanisms towards more generalizable learning-to-adapt strategies.
*   **Enabling Efficient On-Device Adaptation:** The lightweight nature of adapters, combined with meta-learned efficient update rules, could pave the way for on-device personalization and continual learning, enhancing user privacy and reducing reliance on centralized servers.
*   **Extension to Broader AI Challenges:** The principles behind MeLPA—meta-learning initialization and update rules for modular components—could potentially be extended to other challenges in AI, such as few-shot learning in complex systems, lifelong robot learning, or adapting models to entirely new modalities with minimal data.

In conclusion, MeLPA aims to be a significant step towards creating foundation models that are not just knowledgeable but also dynamically and efficiently tailored to the nuanced and evolving requirements of individual users and diverse applications, directly aligning with the goals of fostering efficient and adaptive foundation models.
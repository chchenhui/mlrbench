Title: Dynamic Component Adaptation for Continual Compositional Learning

Introduction  
Background  
Human intelligence excels at building complex concepts by recombining simpler primitives. Compositional learning in machines seeks to replicate this ability, yielding strong out-of-distribution generalization by reusing “building blocks” in novel ways. Recent advances in object-centric representation learning, compositional generalization (e.g., SCAN, CFQ), and compositional reasoning in large-scale models have demonstrated impressive gains across vision, language, and multi-modal tasks. However, most of these systems assume a static world: primitives and their composition rules remain fixed during training. In real-world streams, concepts drift—objects change appearance, semantics evolve, new categories emerge, and relationships among parts shift. Off-the-shelf compositional learners fail under such non-stationarity, either forgetting earlier components (catastrophic forgetting) or misinterpreting new ones (obsolete primitives).  

Continual learning (CL) studies how models can incrementally incorporate new information while retaining past knowledge, but existing CL techniques focus on monolithic representations rather than structured, component-based ones. To date, there is a gap at the intersection of compositional and continual learning: how to continually adapt both primitive components and composition mechanisms in response to streaming, drifting data.  

Research Objectives  
1. To devise concept-drift detection methods tailored for compositional representations that can identify when a primitive or its combinatorial behavior has changed over time.  
2. To develop incremental component-learning algorithms that update or extend a library of primitives without catastrophic forgetting, leveraging techniques such as generative replay and parameter isolation.  
3. To design adaptive composition-mechanisms (e.g., attention or routing networks) that dynamically adjust how primitives are combined in light of new evidence.  
4. To construct and release benchmark suites of evolving compositional tasks (vision, language, multi-modal) for standardized evaluation of continual compositional learners.  

Significance  
Bridging compositional and continual learning is critical for deploying AI agents in dynamic environments—autonomous driving under changing road signs, robotics handling new objects, dialogue systems that encounter novel intents, and scientific discovery where concepts evolve. Our framework will deliver the first end-to-end system for robust compositional reasoning in non-stationary data streams, with broad impact on lifelong learning and adaptive AI.  

Methodology  
Our Dynamic Component Adaptation (DCA) framework consists of three modules: concept-drift detection, incremental component learning, and adaptive composition. We provide mathematical details, algorithmic steps, and evaluation design below.  

1. Representation of Primitives and Compositions  
– We assume a model maintains a library \(\mathcal{C} = \{c_1,\dots,c_K\}\) of primitive components. Each \(c_i\) is parameterized by \(\theta_i\) (e.g., feature extractor or embedding network) producing an embedding \(e_i(x)\in\mathbb{R}^d\) for input \(x\).  
– A composition mechanism \(g\) (parameterized by \(\phi\)) maps input-context \(x\) to combination weights over \(\mathcal{C}\):  
  $$\alpha = g_\phi(x)\in\Delta^{K-1},\quad \text{where } \Delta^{K-1} \text{ is the probability simplex.}$$  
– The composed representation is  
  $$h(x) = \sum_{i=1}^K \alpha_i\,e_i(x).$$  
– Downstream tasks (classification, generation, reasoning) use \(h(x)\).  

2. Concept-Drift Detection  
We extend Maximum Concept Discrepancy (MCD-DD, Wan et al. 2024) to compositional embeddings. Let \(X_t\) be the batch at time \(t\). We compute concept-specific embedding distributions via clustering or label-guided grouping. For each component index \(i\), maintain a running mean \(\mu_i^{(t-1)}\) over past embeddings. For the current batch, compute  
  $$\mu_i^{(t)} = \frac{1}{|X_t^i|}\sum_{x\in X_t^i} e_i(x),$$  
where \(X_t^i\) are samples primarily using component \(i\) (based on highest gating weight). Define the MCD statistic:  
  $$D_{\mathrm{MCD}}^{(t)} = \max_{i}\|\mu_i^{(t)} - \mu_i^{(t-1)}\|_2.$$  
If \(D_{\mathrm{MCD}}^{(t)} > \tau\) (threshold determined via validation), declare drift in component \(i^* = \arg\max_i\|\mu_i^{(t)} - \mu_i^{(t-1)}\|\).  

Drift characterization (real vs. virtual) follows Neighbor-Searching Discrepancy (Gu et al. 2024): compare classification boundaries locally to identify semantic shifts vs. rebalancing.  

3. Incremental Component Learning  
Upon detection of drift for component \(i^*\), we must update or extend \(\theta_{i^*}\) without forgetting:  

a. Generative Replay  
– Train a lightweight generative model \(G_{i^*}\) to sample pseudo-examples from the old distribution of \(c_{i^*}\). Maintain a small memory buffer \(\mathcal{M}_{i^*}\) of real exemplars.  
– Update \(\theta_{i^*}\) with mixed batches \(\mathcal{B} = B_{\textrm{new}}\cup G_{i^*}(\mathcal{N})\cup \mathcal{M}_{i^*}\), minimizing  
  $$\mathcal{L}(\theta_{i^*}) = \mathbb{E}_{x\sim\mathcal{B}}\,\ell\big(f(h(x)), y\big),$$  
  where \(\ell\) is the task loss (e.g., cross-entropy), \(f\) is the downstream head.  

b. Parameter Isolation  
– Alternative: assign a dedicated subspace of parameters for each concept. Use masks \(m_{i^*,t}\) to protect weights updated in earlier tasks (see Korycki & Krawczyk 2021). Only update free parameters \(\bar\theta\), avoid overlap.  

c. Component Addition  
– If drift indicates a novel concept (embedding distance to all existing \(\mu_i\) exceeds a second threshold \(\tau_{\textrm{new}}\)), spawn a new component \(\theta_{K+1}\). Initialize via clustering on \(X_t\).  

4. Adaptive Composition Mechanisms  
The composition network \(g_\phi\) must also adapt to changed primitives and contexts. We employ an attention-based routing:  
  $$\alpha = \mathrm{softmax}\Big(\frac{Q_\phi(x)\,K_\phi(C)^\top}{\sqrt{d}}\Big),$$  
where \(Q_\phi(x)\in\mathbb{R}^{1\times d}\) is a query vector from \(x\), and \(K_\phi(C)\in\mathbb{R}^{K\times d}\) contains keys for each component. Values are the embeddings \(V(C)=\{e_i(x)\}\).  

When drift occurs:  
– Update \(\phi\) via gradient descent on the combined loss over new and replayed data.  
– Optionally, perform a limited meta-learning step (e.g., MAML) to improve fast adaptation:  
  1. Inner update: \(\phi' = \phi - \eta\nabla_\phi \mathcal{L}_{\textrm{new}}(\phi)\).  
  2. Outer update: \(\phi \leftarrow \phi - \beta\nabla_\phi \mathcal{L}_{\textrm{support}}(\phi')\).  

5. Algorithmic Summary  
Algorithm 1: Dynamic Component Adaptation (DCA)  
Input: Initial components \(\{\theta_i\}_{i=1}^K\), composition \(\phi\), memory buffers \(\{\mathcal{M}_i\}\), thresholds \(\tau,\tau_{\textrm{new}}\).  
For each time step \(t\):  
  1. Receive batch \(X_t\).  
  2. Compute embeddings \(e_i(x)\), gating weights \(\alpha(x)\).  
  3. Perform drift detection: if \(D_{\mathrm{MCD}}^{(t)}>\tau\), identify \(i^*\).  
     a. If \(\min_j\|\mu_j^{(t)} - \mu_i^{(t)}\|\!>\!\tau_{\textrm{new}}\), spawn new \(\theta_{K+1}\).  
     b. Else update component \(i^*\) via generative replay or parameter isolation.  
  4. Update composition \(\phi\) on mixed replay and new data.  
  5. Update running means \(\mu_i^{(t)}\) and memory buffers.  

6. Experimental Design  
Datasets and Benchmarks  
– Vision: Evolving CLEVR Scenes – objects gain new shapes/colors over time.  
– Language: Dynamic SCAN – new primitives (verbs/adverbs) appear or semantics drift.  
– Multi-modal: Text-to-Image with changing style distributions (e.g., winter→summer landscapes).  

Baselines  
– Static compositional learners (LoRA-Compositional, Mixture-of-Experts without adaptation).  
– Continual learning models (EWC, GEM, replay-only) applied monolithically.  
– Drift-agnostic compositional CL (no detection, always retrain).  

Metrics  
– Task accuracy over time.  
– Backward Transfer (BWT) and Forward Transfer (FWT) to measure forgetting and learning speed.  
– Drift detection precision, recall, detection delay.  
– Component purity: cluster coherence of updated embeddings.  
– Compositional generalization: performance on held-out combinations of components.  

Implementation Details  
– Feature extractors: ResNet-50 for vision, Transformer encoder for language.  
– Embedding dimension \(d=256\).  
– Learning rates: \(\eta=1e^{-4}\) for component updates, \(\beta=5e^{-5}\) for composition meta-updates.  
– Memory buffer size per component: 200 examples.  
– Generative models: VAEs with latent dimension 64 for replay.  

All experiments run for 10 sequential tasks; each task induces one or more drifts. Evaluate on an unseen test set per task after each update.  

Expected Outcomes & Impact  
We anticipate the following outcomes:

1. Robust Drift Detection for Components  
By tailoring MCD-DD to compositional embeddings, we expect high-precision, low-latency detection of semantic shifts in primitives, outperforming generic drift detectors (DriftLens, Neighbor-Searching Discrepancy) on structured tasks.  

2. Retention with Minimal Forgetting  
Our generative-replay and parameter-isolation schemes should yield strong backward transfer (near zero forgetting) while enabling swift assimilation of new concept variations, surpassing monolithic CL baselines (EWC, GEM).  

3. Improved Compositional Generalization under Non-Stationarity  
Adaptive composition mechanisms are projected to maintain or improve performance on held-out component combinations, even as primitives evolve, demonstrating resilience in dynamic environments.  

4. Benchmark Suite and Open-Source Code  
We will release our evolving compositional datasets, evaluation protocols, and codebase to foster further research at the intersection of compositional and continual learning.  

Broader Impact  
– Lifelong Robotics: Robots can adapt to new tools or objects without forgetting prior skills.  
– Adaptive NLP Systems: Dialogue agents can handle novel user intents and evolving language usage over time.  
– Scientific Discovery: AI models can continually refine conceptual taxonomies as empirical data shifts.  

By unifying compositional and continual learning, this research paves the way for truly adaptive, structured AI systems capable of lifelong reasoning and generalization in the wild.
1. Title  
Reinforcement Learning–Guided Data Curation for Safety-Aligned Foundation Models  

2. Introduction  
2.1 Background  
Foundation Models (FMs), such as GPT-3/4, LLaMA, DALL-E and Stable Diffusion, have revolutionized many fields by demonstrating strong zero-shot and few-shot capabilities on diverse downstream tasks. However, because these models are typically trained on massive, diverse, and largely unlabeled text or image corpora, they often inherit toxic, biased or misaligned content. Such harmful behaviors can undermine user trust, pose societal risks, and trigger legal or ethical concerns. Recent research in data-centric AI emphasizes that high-quality training data plays a pivotal role in improving model safety, robustness, interpretability, and alignment with human values. Manual data filtering and curation can reduce harmful content but does not scale to the terabyte- or petabyte-scale corpora used in FM training.  

2.2 Research Objectives  
This proposal aims to develop an automated, closed-loop data curation framework that:  
  • Dynamically prioritizes or weights raw training samples to maximize safety and alignment metrics.  
  • Learns a data selection policy via reinforcement learning (RL) that balances reduction of harmful content with preservation of general linguistic and cognitive capabilities.  
  • Integrates off-the-shelf safety detectors and small human-labeled probes to define a composite reward signal that reflects toxicity, bias, and proxy alignment.  
  • Demonstrates scalability to large corpora and validates improvements on both safety benchmarks and standard performance metrics.  

2.3 Significance  
A successful RL-guided data curation pipeline will:  
  • Dramatically reduce the labor and time costs of manual dataset cleaning at FM scale.  
  • Provide a principled mechanism to balance safety and performance, mitigating the common trade-off between stricter filtering and degraded generalization.  
  • Offer a modular framework that can incorporate emerging safety detectors, alignment probes, or legal constraints (e.g., copyright flags).  
  • Advance data-centric AI by showing how dynamic, feedback-driven curation can serve as a powerful complement or alternative to parameter-centric fine-tuning.  

3. Methodology  
3.1 Data Collection and Preprocessing  
  1. Candidate Pool Construction  
     • Collect raw textual data from diverse web crawls (e.g., Common Crawl, OpenWebText) and domain-specific corpora (e.g., medical, legal) to form a large candidate pool \( \mathcal{D}_{raw} \).  
  2. Pre-Filtering  
     • Remove duplicates, boilerplate, and extremely short (< 10 tokens) or excessively long (> 1,024 tokens) samples.  
     • Language detection to retain only English (or target languages).  
  3. Annotation Probes  
     • Create a small human-labeled dataset \( \mathcal{D}_{probe} \) (e.g., 5 K samples) with multi-dimensional labels: toxicity, bias, factuality, or alignment with instructed tasks.  
     • Train lightweight proxy classifiers (e.g., fine-tuned RoBERTa) on \( \mathcal{D}_{probe} \) to generate continuous alignment scores for new samples.  

3.2 RL-Based Data Curation Framework  
3.2.1 Overview  
We model data curation as a Markov Decision Process (MDP), where at each iteration the agent selects a mini-batch of training samples to maximize a cumulative safety-alignment reward. The pipeline alternates between: (a) RL policy updates to refine sample selection probabilities; (b) fine-tuning a small FM on the curated mini-batches; (c) evaluating safety and alignment to update the reward model.  

3.2.2 MDP Formulation  
  • State \( s_t \): summarizes the current policy parameters \( \theta_t \), recent safety/alignment statistics, and optionally features of candidate samples.  
  • Action \( a_t \): a selection or weight vector \( w_t \in [0,1]^{k} \) for a candidate mini-batch of size \( k \). Concretely, \( w_{t,i} = \pi_\theta(x_i) \) where \( \pi_\theta \) is the data-selection policy and \( x_i \in \mathcal{D}_{raw} \).  
  • Transition: after selecting \( w_t \), we sample a mini-batch \( \mathcal{B}_t \) from \( \mathcal{D}_{raw} \) according to \( w_t \), fine-tune the FM (denote parameters \( \phi_{t+1} \)), then compute new safety/alignment statistics.  
  • Reward \( r_t = R(\mathcal{B}_t; \phi_{t+1}) \) captures improvements in safety/alignment relative to a baseline.  

3.2.3 Reward Model  
We design a composite reward that trades off reduced toxicity and improved alignment:  
  $$ r_t = \alpha \cdot \bigl(1 - \mathrm{Toxicity}(\mathcal{B}_t)\bigr) + \beta \cdot \mathrm{AlignScore}(\mathcal{B}_t)\,. $$  
Here  
  • \( \mathrm{Toxicity}(\mathcal{B}_t) = \frac{1}{|\mathcal{B}_t|}\sum_{x\in\mathcal{B}_t} D_{\mathrm{tox}}(x) \), where \( D_{\mathrm{tox}}(\cdot)\in[0,1] \) is a pretrained toxicity detector (e.g., Perspective API or Detoxify).  
  • \( \mathrm{AlignScore}(\mathcal{B}_t) = \frac{1}{|\mathcal{B}_t|}\sum_{x\in\mathcal{B}_t} D_{\mathrm{align}}(x) \), where \( D_{\mathrm{align}} \) is the proxy alignment classifier fine-tuned on \( \mathcal{D}_{probe} \).  
  • Hyperparameters \( \alpha,\beta\ge0 \) balance safety and alignment. We will tune them via grid search.  

3.2.4 Policy Learning via PPO  
We parameterize the data-selection policy \( \pi_\theta(x) \) as a lightweight neural network taking as input simple features of \( x \) (e.g., toxicity score, length, topic embedding) and outputting a scalar logit. We sample mini-batches proportionally to \( \exp(\pi_\theta(x)) \). The objective is to maximize the expected cumulative discounted reward:  
  $$ J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\Bigl[\sum_{t=0}^T \gamma^t r_t\Bigr], $$  
where \( \gamma\in(0,1) \) is a discount factor and \( \tau \) denotes a trajectory of states, actions, and rewards. We employ Proximal Policy Optimization (PPO) to update \( \theta \):  
  $$ L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\Bigl[\min\bigl(r_t(\theta)\hat{A}_t\,,\,\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\bigr)\Bigr], $$  
  $$ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\,,\quad \hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}, $$  
with advantage estimate \( \hat{A}_t \), GAE parameter \( \lambda \), and clip threshold \( \epsilon \).  

3.2.5 Closed-Loop Iteration  
  1. Initialize FM parameters \( \phi_0 \) (e.g., a pretrained LLaMA-small). Initialize policy parameters \( \theta_0 \).  
  2. For iteration \( t=0,\dots,T\):  
     a. Sample a candidate pool \( \mathcal{C}_t \subset \mathcal{D}_{raw} \).  
     b. Compute selection logits \( \pi_{\theta_t}(x) \) for \( x\in \mathcal{C}_t \).  
     c. Draw mini-batch \( \mathcal{B}_t \) of size \( k \) via weighted sampling.  
     d. Fine-tune FM: \( \phi_{t+1} \leftarrow \mathrm{FineTune}(\phi_t,\mathcal{B}_t) \).  
     e. Evaluate safety/alignment metrics on held-out probes to obtain \( r_t \).  
     f. Update policy via PPO to obtain \( \theta_{t+1} \).  
     g. Every \( M \) iterations, update proxy classifiers \( D_{\mathrm{align}} \) or calibrate \( D_{\mathrm{tox}} \) with new labels.  

3.3 Experimental Design  
3.3.1 Datasets  
  • Training corpora: ~100 M raw text samples from Common Crawl and domain loans.  
  • Probe set: 5 K human-labeled samples spanning toxicity, bias, factuality, instruction fidelity.  
  • Evaluation benchmarks:  
     – SafetyBench (newly proposed comprising adversarial prompts, bias tests).  
     – Standard language benchmarks (HellaSwag, LAMBADA, Wikitext perplexity).  

3.3.2 Baselines  
  1. Random sampling (no curation).  
  2. Heuristic filtering (remove top-5% toxicity scores).  
  3. RAFT (reward-ranked fine-tuning).  
  4. Safety Pretraining (Maini et al., 2025).  
  5. Safer-Instruct (Shi et al., 2023).  

3.3.3 Evaluation Metrics  
  • Safety metrics:  
     – Attack Success Rate (ASR) on adversarial prompts.  
     – Mean toxicity score via a held-out detector.  
  • Alignment metrics:  
     – Proxy alignment accuracy on held-out probes.  
     – Human preference rate in pairwise comparisons.  
  • Performance metrics:  
     – Perplexity on Wikitext.  
     – Accuracy on HellaSwag and LAMBADA.  
  • Efficiency metrics:  
     – Data computation overhead.  
     – Convergence speed (iterations to reach target safety threshold).  

3.3.4 Implementation Details  
  • FM: LLaMA-7B derivatives for fine-tuning; adapters to reduce compute.  
  • Hardware: 16 × A100 GPUs for joint RL and fine-tuning.  
  • Hyperparameters:  
     – PPO: learning rate \(2\times10^{-5}\), clip \(\epsilon=0.2\), \(\gamma=0.99\), \(\lambda=0.95\).  
     – Batch size \(k=256\), candidate pool \(|\mathcal{C}_t|=10{,}000\).  
     – Reward weights: grid search over \(\alpha,\beta\in\{0.25,0.5,0.75,1.0\}\).  
  • Training schedule: \(T=5{,}000\) iterations, periodic checkpointing every 250 steps.  

4. Expected Outcomes & Impact  
4.1 Expected Outcomes  
  • Safety Improvement  
     – 50–70% reduction in mean toxicity score compared to baseline fine-tuning.  
     – 30–50% decrease in ASR on adversarial prompt sets.  
  • Alignment Gains  
     – +10–15% proxy alignment accuracy on human probes.  
     – Human preference for RL-curated model in ≥ 70% pairwise tests.  
  • Performance Preservation  
     – ≤ 2% degradation in perplexity; ≤ 3% drop in HellaSwag accuracy.  
  • Efficiency and Scalability  
     – Data curation overhead < 20% additional compute relative to baseline fine-tuning.  
     – Stable convergence in ≈ 3{,}000 RL iterations.  

4.2 Broader Impact  
  • Automating safe data curation at FM scale will significantly reduce human labor and accelerate deployment of aligned AI systems in sensitive domains (education, healthcare, finance).  
  • The modular RL pipeline can integrate new detectors (e.g., hate-speech, disinformation) or legal constraints (e.g., copyright flags), making it adaptable to evolving safety requirements.  
  • By demonstrating that data-centric RL can balance safety and utility, this work may inspire a shift away from purely parameter-centric alignment methods (e.g., RLHF), fostering research on closed-loop, data-driven alignment.  
  • Ethical considerations: we will audit the reward model for unintended biases, ensure transparency in data selection, and open-source code and dataset statistics to support community scrutiny.  

5. References  
[1] Maini, P., Goyal, S., Sam, D., Robey, A., Savani, Y., Jiang, Y., Zou, A., Lipton, Z. C., & Kolter, J. Z. (2025). Safety Pretraining: Toward the Next Generation of Safe AI. arXiv:2504.16980.  
[2] Shi, T., Chen, K., & Zhao, J. (2023). Safer-Instruct: Aligning Language Models with Automated Preference Data. arXiv:2311.08685.  
[3] Dong, H., Xiong, W., Goyal, D., Zhang, Y., Chow, W., Pan, R., Diao, S., Zhang, J., Shum, K., & Zhang, T. (2023). RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment. arXiv:2304.06767.  
[4] Zhang, J., Elgohary, A., Magooda, A., Khashabi, D., & Van Durme, B. (2024). Controllable Safety Alignment: Inference-Time Adaptation to Diverse Safety Requirements. arXiv:2410.08968.  
[5] Henderson, P., Islam, R., Bach, S. H., Pineau, J., Precup, D., & Meger, D. (2018). Deep Reinforcement Learning That Matters. AAAI.  
[6] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
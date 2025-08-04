1. **Title**: LlamaFirewall: An Open Source Guardrail System for Building Secure AI Agents (arXiv:2505.03574)
   - **Authors**: Sahana Chennabasappa, Cyrus Nikolaidis, Daniel Song, David Molnar, Stephanie Ding, Shengye Wan, Spencer Whitman, Lauren Deason, Nicholas Doucette, Abraham Montilla, Alekhya Gampa, Beto de Paola, Dominik Gabi, James Crnkovich, Jean-Christophe Testud, Kat He, Rashnil Chaturvedi, Wu Zhou, Joshua Saxe
   - **Summary**: LlamaFirewall introduces an open-source security-focused guardrail framework designed to serve as a final layer of defense against security risks associated with AI agents. It mitigates risks such as prompt injection, agent misalignment, and insecure code generation through three guardrails: PromptGuard 2, Agent Alignment Checks, and CodeShield.
   - **Year**: 2025

2. **Title**: NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails (arXiv:2310.10501)
   - **Authors**: Traian Rebedea, Razvan Dinu, Makesh Sreedhar, Christopher Parisien, Jonathan Cohen
   - **Summary**: NeMo Guardrails is an open-source toolkit that allows developers to add programmable guardrails to LLM-based conversational systems. These guardrails control the output of an LLM, ensuring it adheres to predefined guidelines, such as avoiding harmful topics or following specific dialogue paths.
   - **Year**: 2023

3. **Title**: SafeInfer: Context Adaptive Decoding Time Safety Alignment for Large Language Models (arXiv:2406.12274)
   - **Authors**: Somnath Banerjee, Sayan Layek, Soham Tripathy, Shanu Kumar, Animesh Mukherjee, Rima Hazra
   - **Summary**: SafeInfer proposes a context-adaptive, decoding-time safety alignment strategy for generating safe responses from LLMs. It employs safe demonstration examples to adjust the model's hidden states and influences token selection based on safety-optimized distributions, ensuring compliance with ethical guidelines.
   - **Year**: 2024

4. **Title**: AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection (arXiv:2502.11448)
   - **Authors**: Weidi Luo, Shenghong Dai, Xiaogeng Liu, Suman Banerjee, Huan Sun, Muhao Chen, Chaowei Xiao
   - **Summary**: AGrail introduces a lifelong agent guardrail to enhance LLM agent safety. It features adaptive safety check generation, effective safety check optimization, and tool compatibility, demonstrating strong performance against task-specific and systemic risks.
   - **Year**: 2025

5. **Title**: RSafe: Incentivizing Proactive Reasoning to Build Robust and Adaptive LLM Safeguards (arXiv:2506.07736)
   - **Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua
   - **Summary**: RSafe proposes an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within specified safety policies. It operates in two stages: guided reasoning and reinforced alignment, enabling generalization over unseen or adversarial safety violation scenarios.
   - **Year**: 2025

6. **Title**: LLM Access Shield: Domain-Specific LLM Framework for Privacy Policy Compliance (arXiv:2505.17145)
   - **Authors**: Yu Wang, Cailing Cai, Zhihua Xiao, Peifung E. Lam
   - **Summary**: LLM Access Shield introduces a security framework to enforce policy compliance and mitigate risks in LLM interactions. It features LLM-based policy enforcement, dynamic policy customization, and sensitive data anonymization, effectively mitigating security risks while preserving functional accuracy.
   - **Year**: 2025

7. **Title**: Lifelong Safety Alignment for Language Models (arXiv:2505.20259)
   - **Authors**: Haoyu Wang, Zeyu Qin, Yifei Zhao, Chao Du, Min Lin, Xueqian Wang, Tianyu Pang
   - **Summary**: This paper proposes a lifelong safety alignment framework that enables LLMs to continuously adapt to new and evolving jailbreaking strategies. It introduces a competitive setup between a Meta-Attacker and a Defender, progressively improving robustness and reducing attack success rates.
   - **Year**: 2025

8. **Title**: GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning (arXiv:2406.09187)
   - **Authors**: Zhen Xiang, Linzhi Zheng, Yanjie Li, Junyuan Hong, Qinbin Li, Han Xie, Jiawei Zhang, Zidi Xiong, Chulin Xie, Carl Yang, Dawn Song, Bo Li
   - **Summary**: GuardAgent introduces an LLM agent as a guardrail to other LLM agents. It oversees a target LLM agent by checking whether its inputs/outputs satisfy user-defined guard requests, comprising task planning and guardrail code generation.
   - **Year**: 2024

9. **Title**: Learning Safety Constraints for Large Language Models (arXiv:2505.24445)
   - **Authors**: Xin Chen, Yarden As, Andreas Krause
   - **Summary**: This paper proposes SaP (Safety Polytope), a geometric approach to LLM safety that learns and enforces multiple safety constraints directly in the model's representation space. It enables detection and correction of unsafe outputs through geometric steering without modifying model weights.
   - **Year**: 2025

10. **Title**: Almost Surely Safe Alignment of Large Language Models at Inference-Time (arXiv:2502.01208)
    - **Authors**: Xiaotong Ji, Shyam Sundhar Ramesh, Matthieu Zimmer, Ilija Bogunovic, Jun Wang, Haitham Bou Ammar
    - **Summary**: This paper introduces InferenceGuard, an inference-time alignment approach that ensures LLMs generate safe responses almost surely. It frames safe generation as a constrained Markov decision process within the LLM's latent space, providing formal safety guarantees without modifying model weights.
    - **Year**: 2025

**Key Challenges:**

1. **Adaptability to Evolving Policies**: Ensuring that LLM guardrails can dynamically adapt to new regulations, emerging threats, and context-specific safety requirements without necessitating model retraining.

2. **Real-Time Monitoring and Validation**: Developing efficient mechanisms for real-time monitoring and validation of LLM outputs to promptly detect and mitigate unsafe or non-compliant content.

3. **Balancing Safety and Performance**: Maintaining a balance between enforcing strict safety measures and preserving the functional accuracy and utility of LLM applications.

4. **Generalization Across Domains**: Creating guardrails that are effective across diverse applications and domains, each with unique safety and compliance requirements.

5. **Interpretability and Transparency**: Ensuring that the decision-making processes of policy enforcers are interpretable and transparent to build trust and facilitate compliance audits. 
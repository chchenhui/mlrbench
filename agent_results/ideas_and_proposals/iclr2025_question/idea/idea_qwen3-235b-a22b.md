1. **Title**: *Self-Limited Reasoning: Uncertainty-Guided Knowledge Disengagement for Mitigating Hallucinations in Autoregressive Models*  

2. **Motivation**:  
Large autoregressive models often hallucinate due to their latent knowledge retrieval mechanism, which conflates factual recall with creative generation. Existing methods to detect hallucinations rely on external knowledge sources or post-hoc checks, which are computationally costly and incompatible with real-time generation. This work addresses the critical gap of *controlling knowledge retrieval during decoding itself* to suppress unfounded statements while preserving the model's expressive power.  

3. **Main Idea**:  
We propose a self-referential mechanism to dynamically disengage knowledge retrieval during generation. Specifically:  
- Train a lightweight, auxiliary network to predict per-token uncertainty via entropy bounds derived from contrastive input perturbations (e.g., masking, paraphrasing).  
- Use these uncertainty scores to gate the contribution of internal knowledge: when uncertainty exceeds a context-adaptive threshold, the model transitions into a *fact-free generative mode*, prioritizing linguistic coherence over knowledge retrieval.  
- Introduce a hybrid training objective combining language modeling, uncertainty calibration, and a reinforcement learning component to penalize high-uncertainty sequences while rewarding factuality.  

Expected outcomes include reduced hallucinations in QA and factual generation benchmarks (e.g., TruthfulQA, FActScore) without sacrificing performance on creative tasks (e.g., story writing). The method’s lightweight design ensures scalability, with minimal overhead during inference. Impacting safe deployment in legal/medical domains, this approach redefines hallucination control by directly encoding knowledge limitations into the model’s generation process, rather than relying on external oversight.
Title: LLM-Guided Tactic Autogeneration for Interactive Theorem Provers

Motivation:  
Interactive theorem proving (ITP) demands intensive manual tactic engineering, creating bottlenecks in formalizing large-scale mathematical libraries and verified software. Automating tactic discovery can dramatically accelerate proof development and broaden the adoption of formal methods.

Main Idea:  
We propose a two-stage framework, LLM-TAC, that leverages fine-tuned large language models to generate and refine proof tactics for systems like Coq or Lean.  
1. Contextual Encoding: For each proof obligation, we encode the goal state, local hypotheses, and project libraries using a retrieval-augmented transformer.  
2. Tactic Generation & Verification: The LLM proposes candidate tactic sequences, which are mechanically executed inside the prover. Successful sequences that close subgoals are logged as new training data; failing ones generate counter-examples.  
3. Reinforcement Loop: We apply reinforcement learning from proof feedback to iteratively improve generation accuracy.  

Expected Outcomes:  
• 50% reduction in manual tactic writing on standard Coq benchmarks (mathcomp, stdlib)  
• Public release of trained models and scripts for integration  

Potential Impact:  
By fusing probabilistic generation with formal verification checks, LLM-TAC can substantially lower the barrier to ITP, paving the way toward scalable, AI-driven proof engineering.
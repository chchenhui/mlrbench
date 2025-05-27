Title  
LLM-Guided Tactic Autogeneration for Interactive Theorem Provers (LLM-TAC)

1. Introduction  
Background  
Interactive theorem provers (ITPs) such as Coq and Lean provide a rigorous framework for developing formally verified proofs in mathematics and software. Building large-scale formal libraries (e.g., Mathematical Components in Coq or mathlib in Lean) requires engineering proof scripts by hand—writing sequences of tactics that gradually transform proof goals into trivially solvable subgoals. While formal methods guarantee correctness by construction, they struggle with the “last‐mile” puzzle of human effort: crafting tactics for each new theorem is labor‐intensive and demands deep expertise in the proof assistant’s logic and tactics language.

In parallel, large language models (LLMs) have exhibited impressive proficiency in natural language generation, code synthesis, and—recently—generating proof steps in ITPs. Works such as LeanDojo (Yang et al., 2023), LLMSTEP (Welleck & Saha, 2023), COPRA (Thakur et al., 2023), and Lean Copilot (Song et al., 2024) demonstrate that retrieval‐augmented transformers can suggest proof steps or complete subgoals. However, these systems are primarily synchronous assistants: they suggest individual proof steps but do not autonomously generate full tactic scripts, nor do they close the loop via reinforcement learning guided by proof success or failure.

Research Objectives  
We propose LLM-TAC, a two‐stage framework for LLM-guided tactic autogeneration in ITPs. Our objectives are:  
• To design a contextual encoding scheme that captures proof states (goals, local hypotheses, library context) in a format digestible by LLMs.  
• To fine-tune and deploy an LLM that emits complete tactic sequences for Coq and/or Lean proof obligations.  
• To close the loop via reinforcement learning from proof‐execution feedback, treating successful proof closures as high‐reward events.  
• To integrate LLM-TAC seamlessly with Coq and Lean, providing a developer‐friendly plugin.  
• To evaluate LLM-TAC on standard benchmarks (e.g., mathcomp, stdlib) and demonstrate at least a 50% reduction in manual tactic writing.

Significance  
By marrying probabilistic generation with deterministic proof checking, LLM-TAC aims to democratize formal verification. A robust AI‐driven tactic generator will lower the bar for newcomers, accelerate proof library development, and pave the way toward large‐scale, machine‐assisted formal mathematics and software verification.

2. Methodology  
Our methodology comprises four components: (A) Data Collection & Preprocessing, (B) Contextual Encoding, (C) Tactic Generation & Verification, and (D) Reinforcement Learning Loop.  

A. Data Collection & Preprocessing  
1. Proof Corpus Assembly  
   • Collect proof scripts from Coq’s standard library (stdlib), Mathematical Components (mathcomp), and prominent Lean libraries (mathlib), totaling over 200 K theorems.  
   • Extract proof obligations: for each tactic invocation in a script, record the proof state (goal + hypotheses) before invocation and the subsequent proof state(s) after.  
   • Partition the dataset into train (80%), validation (10%), and test (10%) splits at the theorem level, ensuring no overlap of theorems.  

2. Serialization & Tokenization  
   • Serialize goals and contexts into a linearized format using the tactic language’s concrete syntax.  
   • Apply a subword tokenizer (e.g., Byte-Pair Encoding with vocabulary size ~50 K tokens) adapted to Coq/Lean syntax (keywords, identifiers, punctuation).  

B. Contextual Encoding  
We represent each proof obligation as an ordered tuple  
$$s = (\mathit{goal}, \{\mathit{hyp}_i\}_{i=1}^H, \mathit{context\_snippets}),$$  
where “context_snippets” are retrieved lemmas and definitions relevant to the goal.  

1. Retrieval‐Augmented Context  
   • Train a dual‐encoder retriever (as in DPR) over the proof corpus. Given a goal $g$, retrieve the top-$k$ most semantically similar proof contexts.  
   • Let $\mathcal{R}(g) = \{c_1,\dots,c_k\}$ be the retrieved snippets.  

2. Transformer Encoder Input  
   • Concatenate the serialized goal, local hypotheses, and retrieved snippets with special separator tokens:  
     [CLS] goal [SEP] hyp_1 [SEP] … [SEP] hyp_H [CTX] c_1 [SEP] … [SEP] c_k [EOS].  
   • Feed the sequence into a pretrained LLM (e.g., Code-T5 or GPT-3.5), then fine‐tune end‐to‐end.

C. Tactic Generation & Verification  
1. Sequence Generation  
   • Given encoded state $s$, the model $\pi_\theta$ generates a sequence of tactics $\tau = (t_1, \dots, t_L)$ via left‐to‐right sampling with temperature $T$ and top-$p$ filtering.  

2. Mechanical Proof Execution  
   • Invoke the ITP’s tactic interpreter (e.g., Coq’s “vernac” or Lean’s “elab”) to apply $(t_1,\dots,t_L)$ to $s$.  
   • Observe the result:  
     – Success if all subgoals are closed.  
     – Partial success if some subgoals remain.  
     – Failure if syntax or type errors occur.  
   • Extract reward $R(\tau)$:  
     $$R(\tau) = 
       \begin{cases}
         1.0, & \text{if full proof closure};\\
         \alpha \cdot \frac{\#\text{closed\_subgoals}}{\#\text{initial\_subgoals}}, & \text{if partial};\\
         0, & \text{on failure},
       \end{cases}$$  
   where $\alpha\in(0,1)$ (e.g., $\alpha=0.5$) trades off partial progress.  

D. Reinforcement Learning from Proof Feedback  
We frame tactic generation as a policy optimization problem. Let $\pi_\theta(\tau\,|\,s)$ be our model’s probability of generating $\tau$ in state $s$. We seek to maximize expected reward  
$$J(\theta)=\mathbb{E}_{s\sim\mathcal{D}}\bigl[\mathbb{E}_{\tau\sim\pi_\theta(\cdot|s)}[R(\tau)]\bigr].$$  
Using the REINFORCE algorithm, the policy gradient is  
$$\nabla_\theta J(\theta) 
  = \mathbb{E}_{s,\tau}\Bigl[R(\tau)\,\nabla_\theta\log\pi_\theta(\tau|s)\Bigr].$$  
In practice we apply variance reduction with a learned baseline $b(s)$, so that updates are proportional to $[R(\tau)-b(s)]\nabla_\theta\log\pi_\theta(\tau|s)$.  

Training Protocol  
• Stage 1: Supervised Fine‐Tuning. Train $\pi_\theta$ on $(s,\tau^*)$ pairs extracted from human proofs via maximum likelihood.  
• Stage 2: Reinforcement Learning. Roll out $\pi_\theta$ on held‐out states, collect $(s,\tau,R(\tau))$ tuples, and update via policy gradient. Alternate batches of supervised and RL updates to stabilize training.  

Implementation Details  
• Model backbone: a transformer with 12 layers, 12 attention heads, hidden size 768 (≈110 M parameters), initialized from Code-T5 or GPT-Neo.  
• Optimizer: AdamW, learning rate $1e$-5 (fine-tuning), $5e$-6 (RL), warmup 1 K steps, linear decay.  
• Batch size: 16 sequences per GPU; training on 8 A100 GPUs.  

Experimental Design & Evaluation Metrics  
Benchmarks  
• Coq mathcomp benchmark: 10 K theorems not seen during training.  
• Coq stdlib: 5 K hold‐out theorems.  
• Lean mathlib subset: 3 K theorems.  

Baselines  
• Human‐written tactics (oracle).  
• LLMSTEP (Welleck & Saha, 2023).  
• COPRA (Thakur et al., 2023).  
• Pure supervised model (no RL).  

Metrics  
1. Proof Success Rate (PSR): fraction of theorems fully closed by $\tau$.  
2. Partial Closure Rate (PCR): fraction with ≥ 50% subgoals solved.  
3. Average Tactic Length (ATL): mean number of tactic steps in $\tau$.  
4. Manual Intervention Reduction (MIR): relative decrease in human‐written tactics required when using LLM-TAC.  
   $$\mathrm{MIR} = 1 - \frac{\mathrm{steps\_with\_LLM\_assist}}{\mathrm{steps\_manual}}.$$  
5. Wall‐Clock Time Savings (WTS): time to complete proofs with/without LLM-TAC.  

Statistical Analysis  
• Conduct paired t-tests on PSR and MIR across 10 random splits to assess significance ($p<0.01$).  
• Ablation studies:  
   – Without retrieval (§B),  
   – Without RL (§D),  
   – Varying reward parameter $\alpha$.  

3. Expected Outcomes & Impact  
We anticipate the following outcomes:  
1. A trained LLM-TAC model that achieves ≥ 50% PSR on held‐out Coq benchmarks and ≥ 35% on Lean benchmarks, surpassing existing baselines by 2–3×.  
2. Demonstrated MIR of at least 50% on mathcomp and stdlib: proof engineers write half as many tactics manually.  
3. Open‐source release of LLM-TAC code, pretrained models, proof‐execution harness, and benchmark splits.  

Broader Impact  
• Scalability: By automating low‐level tactic engineering, LLM-TAC can accelerate the construction of large formal libraries, unlocking formal verification for complex mathematical theorems and critical software.  
• Accessibility: Novice users can rely on AI‐driven suggestions to learn tactic patterns and proof strategies, lowering the expertise barrier.  
• Synergy of Formal and Probabilistic Methods: LLM-TAC embodies the workshop’s theme “AI Verification in the Wild” by combining probabilistic generation with formal proof checking, yielding “correctness by feedback.”  
• Foundations for Future Research: The dataset, code, and interface will facilitate follow‐on work in LLM‐based formal methods, including richer proof automation, interactive copilots, and integration with SMT solvers.  

In summary, LLM-TAC will demonstrate that generative AI can be harnessed, verified, and improved via formal proof feedback—paving the way toward AI‐assisted theorem proving at scale.
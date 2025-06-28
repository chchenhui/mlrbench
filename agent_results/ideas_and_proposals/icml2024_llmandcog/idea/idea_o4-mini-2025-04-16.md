Title: Dynamic Curriculum Benchmark for Emergent Planning and Theory-of-Mind in LLMs

Motivation:  
Existing static benchmarks fail to capture how LLMs adapt to progressively complex cognitive tasks such as multi-step planning or theory-of-mind reasoning. Without an adaptive evaluation framework, we cannot pinpoint the emergence thresholds of these abilities or compare models on a level playing field.

Main Idea:  
We propose a “Dynamic Curriculum Benchmark” (DCB) that algorithmically generates sequences of tasks in planning, navigation, and theory-of-mind, scaling difficulty based on an LLM’s previous performance. Using reinforcement‐learning-based task samplers, DCB will:  
1. Start with simple 2-step planning puzzles or first-person navigation prompts.  
2. Monitor LLM success rates and unlock more complex multiagent scenarios (e.g., predicting another agent’s beliefs in a story world).  
3. Record performance trajectories to estimate emergence points for each cognitive skill.  
4. Integrate human-in‐the-loop audits to validate automatic scoring and edge‐case behaviors.  

Expected outcomes include fine‐grained cognitive profiles for LLMs, clearer comparisons between fine‐tuned vs. modular architectures, and actionable insights for designing models that robustly acquire higher‐order reasoning and social cognition.
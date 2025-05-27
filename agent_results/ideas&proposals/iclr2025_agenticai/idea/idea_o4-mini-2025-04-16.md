Title: DecompAI: Multi-Agent Decomposition Framework for Automated Hypothesis Generation

Motivation:  
Existing AI-driven hypothesis generators often produce unfocused or generic suggestions due to monolithic architectures lacking domain specialization. By decomposing the hypothesis generation process into specialized agents, we can improve the relevance, testability, and novelty of proposed scientific hypotheses.

Main Idea:  
We propose a modular multi-agent system where dedicated agents handle domain exploration, knowledge retrieval, inferential reasoning, and experimental validation. Agents share a dynamic knowledge graph and employ game-theoretic utility functions to balance cooperation and divergence. Each agent is fine-tuned on domain-specific corpora to enhance expertise in, for example, chemistry or genetics. The framework is evaluated on chemical synthesis and genetic pathway discovery benchmarks, assessing hypothesis novelty, scientific validity, and resource efficiency. We anticipate improved hypothesis quality, reduced hallucination, and transparent reasoning chains, enabling faster, more reliable scientific discovery with integrated human oversight.
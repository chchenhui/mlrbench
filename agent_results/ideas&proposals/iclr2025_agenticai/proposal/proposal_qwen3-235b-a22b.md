# **DecompAI: A Game-Theoretic Multi-Agent Framework for Scientific Hypothesis Generation**

## **1. Introduction**

### **Background**
Scientific discovery has traditionally relied on iterative cycles of hypothesis generation, experimentation, and validation—a process demanding substantial time, resources, and domain expertise. While artificial intelligence (AI) has accelerated data analysis and modeling in fields like drug discovery and materials science, automating hypothesis generation remains a nascent yet transformative frontier. Recent efforts, such as AstroAgents [1], VirSci [2], and SciAgents [3], demonstrate the potential of multi-agent systems to mimic collaborative human scientific teams. However, these frameworks often treat agents as loosely coupled entities lacking structured domain specialization, leading to redundant suggestions, generic hypotheses, and challenges in validating output novelty.

For instance, AstroAgents applied eight heterogeneous agents to generate hypotheses from astrobiology data but struggled with balancing novel vs. plausible suggestions [1]. Similarly, VirSci demonstrated improved scientific idea generation through agent collaboration but lacked domain-specific fine-tuning mechanisms to enhance hypothesis relevance [2]. These limitations highlight a critical gap: **monolithic architectures fail to decompose hypothesis generation into specialized phases**, such as domain exploration, knowledge synthesis, logic inference, and experimental feasibility. Additionally, existing systems lack formalized coordination mechanisms to manage conflicting agent objectives.

### **Research Objectives**
This work introduces **DecompAI**, a formally structured multi-agent framework addressing the following objectives:

1. **Specialized Decomposition**: Partition hypothesis generation into modular phases managed by domain-specific agents.
2. **Optimized Coordination**: Leverage game-theoretic utility functions to balance agent cooperation/divergence.
3. **Dynamic Knowledge Graphs**: Implement in-situ learning over ontological knowledge graphs to ensure evidence-based hypothesis refinement.
4. **Cross-Domain Validation**: Evaluate performance on **chemical synthesis planning** (chem) and **genetic pathway discovery** (gen).

### **Significance**
DecompAI has the potential to:
- Accelerate discovery cycles by automating hypothesis generation without sacrificing domain fidelity.
- Reduce hallucination through structured evidence retrieval and explainability mechanisms.
- Serve as an open framework for extending hypothesis generation to other domains (e.g., materials science, cosmology).
- Enable new research into agent coordination strategies for AI-driven science.

---

## **2. Methodology**

### **2.1 Framework Architecture**
DecompAI comprises four interdependent agents (Figure 1):
1. **Domain Explorer (DE)**: Surveys domain knowledge (e.g., molecule properties in chem, gene interaction networks in gen) to identify underexplored areas.
2. **Knowledge Retriever (KR)**: Extracts relevant evidence from curated ontologies (e.g., PubChem for chem, Reactome for gen) to support hypothesis viability.
3. **Inferential Reasoner (IR)**: Applies logical/mathematical inference to generate candidate hypotheses (e.g., predicting product yields, gene regulatory relationships).
4. **Validation Planner (VP)**: Quantifies experimental feasibility (cost/time/resources) and identifies falsification criteria.

Agents communicate through a **dynamic global knowledge graph (GKG)** updated after each iteration. The GKG stores nodes for:
- Entities: chemicals, genes, reactions, pathways.
- Relationships: chemical similarity, reaction mechanisms, gene expression modulations.
- Metadata: citations, experimental conditions, error margins.

### **2.2 Agent Coordination via Game Theory**
To synthesize agents' diverse objectives, we model their interaction as an **asymmetric Nash bargaining game** [4], where each agent derives utility from:
- **DE**: Maximizing the novelty of explored regions.
- **KR**: Ensuring the factual support of hypotheses.
- **IR**: Prioritizing logico-mathematical coherence.
- **VP**: Favoring hypotheses with high validation probability.

We define agent utilities as:
$$
U_i = \alpha_i \cdot N_i + \beta_i \cdot V_i \cdot (1 - C_i)
$$
where:
- $N_i$: Novelty score (Jaccard distance from prior knowledge).
- $V_i$: Viability (e.g., whether IR has supporting logical deductions).
- $C_i$: Validation cost (normalization of time/resources estimated by VP).
- $\alpha_i, \beta_i$: Tunable agent-specific coefficients.

Agents negotiate utilities iteratively:
1. **Initialization**: DE proposes a seed hypothesis $H_0$ (e.g., unmapped chemical reaction).
2. **KR augments** $H_0$ with contextual evidence.
3. **IR** generates candidate modifications $H_1, H_2, \dots$ satisfying GKG constraints.
4. **VP** computes $\forall H_j: P(\text{successful validation}), E[\text{cost}]$.
5. Agents select the hypothesis $H^*$ maximizing the **joint utility**:
$$
H^* = \arg\max_{H_j} \left( \sum_{i \in \mathcal{A}} \log U_i(H_j) \right)
$$
where $\mathcal{A} = \{\text{DE}, \text{KR}, \text{IR}, \text{VP}\}$.

This formulation ensures no single agent dominates the hypothesis space. DE-KR interactions reward well-supported yet novel discoveries, while IR-VP balances theoretical plausibility against empirical testability.

### **2.3 Domain-Specific Model Design**
Each agent is a domain-specialized **hybrid model**:
- **KR, DE**: BERT-based language models fine-tuned on domain corpora (e.g., Elsevier Chemistry Journals for chem, OMIM genomic literature for gen).
- **IR**: Symbolic transformer integrating domain rules (e.g., retrosynthetic rules in chem, constraint-based models in gen).
- **VP**: Graph neural network (GNN) trained on historical experiment data from platforms like Reaxys (chem) or KEGG (gen).

Key workflow:
1. *KR initializes* the GKG by extracting triples like (“Ethylene oxide”, “REACTION”, “Propylene carbonate”) from domain texts.
2. *DE explores* GKG using node2vec embeddings [5] and identifies high-uncertainty regions (e.g., reactions with unknown mechanisms, unexplored transcription factor interactions).
3. *IR infers* relationships through probabilistic logical rules. For chem:
$$
P(\text{Reactivity}) = \sigma(\mathbf{W} \cdot [\text{Fukui}_u^{+}, \text{Fukui}_e^{-}] + b)
$$
where $\sigma$ is the sigmoid function, and Fukui indices quantify electron density at reactive sites.
4. *VP quantifies* costs via GNN predictions of DFT-level accuracy or qPCR effort, respectively.

### **2.4 Datasets and Experimental Design**
#### **Datasets**
- **Chemical Synthesis**:
  - Training: USPTO reactions (1M+ entries), Reaxys experimental parameters.
  - Evaluation: Challenging retrosynthesis cases (e.g., Enamine REAL space).
- **Genetic Pathways**:
  - Training: KEGG pathways (500K+ interactions), PhosphoSitePlus phospho-proteomics.
  - Evaluation: Novel SARS-CoV-2 host factor predictions validated by CRISPR screens.

#### **Baselines**
1. Single-agent LLM (e.g., GPT-4 across domains).
2. AstroAgents (asymmetric agent prioritization, no game theory) [1].
3. VirSci (agent debate framework) [2].

#### **Metrics**
1. **Novelty**: 
   - Jaccard distance vs. known hypothesis space ($N = |H^* \cap \text{GKG}| / |H^* \cup \text{GKG}|$).
   - Citation burst detection [6].
2. **Validity**:
   - Domain-specific plausibility checks (e.g., DFT for chem, motif analysis for gen).
   - Cross-verification against public repositories (PubChem, STRINGdb).
3. **Cost**: 
   - Resource estimation normalized to a drug screening benchmark (1M compounds).
4. **Hallucination Rate**:
   - % of unsupported claims detected via Knowledge Graph Data Validation [7].
5. **Human Evaluation** (30 domain scientists):
   - Likert scale ratings for hypothesis quality, readability, and usefulness.

---

## **3. Expected Outcomes & Impact**

### **3.1 Technical Innovations**
1. **First Integration of Game Theory in Scientific Agents**: Introducing Nash bargaining to resolve conflicting agent incentives while preserving domain alignment.
2. **Hybrid Symbolic-Subsymbolic Agents**: Combining interpretable rule-based inference (IR) with statistical GNNs/Vision LLMs.
3. **Cross-Domain Generalization Toolkit**: Demonstrating a framework deployable to chemistry (this work), genetics, and physics with minimal retraining.

### **3.2 Scientific Impact**
We anticipate DecomAI will:
- Reduce discovery cycles by 5–10x through automated hypothesis enumeration.
- Generate 20–30% more testable hypotheses than virSci or AstroAgents.
- Uncover ≥5 unexpected genotype-phenotype relationships in evaluation datasets.

Example breakthrough: In chem, identifying overlooked catalysts for CO₂ conversion through reactant hardness criteria overlooked by human experts.

### **3.3 Societal Impact**
- **Accelerate Interdisciplinary Research**: Lower the barrier for combining domain-specific knowledge (e.g., linking chemical designer drugs with neurogenomics).
- **Combat AI Hallucination**: Structured GKG-based inference mitigates spurious claims.
- **Enable Open Science**: Publishing GKG updates and agent code for transparent hypothesis generation.

---

## **4. Challenges & Mitigation Strategies**

| **Challenge**                     | **DecompAI Solution**                                                  |
|-----------------------------------|--------------------------------------------------------------------------|
| **Agent Coordination Complexity** | Game-theoretic optimization ensures convergence of diverse utility functions. |
| **Domain Adaptation Costs**       | Transfer learning from foundation models (e.g., BioBERT for gen).       |
| **Scalability of GKG**            | Dynamic pruning of low-certainty nodes, distributed vector stores.      |
| **Ethical Risk (Automation Bias)**| Human-in-the-loop workflows that allow experts to override agent outputs. |

---

## **5. Timeline**

| **Phase 1 (0–6 months)**       | Agent implementation, integration with KG base models.                  |
| **Phase 2 (6–12 months)**      | Training, domain fine-tuning, optimization through synthetic benchmarks. |
| **Phase 3 (12–18 months)**     | Real-world evaluations with domain experts (e.g., Novartis chemists).     |
| **Phase 4 (18–24 months)**     | Publish results, open-source tools, establish community benchmarks.     |

---

## **6. Conclusion**
DecompAI reimagines hypothesis generation as a cooperative yet specialized process, where contextualized agents integrate diverse scientific workflows into a unified feedback loop. By anchoring decomposition in game-theoretic coordination and dynamic knowledge graphs, we bridge the gap between creative exploration and rigorous validation—a critical step toward trustworthy AI-driven science. The proposed framework will serve as a blueprint for next-generation research platforms, empowering scientists to harness AI without compromising precision or innovation.

---

**References**  
[1] Saeedi et al., *AstroAgents: Multi-Agent Hypothesis Generation from Mass Spectrometry Data* (2025).  
[2] Su et al., *VirSci: Multi-Agent Scientific Idea Generation* (2024).  
[3] Ghafarollahi and Buehler, *SciAgents: Multi-Agent Intelligent Graph Reasoning* (2024).  
[4] Ghafarollahi et al., *Game-Theoretic Trust Region Optimization* (2021).  
[5] Grover and Leskovec, *node2vec: Scalable Features for Networks* (KDD 2016).  
[6] Chen et al., *CiteSpace: Visualizing Emerging Trends* (2018).  
[7] Wang et al., *KG Validation Techniques* (2023).
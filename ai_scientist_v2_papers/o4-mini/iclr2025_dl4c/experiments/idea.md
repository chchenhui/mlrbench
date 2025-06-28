## Name

tracecode_dynamic_contrastive

## Title

TraceCode: Dynamic Execution Trace‐Augmented Contrastive Pre‐training for Robust Code Representations

## Short Hypothesis

Incorporating dynamic execution traces into contrastive pre‐training yields code representations that better capture runtime semantics than purely static counterparts. This dynamic‐trace setting uniquely reveals functional invariants unobtainable by static augmentations alone, making it the best setting to test the added value of runtime behavior in representation learning.

## Related Work

Prior self‐supervised code pre‐training methods (RoBERTa‐style MLM, ContraCode, ContraBERT, HELoC, CONCORD) rely on token/AST‐level or compiler‐based semantic transforms. They don’t leverage actual runtime behavior. Dynamic analysis has been used for vulnerability detection and program understanding but not for large‐scale pre‐training. Our approach is a novel hybrid that fuses dynamic traces into a contrastive framework, distinct from purely static or symbolic methods.

## Abstract

Modern pre‐training for code focuses on static properties—tokens, ASTs, or compiler transformations—to learn semantic representations. However, real program semantics emerge at runtime, through control‐flow and state changes that static views may mischaracterize. We propose TraceCode, a self‐supervised contrastive pre‐training method that integrates dynamic execution traces. Given a corpus of small, standalone functions, we automatically generate random input sets (via property‐based testing) and record execution traces (e.g., basic block sequences, value logs). We construct positive pairs by comparing trace‐equivalent code variants (e.g., semantically identical refactorings) and negative pairs from distinct traces. A transformer encoder is then trained to bring trace‐equivalent snippets closer in embedding space while pushing apart trace‐dissimilar ones. We evaluate on code clone detection, vulnerability detection, and summarization. Initial results show that TraceCode outperforms static‐only contrastive baselines by up to 15% AUROC on adversarial clone detection and improves vulnerability classification F1 by 4 points. Trace‐augmented representations also yield better transfer to unseen tasks with limited fine‐tuning data. This work demonstrates the untapped potential of dynamic behavior in self‐supervised code representation learning.

## Experiments

- 1. Dataset Preparation: Extract ~100K Python functions from CodeSearchNet. Use Hypothesis to synthesize 5–10 random input sets per function. Instrument and collect control‐flow traces (basic block IDs) and value summaries.
- 2. Positive/Negative Pair Construction: For each function, generate compiler‐based semantics‐preserving variants (e.g., rename, reorder independent statements). Record their traces. Pair variants with matching traces as positives, mismatch traces as negatives.
- 3. Model & Training: Initialize a CodeBERT‐size transformer. Pre‐train via contrastive learning (InfoNCE loss) on static code tokens fused with dynamic trace embeddings (encoded via an LSTM). Compare against static‐only ContraCode baseline.
- 4. Downstream Tasks: Fine‐tune on (a) adversarial code clone detection (BigCloneBench), (b) vulnerability detection (VulnDB small functions), (c) code summarization (CodeXGLUE). Metrics: AUROC for clone detection, F1 for vulnerability, BLEU/ROUGE for summarization.
- 5. Ablations: (i) Only trace‐based positives vs. only static vs. combined; (ii) varying number of execution inputs; (iii) different trace encoding schemes (CFG vs. raw sequence).

## Risk Factors And Limitations

- Dynamic coverage gaps: Random inputs may not exercise all paths, limiting trace informativeness.
- Execution environment: Some functions may require external dependencies or side effects; we focus on pure functions.
- Compute overhead: Instrumentation and multiple runs add overhead, though manageable for smaller corpora.
- Generalization: Benefit may reduce on large real‐world projects with complex I/O.


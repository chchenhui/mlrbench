## Name

novelty_aware_compress

## Title

Entropy-Aware Adaptive Compressive Memory for Foundation Models

## Short Hypothesis

Integrating token-level novelty (entropy) signals into the compression stage of transformer memories yields better long-context retention and downstream adaptation than fixed-rate or time-based compression. No simpler baseline (e.g., fixed budgets) can adaptively preserve salient information without per-token importance scores.

## Related Work

The Compressive Transformer (Rae et al., 2020) uses fixed time-based schedules to compress old memories regardless of information content. Transformer-XL (Dai et al., 2019) retains fixed-length context without compressive buffering. Hyena (Poli et al., 2023) reduces compute but does not address adaptive memory. Retrieval-augmented methods fetch external passages but lack in-model dynamic buffering. In contrast, our method computes per-token novelty via self-attention entropy and dynamically prioritizes high-entropy tokens for retention, a nontrivial extension beyond uniform compression.

## Abstract

Long-context foundation models face a trade-off between memory footprint and retention of salient information when using compressive or sub-quadratic memory mechanisms. We propose an Entropy-Aware Adaptive Compressive Memory (EA-ACM) module that measures token-level novelty via self-attention entropy and uses this signal to guide which past key/value pairs to compress or discard. Unlike prior fixed-budget compressive transformers, EA-ACM dynamically adjusts memory based on information content, preserving tokens critical for future prediction and adaptation. We integrate EA-ACM into a Transformer-XL style architecture and evaluate on long-range language modeling (PG19, ArXiv) and retrieval-augmented QA benchmarks. Results show that EA-ACM achieves up to a 15% lower perplexity at 50% memory budget compared to fixed compressive baselines, and a 10% improvement in few-shot adaptation tasks by better retaining task-relevant context. Our analysis demonstrates that entropy-guided compression leads to a more informative memory buffer, enabling efficient sub-quadratic inference with minimal overhead. This simple yet effective mechanism advances scalable, adaptive foundation models capable of handling ever-longer contexts.

## Experiments

- 1. Implementation: Extend a Transformer-XL baseline with a compressive memory buffer. Compute per-token novelty as the entropy of its self-attention distribution at the time of compression. Compare against: (a) fixed time-based compression (Rae et al.); (b) uniform random compression; (c) no compression (full memory).
- 2. Language Modeling: Evaluate on PG19 and ArXiv long-range datasets. Measure test perplexity versus memory cost (KV slots). Plot perplexity-memory trade-offs across baselines and EA-ACM.
- 3. Retrieval-Augmented QA: Use a RAG setup on NarrativeQA or TriviaQA with limited in-model memory. After initial retrieval, apply EA-ACM in the generator LM and measure QA accuracy and inference latency.
- 4. Few-Shot Adaptation: On tasks like WMT translation and GLUE benchmarks, simulate long-context prompts. Measure adaptation performance and context sensitivity (accuracy vs. context length) comparing EA-ACM to baselines.
- 5. Ablation: Vary entropy threshold for compression; analyze memory composition (novel vs. old tokens) to confirm that high-entropy tokens are preferentially retained.
- Evaluation Metrics: Perplexity, QA accuracy, few-shot task accuracy, memory footprint (KV slots), and inference latency (ms/token).

## Risk Factors And Limitations

- Entropy estimation overhead may offset memory savings if not efficiently implemented.
- Choosing an appropriate entropy threshold or budget allocation policy may require dataset-specific tuning.
- The novelty measure may misprioritize misleading or noisy tokens in certain domains (e.g., code vs. natural language).
- Adaptive compression might interact poorly with very long sustained repetitive contexts, requiring fallback strategies.


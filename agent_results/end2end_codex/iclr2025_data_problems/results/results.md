# Experiment Results

## Experimental Setup
- Dataset: AG News train split (50 samples)
- Retrieval model: SentenceTransformer all-MiniLM-L6-v2
- FAISS index type: IndexFlatL2
- Top-k retrieval (k=5)
- Retrieval attribution: MRIA (linear utility) vs Leave-One-Out baseline
- Number of queries: 10

## Results Summary

Table 1: Correlation between MRIA retrieval attribution and LOO baseline
| Query ID | Kendall τ | Spearman ρ |
|----------|-----------|------------|
| 0        | 1.00      | 1.00       |
| 1        | 1.00      | 1.00       |
| 2        | 1.00      | 1.00       |
| 3        | 1.00      | 1.00       |
| 4        | 1.00      | 1.00       |
| 5        | 1.00      | 1.00       |
| 6        | 1.00      | 1.00       |
| 7        | 1.00      | 1.00       |
| 8        | 1.00      | 1.00       |
| 9        | 1.00      | 1.00       |

In all 10 queries, MRIA retrieval attribution (with linear utility) perfectly matches the Leave-One-Out baseline (Kendall 	au = 1.00, Spearman rho = 1.00), as expected for a linear scoring function.

## Figures
- `attr_comp_<i>.png`: Scatter plot of MRIA vs LOO φ_i values for each query (i = 09).
- `correlations.png`: Line plot of Kendall 	au and Spearman rho across queries.

## Discussion
The perfect correlation arises because the utility function (sum of similarity scores) is linear in document contributions, making Shapley values equal to raw similarity scores. This serves as a sanity check for the experimental pipeline.

## Limitations and Future Work
- This prototype only covers retrieval attribution with a trivial linear utility. Future experiments should implement the full MRIA pipeline, including randomized Shapley estimation with sketching and generation attribution via Jacobian sketches.
- Extend evaluation to larger datasets and real RAG scenarios using open-source LLMs and FAISS indices with embeddings from dual-encoder models.
- Compare against additional baselines (KernelSHAP, SIM-Shapley) and report latency/memory overhead.

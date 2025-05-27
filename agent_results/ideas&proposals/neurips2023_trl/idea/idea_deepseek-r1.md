**Title:** Structure-Aware Dual-Stream Pretraining for Robust Tabular Language Models  

**Motivation:** Large language models (LLMs) struggle to interpret complex table structures (e.g., nested headers, sparse relations), leading to errors in tasks like text-to-SQL and data QA. Current tabular pretraining methods prioritize content over explicit structural semantics, limiting their utility in real-world applications involving heterogeneous schemas.  

**Main Idea:** We propose a dual-stream transformer that jointly encodes tabular content *and* structural metadata (e.g., headers, data types, primary/foreign keys). The model employs:  
1. **Content Stream:** Tokenizes cell values with structural position embeddings.  
2. **Structure Stream:** Encodes metadata via learnable schema graphs, connecting columns hierarchically.  
Pretraining tasks include **masked cell recovery** (content), **schema relation prediction** (structure), and **cross-stream alignment** (e.g., matching SQL queries to schema subgraphs). By explicitly modeling table topology, the framework enhances LLMs’ ability to reason over diverse schemas. Expected outcomes include state-of-the-art performance on benchmarks like Spider (text-to-SQL) and WikiTableQuestions, with broader impact on data-driven NLP systems requiring precise table comprehension.
Title: MRIA – Modular Retrieval Influence Attribution for RAG-based Foundation Models

Motivation: As retrieval-augmented generation (RAG) becomes integral to foundation models, attributing outputs to specific data sources is crucial for transparency, copyright compliance, and fair compensation. Current attribution methods falter at RAG scale, hindering trust and data valuation.

Main Idea: We propose MRIA, a two-stage, scalable attribution framework seamlessly integrated into RAG pipelines.  
1. Retrieval Attribution logs retrieval scores and document embeddings during inference, then applies randomized Shapley estimators with sketching to approximate each document’s marginal contribution to the retrieval context.  
2. Generation Attribution uses low-rank Jacobian sketches to estimate gradient-based influence of retrieved tokens on the generated output.  
By combining these modules, MRIA yields end-to-end source scores for every output. We implement MRIA in an Llama-based RAG setup using CountSketch for memory-efficient streaming. Evaluation against leave-one-out ground truth on text and multimodal benchmarks will assess the trade-off between attribution accuracy and runtime. MRIA enables near-real-time, fine-grained data attribution, enhancing model interpretability, supporting data marketplaces, and ensuring compliance in FM deployments.
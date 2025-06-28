1. **Title**: Quadratic Graph Attention Network (Q-GAT) for Robust Construction of Gene Regulatory Networks (arXiv:2303.14193)
   - **Authors**: Hui Zhang, Xuexin An, Qiang He, Yudong Yao, Yudong Zhang, Feng-Lei Fan, Yueyang Teng
   - **Summary**: This paper introduces Q-GAT, a Quadratic Graph Attention Network designed to construct gene regulatory networks (GRNs) from gene expression data. Q-GAT employs a dual attention mechanism and quadratic neurons to enhance robustness against noise, a common issue in gene expression measurements. The model demonstrates superior performance in robustness compared to nine state-of-the-art baselines on E. coli and S. cerevisiae datasets.
   - **Year**: 2023

2. **Title**: DiscoGen: Learning to Discover Gene Regulatory Networks (arXiv:2304.05823)
   - **Authors**: Nan Rosemary Ke, Sara-Jane Dunn, Jorg Bornschein, Silvia Chiappa, Melanie Rey, Jean-Baptiste Lespiau, Albin Cassirer, Jane Wang, Theophane Weber, David Barrett, Matthew Botvinick, Anirudh Goyal, Mike Mozer, Danilo Rezende
   - **Summary**: DiscoGen presents a neural network-based method for inferring gene regulatory networks (GRNs) that can denoise gene expression data and handle interventional data. The model outperforms existing neural network-based causal discovery methods, addressing challenges such as noisy data and large sample sizes inherent in biological datasets.
   - **Year**: 2023

3. **Title**: GCBLANE: A graph-enhanced convolutional BiLSTM attention network for improved transcription factor binding site prediction (arXiv:2503.12377)
   - **Authors**: Jonas Chris Ferrao, Dickson Dias, Sweta Morajkar, Manisha Gokuldas Fal Dessai
   - **Summary**: GCBLANE integrates convolutional, multi-head attention, and recurrent layers with a graph neural network to predict transcription factor binding sites (TFBS). Evaluated on 690 ENCODE ChIP-Seq datasets, it achieved an average AUC of 0.943, outperforming advanced models that utilize multimodal approaches, including DNA shape information.
   - **Year**: 2025

4. **Title**: Analysis of Gene Regulatory Networks from Gene Expression Using Graph Neural Networks (arXiv:2409.13664)
   - **Authors**: Hakan T. Otal, Abdulhamit Subasi, Furkan Kurt, M. Abdullah Canbaz, Yasin Uzun
   - **Summary**: This study explores the use of Graph Neural Networks (GNNs) for modeling gene regulatory networks (GRNs) from gene expression data. Utilizing a Graph Attention Network v2 (GATv2), the model accurately predicts regulatory interactions and identifies key regulators, suggesting that GNNs can address traditional limitations in GRN analysis and offer richer biological insights.
   - **Year**: 2024

**Key Challenges:**

1. **Noise in Gene Expression Data**: High levels of noise in gene expression measurements can hinder the accurate construction of gene regulatory networks, necessitating robust models capable of denoising data.

2. **Capturing Complex Regulatory Interactions**: Modeling the intricate and dynamic nature of gene regulatory networks, including long-range dependencies and context-specific regulations, remains a significant challenge.

3. **Scalability and Computational Efficiency**: Handling large-scale genomic datasets requires models that are both scalable and computationally efficient to process vast amounts of data effectively.

4. **Integration of Multimodal Data**: Effectively combining diverse types of genomic data (e.g., gene expression, epigenetic modifications) to construct comprehensive regulatory networks is complex and challenging.

5. **Interpretability of Models**: Ensuring that models are interpretable and provide biologically meaningful insights is crucial for their adoption in the biomedical field. 
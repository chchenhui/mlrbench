1. **Title**: Extracting Molecular Properties from Natural Language with Multimodal Contrastive Learning (arXiv:2307.12996)
   - **Authors**: Romain Lacombe, Andrew Gaut, Jeff He, David Lüdeke, Kateryna Pistunova
   - **Summary**: This study explores the integration of molecular graph representations with textual descriptions to enhance molecular property prediction. By employing contrastive learning, the authors align graph-based and text-based embeddings, leading to improved performance on MoleculeNet property classification tasks.
   - **Year**: 2023

2. **Title**: Graph Contrastive Learning for Multi-omics Data (arXiv:2301.02242)
   - **Authors**: Nishant Rajadhyaksha, Aarushi Chitkara
   - **Summary**: The paper introduces MOGCL, a framework that applies graph contrastive learning to multi-omics datasets. By pre-training graph models with contrastive methods and fine-tuning them in a supervised manner, MOGCL achieves superior classification performance, demonstrating the efficacy of contrastive learning in integrating multi-omics data.
   - **Year**: 2023

3. **Title**: Multimodal Contrastive Representation Learning in Augmented Biomedical Knowledge Graphs (arXiv:2501.01644)
   - **Authors**: Tien Dang, Viet Thanh Duy Nguyen, Minh Tuan Le, Truong-Son Hy
   - **Summary**: This work presents a multimodal approach that combines embeddings from specialized language models with graph contrastive learning to enhance link prediction in biomedical knowledge graphs. The authors introduce PrimeKG++, an enriched knowledge graph incorporating multimodal data, demonstrating improved generalizability and accuracy in link prediction tasks.
   - **Year**: 2025

4. **Title**: HyperGCL: Multi-Modal Graph Contrastive Learning via Learnable Hypergraph Views (arXiv:2502.13277)
   - **Authors**: Khaled Mohammed Saifuddin, Shihao Ji, Esra Akbas
   - **Summary**: HyperGCL introduces a multimodal graph contrastive learning framework that constructs hypergraph views by integrating graph structure and attributes. The model employs adaptive topology augmentation and network-aware contrastive loss to capture essential characteristics, achieving state-of-the-art node classification performance.
   - **Year**: 2025

5. **Title**: Causal Representation Learning from Multimodal Biological Observations (arXiv:2411.06518)
   - **Authors**: Yuewen Sun, Lingjing Kong, Guangyi Chen, Loka Li, Gongxu Luo, Zijian Li, Yixuan Zhang, Yujia Zheng, Mengyue Yang, Petar Stojanov, Eran Segal, Eric P. Xing, Kun Zhang
   - **Summary**: The authors develop a framework for identifying latent causal variables from multimodal biological data without restrictive parametric assumptions. By leveraging structural sparsity in causal connections across modalities, the approach provides interpretable insights into biological mechanisms, validated through experiments on human phenotype datasets.
   - **Year**: 2024

6. **Title**: Enhancing Multimodal Medical Image Classification using Cross-Graph Modal Contrastive Learning (arXiv:2410.17494)
   - **Authors**: Jun-En Ding, Chien-Chin Hsu, Feng Liu
   - **Summary**: This paper proposes the Cross-Graph Modal Contrastive Learning (CGMCL) framework, which integrates image and non-image data by constructing cross-modality graphs. Utilizing contrastive learning, CGMCL aligns multimodal features in a shared latent space, improving accuracy and interpretability in medical image classification tasks.
   - **Year**: 2024

7. **Title**: TopoGCL: Topological Graph Contrastive Learning (arXiv:2406.17251)
   - **Authors**: Yuzhou Chen, Jose Frias, Yulia R. Gel
   - **Summary**: TopoGCL introduces topological invariance and extended persistence into graph contrastive learning. By extracting latent shape properties of graphs at multiple resolutions, the model enhances unsupervised graph classification performance across various datasets, including biological networks.
   - **Year**: 2024

8. **Title**: Multimodal Contrastive Learning for Spatial Gene Expression Prediction Using Histology Images (arXiv:2407.08216)
   - **Authors**: Wenwen Min, Zhiceng Shi, Jun Zhang, Jun Wan, Changmiao Wang
   - **Summary**: The authors present mclSTExp, a framework that predicts spatial gene expression from histology images using multimodal contrastive learning. By integrating image features with spatial context through a Transformer encoder, the model demonstrates superior performance in predicting spatial gene expression and interpreting cancer-specific genes.
   - **Year**: 2024

9. **Title**: FormNetV2: Multimodal Graph Contrastive Learning for Form Document Information Extraction (arXiv:2305.02549)
   - **Authors**: Chen-Yu Lee, Chun-Liang Li, Hao Zhang, Timothy Dozat, Vincent Perot, Guolong Su, Xiang Zhang, Kihyuk Sohn, Nikolai Glushnev, Renshen Wang, Joshua Ainslie, Shangbang Long, Siyang Qin, Yasuhisa Fujii, Nan Hua, Tomas Pfister
   - **Summary**: FormNetV2 introduces a multimodal graph contrastive learning strategy for form document understanding. By unifying self-supervised pre-training across modalities, the model achieves state-of-the-art performance on various benchmarks with a compact model size.
   - **Year**: 2023

10. **Title**: Causal Machine Learning for Single-Cell Genomics (arXiv:2310.14935)
    - **Authors**: Alejandro Tejada-Lapuerta, Paul Bertin, Stefan Bauer, Hananeh Aliee, Yoshua Bengio, Fabian J. Theis
    - **Summary**: This perspective discusses the application of causal machine learning methodologies to single-cell genomics. The authors highlight challenges such as generalization to unseen environments and learning interpretable models, proposing research directions to advance the understanding of gene regulation and cellular development.
    - **Year**: 2023

**Key Challenges:**

1. **Data Integration Complexity**: Effectively integrating heterogeneous biological data modalities (e.g., molecular graphs, imaging data) into a unified representation remains challenging due to differences in data structures and scales.

2. **Causal Inference in High-Dimensional Spaces**: Identifying causal relationships within high-dimensional, multimodal biological datasets is complex, often requiring assumptions that may not hold across all modalities.

3. **Generalization to Unseen Perturbations**: Developing models that generalize well to unseen biological perturbations or conditions is difficult, limiting their applicability in real-world scenarios.

4. **Interpretability of Learned Representations**: Ensuring that the representations learned by models are interpretable and provide meaningful insights into biological mechanisms is a significant challenge.

5. **Scalability and Computational Efficiency**: Processing large-scale multimodal biological datasets requires scalable and computationally efficient algorithms, which can be difficult to design and implement. 
1. **Title**: RiboDiffusion: Tertiary Structure-based RNA Inverse Folding with Generative Diffusion Models (arXiv:2404.11199)
   - **Authors**: Han Huang, Ziqian Lin, Dongchen He, Liang Hong, Yu Li
   - **Summary**: This paper introduces RiboDiffusion, a generative diffusion model designed for RNA inverse folding. The model learns the conditional distribution of RNA sequences given 3D backbone structures, utilizing a graph neural network-based structure module and a Transformer-based sequence module. It iteratively transforms random sequences into desired ones, balancing sequence recovery and diversity. RiboDiffusion outperforms existing methods in sequence recovery and demonstrates consistent performance across various RNA lengths and types.
   - **Year**: 2024

2. **Title**: A Survey of Generative AI for de novo Drug Design: New Frontiers in Molecule and Protein Generation (arXiv:2402.08703)
   - **Authors**: Xiangru Tang, Howard Dai, Elizabeth Knight, Fang Wu, Yunyang Li, Tianxiao Li, Mark Gerstein
   - **Summary**: This survey provides a comprehensive overview of generative AI methods applied to de novo drug design, focusing on both small molecule and protein generation. It discusses various datasets, benchmarks, and model architectures, comparing the performance of top models. The paper highlights the rapid development in the field and identifies future directions for AI-driven drug design.
   - **Year**: 2024

3. **Title**: Structure-based Drug Design with Equivariant Diffusion Models (arXiv:2210.13695)
   - **Authors**: Arne Schneuing, Charles Harris, Yuanqi Du, Kieran Didi, Arian Jamasb, Ilia Igashov, Weitao Du, Carla Gomes, Tom Blundell, Pietro Lio, Max Welling, Michael Bronstein, Bruno Correia
   - **Summary**: The authors present DiffSBDD, an SE(3)-equivariant diffusion model for structure-based drug design. The model generates novel ligands conditioned on protein pockets and can be applied to various tasks such as property optimization and partial molecular design. DiffSBDD captures the statistics of ground truth data effectively and incorporates additional design objectives through modified sampling strategies.
   - **Year**: 2022

4. **Title**: A 3D Generative Model for Structure-Based Drug Design (arXiv:2203.10446)
   - **Authors**: Shitong Luo, Jiaqi Guan, Jianzhu Ma, Jian Peng
   - **Summary**: This paper introduces a 3D generative model that generates molecules binding to specific protein binding sites. The model estimates the probability density of atom occurrences in 3D space and employs an auto-regressive sampling scheme to generate valid and diverse molecules. Experimental results show that the generated molecules exhibit high binding affinity and favorable drug properties.
   - **Year**: 2022

5. **Title**: trRosettaRNA: Automated Prediction of RNA 3D Structure with Transformer Network
   - **Authors**: [Not specified]
   - **Summary**: trRosettaRNA is an algorithm for automated prediction of RNA 3D structures. It builds RNA structures using Rosetta energy minimization, guided by deep learning restraints from a transformer network (RNAformer). The method has been validated in blind tests, including CASP15 and RNA-Puzzles, demonstrating competitive performance with top human groups on natural RNAs.
   - **Year**: 2023

6. **Title**: UFold: Fast and Accurate RNA Secondary Structure Prediction with Deep Learning
   - **Authors**: [Not specified]
   - **Summary**: UFold is a deep learning-based method for predicting RNA secondary structures, including pseudoknots. It utilizes a two-dimensional deep neural network and transfer learning to achieve high accuracy and speed in structure prediction.
   - **Year**: 2022

7. **Title**: SPOT-RNA: Predicting RNA Secondary Structure with Deep Learning
   - **Authors**: [Not specified]
   - **Summary**: SPOT-RNA is the first RNA secondary structure predictor capable of predicting all types of base pairs, including canonical, noncanonical, pseudoknots, and base triplets. It employs deep learning techniques to achieve this comprehensive prediction capability.
   - **Year**: 2022

8. **Title**: RNAComposer: Fully Automated Prediction of Large RNA 3D Structures
   - **Authors**: [Not specified]
   - **Summary**: RNAComposer is a web server that provides fully automated prediction of large RNA 3D structures. It utilizes a knowledge-based approach to assemble RNA 3D models from sequence data, facilitating the modeling of complex RNA structures.
   - **Year**: 2022

9. **Title**: FARFAR2: Improved De Novo Rosetta Prediction of Complex Global RNA Folds
   - **Authors**: [Not specified]
   - **Summary**: FARFAR2 is an automated method for de novo prediction of native-like RNA tertiary structures. It builds upon the Rosetta framework to improve the accuracy of RNA fold predictions, particularly for complex global folds.
   - **Year**: 2020

10. **Title**: EternaFold: A Multitask-Learning-Based Model Trained on Data from the Eterna Project
    - **Authors**: [Not specified]
    - **Summary**: EternaFold is a multitask-learning-based model trained on data from the Eterna project. It aims to improve RNA secondary structure prediction by leveraging a large dataset of RNA sequences and structures, enhancing the accuracy of predictions.
    - **Year**: 2022

**Key Challenges:**

1. **Data Scarcity and Quality**: The limited availability of high-quality RNA 3D structures hampers the training of robust generative models.

2. **Complexity of RNA Folding**: Accurately modeling the intricate folding patterns of RNA, including secondary and tertiary structures, remains a significant challenge.

3. **Sequence-Structure Relationship**: Understanding and predicting the non-unique mapping between RNA sequences and their 3D structures is complex.

4. **Computational Efficiency**: Developing models that are both accurate and computationally efficient is essential for practical applications.

5. **Generalization to Novel Targets**: Ensuring that generative models can generalize to novel RNA targets and binding pockets without extensive retraining is a critical hurdle. 
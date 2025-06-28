1. **Title**: Unbalanced minibatch Optimal Transport; applications to Domain Adaptation (arXiv:2103.03606)
   - **Authors**: Kilian Fatras, Thibault Séjourné, Nicolas Courty, Rémi Flamary
   - **Summary**: This paper addresses the computational challenges of optimal transport (OT) in large-scale datasets by introducing an unbalanced minibatch OT approach. The authors highlight the limitations of standard minibatch strategies, which can lead to undesirable smoothing effects, and propose using unbalanced OT to achieve more robust behavior. Theoretical properties such as unbiased estimators and concentration bounds are discussed. Experimental results demonstrate that unbalanced OT significantly improves performance in domain adaptation tasks, surpassing recent baselines.
   - **Year**: 2021

2. **Title**: Optimal Transport for Conditional Domain Matching and Label Shift (arXiv:2006.08161)
   - **Authors**: Alain Rakotomamonjy, Rémi Flamary, Gilles Gasso, Mokhtar Z. Alaya, Maxime Berar, Nicolas Courty
   - **Summary**: The authors tackle unsupervised domain adaptation under generalized target shift, encompassing both class-conditional and label shifts. They theoretically demonstrate the necessity of aligning both marginals and class-conditional distributions across domains for effective generalization. The proposed method minimizes importance-weighted loss in the source domain and employs Wasserstein distance between weighted marginals. An estimator for target label proportions is introduced, blending mixture estimation with optimal transport, and is supported by theoretical guarantees. Experiments show superior performance across various domain adaptation problems.
   - **Year**: 2020

3. **Title**: Optimal transport meets noisy label robust loss and MixUp regularization for domain adaptation (arXiv:2206.11180)
   - **Authors**: Kilian Fatras, Hiroki Naganuma, Ioannis Mitliagkas
   - **Summary**: This work addresses domain adaptation challenges arising from domain shifts in computer vision. The authors identify that optimal transport (OT) can lead to negative transfer by aligning samples with different labels, especially under label shift conditions. To mitigate this, they propose combining MixUp regularization with a loss function robust to noisy labels, resulting in the \textsc{mixunbot} method. An extensive ablation study underscores the importance of this combination, and evaluations on multiple benchmarks and real-world problems demonstrate improved domain adaptation performance.
   - **Year**: 2022

4. **Title**: Unbalanced CO-Optimal Transport (arXiv:2205.14923)
   - **Authors**: Quang Huy Tran, Hicham Janati, Nicolas Courty, Rémi Flamary, Ievgen Redko, Pinar Demetci, Ritambhara Singh
   - **Summary**: The paper introduces unbalanced CO-optimal transport (COOT), extending OT by inferring alignments between both samples and features. The authors highlight the sensitivity of COOT to outliers and propose an unbalanced version to enhance robustness. Theoretical results demonstrate the method's resilience to noise in datasets from incomparable spaces. Empirical evidence supports its effectiveness in heterogeneous domain adaptation tasks with varying class proportions and in aligning samples and features across single-cell measurements.
   - **Year**: 2022

**Key Challenges:**

1. **Computational Complexity**: Optimal transport methods, especially in large-scale datasets, face significant computational challenges. Efficient algorithms are required to make these methods practical for real-world applications.

2. **Sensitivity to Outliers**: Standard OT approaches can be sensitive to outliers, leading to misalignments and degraded performance. Developing robust OT formulations that can handle noisy data is essential.

3. **Negative Transfer**: In domain adaptation, OT can inadvertently align samples from different classes, causing negative transfer. Strategies to mitigate this issue are crucial for effective adaptation.

4. **Label Shift Handling**: Traditional OT assumes balanced class distributions, which is often violated in practice due to label shifts. Methods that can adapt to unknown label shifts are necessary for robust performance.

5. **Parameter Selection**: Unbalanced OT requires predefined marginal relaxation parameters, which can be challenging to set appropriately. Adaptive methods that learn these parameters from data are needed to improve flexibility and performance. 
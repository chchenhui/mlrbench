1. **Title**: Online Drift Detection with Maximum Concept Discrepancy (arXiv:2407.05375)
   - **Authors**: Ke Wan, Yi Liang, Susik Yoon
   - **Summary**: This paper introduces MCD-DD, a novel concept drift detection method based on maximum concept discrepancy. It employs contrastive learning of concept embeddings to identify various forms of concept drift without relying on labels or statistical properties, effectively addressing high-dimensional data streams with complex distribution shifts.
   - **Year**: 2024

2. **Title**: Unsupervised Concept Drift Detection from Deep Learning Representations in Real-time (arXiv:2406.17813)
   - **Authors**: Salvatore Greco, Bartolomeo Vacchetti, Daniele Apiletti, Tania Cerquitelli
   - **Summary**: The authors propose DriftLens, an unsupervised real-time concept drift detection framework that leverages distribution distances of deep learning representations. It operates on unstructured data and provides drift characterization by analyzing each label separately, demonstrating superior performance and efficiency across multiple deep learning classifiers for text, image, and speech.
   - **Year**: 2024

3. **Title**: A Neighbor-Searching Discrepancy-based Drift Detection Scheme for Learning Evolving Data (arXiv:2405.14153)
   - **Authors**: Feng Gu, Jie Lu, Zhen Fang, Kun Wang, Guangquan Zhang
   - **Summary**: This work presents a novel real concept drift detection method based on Neighbor-Searching Discrepancy, measuring classification boundary differences between samples. The method accurately detects real concept drift while ignoring virtual drift and indicates the direction of classification boundary changes, enhancing model maintenance in evolving data streams.
   - **Year**: 2024

4. **Title**: Concept Drift Detection from Multi-Class Imbalanced Data Streams (arXiv:2104.10228)
   - **Authors**: Łukasz Korycki, Bartosz Krawczyk
   - **Summary**: The authors propose a trainable concept drift detector based on Restricted Boltzmann Machine, capable of monitoring multiple classes and using reconstruction error to detect changes independently. It addresses challenges posed by concept drift in multi-class imbalanced data streams, handling evolving class roles and local drifts in minority classes.
   - **Year**: 2021

**Key Challenges:**

1. **Dynamic Component Adaptation**: Developing mechanisms that allow compositional learning models to adapt their primitive components and composition rules in response to evolving data streams without compromising previously learned knowledge.

2. **Concept Drift Detection**: Creating effective methods to identify and respond to shifts in component semantics or relationships within compositional representations, ensuring models remain accurate over time.

3. **Incremental Component Learning**: Implementing techniques such as generative replay or parameter isolation to update or add components incrementally, mitigating the risk of catastrophic forgetting.

4. **Adaptive Composition Mechanisms**: Designing flexible composition mechanisms, like attention or routing, that can adjust how components are combined based on new evidence, facilitating robust reasoning in non-stationary environments.

5. **Evaluation on Evolving Benchmarks**: Establishing benchmarks that feature evolving tasks or changing object appearances/relationships to effectively assess the performance of compositional learning models in dynamic settings. 
**Title:** Iterative Generative Design with Active Learning for Optimized Antibody Affinity Maturation

**Related Papers:**

1. **Title:** Active Learning for Affinity Prediction of Antibodies (arXiv:2406.07263)
   - **Authors:** Alexandra Gessner, Sebastian W. Ober, Owen Vickery, Dino Oglić, Talip Uçar
   - **Summary:** This paper presents an active learning framework that iteratively proposes promising antibody sequences for simulation, aiming to enhance binding affinity predictions. The approach combines surrogate modeling with active learning to efficiently explore the sequence space, reducing reliance on costly wet-lab experiments.
   - **Year:** 2024

2. **Title:** Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences (arXiv:2412.07763)
   - **Authors:** Alan Nawzad Amin, Nate Gruver, Yilun Kuang, Lily Li, Hunter Elliott, Calvin McCarter, Aniruddh Raghu, Peyton Greenside, Andrew Gordon Wilson
   - **Summary:** The authors introduce CloneBO, a Bayesian optimization procedure that leverages a generative model trained on clonal families to design antibody sequences with improved binding and stability. This method efficiently optimizes antibodies by mimicking the natural evolutionary process of the immune system.
   - **Year:** 2024

3. **Title:** AffinityFlow: Guided Flows for Antibody Affinity Maturation (arXiv:2502.10365)
   - **Authors:** Can Chen, Karla-Luise Herpoldt, Chenchao Zhao, Zichen Wang, Marcus Collins, Shang Shang, Ron Benson
   - **Summary:** This study proposes AffinityFlow, an alternating optimization framework that integrates structure-based and sequence-based affinity predictors to guide antibody design. The method employs a co-teaching module to refine predictors using noisy biophysical data, achieving state-of-the-art performance in affinity maturation experiments.
   - **Year:** 2025

4. **Title:** AbGPT: De Novo Antibody Design via Generative Language Modeling (arXiv:2409.06090)
   - **Authors:** Desmond Kuan, Amir Barati Farimani
   - **Summary:** AbGPT is a generative language model fine-tuned on a large dataset of B-cell receptor sequences. It successfully generates high-quality antibody sequences, demonstrating an understanding of intrinsic variability and conserved regions within the antibody repertoire, which is crucial for effective de novo antibody design.
   - **Year:** 2024

5. **Title:** Active Learning for Energy-Based Antibody Optimization and Enhanced Screening (arXiv:2409.10964)
   - **Authors:** Kairi Furui, Masahito Ohue
   - **Summary:** The authors propose an active learning workflow that integrates deep learning models with energy-based methods to efficiently explore antibody mutants. This approach significantly improves screening performance and identifies mutants with better binding properties without experimental ΔΔG data.
   - **Year:** 2024

6. **Title:** Antigen-Specific Antibody Design via Direct Energy-Based Preference Optimization (arXiv:2403.16576)
   - **Authors:** Xiangxin Zhou, Dongyu Xue, Ruizhe Chen, Zaixiang Zheng, Liang Wang, Quanquan Gu
   - **Summary:** This paper addresses antigen-specific antibody sequence-structure co-design by fine-tuning a pre-trained diffusion model using energy-based preference optimization. The method effectively optimizes the energy of generated antibodies, achieving high binding affinity and rational structures.
   - **Year:** 2024

7. **Title:** Retrieval Augmented Diffusion Model for Structure-Informed Antibody Design and Optimization (arXiv:2410.15040)
   - **Authors:** Zichen Wang, Yaokun Ji, Jianing Tian, Shuangjia Zheng
   - **Summary:** The authors introduce RADAb, a retrieval-augmented diffusion framework that utilizes structural homologous motifs to guide generative models in antibody design. This approach integrates structural and evolutionary information, achieving state-of-the-art performance in antibody inverse folding and optimization tasks.
   - **Year:** 2024

8. **Title:** Optimizing Drug Design by Merging Generative AI with Active Learning Frameworks (arXiv:2305.06334)
   - **Authors:** Isaac Filella-Merce, Alexis Molina, Marek Orzechowski, Lucía Díaz, Yang Ming Zhu, Julia Vilalta Mor, Laura Malo, Ajay S Yekkirala, Soumya Ray, Victor Guallar
   - **Summary:** This study develops a workflow combining a variational autoencoder with active learning steps to design molecules with high predicted affinity toward targets. The method demonstrates the potential of generative AI and active learning in exploring novel chemical spaces for drug discovery.
   - **Year:** 2023

9. **Title:** De Novo Antibody Design with SE(3) Diffusion (arXiv:2405.07622)
   - **Authors:** Daniel Cutting, Frédéric A. Dreyer, David Errington, Constantin Schneider, Charlotte M. Deane
   - **Summary:** IgDiff is an antibody variable domain diffusion model that generates highly designable antibodies with novel binding regions. The model's generated structures show good agreement with reference antibody distributions, and experimental validation confirms high expression yields.
   - **Year:** 2024

10. **Title:** Antibody Design Using a Score-Based Diffusion Model Guided by Evolutionary, Physical, and Geometric Constraints
    - **Authors:** Tian Zhu, Milong Ren, Haicang Zhang
    - **Summary:** AbX is a score-based diffusion generative model that incorporates evolutionary, physical, and geometric constraints for antibody design. The model jointly models sequence and structure spaces, achieving high accuracy in sequence and structure generation and enhanced antibody-antigen binding affinity.
    - **Year:** 2024

**Key Challenges:**

1. **Data Scarcity:** The limited availability of labeled antibody-antigen complex data and binding affinity measurements hampers the training and validation of generative and predictive models.

2. **Computational Complexity:** Accurately simulating the physics of large molecules like antibodies is computationally intensive, making large-scale screening and optimization challenging.

3. **Model Generalization:** Ensuring that generative models can generalize to diverse antibody sequences and structures without overfitting to specific datasets remains a significant hurdle.

4. **Integration of Experimental Data:** Effectively incorporating wet-lab experimental results into the iterative design process to refine models and guide subsequent experiments is complex and requires robust frameworks.

5. **Balancing Exploration and Exploitation:** Developing active learning strategies that balance exploring new sequence spaces and exploiting known high-affinity regions is critical for efficient antibody design. 
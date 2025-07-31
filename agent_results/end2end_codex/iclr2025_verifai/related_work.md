**1. Related Papers**

1. **Title**: APOLLO: Automated LLM and Lean Collaboration for Advanced Formal Reasoning
   - **Authors**: Azim Ospanov, Roozbeh Yousefzadeh
   - **Summary**: APOLLO introduces a modular pipeline that integrates large language models (LLMs) with the Lean theorem prover. The system automates proof generation by analyzing and repairing LLM-generated proofs, utilizing Lean's feedback to iteratively improve accuracy. This approach achieves state-of-the-art results on benchmarks like miniF2F, demonstrating significant efficiency and correctness gains.
   - **Year**: 2025

2. **Title**: LemmaHead: RAG Assisted Proof Generation Using Large Language Models
   - **Authors**: Tianbo Yang, Mingqi Yan, Hongyi Zhao, Tianshuo Yang
   - **Summary**: LemmaHead explores the use of retrieval-augmented generation (RAG) to enhance LLMs' mathematical reasoning capabilities. By supplementing model queries with relevant mathematical context from textbooks, the system aims to improve automated theorem proving in the Lean formal language.
   - **Year**: 2025

3. **Title**: LeanDojo: Theorem Proving with Retrieval-Augmented Language Models
   - **Authors**: Kaiyu Yang, Aidan M. Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan Prenger, Anima Anandkumar
   - **Summary**: LeanDojo presents an open-source platform that combines LLMs with the Lean theorem prover. It includes toolkits, data, models, and benchmarks, facilitating research in machine learning for theorem proving. The system features fine-grained annotations for premise selection and introduces ReProver, an LLM-based prover augmented with retrieval mechanisms.
   - **Year**: 2023

4. **Title**: Lemmanaid: Neuro-Symbolic Lemma Conjecturing
   - **Authors**: Yousef Alhessi, Sólrún Halla Einarsdóttir, George Granberry, Emily First, Moa Johansson, Sorin Lerner, Nicholas Smallbone
   - **Summary**: Lemmanaid introduces a neuro-symbolic approach to lemma conjecturing, combining LLMs with symbolic methods. The system generates lemma templates using LLMs and fills in details with symbolic techniques, aiming to improve automated reasoning tools and facilitate formalization in proof assistants.
   - **Year**: 2025

5. **Title**: STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving
   - **Authors**: Kefan Dong, Tengyu Ma
   - **Summary**: STP addresses the challenge of limited high-quality training data in formal theorem proving by introducing a self-play framework. The system alternates between generating novel conjectures and attempting to prove them, iteratively improving the model's capabilities. This approach achieves state-of-the-art performance on several benchmarks.
   - **Year**: 2025

6. **Title**: Neural Theorem Proving: Generating and Structuring Proofs for Formal Verification
   - **Authors**: Balaji Rao, William Eiers, Carlo Lipizzi
   - **Summary**: This paper introduces a framework for generating formal proofs using LLMs within systems that utilize built-in tactics and automated theorem provers. The approach includes generating natural language statements, formal proofs, and employing heuristics to build the final proof, aiming to enhance formal verification processes.
   - **Year**: 2025

7. **Title**: Proof Automation with Large Language Models
   - **Authors**: Minghai Lu, Benjamin Delaware, Tianyi Zhang
   - **Summary**: The authors propose PALM, a generate-then-repair approach that combines LLMs with symbolic methods to automate proof generation in Coq. By first generating an initial proof and then iteratively repairing low-level issues, PALM significantly outperforms existing methods, proving a higher percentage of theorems.
   - **Year**: 2024

8. **Title**: Faithful and Robust LLM-Driven Theorem Proving for NLI Explanations
   - **Authors**: Xin Quan, Marco Valentino, Louise A. Dennis, Andre Freitas
   - **Summary**: This paper investigates strategies to enhance the faithfulness and robustness of LLM-driven theorem proving for natural language inference (NLI) explanations. The authors propose methods to reduce semantic loss during autoformalization, correct syntactic errors, and guide LLMs in generating structured proof sketches, leading to significant improvements over state-of-the-art models.
   - **Year**: 2025

9. **Title**: LEGO-Prover: Neural Theorem Proving with Growing Libraries
   - **Authors**: Amitayush Thakur, Yeming Wen, Swarat Chaudhuri
   - **Summary**: LEGO-Prover introduces a neural theorem proving approach that constructs a continuously growing library of proven theorems. By leveraging prompt engineering and focusing on building an expanding theorem library, the system aims to improve proof generation capabilities.
   - **Year**: 2024

10. **Title**: Lyra: Orchestrating Dual Correction in Automated Theorem Proving
    - **Authors**: Haiming Wang, Ye Yuan, Zhengying Liu, Jianhao Shen, Yichun Yin, Jing Xiong, Enze Xie, Han Shi, Yujun Li, Lin Li, Jian Yin, Zhenguo Li, Xiaodan Liang
    - **Summary**: Lyra presents a dual correction mechanism in automated theorem proving, combining neural and symbolic methods. The system focuses on correcting both the proof generation process and the generated proofs themselves, aiming to enhance the accuracy and reliability of automated theorem proving.
    - **Year**: 2023

**2. Key Challenges**

1. **Semantic Loss during Autoformalization**: Translating natural language statements into formal representations can lead to semantic inaccuracies, affecting the correctness of generated proofs.

2. **Limited High-Quality Training Data**: The scarcity of annotated formal proofs hampers the training of LLMs, limiting their effectiveness in theorem proving tasks.

3. **Error Propagation in Proof Generation**: Initial errors in LLM-generated proofs can propagate through the proof process, making it challenging to correct and verify proofs.

4. **Balancing Neural and Symbolic Methods**: Integrating LLMs with symbolic reasoning tools requires careful coordination to leverage the strengths of both approaches without introducing conflicts.

5. **Scalability and Efficiency**: Ensuring that LLM-assisted theorem proving systems can handle complex proofs efficiently without excessive computational resources remains a significant challenge. 
1. **Title**: InternalInspector $I^2$: Robust Confidence Estimation in LLMs through Internal States (arXiv:2406.12053)
   - **Authors**: Mohammad Beigi, Ying Shen, Runing Yang, Zihao Lin, Qifan Wang, Ankith Mohan, Jianfeng He, Ming Jin, Chang-Tien Lu, Lifu Huang
   - **Summary**: This paper introduces InternalInspector, a framework that enhances confidence estimation in Large Language Models (LLMs) by leveraging contrastive learning on internal states, including attention and activation patterns across all layers. Unlike methods focusing solely on final activation states, InternalInspector analyzes comprehensive internal states to accurately identify both correct and incorrect predictions. The approach demonstrates improved accuracy in aligning confidence scores with prediction correctness and reduces calibration error across various natural language understanding and generation tasks.
   - **Year**: 2024

2. **Title**: Unsupervised Real-Time Hallucination Detection based on the Internal States of Large Language Models (arXiv:2403.06448)
   - **Authors**: Weihang Su, Changyue Wang, Qingyao Ai, Yiran Hu, Zhijing Wu, Yujia Zhou, Yiqun Liu
   - **Summary**: The authors present MIND, an unsupervised training framework that utilizes the internal states of LLMs for real-time hallucination detection without requiring manual annotations. MIND addresses limitations of post-processing techniques by integrating detection within the inference process, enhancing efficiency and effectiveness. The paper also introduces HELM, a benchmark for evaluating hallucination detection across multiple LLMs, demonstrating that MIND outperforms existing state-of-the-art methods in this area.
   - **Year**: 2024

3. **Title**: Prompt-Guided Internal States for Hallucination Detection of Large Language Models (arXiv:2411.04847)
   - **Authors**: Fujie Zhang, Peiqi Yu, Biao Yi, Baolei Zhang, Tong Li, Zheli Liu
   - **Summary**: This work proposes PRISM, a framework that enhances cross-domain performance of supervised hallucination detectors by utilizing prompts to guide changes in the structure related to text truthfulness within LLMs' internal states. By making this structure more salient and consistent across different domains, PRISM significantly improves the generalization of existing hallucination detection methods, as evidenced by experiments on datasets from various domains.
   - **Year**: 2024

4. **Title**: Calibrating Large Language Models with Sample Consistency (arXiv:2402.13904)
   - **Authors**: Qing Lyu, Kumar Shridhar, Chaitanya Malaviya, Li Zhang, Yanai Elazar, Niket Tandon, Marianna Apidianaki, Mrinmaya Sachan, Chris Callison-Burch
   - **Summary**: The authors explore deriving confidence from the distribution of multiple randomly sampled model generations through measures of consistency. They perform extensive evaluations across various models and reasoning datasets, showing that consistency-based calibration methods outperform existing post-hoc approaches. The study also examines factors such as intermediate explanations, model scaling, and sample sizes, providing practical guidance on choosing suitable consistency metrics for calibration tailored to different LLMs.
   - **Year**: 2024

5. **Title**: Graph-based Confidence Calibration for Large Language Models (arXiv:2411.02454)
   - **Authors**: Yukun Li, Sijia Wang, Lifu Huang, Li-Ping Liu
   - **Summary**: This paper introduces a method combining LLMs' self-consistency with labeled data to train an auxiliary model that estimates the correctness of responses. Using a weighted graph to represent consistency among multiple responses, a graph neural network is trained to estimate the probability of correct responses. Experiments demonstrate that this approach substantially outperforms recent methods in confidence calibration across multiple benchmark datasets and improves generalization on out-of-domain data.
   - **Year**: 2024

6. **Title**: Refine Knowledge of Large Language Models via Adaptive Contrastive Learning (arXiv:2502.07184)
   - **Authors**: Yinghui Li, Haojing Huang, Jiayi Kuang, Yangning Li, Shu-Yu Guo, Chao Qu, Xiaoyu Tan, Hai-Tao Zheng, Ying Shen, Philip S. Yu
   - **Summary**: The authors design an Adaptive Contrastive Learning strategy that flexibly constructs positive and negative samples based on LLMs' actual knowledge mastery. This strategy helps LLMs consolidate correct knowledge, deepen understanding of partially grasped knowledge, forget incorrect knowledge, and acknowledge knowledge gaps. Extensive experiments demonstrate the effectiveness of this method in reducing hallucinations.
   - **Year**: 2025

7. **Title**: Contrastive Learning to Improve Retrieval for Real-world Fact Checking (arXiv:2410.04657)
   - **Authors**: Aniruddh Sriram, Fangyuan Xu, Eunsol Choi, Greg Durrett
   - **Summary**: This work presents Contrastive Fact-Checking Reranker (CFR), an improved retriever for fact-checking that leverages contrastive learning. By fine-tuning a retriever with multiple training signals, including distillation from GPT-4 and evaluation of subquestion answers, CFR enhances retrieval effectiveness. Experiments show a 6% improvement in veracity classification accuracy on the AVeriTeC dataset and demonstrate transferability to other datasets.
   - **Year**: 2024

8. **Title**: Mind the Confidence Gap: Overconfidence, Calibration, and Distractor Effects in Large Language Models (arXiv:2502.11028)
   - **Authors**: Prateek Chhikara
   - **Summary**: This empirical study examines how model size, distractors, and question types affect confidence calibration in LLMs. The findings indicate that while larger models are better calibrated overall, they are more prone to distraction, whereas smaller models benefit more from answer choices but struggle with uncertainty estimation. The study highlights the need for calibration-aware interventions and improved uncertainty estimation methods.
   - **Year**: 2025

9. **Title**: Calibrating the Confidence of Large Language Models by Eliciting Fidelity (ACL Anthology: 2024.emnlp-main.173)
   - **Authors**: Mozhi Zhang, Mianqiu Huang, Rundong Shi, Linsen Guo, Chong Peng, Peng Yan, Yaqian Zhou, Xipeng Qiu
   - **Summary**: The authors propose UF Calibration, a plug-and-play method to estimate LLMs' confidence by decomposing it into uncertainty about the question and fidelity to the generated answer. Experiments with multiple RLHF-LMs on MCQA datasets show that UF Calibration achieves good calibration performance. The paper also introduces two novel metrics, IPR and CE, to evaluate model calibration and discusses truly well-calibrated confidence for LLMs.
   - **Year**: 2024

10. **Title**: TrueTeacher: Learning Factual Consistency Evaluation with Large Language Models (arXiv:2305.11171)
    - **Authors**: Zorik Gekhman, Jonathan Herzig, Roee Aharoni, Chen Elkind, Idan Szpektor
    - **Summary**: TrueTeacher introduces a method for generating synthetic data by annotating diverse model-generated summaries using a LLM. Unlike prior work relying on human-written summaries, TrueTeacher is multilingual and does not require human annotations. Experiments on the TRUE benchmark show that a student model trained using this data substantially outperforms state-of-the-art models and the LLM teacher, demonstrating the method's superiority and robustness to domain shifts.
    - **Year**: 2023

**Key Challenges:**

1. **Calibration of Internal Confidence Metrics**: Accurately aligning internal confidence scores with factual accuracy remains challenging due to the complexity of LLMs' internal states and the variability in their outputs.

2. **Generalization Across Domains**: Ensuring that hallucination detection methods generalize well across different domains and tasks is difficult, as models trained on specific data may struggle with unseen contexts.

3. **Real-Time Detection Efficiency**: Implementing real-time hallucination detection without significantly impacting the efficiency and speed of LLMs during inference poses a significant challenge.

4. **Dependence on Large-Scale Annotated Data**: Many approaches require extensive annotated datasets for training, which can be resource-intensive and may not cover all possible hallucination scenarios.

5. **Balancing Model Complexity and Interpretability**: Developing methods that effectively detect hallucinations while maintaining model interpretability and avoiding excessive complexity is a persistent challenge in the field. 
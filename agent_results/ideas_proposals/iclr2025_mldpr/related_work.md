1. **Title**: Holistic Evaluation Metrics: Use Case Sensitive Evaluation Metrics for Federated Learning (arXiv:2405.02360)
   - **Authors**: Yanli Li, Jehad Ibrahim, Huaming Chen, Dong Yuan, Kim-Kwang Raymond Choo
   - **Summary**: This paper introduces Holistic Evaluation Metrics (HEM) for federated learning, emphasizing the need for comprehensive evaluation beyond single metrics like accuracy. HEM considers various aspects such as accuracy, convergence, computational efficiency, fairness, and personalization, tailored to specific use cases like IoT, smart devices, and institutions. The proposed framework assigns importance vectors to these metrics based on the distinct requirements of each use case, facilitating the identification of the most suitable federated learning algorithms for real-world applications.
   - **Year**: 2024

2. **Title**: Holistic Evaluation of Language Models (arXiv:2211.09110)
   - **Authors**: Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda
   - **Summary**: The authors present HELM (Holistic Evaluation of Language Models), a framework designed to improve the transparency of language models by evaluating them across a broad set of scenarios and metrics. HELM assesses models on seven metrics—accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency—across 16 core scenarios, ensuring a comprehensive understanding of model capabilities and limitations. This approach highlights the importance of multi-metric evaluation in language model assessment.
   - **Year**: 2022

3. **Title**: Model Cards for Model Reporting (arXiv:1810.03993)
   - **Authors**: Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, Timnit Gebru
   - **Summary**: This paper introduces "Model Cards," a framework for documenting machine learning models to enhance transparency and accountability. Model Cards provide detailed information on a model's intended use, performance across different conditions, and potential biases. By standardizing model documentation, this approach aims to inform users about appropriate use cases and limitations, promoting responsible deployment of machine learning models.
   - **Year**: 2018

**Key Challenges**:

1. **Overemphasis on Single Metrics**: Current benchmarking practices often focus on aggregate metrics like accuracy, neglecting other critical factors such as fairness, robustness, and efficiency. This narrow focus can lead to models that perform well on leaderboards but fail in real-world applications.

2. **Lack of Standardized Documentation**: The absence of standardized documentation for benchmarks results in inconsistent reporting of evaluation contexts, dataset characteristics, and performance metrics. This inconsistency hampers the reproducibility and comparability of model assessments.

3. **Insufficient Consideration of Contextual Factors**: Benchmarks frequently overlook the specific contexts in which models will be deployed, including potential biases in datasets and the diverse needs of different user groups. This oversight can lead to models that are unsuitable or even harmful in certain applications.

4. **Inadequate Evaluation of Model Limitations**: There is often a lack of thorough evaluation regarding the limitations and potential misuse scenarios of models. Without this critical analysis, models may be deployed in inappropriate contexts, leading to unintended consequences.

5. **Challenges in Implementing Holistic Evaluation Frameworks**: Developing and adopting comprehensive evaluation frameworks that encompass multiple metrics and contextual factors is complex. It requires consensus within the research community and the establishment of best practices, which can be difficult to achieve. 
1. **Title**: Modality-aware Transformer for Financial Time Series Forecasting (2310.01232)
   - **Authors**: Hajar Emami, Xuan-Hong Dang, Yousaf Shah, Petros Zerfos
   - **Summary**: This paper introduces a multimodal transformer-based model designed to enhance financial time series forecasting by integrating both categorical text and numerical data. The model employs feature-level attention layers to focus on relevant features within each modality and incorporates intra-modal, inter-modal, and target-modal multi-head attention mechanisms. This approach effectively captures temporal and modality-specific information, leading to improved forecasting accuracy.
   - **Year**: 2023

2. **Title**: Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting (2502.04395)
   - **Authors**: Siru Zhong, Weilin Ruan, Ming Jin, Huan Li, Qingsong Wen, Yuxuan Liang
   - **Summary**: The authors propose Time-VLM, a framework that leverages pre-trained Vision-Language Models (VLMs) to integrate temporal, visual, and textual modalities for enhanced time series forecasting. The framework includes a Retrieval-Augmented Learner, a Vision-Augmented Learner, and a Text-Augmented Learner, which together produce multimodal embeddings fused with temporal features for final prediction. Experiments demonstrate superior performance, especially in few-shot and zero-shot scenarios.
   - **Year**: 2025

3. **Title**: MST-GAT: A Multimodal Spatial-Temporal Graph Attention Network for Time Series Anomaly Detection (2310.11169)
   - **Authors**: Chaoyue Ding, Shiliang Sun, Jing Zhao
   - **Summary**: This study presents MST-GAT, a multimodal spatial-temporal graph attention network aimed at improving anomaly detection in multimodal time series data. The model utilizes a multimodal graph attention network and a temporal convolution network to capture spatial-temporal correlations, employing intra- and inter-modal attention mechanisms to explicitly model modal relationships. The approach enhances interpretability by identifying the most anomalous univariate time series.
   - **Year**: 2023

4. **Title**: Multi-Modal Forecaster: Jointly Predicting Time Series and Textual Data (2411.06735)
   - **Authors**: Kai Kim, Howard Tsai, Rajat Sen, Abhimanyu Das, Zihao Zhou, Abhishek Tanpure, Mathew Luo, Rose Yu
   - **Summary**: The authors develop the TimeText Corpus (TTC), a curated dataset combining time-aligned text and numerical data from climate science and healthcare domains. They propose the Hybrid Multi-Modal Forecaster (Hybrid-MMF), a multimodal language model that jointly forecasts text and time series data using shared embeddings. Despite the innovative approach, the model does not outperform existing baselines, highlighting challenges in multimodal forecasting.
   - **Year**: 2024

**Key Challenges**:

1. **Modality Integration Complexity**: Effectively combining diverse data modalities (e.g., numerical, textual, visual) poses significant challenges due to differences in data structures, scales, and temporal alignments.

2. **Attention Mechanism Design**: Developing attention mechanisms that can dynamically and accurately weigh the importance of different modalities and features is complex, especially when dealing with varying data quality and relevance.

3. **Data Quality and Availability**: Ensuring the availability of high-quality, synchronized multimodal datasets is challenging, as inconsistencies and missing data can adversely affect model performance.

4. **Model Interpretability**: As models become more complex with multimodal inputs, interpreting their decisions and understanding the contribution of each modality to the final prediction becomes increasingly difficult.

5. **Computational Resources**: Training and deploying multimodal models, especially those incorporating large pre-trained components, require substantial computational resources, which may not be feasible in all settings. 
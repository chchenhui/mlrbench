1. Title  
Multimodal Adaptive Fusion Transformer (MAFFT) for Enhanced Time Series Forecasting

2. Introduction  
Background  
Time series forecasting is a foundational tool in domains ranging from finance and energy to healthcare and supply chain management. Traditional models (ARIMA, ETS) and more recent deep learning architectures (LSTMs, temporal convolutional networks, pure Transformers) have mostly operated on numerical sequences in isolation. However, many real-world phenomena are driven or modulated by external contextual signals—news articles, social media sentiment, static metadata, or even satellite imagery—that cannot be captured by numeric history alone.  
Foundation models pretrained on massive text or vision corpora (e.g., BERT, ViT) have revolutionized natural language processing (NLP) and computer vision (CV). Recent efforts (Emami et al., 2023; Zhong et al., 2025) have started to adapt these large pretrained models for time series tasks. Multimodal approaches such as MST-GAT (Ding et al., 2023) and Hybrid-MMF (Kim et al., 2024) demonstrate that integrating heterogenous data can improve anomaly detection and forecasting in specialized settings. Yet challenges remain in:  
• Dynamically weighting modalities according to relevance and data quality.  
• Designing cross-modal attention that is both expressive and computationally efficient.  
• Evaluating model behavior during regime changes or anomalous events when external context is most critical.  

Research Objectives  
This proposal aims to develop MAFFT, a new architecture that:  
1. Leverages pretrained Transformers for text and vision, plus a temporal encoder for numeric data.  
2. Employs a novel cross-modal attention module with adaptive weighting to fuse contextual embeddings and numerical features.  
3. Quantifies gains in forecasting accuracy and robustness, especially during regime shifts, via comprehensive experiments on benchmark and real-world datasets.  

Significance  
By marrying the strengths of large pretrained language and vision models with specialized time series encoders, MAFFT is expected to:  
— Improve point and probabilistic forecasting accuracy, particularly under external shocks.  
— Offer interpretable modality-importance scores through attention analyses.  
— Provide a blueprint for future multimodal foundation models in time series, driving advances in energy forecasting, financial risk management, and other mission-critical domains.

3. Methodology  
3.1 Overview  
MAFFT consists of three main stages:  
(a) Modality-Specific Encoding  
(b) Cross-Modal Attention Fusion with Adaptive Weights  
(c) Forecasting Head and Loss Computation  

3.2 Data Collection and Preprocessing  
Datasets  
1. Public Benchmarks: M4, electricity consumption, traffic flow.  
2. TimeText Corpus (Kim et al., 2024): aligned climate science and healthcare time series with text annotations.  
3. Custom NewsTime: a scraped dataset linking daily economic indicators to financial news headlines and stock chart images.  

Preprocessing Steps  
• Numeric Series: normalization to zero mean/unit variance per series. Missing values imputed via linear interpolation or historical average.  
• Text Data: use a pretrained tokenizer (BERT-base). Truncate/pad headlines to 128 tokens.  
• Image Data: resize to 224×224, standard ImageNet normalization, and feed into a pretrained ViT or ResNet backbone.  
• Time Alignment: ensure that at each forecast time point $t$, we have numeric history $\mathbf{x}_{t-L:t-1}$, text embeddings $T_t$, and image embeddings $V_t$.  

3.3 Modality-Specific Encoders  
• Numeric Encoder: A temporal Transformer with $N$ layers. Input $\mathbf{x}_{t-L:t-1}\in \mathbb{R}^{L\times d_x}$ is projected to dimension $d$. Positional encodings $P\in\mathbb{R}^{L\times d}$ are added. Each layer computes self-attention:  
$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\Bigl(\frac{QK^\top}{\sqrt{d_k}}\Bigr)V
$$  
followed by a feed-forward block. The final numeric representation is $H^x_t\in\mathbb{R}^{L\times d}$. We take the last time step embedding $h^x_t\in\mathbb{R}^d$.  

• Text Encoder: A frozen or fine-tuned BERT model that maps token sequence to CLS token embedding $h^T_t\in\mathbb{R}^d$.  

• Vision Encoder: A pretrained ViT whose [CLS] token gives $h^V_t\in\mathbb{R}^d$.  

3.4 Cross-Modal Attention Fusion  
We introduce a multi-head cross-modal attention module. For each modality pair $(m,n)\in\{x,T,V\}^2$, we compute:  
$$
\mathrm{CrossAttn}(h^m_t,h^n_t) = \mathrm{softmax}\!\Bigl(\tfrac{W_Q^mh^m_t\,(W_K^n h^n_t)^\top}{\sqrt{d_k}}\Bigr)\,(W_V^n h^n_t)
$$  
where $W_Q^m,W_K^n,W_V^n\in\mathbb{R}^{d\times d_k}$. We then aggregate all pairwise cross-modal outputs into modality-specific fused vectors $\tilde h^x_t,\tilde h^T_t,\tilde h^V_t\in\mathbb{R}^d$.  

3.5 Adaptive Modality Weighting  
To allow the model to adjust reliance on each modality based on context and data quality, we learn a gating network:  
$$
\alpha = \mathrm{softmax}(W_g[h^x_t;\tilde h^x_t;\tilde h^T_t;\tilde h^V_t]+b_g)\in\mathbb{R}^3
$$  
where $[\,;\,]$ denotes concatenation, and $\alpha=(\alpha_x,\alpha_T,\alpha_V)$. The final fused representation is:  
$$
h^{\text{fused}}_t = \alpha_x\,\tilde h^x_t + \alpha_T\,\tilde h^T_t + \alpha_V\,\tilde h^V_t.
$$  

3.6 Forecasting Head and Loss  
We feed $h^{\text{fused}}_t$ into a two-layer MLP with ReLU activation to predict the next $H$ time steps $\hat{\mathbf y}_{t+1:t+H}\in\mathbb{R}^{H}$. For probabilistic forecasting, the head outputs both mean and variance vectors $(\mu,\sigma^2)$. We train by minimizing the combination:  
$$
\mathcal{L} = \frac{1}{H}\sum_{i=1}^H\bigl|\hat y_{t+i}-y_{t+i}\bigr| \;+\;\lambda\,\mathrm{CRPS}\bigl(\mu,\sigma; y_{t+i}\bigr)
$$  
where CRPS is the Continuous Ranked Probability Score.  

3.7 Training and Hyperparameters  
• Batch size: 64; learning rate: $1e^{-4}$ with cosine decay; weight decay: $1e^{-5}$.  
• AdamW optimizer.  
• Number of Transformer layers ($N$): 4 for numeric encoder; number of heads: 8; hidden dim $d=256$; feed-forward dim: 512.  
• Fine-tuning depth for BERT/ViT: last two layers by default.  

3.8 Experimental Design  
Baselines  
– Unimodal Transformer (numeric only)  
– Emami et al. (2023) modality-aware transformer  
– Time-VLM (Zhong et al., 2025)  
– Hybrid-MMF (Kim et al., 2024), MST-GAT (Ding et al., 2023)  

Metrics  
– Point Forecasting: MAE, RMSE, MAPE  
– Probabilistic Forecasting: CRPS, Prediction Interval Coverage Probability (PICP)  
– Robustness: performance during flagged anomalous windows (e.g., economic shocks), measured by relative MAE improvement over baselines  
– Inference Latency: ms per forecast on a single GPU  

Ablation Studies  
1. Remove adaptive gating ($\alpha$ fixed uniform).  
2. Remove cross-modal attention (concatenate modality embeddings).  
3. Exclude one modality at a time to quantify its contribution.  

Interpretability  
We will visualize attention weights in both cross-modal and gating modules to interpret which modalities or time steps drive predictions, especially during regime shifts.  

3.9 Implementation and Reproducibility  
– Codebase in PyTorch, released under an open-source license.  
– Pretrained modality encoders obtained from HuggingFace.  
– Docker containers with environment specifications.  
– Detailed logs and seed control for full reproducibility.  

4. Expected Outcomes & Impact  
We anticipate that MAFFT will:  
• Achieve 5–15% relative improvements in MAE/RMSE over state-of-the-art multimodal baselines across diverse datasets.  
• Significantly reduce forecasting errors (up to 20%) during anomalous periods thanks to dynamic weighting of external modalities.  
• Provide interpretable modality-importance scores that can guide domain experts in understanding external drivers of time series behavior.  
• Introduce a new public benchmark (“NewsTime”) linking numeric and multimodal context for financial forecasting.  

Broader Impacts  
• Energy Sector: More accurate demand forecasts that incorporate weather reports or satellite imagery, enabling better grid management.  
• Finance: Improved risk assessment by fusing market news sentiment with price histories.  
• Healthcare: Enhanced epidemic forecasting by combining case counts with social media or news trends.  

By establishing a clear methodology for fusing pretrained large models from NLP and CV into time series forecasting, MAFFT will serve as a catalyst for future research on foundation models in the time series domain.  

5. References  
[1] H. Emami et al., “Modality-aware Transformer for Financial Time Series Forecasting,” arXiv:2310.01232, 2023.  
[2] S. Zhong et al., “Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting,” arXiv:2502.04395, 2025.  
[3] C. Ding et al., “MST-GAT: A Multimodal Spatial-Temporal Graph Attention Network,” arXiv:2310.11169, 2023.  
[4] K. Kim et al., “Multi-Modal Forecaster: Jointly Predicting Time Series and Textual Data,” arXiv:2411.06735, 2024.
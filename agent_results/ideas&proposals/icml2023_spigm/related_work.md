1. **Title**: Uncertainty Modeling in Graph Neural Networks via Stochastic Differential Equations (arXiv:2408.16115)
   - **Authors**: Richard Bergna, Sergio Calvo-Ordoñez, Felix L. Opolka, Pietro Liò, Jose Miguel Hernandez-Lobato
   - **Summary**: This paper introduces Latent Graph Neural Stochastic Differential Equations (LGNSDE), enhancing Graph Neural Ordinary Differential Equations by embedding randomness through Brownian motion to quantify uncertainty. The authors provide theoretical guarantees for LGNSDE and demonstrate improved performance in uncertainty quantification.
   - **Year**: 2024

2. **Title**: Uncertainty Quantification in Graph Neural Networks with Shallow Ensembles (arXiv:2504.12627)
   - **Authors**: Tirtha Vinchurkar, Kareem Abdelmaqsoud, John R. Kitchin
   - **Summary**: The authors explore Uncertainty Quantification (UQ) techniques, focusing on Direct Propagation of Shallow Ensembles (DPOSE) as a computationally efficient alternative to deep ensembles. By integrating DPOSE into the SchNet model, they assess its ability to provide reliable uncertainty estimates across diverse datasets, demonstrating that DPOSE successfully distinguishes between in-domain and out-of-domain samples.
   - **Year**: 2025

3. **Title**: Evidential Uncertainty Probes for Graph Neural Networks (arXiv:2503.08097)
   - **Authors**: Linlin Yu, Kangshuo Li, Pritom Kumar Saha, Yifei Lou, Feng Chen
   - **Summary**: This paper proposes a plug-and-play framework for uncertainty quantification in GNNs that works with pre-trained models without the need for retraining. The Evidential Probing Network (EPN) uses a lightweight Multi-Layer-Perceptron head to extract evidence from learned representations, allowing efficient integration with various GNN architectures. The authors introduce evidence-based regularization techniques to enhance the estimation of epistemic uncertainty, achieving state-of-the-art performance in accurate and efficient uncertainty quantification.
   - **Year**: 2025

4. **Title**: Uncertainty in Graph Neural Networks: A Survey (arXiv:2403.07185)
   - **Authors**: Fangxin Wang, Yuqing Liu, Kay Liu, Yibo Wang, Sourav Medya, Philip S. Yu
   - **Summary**: This survey provides a comprehensive overview of GNNs from the perspective of uncertainty, emphasizing its integration in graph learning. The authors compare and summarize existing graph uncertainty theories and methods, bridging the gap between theory and practice, and offering valuable insights into promising directions in this field.
   - **Year**: 2024

5. **Title**: Uncertainty Quantification over Graph with Conformalized Graph Neural Networks (arXiv:2305.14535)
   - **Authors**: Kexin Huang, Ying Jin, Emmanuel Candès, Jure Leskovec
   - **Summary**: The authors propose conformalized GNN (CF-GNN), extending conformal prediction to graph-based models for guaranteed uncertainty estimates. CF-GNN produces prediction sets/intervals that provably contain the true label with a pre-defined coverage probability. They establish a permutation invariance condition that enables the validity of conformal prediction on graph data and provide an exact characterization of the test-time coverage.
   - **Year**: 2023

6. **Title**: On the Temperature of Bayesian Graph Neural Networks for Conformal Prediction (arXiv:2310.11479)
   - **Authors**: Seohyeon Cha, Honggu Kang, Joonhyuk Kang
   - **Summary**: This study explores the advantages of incorporating a temperature parameter into Bayesian GNNs within the conformal prediction framework. The authors empirically demonstrate the existence of temperatures that result in more efficient prediction sets and conduct an analysis to identify factors contributing to inefficiency, offering insights into the relationship between conformal prediction performance and model calibration.
   - **Year**: 2023

7. **Title**: Uncertainty Quantification for Molecular Property Predictions with Graph Neural Architecture Search (arXiv:2307.10438)
   - **Authors**: Shengli Jiang, Shiyi Qin, Reid C. Van Lehn, Prasanna Balaprakash, Victor M. Zavala
   - **Summary**: The authors introduce AutoGNNUQ, an automated uncertainty quantification approach for molecular property prediction. AutoGNNUQ leverages architecture search to generate an ensemble of high-performing GNNs, enabling the estimation of predictive uncertainties. They employ variance decomposition to separate data (aleatoric) and model (epistemic) uncertainties, providing valuable insights for reducing them.
   - **Year**: 2023

8. **Title**: Energy-based Epistemic Uncertainty for Graph Neural Networks (arXiv:2406.04043)
   - **Authors**: Dominik Fuchsgruber, Tom Wollschläger, Stephan Günnemann
   - **Summary**: This paper proposes GEBM, an energy-based model that provides high-quality uncertainty estimates by aggregating energy at different structural levels that naturally arise from graph diffusion. The authors introduce an evidential interpretation of their EBM that significantly improves the predictive robustness of the GNN. Their framework is a simple and effective post hoc method applicable to any pre-trained GNN, sensitive to various distribution shifts.
   - **Year**: 2024

9. **Title**: Uncertainty Quantification of Spatiotemporal Travel Demand with Probabilistic Graph Neural Networks (arXiv:2303.04040)
   - **Authors**: Qingyi Wang, Shenhao Wang, Dingyi Zhuang, Haris Koutsopoulos, Jinhua Zhao
   - **Summary**: The authors propose a framework of probabilistic graph neural networks (Prob-GNN) to quantify the spatiotemporal uncertainty of travel demand. They empirically apply this framework to predict transit and ridesharing demand in Chicago, demonstrating that Prob-GNNs can predict ridership uncertainty in a stable manner, even under significant domain shifts.
   - **Year**: 2023

10. **Title**: Deep-Ensemble-Based Uncertainty Quantification in Spatiotemporal Graph Neural Networks for Traffic Forecasting (arXiv:2204.01618)
    - **Authors**: Tanwi Mallick, Prasanna Balaprakash, Jane Macfarlane
    - **Summary**: This paper develops a scalable deep ensemble approach to quantify uncertainties for diffusion convolutional recurrent neural networks (DCRNN) in traffic forecasting. The authors demonstrate the efficacy of their methods by comparing them with other uncertainty estimation techniques, showing that their approach outperforms current state-of-the-art Bayesian and other commonly used frequentist techniques.
    - **Year**: 2022

**Key Challenges**:

1. **Integration of Uncertainty Quantification into GNN Architectures**: Many existing methods treat uncertainty quantification as an add-on rather than integrating it into the core GNN architecture, leading to less reliable confidence estimates.

2. **Distinguishing Between Aleatoric and Epistemic Uncertainty**: Separating data-related uncertainty (aleatoric) from model-related uncertainty (epistemic) remains challenging, affecting the interpretability and reliability of uncertainty estimates.

3. **Scalability and Computational Efficiency**: Implementing uncertainty quantification methods, especially those involving ensembles or Bayesian approaches, can be computationally intensive, hindering their scalability to large datasets.

4. **Handling Out-of-Distribution Data**: Ensuring that GNNs provide reliable uncertainty estimates when encountering out-of-distribution data is crucial for robust decision-making but remains a significant challenge.

5. **Empirical Validation Across Diverse Applications**: Demonstrating the effectiveness of uncertainty-aware GNNs across various real-world applications, such as drug discovery, traffic forecasting, and social network analysis, requires extensive empirical validation. 
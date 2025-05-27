1. **Title**: A Self-Supervised Learning-based Approach to Clustering Multivariate Time-Series Data with Missing Values (SLAC-Time): An Application to TBI Phenotyping (arXiv:2302.13457)
   - **Authors**: Hamid Ghaderi, Brandon Foreman, Amin Nayebi, Sindhu Tipirneni, Chandan K. Reddy, Vignesh Subbian
   - **Summary**: SLAC-Time introduces a Transformer-based clustering method that leverages self-supervised learning to handle multivariate time-series data with missing values. By using time-series forecasting as a proxy task, it learns robust representations without the need for imputation, which is particularly beneficial in clinical settings where missing data is common. The approach was validated on Traumatic Brain Injury (TBI) patient data, demonstrating superior clustering performance and the identification of clinically significant phenotypes.
   - **Year**: 2023

2. **Title**: Multi-Modal Contrastive Learning for Online Clinical Time-Series Applications (arXiv:2403.18316)
   - **Authors**: Fabian Baldenweg, Manuel Burger, Gunnar Rätsch, Rita Kuznetsova
   - **Summary**: This study applies self-supervised multi-modal contrastive learning techniques to ICU data, focusing on integrating clinical notes and time-series data for online prediction tasks. The proposed Multi-Modal Neighborhood Contrastive Loss (MM-NCL) enhances the learning of representations that are effective for downstream clinical applications, showcasing strong performance in both linear probe and zero-shot settings.
   - **Year**: 2024

3. **Title**: Self-Supervised Transformer for Sparse and Irregularly Sampled Multivariate Clinical Time-Series (arXiv:2107.14293)
   - **Authors**: Sindhu Tipirneni, Chandan K. Reddy
   - **Summary**: The authors propose STraTS, a self-supervised Transformer model designed to handle sparse and irregularly sampled multivariate clinical time-series data. By treating time-series as a set of observation triplets and employing a Continuous Value Embedding technique, STraTS effectively learns contextual embeddings without the need for imputation. The model demonstrated improved mortality prediction performance, especially in scenarios with limited labeled data.
   - **Year**: 2021

4. **Title**: As easy as APC: overcoming missing data and class imbalance in time series with self-supervised learning (arXiv:2106.15577)
   - **Authors**: Fiorella Wever, T. Anderson Keller, Laura Symul, Victor Garcia
   - **Summary**: This work demonstrates how Autoregressive Predictive Coding (APC), a self-supervised training method, can address challenges of missing data and class imbalance in time-series data without making strong assumptions about the data generation process. The approach showed substantial improvements over standard baselines, particularly in settings with high missingness and severe class imbalance, and achieved state-of-the-art results on the Physionet benchmark.
   - **Year**: 2021

**Key Challenges:**

1. **Handling Missing and Irregular Data**: Clinical time-series data often contain missing values and are sampled irregularly, complicating the development of robust models.

2. **Limited Labeled Data**: The scarcity of labeled data in clinical settings hinders the training of supervised models, necessitating label-efficient learning methods.

3. **Interpretability and Trust**: Ensuring that models provide interpretable and trustworthy outputs is crucial for clinician adoption and effective decision-making.

4. **High Dimensionality**: The high-dimensional nature of multivariate time-series data poses challenges in learning meaningful representations without overfitting.

5. **Class Imbalance**: Clinical datasets often exhibit class imbalance, which can bias model predictions and affect performance on minority classes. 
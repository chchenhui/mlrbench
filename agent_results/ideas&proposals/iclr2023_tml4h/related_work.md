1. **Title**: MDA: An Interpretable and Scalable Multi-Modal Fusion under Missing Modalities and Intrinsic Noise Conditions (arXiv:2406.10569)
   - **Authors**: Lin Fan, Yafei Ou, Cenyang Zheng, Pengyu Dai, Tamotsu Kamishima, Masayuki Ikebe, Kenji Suzuki, Xun Gong
   - **Summary**: This paper introduces the Modal-Domain Attention (MDA) model, designed to address challenges in multi-modal learning, including modality heterogeneity, missing data, intrinsic noise, and interpretability. MDA employs continuous attention mechanisms to dynamically allocate attention to different modalities, reducing the impact of low-correlation data, missing modalities, or noisy inputs. The model demonstrates state-of-the-art performance across various tasks on multiple public datasets and aligns with established clinical diagnostic standards.
   - **Year**: 2024

2. **Title**: Multimodal Fusion Learning with Dual Attention for Medical Imaging (arXiv:2412.01248)
   - **Authors**: Joy Dhar, Nayyar Zaidi, Maryam Haghighat, Puneet Goyal, Sudipta Roy, Azadeh Alavi, Vikas Kumar
   - **Summary**: The authors propose DRIFA-Net, a multimodal fusion learning framework that integrates dual attention mechanisms: a multi-branch fusion attention module and a multimodal information fusion attention module. DRIFA-Net enhances representations for each modality and refines shared multimodal representations, improving generalization across multiple diagnostic tasks. An ensemble Monte Carlo dropout strategy is employed to estimate prediction uncertainty. Experiments on five publicly available datasets demonstrate consistent outperformance over state-of-the-art methods.
   - **Year**: 2024

3. **Title**: HEALNet: Multimodal Fusion for Heterogeneous Biomedical Data (arXiv:2311.09115)
   - **Authors**: Konstantin Hemker, Nikola Simidjievski, Mateja Jamnik
   - **Summary**: HEALNet is a flexible multimodal fusion architecture that preserves modality-specific structural information and captures cross-modal interactions in a shared latent space. It effectively handles missing modalities during training and inference and enables intuitive model inspection by learning directly from raw data inputs. The model achieves state-of-the-art performance in multimodal survival analysis on four cancer datasets from The Cancer Genome Atlas (TCGA), demonstrating robustness in scenarios with missing modalities.
   - **Year**: 2023

4. **Title**: DrFuse: Learning Disentangled Representation for Clinical Multi-Modal Fusion with Missing Modality and Modal Inconsistency (arXiv:2403.06197)
   - **Authors**: Wenfang Yao, Kejing Yin, William K. Cheung, Jia Liu, Jing Qin
   - **Summary**: DrFuse addresses challenges in fusing electronic health records (EHR) and medical images, such as missing modalities and modal inconsistency. The model disentangles shared and unique features across modalities and employs a disease-wise attention layer to produce patient- and disease-specific modality weightings for predictions. Validation on large-scale datasets (MIMIC-IV and MIMIC-CXR) shows significant performance improvements over state-of-the-art models.
   - **Year**: 2024

**Key Challenges:**

1. **Modality Heterogeneity**: Integrating diverse medical data types (e.g., imaging, EHRs) poses challenges due to differences in data structures, scales, and distributions.

2. **Missing Data**: Real-world medical datasets often have incomplete modalities, requiring models to handle missing information without compromising performance.

3. **Intrinsic Noise**: Medical data can be noisy due to factors like imaging artifacts or recording errors, necessitating robust models that can mitigate the impact of such noise.

4. **Interpretability**: Ensuring that multi-modal fusion models provide interpretable outputs is crucial for clinical trust and adoption.

5. **Uncertainty Estimation**: Accurately quantifying uncertainty in predictions is essential for identifying low-confidence cases and enhancing model reliability in clinical settings. 
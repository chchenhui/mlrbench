1. **Title**: Mining Gaze for Contrastive Learning toward Computer-Assisted Diagnosis (arXiv:2312.06069)
   - **Authors**: Zihao Zhao, Sheng Wang, Qian Wang, Dinggang Shen
   - **Summary**: This paper introduces the Medical contrastive Gaze Image Pre-training (McGIP) framework, which leverages radiologists' eye-tracking data to guide contrastive learning in medical imaging. By treating images with similar gaze patterns as positive pairs, McGIP enhances the pre-training of computer-assisted diagnosis networks without relying on extensive labeled datasets.
   - **Year**: 2023

2. **Title**: Learning Better Contrastive View from Radiologist's Gaze (arXiv:2305.08826)
   - **Authors**: Sheng Wang, Zixu Zhuang, Xi Ouyang, Lichi Zhang, Zheren Li, Chong Ma, Tianming Liu, Dinggang Shen, Qian Wang
   - **Summary**: The authors propose FocusContrast, an augmentation method that utilizes radiologists' gaze data to generate attention-aware views for contrastive learning. This approach aims to preserve disease-related abnormalities during augmentation, thereby improving the performance of self-supervised models in medical image analysis.
   - **Year**: 2023

3. **Title**: GazeGNN: A Gaze-Guided Graph Neural Network for Chest X-ray Classification (arXiv:2305.18221)
   - **Authors**: Bin Wang, Hongyi Pan, Armstrong Aboah, Zheyuan Zhang, Elif Keles, Drew Torigian, Baris Turkbey, Elizabeth Krupinski, Jayaram Udupa, Ulas Bagci
   - **Summary**: GazeGNN introduces a graph neural network that integrates raw eye-gaze data directly into the classification of chest X-ray images. By constructing a unified representation graph of images and gaze patterns, the model enables real-time, end-to-end disease classification without the need for preprocessing gaze data into visual attention maps.
   - **Year**: 2023

4. **Title**: Enhancing Contrastive Learning for Retinal Imaging via Adjusted Augmentation Scales (arXiv:2501.02451)
   - **Authors**: Zijie Cheng, Boxuan Li, André Altmann, Pearse A Keane, Yukun Zhou
   - **Summary**: This study investigates the impact of augmentation strategies on contrastive learning in retinal imaging. The authors find that optimizing the scale of augmentation is crucial, as weak augmentations outperform strong ones, leading to improved model performance across multiple datasets.
   - **Year**: 2025

**Key Challenges**:

1. **Limited Availability of Eye-Tracking Data**: Collecting comprehensive eye-tracking datasets from radiologists is resource-intensive and may not cover all medical imaging modalities, limiting the generalizability of gaze-guided models.

2. **Variability in Gaze Patterns**: Individual differences in radiologists' gaze behaviors can introduce variability, making it challenging to develop models that generalize well across different practitioners.

3. **Integration Complexity**: Incorporating gaze data into existing machine learning frameworks requires sophisticated integration techniques, which can complicate model development and deployment.

4. **Data Privacy Concerns**: Utilizing eye-tracking data raises privacy issues, as gaze patterns may inadvertently reveal sensitive information about the radiologists or patients.

5. **Scalability of Gaze-Guided Models**: Ensuring that models trained with gaze data can scale effectively to large datasets and diverse clinical settings remains a significant challenge. 
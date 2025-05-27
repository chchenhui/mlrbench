1. **Title**: InvisMark: Invisible and Robust Watermarking for AI-generated Image Provenance (arXiv:2411.07795)
   - **Authors**: Rui Xu, Mengya Hu, Deren Lei, Yaxi Li, David Lowe, Alex Gorevski, Mingyu Wang, Emily Ching, Alex Deng
   - **Summary**: InvisMark introduces a watermarking technique for high-resolution AI-generated images, embedding imperceptible yet robust watermarks using advanced neural network architectures. The method achieves high imperceptibility (PSNR ~51, SSIM ~0.998) and maintains over 97% bit accuracy across various image manipulations, enabling reliable media provenance.
   - **Year**: 2024

2. **Title**: GenPTW: In-Generation Image Watermarking for Provenance Tracing and Tamper Localization (arXiv:2504.19567)
   - **Authors**: Zhenliang Gan, Chunya Liu, Yichao Tang, Binghao Wang, Weiqiang Wang, Xinpeng Zhang
   - **Summary**: GenPTW presents an in-generation image watermarking framework for latent diffusion models, embedding structured watermark signals during image generation. This approach enables unified provenance tracing and tamper localization, demonstrating superior performance in image fidelity, watermark extraction accuracy, and tamper localization compared to existing methods.
   - **Year**: 2025

3. **Title**: Provenance Detection for AI-Generated Images: Combining Perceptual Hashing, Homomorphic Encryption, and AI Detection Models (arXiv:2503.11195)
   - **Authors**: Shree Singhi, Aayan Yadav, Aayush Gupta, Shariar Ebrahimi, Parisa Hassanizadeh
   - **Summary**: This paper introduces a three-part framework for secure, transformation-resilient AI content provenance detection. It combines an adversarially robust perceptual hashing model (DinoHash), a Multi-Party Fully Homomorphic Encryption scheme, and an improved AI-generated media detection model, achieving significant improvements in robustness and privacy over previous methods.
   - **Year**: 2025

4. **Title**: Watermark-based Attribution of AI-Generated Content (arXiv:2404.04254)
   - **Authors**: Zhengyuan Jiang, Moyang Guo, Yuepeng Hu, Neil Zhenqiang Gong
   - **Summary**: This study explores watermark-based user-level attribution of AI-generated content by assigning unique watermarks to each user of a generative AI service. The approach demonstrates high accuracy in attribution, even under common post-processing techniques like JPEG compression, highlighting the potential for tracing AI-generated content back to individual users.
   - **Year**: 2024

5. **Title**: Watermarking across Modalities for Content Tracing and Generative AI (arXiv:2502.05215)
   - **Authors**: Pierre Fernandez
   - **Summary**: This thesis develops new watermarking techniques for images, audio, and text, focusing on AI-generated content. It introduces methods to adapt latent generative models to embed watermarks across different modalities, enhancing content tracing and monitoring the usage of AI models, thereby addressing challenges posed by the increasing use of generative AI.
   - **Year**: 2025

6. **Title**: Generative Models are Self-Watermarked: Declaring Model Authentication through Re-Generation (arXiv:2402.16889)
   - **Authors**: Aditya Desu, Xuanli He, Qiongkai Xu, Wei Lu
   - **Summary**: This work proposes a methodology to detect data reuse from individual samples by identifying latent fingerprints inherently present within the outputs of generative models through re-generation. The approach provides a tool to ensure the integrity of sources and authorship, expanding its application in scenarios where authenticity and ownership verification are essential.
   - **Year**: 2024

7. **Title**: Evading Watermark based Detection of AI-Generated Content (arXiv:2305.03807)
   - **Authors**: Zhengyuan Jiang, Jinghuai Zhang, Neil Zhenqiang Gong
   - **Summary**: This study systematically examines the robustness of watermark-based AI-generated content detection. It demonstrates that attackers can add small, human-imperceptible perturbations to watermarked images, effectively evading detection while maintaining visual quality, thereby highlighting the insufficiency of existing watermark-based detection methods.
   - **Year**: 2023

8. **Title**: Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models (Cryptology ePrint Archive: 2023/1776)
   - **Authors**: Hanlin Zhang, Benjamin L. Edelman, Danilo Francati, Daniele Venturi, Giuseppe Ateniese, Boaz Barak
   - **Summary**: This paper proves that, under natural assumptions, strong watermarking schemes for generative models are impossible to achieve. The findings suggest that any computationally bounded attacker can remove watermarks without significant quality degradation, even when detection algorithms share a secret key unknown to the attacker.
   - **Year**: 2023

9. **Title**: Copyright Protection and Accountability of Generative AI: Attack, Watermarking and Attribution (arXiv:2303.09272)
   - **Authors**: Haonan Zhong, Jiamin Chang, Ziyue Yang, Tingmin Wu, Pathum Chamikara Mahawaga Arachchige, Chehara Pathmabandu, Minhui Xue
   - **Summary**: This paper evaluates the current state of copyright protection measures for generative adversarial networks (GANs), assessing their performance across various architectures. It identifies that while existing methods are satisfactory for input images and model watermarking, they fall short in protecting training sets, indicating a need for robust IPR protection and provenance tracing in this area.
   - **Year**: 2023

10. **Title**: Watermarking Vision-Language Pre-trained Models for Multi-modal Embedding as a Service (arXiv:2311.05863)
    - **Authors**: Yuanmin Tang, et al.
    - **Summary**: This study introduces VLPMarker, a backdoor-based embedding watermarking method for vision-language pre-trained models. It utilizes embedding orthogonal transformation to inject triggers without interfering with model parameters, achieving high-quality copyright verification and minimal impact on model performance, and demonstrates robustness against model extraction attacks.
    - **Year**: 2023

**Key Challenges:**

1. **Cross-Modal Watermarking**: Developing watermarking techniques that are effective across different modalities (text, image, video, audio) remains a significant challenge. Ensuring that a watermark embedded in one modality can be reliably detected in another requires sophisticated encoding and decoding strategies.

2. **Robustness Against Manipulations**: Watermarks must withstand various post-processing operations such as compression, cropping, and format changes. Designing watermarks that are imperceptible yet resilient to such manipulations is complex and often leads to trade-offs between robustness and imperceptibility.

3. **Scalability and Efficiency**: Embedding and detecting watermarks in large-scale multi-modal generative models require efficient algorithms that do not significantly impact the performance or output quality of the models. Balancing computational efficiency with watermark robustness is a critical challenge.

4. **Security Against Adversarial Attacks**: Watermarking schemes are vulnerable to adversarial attacks aimed at removing or altering the watermark without degrading content quality. Developing methods that can resist such attacks is essential for reliable content provenance.

5. **Standardization and Adoption**: The lack of standardized protocols for watermarking AI-generated content hinders widespread adoption. Establishing industry standards and ensuring interoperability between different systems is necessary for effective implementation. 
1. **Title**: MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators (arXiv:2504.02658)
   - **Authors**: Beichen Huang, Yueming Yuan, Zelei Shao, Minjia Zhang
   - **Summary**: MiLo introduces a method to enhance highly quantized Mixture-of-Experts (MoE) models by integrating low-rank compensators. These compensators, requiring minimal additional memory, effectively recover accuracy lost due to extreme quantization. The approach includes adaptive rank selection and iterative optimization, eliminating the need for calibration data and ensuring generalizability across various MoE models and datasets. Additionally, MiLo develops hardware-friendly 3-bit kernels to address inefficiencies associated with extreme quantization, achieving notable latency speedups.
   - **Year**: 2025

2. **Title**: MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More (arXiv:2410.06270)
   - **Authors**: Wei Huang, Yue Liao, Jianhui Liu, Ruifei He, Haoru Tan, Shiming Zhang, Hongsheng Li, Si Liu, Xiaojuan Qi
   - **Summary**: MC-MoE addresses the challenges of memory consumption and loading latency in MoE large language models by introducing a training-free Mixture-Compressor. The method employs Pre-Loading Mixed-Precision Quantization, formulating adaptive bit-width allocation as a Linear Programming problem to balance multiple factors reflecting each expert's importance. Additionally, Online Dynamic Pruning identifies and retains critical tokens, dynamically selecting activated experts during inference to optimize efficiency while maintaining performance. This integrated approach achieves significant compression with minimal accuracy loss.
   - **Year**: 2024

3. **Title**: MoQa: Rethinking MoE Quantization with Multi-stage Data-model Distribution Awareness (arXiv:2503.21135)
   - **Authors**: Zihao Zheng, Xiuping Cui, Size Zheng, Maoliang Li, Jiayu Chen, Yun Liang, Xiang Chen
   - **Summary**: MoQa presents a quantization framework that decouples the complexities of data-model distribution in MoEs through multi-stage analysis. By quantitatively revealing dynamics during sparse data activation, data-parameter mapping, and inter-expert correlations, MoQa identifies the significance of specific experts and parameters. The framework proposes fine-grained mix-quantization strategies adaptive to various data activation and expert combination scenarios, leading to improved performance in language modeling and zero-shot inference tasks.
   - **Year**: 2025

4. **Title**: Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness (arXiv:2310.02410)
   - **Authors**: Young Jin Kim, Raffy Fahim, Hany Hassan Awadalla
   - **Summary**: MoQE introduces a weight-only quantization method applying ultra low-bit (down to 2-bit) quantization to expert weights in MoE models. This approach mitigates increased memory and latency issues without additional training in most cases. The study finds that expert layers in MoE models are more robust to quantization than conventional feedforward network layers, achieving significant model size reduction and inference speedup while maintaining or even improving performance compared to dense models trained on the same dataset.
   - **Year**: 2023

5. **Title**: Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints (arXiv:2302.01717)
   - **Authors**: Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa
   - **Summary**: This paper introduces a method for converting dense model checkpoints into sparse Mixture-of-Experts (MoE) models, termed "sparse upcycling." The approach involves training MoE models from existing dense checkpoints, leveraging the benefits of sparsity to improve efficiency and scalability. The study demonstrates that sparse upcycling can achieve competitive performance with reduced computational resources, facilitating the deployment of large-scale models on resource-constrained hardware.
   - **Year**: 2023

6. **Title**: OLMoE: Open Mixture-of-Experts Language Models (arXiv:2409.00345)
   - **Authors**: Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison
   - **Summary**: OLMoE presents an open-source Mixture-of-Experts language model designed to facilitate research and development in the field. The model incorporates sparsity and modularity, enabling efficient scaling and adaptability. The paper discusses the architecture, training methodology, and potential applications of OLMoE, highlighting its performance on various language tasks and its suitability for deployment on resource-constrained hardware.
   - **Year**: 2024

7. **Title**: Scaling Vision with Sparse Mixture of Experts (arXiv:2106.05974)
   - **Authors**: Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton
   - **Summary**: This study explores the application of sparse Mixture-of-Experts (MoE) models in computer vision tasks. By leveraging sparsity, the authors demonstrate that MoE models can scale efficiently, achieving state-of-the-art performance on various vision benchmarks. The paper discusses the architectural considerations, training strategies, and potential benefits of using sparse MoE models in vision applications.
   - **Year**: 2021

8. **Title**: GLaM: Efficient Scaling of Language Models with Mixture-of-Experts (arXiv:2112.06905)
   - **Authors**: Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin
   - **Summary**: GLaM introduces a Mixture-of-Experts model that efficiently scales language models by leveraging sparsity. The model activates only a subset of experts for each input, reducing computational requirements while maintaining high performance. The paper details the architecture, training process, and evaluation results, demonstrating the effectiveness of GLaM in scaling language models efficiently.
   - **Year**: 2021

9. **Title**: No Language Left Behind: Scaling Human-Centered Machine Translation (arXiv:2207.04672)
   - **Authors**: NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad
   - **Summary**: This paper presents a large-scale multilingual machine translation model that utilizes Mixture-of-Experts (MoE) to efficiently handle numerous languages. The model achieves high-quality translations across 200 languages by leveraging sparsity and modularity inherent in MoE architectures. The study discusses the challenges, methodologies, and outcomes of scaling machine translation models using MoE.
   - **Year**: 2022

10. **Title**: Mixture-of-Experts Meets Instruction Tuning: A Winning Combination for Large Language Models (arXiv:2303.01547)
    - **Authors**: Sheng Shen, Le Hou, Yanqi Zhou, Nan Du, Shayne Longpre
    - **Summary**: This paper explores the integration of Mixture-of-Experts (MoE) architectures with instruction tuning to enhance the performance of large language models. By combining MoE's efficiency with instruction tuning's adaptability, the study demonstrates improvements in various language tasks. The paper provides insights into the training process, architectural choices, and evaluation metrics, highlighting the benefits of this combined approach.
    - **Year**: 2023

**Key Challenges:**

1. **Quantization-Induced Accuracy Degradation**: Applying aggressive quantization to MoE models often leads to significant accuracy loss, especially when reducing precision to extreme levels (e.g., 2-bit or 3-bit). Developing methods to mitigate this degradation while maintaining computational efficiency remains a critical challenge.

2. **Adaptive Bit-Width Allocation**: Determining the optimal bit-width for each expert based on activation frequency and contribution to model outputs is complex. Static quantization schemes fail to exploit the variability among experts, necessitating dynamic and adaptive approaches to balance performance and efficiency.

3. **Hardware Inefficiencies with Extreme Quantization**: Implementing extreme quantization can introduce hardware inefficiencies, such as increased latency or incompatibility with existing hardware accelerators. Developing hardware-friendly quantization methods that align with current hardware capabilities is essential.

4. **Data-Model Distribution Complexity**: MoE models exhibit intricate data-model distributions due to their sparse activation patterns and dynamic expert selection. Accurately modeling and quantizing these distributions without extensive calibration data poses a significant challenge.

5. **Balancing Compression and Performance**: Achieving substantial model compression through quantization and pruning while preserving or enhancing model performance is a delicate balance. Over-compression can lead to performance degradation, whereas under-compression may not yield the desired efficiency gains. 
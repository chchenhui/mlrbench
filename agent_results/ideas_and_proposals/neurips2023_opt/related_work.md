1. **Title**: Predictable Scale: Part I -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining (arXiv:2503.04715)
   - **Authors**: Houyi Li, Wenzheng Zheng, Jingcheng Hu, Qiufeng Wang, Hanshan Zhang, Zili Wang, Shijie Xuyang, Yuantao Fan, Shuigeng Zhou, Xiangyu Zhang, Daxin Jiang
   - **Summary**: This paper presents universal scaling laws for hyperparameters in LLM pretraining, revealing that optimal learning rates follow a power-law relationship with model parameters and data sizes. The authors provide a plug-and-play tool that estimates optimal hyperparameters, achieving performance within 0.09% of the global optimum found via exhaustive search.
   - **Year**: 2025

2. **Title**: Optimization Hyper-parameter Laws for Large Language Models (arXiv:2409.04777)
   - **Authors**: Xingyu Xie, Kuangyu Ding, Shuicheng Yan, Kim-Chuan Toh, Tianwen Wei
   - **Summary**: The authors introduce Optimization Hyper-parameter Laws (Opt-Laws), a framework that captures the relationship between hyperparameters and training outcomes in LLMs. Grounded in stochastic differential equations, Opt-Laws enable the pre-selection of optimal learning rate schedules, reducing computational costs and enhancing model performance.
   - **Year**: 2024

3. **Title**: Scaling Optimal LR Across Token Horizons (arXiv:2409.19913)
   - **Authors**: Johan Bjorck, Alon Benhaim, Vishrav Chaudhary, Furu Wei, Xia Song
   - **Summary**: This study examines how optimal learning rates change with token horizons in LLM training. The authors demonstrate that longer training necessitates smaller learning rates and propose scaling laws to accurately estimate optimal learning rates for extended training durations.
   - **Year**: 2024

4. **Title**: Language models scale reliably with over-training and on downstream tasks (arXiv:2403.08540)
   - **Authors**: Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, Suchin Gururangan, Mitchell Wortsman, Rulin Shao, Jean Mercat, Alex Fang, Jeffrey Li, Sedrick Keh, Rui Xin, Marianna Nezhurina, Igor Vasiljevic, Jenia Jitsev, Luca Soldaini, Alexandros G. Dimakis, Gabriel Ilharco, Pang Wei Koh, Shuran Song, Thomas Kollar, Yair Carmon, Achal Dave, Reinhard Heckel, Niklas Muennighoff, Ludwig Schmidt
   - **Summary**: The authors investigate scaling laws in the context of over-training and downstream task performance. They fit scaling laws that extrapolate in both the amount of over-training and the number of model parameters, enabling predictions of validation loss and downstream task performance with reduced computational resources.
   - **Year**: 2024

5. **Title**: Scaling Laws for Neural Machine Translation (arXiv:2109.00102)
   - **Authors**: Behrooz Ghorbani, Orhan Firat, Markus Freitag, Ankur Bapna, Maxim Krikun
   - **Summary**: This paper explores scaling laws in neural machine translation, analyzing how model performance scales with parameters and data size. The authors provide insights into optimal model configurations and training strategies for efficient scaling in machine translation tasks.
   - **Year**: 2021

6. **Title**: Scaling Vision Transformers (arXiv:2106.04560)
   - **Authors**: Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, Lucas Beyer
   - **Summary**: The authors study scaling laws for vision transformers, examining how model performance scales with size and data. They provide empirical evidence and theoretical insights into the efficient scaling of vision transformers, which can inform strategies for scaling LLMs.
   - **Year**: 2021

7. **Title**: Scaling Laws for Transfer (arXiv:2102.01293)
   - **Authors**: Danny Hernandez, Jared Kaplan, Tom Henighan, Sam McCandlish
   - **Summary**: This study investigates scaling laws in the context of transfer learning, analyzing how pretraining on different data distributions affects downstream performance. The authors provide guidelines for efficient transfer learning strategies in LLMs.
   - **Year**: 2021

8. **Title**: Scaling Data-Constrained Language Models (arXiv:2305.16264)
   - **Authors**: Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi
   - **Summary**: The authors examine scaling laws when data is limited, proposing methods to optimize model performance under data constraints. Their findings are relevant for developing efficient learning rate schedules in data-scarce scenarios.
   - **Year**: 2023

9. **Title**: Beyond neural scaling laws: beating power law scaling via data pruning (arXiv:2304.10439)
   - **Authors**: Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, Ari S. Morcos
   - **Summary**: This paper challenges traditional scaling laws by demonstrating that data pruning can lead to improved performance beyond expected power-law scaling. The findings suggest that strategic data selection can enhance training efficiency in LLMs.
   - **Year**: 2023

10. **Title**: Scaling Laws for Precision (arXiv:2411.12345)
    - **Authors**: Tanishq Kumar, Zachary Ankner, Benjamin F. Spector, Blake Bordelon, Niklas Muennighoff
    - **Summary**: The authors study scaling laws related to numerical precision in LLM training, analyzing how precision affects model performance and training efficiency. Their insights can inform decisions on precision settings for optimal learning rate schedules.
    - **Year**: 2024

**Key Challenges:**

1. **Hyperparameter Sensitivity**: LLM training is highly sensitive to hyperparameter choices, particularly learning rates. Developing adaptive scaling laws requires precise modeling of these sensitivities to ensure optimal performance across different model sizes and architectures.

2. **Computational Cost**: Establishing scaling laws often involves extensive empirical studies, which are computationally expensive. Efficiently deriving these laws without prohibitive resource consumption remains a significant challenge.

3. **Generalization Across Architectures**: Scaling laws derived for specific model architectures may not generalize well to others. Ensuring that adaptive learning rate scaling methods are applicable across diverse architectures is crucial for their broad utility.

4. **Data Quality and Quantity**: The effectiveness of scaling laws can be influenced by the quality and quantity of training data. Variations in data distributions and availability can impact the applicability of derived scaling laws.

5. **Overfitting and Underfitting**: Balancing the risk of overfitting and underfitting when applying scaling laws is challenging. Adaptive methods must account for these risks to maintain model generalization and performance. 
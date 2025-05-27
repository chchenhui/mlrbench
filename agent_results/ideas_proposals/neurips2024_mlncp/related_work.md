1. **Title**: Towards training digitally-tied analog blocks via hybrid gradient computation (arXiv:2409.03306)
   - **Authors**: Timothy Nest, Maxence Ernoult
   - **Summary**: This paper introduces Feedforward-tied Energy-based Models (ff-EBMs), a hybrid model combining feedforward and energy-based blocks to account for digital and analog circuits. The authors derive a novel algorithm to compute gradients end-to-end in ff-EBMs by backpropagating and "eq-propagating" through feedforward and energy-based parts, enabling Equilibrium Propagation to be applied to more flexible architectures. They demonstrate the effectiveness of this approach on ImageNet32, establishing new state-of-the-art performance in the Equilibrium Propagation literature.
   - **Year**: 2024

2. **Title**: The Promise of Analog Deep Learning: Recent Advances, Challenges and Opportunities (arXiv:2406.12911)
   - **Authors**: Aditya Datar, Pramit Saha
   - **Summary**: This survey evaluates eight distinct analog deep learning methodologies across multiple parameters, including accuracy, application domains, algorithmic advancements, computational speed, and energy efficiency. The authors identify neural network-based experiments implemented using these hardware devices and discuss comparative performance achieved by different analog deep learning methods, along with an analysis of their current limitations. They conclude that while analog deep learning has great potential for future applications, scalability remains a significant challenge.
   - **Year**: 2024

3. **Title**: Physics-Informed Machine Learning: A Survey on Problems, Methods and Applications (arXiv:2211.08064)
   - **Authors**: Zhongkai Hao, Songming Liu, Yichi Zhang, Chengyang Ying, Yao Feng, Hang Su, Jun Zhu
   - **Summary**: This survey presents the learning paradigm of Physics-Informed Machine Learning (PIML), which integrates empirical data and physical prior knowledge to improve performance on tasks involving physical mechanisms. The authors systematically review recent developments in PIML from perspectives of machine learning tasks, representation of physical prior, and methods for incorporating physical prior. They propose several open research problems and argue that encoding different forms of physical prior into model architectures, optimizers, and inference algorithms is far from being fully explored.
   - **Year**: 2022

4. **Title**: Deep physical neural networks enabled by a backpropagation algorithm for arbitrary physical systems (arXiv:2104.13386)
   - **Authors**: Logan G. Wright, Tatsuhiro Onodera, Martin M. Stein, Tianyu Wang, Darren T. Schachter, Zoey Hu, Peter L. McMahon
   - **Summary**: This paper introduces a hybrid physical-digital algorithm called Physics-Aware Training to efficiently train sequences of controllable physical systems to act as deep neural networks. The method automatically trains the functionality of any sequence of real physical systems directly using backpropagation. The authors demonstrate physical neural networks with optical, mechanical, and electrical systems, suggesting that such networks may facilitate unconventional machine learning hardware that is orders of magnitude faster and more energy-efficient than conventional electronic processors.
   - **Year**: 2021

**Key Challenges**:

1. **Hardware Imperfections**: Analog and neuromorphic hardware often suffer from noise, device mismatch, and limited precision, which can degrade the performance of machine learning models deployed on such systems.

2. **Scalability**: While analog deep learning shows promise, scaling these systems to handle large-scale models and datasets remains a significant challenge due to hardware constraints and design complexities.

3. **Integration of Physical Priors**: Effectively incorporating physical laws and constraints into machine learning models to guide learning and ensure physically plausible solutions is an ongoing research challenge.

4. **Training Algorithms**: Developing efficient and robust training algorithms that can handle the unique characteristics of analog hardware, such as noise and non-linearity, is crucial for the success of analog deep learning systems.

5. **Energy Efficiency**: While analog hardware has the potential for energy-efficient computation, optimizing these systems to achieve significant energy savings without compromising performance is a complex task. 
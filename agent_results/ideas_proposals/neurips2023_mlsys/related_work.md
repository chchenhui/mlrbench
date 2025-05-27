1. **Title**: CarbonClipper: Optimal Algorithms for Carbon-Aware Spatiotemporal Workload Management (arXiv:2408.07831)
   - **Authors**: Adam Lechowicz, Nicolas Christianson, Bo Sun, Noman Bashir, Mohammad Hajiesmaili, Adam Wierman, Prashant Shenoy
   - **Summary**: This paper introduces CarbonClipper, an online algorithm designed for carbon-aware spatiotemporal workload management in data centers. The algorithm addresses the challenge of scheduling workloads across a network with varying carbon intensities and movement costs, aiming to minimize carbon emissions while considering deadline constraints. The authors provide theoretical guarantees for the algorithm's performance and demonstrate its effectiveness in reducing carbon emissions through simulations on a global data center network.
   - **Year**: 2024

2. **Title**: Carbon- and Precedence-Aware Scheduling for Data Processing Clusters (arXiv:2502.09717)
   - **Authors**: Adam Lechowicz, Rohan Shenoy, Noman Bashir, Mohammad Hajiesmaili, Adam Wierman, Christina Delimitrou
   - **Summary**: The authors present PCAPS, a scheduler that integrates carbon-awareness with task precedence constraints in data processing clusters. By considering both the carbon intensity of energy sources and the dependencies between tasks, PCAPS aims to reduce the carbon footprint of data processing jobs without significantly impacting job completion times. The scheduler is evaluated on a Kubernetes cluster, showing up to a 32.9% reduction in carbon emissions.
   - **Year**: 2025

3. **Title**: CarbonScaler: Leveraging Cloud Workload Elasticity for Optimizing Carbon-Efficiency (arXiv:2302.08681)
   - **Authors**: Walid A. Hanafy, Qianlin Liang, Noman Bashir, David Irwin, Prashant Shenoy
   - **Summary**: This work introduces CarbonScaler, a system that exploits the elasticity of cloud workloads to optimize carbon efficiency. By dynamically adjusting server allocations based on real-time carbon intensity data, CarbonScaler reduces the carbon footprint of batch workloads. The system is implemented in Kubernetes and evaluated using real-world machine learning training and MPI jobs, achieving significant carbon savings compared to baseline methods.
   - **Year**: 2023

4. **Title**: Sustainable AIGC Workload Scheduling of Geo-Distributed Data Centers: A Multi-Agent Reinforcement Learning Approach (arXiv:2304.07948)
   - **Authors**: Siyue Zhang, Minrui Xu, Wei Yang Bryan Lim, Dusit Niyato
   - **Summary**: The paper proposes a multi-agent reinforcement learning algorithm for scheduling AI-generated content (AIGC) workloads across geographically distributed data centers. The approach aims to maximize GPU utilization while minimizing operational costs and carbon emissions. The proposed method demonstrates up to a 28.6% improvement in system utility over baseline algorithms.
   - **Year**: 2023

**Key Challenges**:

1. **Balancing Performance and Sustainability**: Achieving a trade-off between reducing energy consumption and carbon emissions while maintaining or improving job performance metrics such as throughput and latency remains a significant challenge.

2. **Dynamic and Unpredictable Energy Markets**: The variability in energy prices and carbon intensity over time and across regions complicates the development of effective scheduling algorithms that can adapt to these fluctuations.

3. **Complexity of Workload Dependencies**: Incorporating task precedence constraints and interdependencies into scheduling decisions adds complexity, as delaying one task can have cascading effects on subsequent tasks and overall job completion times.

4. **Scalability of Reinforcement Learning Solutions**: Implementing reinforcement learning-based schedulers in large-scale, real-world data center environments poses challenges related to scalability, convergence times, and computational overhead.

5. **Integration with Existing Infrastructure**: Deploying new scheduling algorithms requires seamless integration with existing data center management systems, which may have limitations or require significant modifications to accommodate advanced scheduling techniques. 